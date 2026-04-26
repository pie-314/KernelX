"""
Manual rule-based policy engine for KernelX.

This provides heuristic-based scheduling decisions until the AI model is trained.
It uses the 24D observation vector to make priority adjustment decisions.

Feature vector breakdown (24 dimensions):
  [0-2]:   Per-CPU metrics (load, runqueue, etc.)
  [3-6]:   Memory metrics (used, free, buffers, cache)
  [7-10]:  I/O metrics (read ops, write ops, read bytes, write bytes)
  [11-14]: Interrupt/context metrics
  [15-18]: Process-specific metrics
  [19-23]: Latency/performance metrics
  
The policy outputs 4 action weights (for 4 process groups), each in range [-100, 100].
"""

from typing import List
import numpy as np
import sys
import os

from brain.models import Action, Observation


class ManualPolicy:
    """Rule-based policy that makes scheduling decisions based on kernel metrics."""
    
    def __init__(self):
        self.last_latency = 0.0
        self.latency_history = []
        self.max_history = 10
        self.last_reasoning = "Initializing heuristic policy..."
        
    def decide(self, obs: Observation) -> Action:
        """
        Given a kernel observation, produce scheduling action weights.
        
        Args:
            obs: Observation with 24D features from kernel
            
        Returns:
            Action with 4 priority weights for process groups
        """
        features = obs.features
        
        # Extract key metrics from feature vector
        cpu_load = self._extract_cpu_load(features)
        memory_pressure = self._extract_memory_pressure(features)
        io_activity = self._extract_io_activity(features)
        latency = self._extract_latency(features)
        
        # Track latency trend
        self.latency_history.append(latency)
        if len(self.latency_history) > self.max_history:
            self.latency_history.pop(0)
        
        latency_trend = self._calc_latency_trend()
        
        # Compute weights for 4 groups: [real-time, interactive, batch, idle]
        weights = self._compute_weights(
            cpu_load=cpu_load,
            memory_pressure=memory_pressure,
            io_activity=io_activity,
            latency=latency,
            latency_trend=latency_trend
        )
        
        self.last_reasoning = f"Heuristic: CPU={cpu_load:.2f}, Latency={latency:.1f}us. Applying priority shift."
        
        return Action(weights=weights)
    
    def _extract_cpu_load(self, features: List[float]) -> float:
        """Extract normalized CPU load (0.0 = idle, 1.0+ = overloaded)."""
        # features[0] is typically CPU load or runqueue depth
        # Normalize to 0-1 scale (assume max load ~4 on typical systems)
        cpu_metric = features[0] if len(features) > 0 else 0.0
        return min(cpu_metric / 4.0, 1.0)
    
    def _extract_memory_pressure(self, features: List[float]) -> float:
        """Extract memory pressure (0.0 = plenty, 1.0 = critical)."""
        # features[3-6] are memory metrics
        # Simple heuristic: use ratio of used to total memory
        if len(features) > 6:
            mem_used = features[4]
            mem_free = features[5]
            total = mem_used + mem_free
            if total > 0:
                return mem_used / total
        return 0.0
    
    def _extract_io_activity(self, features: List[float]) -> float:
        """Extract I/O activity level (0.0 = idle, 1.0 = saturated)."""
        # features[7-10] are I/O metrics
        # Use read+write ops as proxy for I/O load
        if len(features) > 9:
            read_ops = features[7]
            write_ops = features[8]
            total_ops = read_ops + write_ops
            # Normalize to 0-1 (assume ~10k ops/sec is moderate)
            return min(total_ops / 10000.0, 1.0)
        return 0.0
    
    def _extract_latency(self, features: List[float]) -> float:
        """Extract latency metric (lower is better)."""
        # features[23] is typically a latency or performance metric
        if len(features) > 23:
            return features[23]
        return 0.0
    
    def _calc_latency_trend(self) -> float:
        """
        Calculate latency trend: 0 = stable, >0 = increasing, <0 = decreasing.
        """
        if len(self.latency_history) < 2:
            return 0.0
        
        recent = self.latency_history[-5:]  # Last 5 samples
        if len(recent) < 2:
            return 0.0
        
        deltas = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        return sum(deltas) / len(deltas)
    
    def _compute_weights(
        self,
        cpu_load: float,
        memory_pressure: float,
        io_activity: float,
        latency: float,
        latency_trend: float,
    ) -> List[float]:
        """
        Compute priority weights for 4 process groups.
        
        Group layout:
          [0] = real-time/critical
          [1] = interactive/UI
          [2] = batch/background
          [3] = idle/best-effort
        
        Weight scale: -100 (lowest priority) to +100 (highest priority)
        """
        weights = [0.0, 0.0, 0.0, 0.0]
        
        # 1. Very High Load: Emergency Throttling of the current task to save system stability
        if cpu_load > 0.9:
            weights = [40.0, 20.0, 10.0, 0.0] # POSITIVE = Throttle/Demote
        
        # 2. High CPU load: aggressive prioritization
        elif cpu_load > 0.7:
            weights = [-50.0, -30.0, 50.0, 30.0]
        
        # 3. Moderate CPU load: prioritize interactive
        elif cpu_load > 0.3:
            weights = [-20.0, -15.0, 20.0, 15.0]
            
        # 4. Under normal load: light boost
        else:
            weights = [-10.0, -5.0, 5.0, 10.0]
        
        # 4. High memory pressure: reduce batch workloads
        if memory_pressure > 0.8:
            weights[2] += 30.0  # penalize batch
            weights[0] -= 10.0  # boost critical for cleanup
        
        # 5. High I/O activity: boost I/O-bound processes slightly
        if io_activity > 0.7:
            weights[1] -= 15.0  # interactive often does I/O
        
        # 6. Rising latency: boost critical processes
        if latency_trend > 0.5:
            weights[0] -= 20.0
            weights[1] -= 10.0
            weights[2] += 10.0
        
        # 7. High latency spike: emergency boost
        if latency > 100.0:  # arbitrary threshold
            weights[0] = -80.0
            weights[1] = -40.0
            weights[2] = 60.0
            weights[3] = 80.0
        
        # Clamp all weights to valid range
        weights = [float(np.clip(w, -100.0, 100.0)) for w in weights]
        
        return weights
