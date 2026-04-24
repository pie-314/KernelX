import uuid
import time
import os
import numpy as np
import mmap
import zmq
from typing import Tuple, Optional
from openenv_core import Environment
from ..models import Observation, Action, State
from .policy import ManualPolicy

SHM_PATH = "/dev/shm/kernelx_state"
SHM_SIZE = 340
ZMQ_BRIDGE_URL = "tcp://127.0.0.1:5555"

class KernelXEnvironment(Environment[Observation, Action, State]):
    def __init__(self):
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.shm = None
        self.policy = ManualPolicy()  # Initialize manual policy
        
        # Initialize ZMQ Socket to talk to Rust Bridge
        try:
            self.zmq_ctx = zmq.Context()
            self.zmq_socket = self.zmq_ctx.socket(zmq.PUSH)
            self.zmq_socket.connect(ZMQ_BRIDGE_URL)
            print(f"[KernelX] Connected to Bridge Actuator at {ZMQ_BRIDGE_URL}")
        except Exception as e:
            print(f"[Warn] ZMQ Connection failed: {e}")
        
        # Try to connect to Real Metal (shared memory from kernel)
        if os.path.exists(SHM_PATH):
            try:
                fd = os.open(SHM_PATH, os.O_RDONLY)
                self.shm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
                print(f"[KernelX] Connected to live Telemetry SHM.")
            except Exception as e:
                print(f"[KernelX] SHM Access Failed: {e}. Falling back to Simulator.")
        else:
            print(f"[Warn] SHM file not found. Using simulated data.")
        
        print(f"[KernelX] OpenEnv Server Initialized. Session: {self.episode_id}")

    def reset(self) -> Observation:
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())
        return self._get_observation()

    def step(self, action: Optional[Action] = None) -> Observation:
        # Get current observation
        obs = self._get_observation()
        
        # If no action provided, use the manual policy to generate one
        if action is None:
            action = self.policy.decide(obs)
        
        # Apply Action to the Bridge (and then to the Kernel)
        self._apply_action(action)
        
        # Collect next observation
        time.sleep(0.01) 
        next_obs = self._get_observation()
        self.step_count += 1
        
        # Attach reward for telemetry
        next_obs.reward = self._calculate_reward(next_obs)
        return next_obs

    def _get_observation(self) -> Observation:
        if self.shm:
            self.shm.seek(0)
            data = self.shm.read(SHM_SIZE)
            
            # Unpack the 24D features (first 192 bytes, uint64)
            features = np.frombuffer(data[:192], dtype=np.uint64).astype(float).tolist()
            
            # Unpack PID (at offset 196)
            pid = int.from_bytes(data[196:200], "little")
            
            return Observation(
                features=features,
                timestamp=int(time.time_ns()),
                pid=pid,
                cpu=0
            )
        
        # Fallback to random if SHM is missing
        return Observation(
            features=list(np.random.rand(24).astype(float)),
            timestamp=int(time.time_ns()),
            pid=1234,
            cpu=0
        )

    def _apply_action(self, action: Action):
        """Send the priority weights to the Rust Bridge."""
        if self.shm:
            try:
                # Convert float decision [-100, 100] to weight
                weight = float(action.weights[0])
                
                # Peek at the current active PID from SHM
                self.shm.seek(196)
                pid_bytes = self.shm.read(4)
                active_pid = int.from_bytes(pid_bytes, "little")
                
                if active_pid > 0:
                    # Format: "PID:WEIGHT:CONFIDENCE:DRIFT:REASONING"
                    # Mocking confidence/drift here as placeholders for the AI to fill
                    confidence = 0.92
                    drift = 0.03
                    reason = "Priority boost for latency-sensitive task"
                    
                    cmd = f"{active_pid}:{weight}:{confidence}:{drift}:{reason}"
                    self.zmq_socket.send_string(cmd, zmq.NOBLOCK)
            except Exception as e:
                pass  # Silently ignore ZMQ errors

    def _calculate_reward(self, obs: Observation) -> float:
        # R = -latency (simplified reward)
        latency = obs.features[23] if len(obs.features) > 23 else 0.0
        return -float(latency)
    
    @property
    def state(self) -> State:
        """Return current state metadata."""
        return State(
            episode_id=self.episode_id,
            step_count=self.step_count,
            latency_p99=0.0,
            cpu_usage=0.0
        )
