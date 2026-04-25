"""
KernelX Intelligence Layer — RL Environment (OpenEnv structure)

Provides reset/step interface for training the Strategist policy via GRPO.
Replays recorded transitions from the preprocessed JSONL dataset and
computes multi-objective rewards.
"""

import json
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from .rewards import RewardComputer


@dataclass
class KernelState:
    """Observation wrapper for the RL environment."""
    features: List[float]   # active features (10D after preprocessing)
    pid: int
    cpu: int
    timestep: int
    prev_action: float


@dataclass
class KernelAction:
    """Action output from the Strategist."""
    value: float            # scheduling weight in [-1.0, 1.0]


class KernelSchedulerEnv:
    """Offline RL environment that replays recorded kernel transitions.

    Each episode starts at a random position in the dataset and runs for
    max_steps transitions.  The reward is computed from the multi-objective
    RewardComputer.
    """

    def __init__(
        self,
        data_path: str = "training/data/train.jsonl",
        max_steps: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 0.5,
    ):
        self.records = [json.loads(l) for l in open(data_path) if l.strip()]
        self.max_steps = max_steps
        self.reward_computer = RewardComputer(alpha=alpha, beta=beta, gamma=gamma)

        # Episode state
        self.timestep = 0
        self.current_idx = 0
        self.prev_action = 0.0

        if len(self.records) < max_steps + 1:
            raise ValueError(
                f"Dataset has {len(self.records)} records but max_steps={max_steps} "
                f"requires at least {max_steps + 1}"
            )

    def reset(self) -> KernelState:
        """Start a fresh episode from a random point in the dataset."""
        self.timestep = 0
        self.current_idx = random.randint(0, len(self.records) - self.max_steps - 1)
        self.prev_action = 0.0
        return self._get_state()

    def step(self, action: KernelAction) -> Tuple[KernelState, dict, bool]:
        """Apply action, compute reward, advance to next state.

        Returns:
            next_state: The new KernelState after the transition
            reward_breakdown: Dict with 'total' and per-component rewards
            done: Whether the episode has ended
        """
        current = self.records[self.current_idx + self.timestep]
        next_idx = self.current_idx + self.timestep + 1
        next_rec = self.records[next_idx] if next_idx < len(self.records) else current

        reward_breakdown = self.reward_computer.compute_total(
            state=current["state"],
            action=action,
            prev_action=self.prev_action,
            next_state=next_rec["state"],
        )

        self.timestep += 1
        self.prev_action = action.value
        done = self.timestep >= self.max_steps

        return self._get_state(), reward_breakdown, done

    def _get_state(self) -> KernelState:
        """Read the current state from the dataset."""
        rec = self.records[self.current_idx + self.timestep]
        return KernelState(
            features=rec["state"],
            pid=rec["pid"],
            cpu=rec["cpu"],
            timestep=self.timestep,
            prev_action=self.prev_action,
        )

    def simulate(self, state_features: list, action_value: float) -> list:
        """Lightweight next-state lookup for reward_fn during GRPO.

        Finds the nearest recorded state in the dataset and returns
        its recorded next_state. This is a simple approximation;
        the World Model provides higher-fidelity simulation.
        """
        import numpy as np

        state_arr = np.array(state_features)
        best_dist = float("inf")
        best_next = state_features  # fallback

        # Sample a subset to keep this fast
        sample_size = min(500, len(self.records))
        indices = random.sample(range(len(self.records)), sample_size)

        for idx in indices:
            rec = self.records[idx]
            dist = float(np.linalg.norm(state_arr - np.array(rec["state"])))
            if dist < best_dist:
                best_dist = dist
                best_next = rec["next_state"]

        return best_next
