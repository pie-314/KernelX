"""
KernelX Intelligence Layer — RL Environment (OpenEnv structure)

Provides reset/step interface for training the Strategist policy via GRPO.
Uses the trained World Model to predict next_state given (state, action),
so the action actually drives dynamics during RL training.
"""

import json
import re
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

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


class WorldModelSimulator:
    """Uses the trained World Model to predict S_{t+1} given (S_t, action).

    This is the key fix: instead of nearest-neighbor lookup (which ignores
    the action), we run the World Model so the action actually drives
    dynamics during GRPO training.
    """

    def __init__(self, model_path: str, feature_names: list):
        self.feature_names = feature_names
        self.model = None
        self.tokenizer = None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import os

            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                # LoRA adapter — load base + adapter
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "data", "preprocessing_config.json")
                config = json.load(open(config_path))
                base_name = config["model"]["name"]
                base = AutoModelForCausalLM.from_pretrained(base_name, device_map="cpu")
                self.model = PeftModel.from_pretrained(base, model_path)
            else:
                # Merged model
                self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
            print(f"[WorldModelSimulator] Loaded from {model_path}")
        except Exception as e:
            print(f"[WorldModelSimulator] Failed to load: {e}. Falling back to analytical model.")
            self.model = None

    def _format_state(self, features: list) -> str:
        parts = []
        for name, val in zip(self.feature_names, features):
            if val == int(val):
                parts.append(f"{name}:{int(val)}")
            else:
                parts.append(f"{name}:{val:.2f}")
        return " | ".join(parts)

    def _parse_state(self, text: str) -> list:
        """Parse [NEXT_STATE] from model output back to float vector."""
        marker = "[NEXT_STATE]"
        idx = text.rfind(marker)
        if idx == -1:
            raise ValueError("No [NEXT_STATE] in output")
        payload = text[idx + len(marker):].split("<|end|>")[0].strip()
        values = []
        for part in payload.split("|"):
            part = part.strip()
            if ":" in part:
                values.append(float(part.split(":")[1]))
        return values

    def predict(self, state_features: list, action_value: float, pid: int = 0) -> list:
        """Predict next state using the World Model.

        The action is part of the prompt, so the model's prediction
        actually depends on what action was taken.
        """
        if self.model is None:
            return self._analytical_predict(state_features, action_value)

        state_str = self._format_state(state_features)
        prompt = (
            "<|system|>You are a Linux kernel simulator. "
            "Predict the next system state.<|end|>\n"
            f"<|user|>[STATE] {state_str}\n"
            f"[ACTION] {action_value:.4f}\n"
            f"[PID] {pid}\n"
            "Predict [NEXT_STATE]<|end|>\n"
            "<|assistant|>"
        )

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs, max_new_tokens=128,
                temperature=0.1, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            predicted = self._parse_state(text)
            if len(predicted) == len(state_features):
                return predicted
        except Exception:
            pass

        # Fallback to analytical model if parsing fails
        return self._analytical_predict(state_features, action_value)

    def _analytical_predict(self, state_features: list, action_value: float) -> list:
        """Physics-inspired fallback: action directly affects wait_us and exec_ns.

        This is NOT a nearest-neighbor lookup. The action changes the outcome:
          - Negative action (boost) → reduces wait_us, increases exec_ns
          - Positive action (demote) → increases wait_us slightly
          - Zero action → state evolves with small noise
        """
        import numpy as np
        predicted = list(state_features)
        wait_us = state_features[9]  # IDX_WAIT_US
        exec_ns = state_features[4]  # IDX_EXEC_NS

        if action_value < -0.1:
            # Boosting priority reduces wait time
            predicted[9] = max(1.0, wait_us * (1.0 - abs(action_value) * 0.4))
            predicted[4] = exec_ns + abs(action_value) * 0.05
        elif action_value > 0.1:
            # Demoting increases wait time slightly
            predicted[9] = wait_us * (1.0 + action_value * 0.15)
        else:
            # No intervention — small random walk
            predicted[9] = max(0.0, wait_us + np.random.normal(0, 1))

        # Context switches: boosting reduces churn, demoting may increase
        csw = state_features[8]  # IDX_CTX_SWITCHES
        predicted[8] = max(0.0, csw + action_value * 2.0 + np.random.normal(0, 0.5))

        return predicted


class KernelSchedulerEnv:
    """RL environment for training the Strategist via GRPO.

    Uses the trained World Model to simulate next_state, so the agent's
    action actually drives dynamics. Falls back to an analytical model
    if the World Model isn't available.
    """

    def __init__(
        self,
        data_path: str = "training/data/train.jsonl",
        max_steps: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 0.5,
        world_model_path: Optional[str] = None,
    ):
        self.records = [json.loads(l) for l in open(data_path) if l.strip()]
        self.max_steps = max_steps
        self.reward_computer = RewardComputer(alpha=alpha, beta=beta, gamma=gamma)

        # Load feature names from config
        import os
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "data", "preprocessing_config.json")
        try:
            config = json.load(open(config_path))
            feature_names = config["feature_names"]
        except Exception:
            feature_names = ["cpu", "prio", "sprio", "nprio", "exec_ns", "vrt", "migr", "cpus", "csw", "wt_us"]

        # World Model simulator
        if world_model_path:
            self.world_model = WorldModelSimulator(world_model_path, feature_names)
        else:
            # Try default paths
            for wm_path in ["training/models/world_model_final", "world_model_final"]:
                if os.path.exists(wm_path):
                    self.world_model = WorldModelSimulator(wm_path, feature_names)
                    break
            else:
                self.world_model = WorldModelSimulator("__none__", feature_names)
                print("[KernelSchedulerEnv] No World Model found. Using analytical simulator.")

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
        """Apply action, compute reward using World Model prediction.

        The World Model predicts next_state conditioned on the action,
        so different actions produce different outcomes and rewards.
        """
        current = self.records[self.current_idx + self.timestep]

        # Use World Model to predict next_state (action-dependent)
        predicted_next = self.simulate(current["state"], action.value)

        reward_breakdown = self.reward_computer.compute_total(
            state=current["state"],
            action=action,
            prev_action=self.prev_action,
            next_state=predicted_next,
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
        """Predict next_state using the World Model.

        The action is fed into the model's prompt, so the prediction
        actually depends on what scheduling decision was made. This is
        what makes GRPO training meaningful — the agent sees different
        outcomes for different actions on the same state.
        """
        return self.world_model.predict(state_features, action_value)
