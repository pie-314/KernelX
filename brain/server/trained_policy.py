"""
KernelX Trained ML Policy — GGUF Strategist Integration

Replaces the ManualPolicy with the trained SmolLM2-360M strategist model.
Loads the quantized GGUF model and converts kernel observations into
scheduling actions via the same interface the environment expects.

Usage:
    Replace ManualPolicy in kernelx_environment.py:
        from .trained_policy import TrainedPolicy
        self.policy = TrainedPolicy("path/to/strategist-q4km.gguf")
"""

import re
import numpy as np

try:
    from ..models import Action, Observation
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import Action, Observation


# ---------------------------------------------------------------------------
# Feature mapping (must match training/data/preprocessing_config.json)
# ---------------------------------------------------------------------------

SYMLOG_FEATURES = [4, 5, 6]
ACTIVE_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7, 12, 23]
FEATURE_NAMES = ["cpu", "prio", "sprio", "nprio", "exec_ns", "vrt", "migr", "cpus", "csw", "wt_us"]


def symmetric_log(x):
    return float(np.sign(x) * np.log1p(np.abs(x)))


def preprocess_observation(features):
    """Convert raw 24D kernel features to 10D active features with scaling."""
    f = list(features)
    for idx in SYMLOG_FEATURES:
        f[idx] = symmetric_log(f[idx])
    return [f[i] for i in ACTIVE_FEATURES]


def format_state(active_features):
    """Format 10D features as compact text for the LLM prompt."""
    parts = []
    for name, val in zip(FEATURE_NAMES, active_features):
        if val == int(val):
            parts.append(f"{name}:{int(val)}")
        else:
            parts.append(f"{name}:{val:.2f}")
    return " | ".join(parts)


def action_to_weights(action_value):
    """Convert a single action float [-1, 1] to 4 priority weights [-100, 100].

    Mapping:
        action < 0 = boost priority (negative = promote current task)
        action > 0 = demote priority (positive = yield to others)
        action ~ 0 = neutral

    Weight groups: [real-time, interactive, batch, idle]
    """
    a = float(np.clip(action_value, -1.0, 1.0))
    scale = abs(a) * 100.0

    if a < -0.1:
        # Boost: promote real-time and interactive
        return [scale, scale * 0.6, -scale * 0.5, -scale * 0.3]
    elif a > 0.1:
        # Demote: suppress real-time, promote batch
        return [-scale * 0.3, -scale * 0.5, scale * 0.6, scale]
    else:
        # Neutral: minimal adjustment
        return [5.0, 2.0, -2.0, -5.0]


class TrainedPolicy:
    """GGUF-based trained scheduling policy for KernelX."""

    def __init__(self, model_path: str, n_threads: int = 4, max_tokens: int = 8):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=n_threads,
            verbose=False,
        )
        self.max_tokens = max_tokens
        self.last_action = 0.0
        print(f"[TrainedPolicy] Loaded GGUF model: {model_path}")

    def decide(self, obs: Observation) -> Action:
        """Make a scheduling decision from a kernel observation.

        Args:
            obs: Observation with 24D features from kernel

        Returns:
            Action with 4 priority weights [-100, 100]
        """
        # Preprocess: 24D raw -> 10D active (symlog scaled)
        active = preprocess_observation(obs.features)
        state_str = format_state(active)

        # Build prompt
        prompt = (
            "<|system|>You are a Linux kernel scheduling strategist. "
            "Given the current system state, output a scheduling action.<|end|>\n"
            f"<|user|>[STATE] {state_str}\n"
            f"[PID] {obs.pid} [CPU] {obs.cpu}\n"
            "[ACTION]<|end|>\n"
            "<|assistant|>"
        )

        # Run inference
        output = self.llm(prompt, max_tokens=self.max_tokens, temperature=0.2)
        text = output["choices"][0]["text"]

        # Parse action
        match = re.search(r"([-+]?\d*\.?\d+)", text)
        if match:
            action_val = float(match.group(1))
            action_val = max(-1.0, min(1.0, action_val))
        else:
            action_val = 0.0

        self.last_action = action_val

        # Convert single action to 4 weights
        weights = action_to_weights(action_val)
        weights = [float(np.clip(w, -100.0, 100.0)) for w in weights]

        return Action(weights=weights)
