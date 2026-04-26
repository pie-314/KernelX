"""
KernelX Intelligence Layer — LLM-Based Grader

Evaluates the Strategist's performance by analyzing both the system
telemetry (quantitative) and the reasoning CoT (qualitative).
Returns a normalized OpenEnv score [0.01, 0.99].
"""

import re
import numpy as np
from typing import Dict, Any
from brain.models import Observation, Action

class LLMGrader:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.llm = None
        if model_path:
            from llama_cpp import Llama
            self.llm = Llama(model_path=model_path, n_ctx=512, verbose=False)

    def calculate_score(self, 
                        obs: Observation, 
                        next_obs: Observation, 
                        reasoning: str) -> Dict[str, Any]:
        """
        Calculates a multi-faceted score for an action.
        
        Returns:
            {
                "score": float (0.01 - 0.99),
                "feedback": str,
                "metrics": dict
            }
        """
        # 1. Quantitative: Latency Improvement
        # Lower latency = higher score
        wait_t0 = obs.features[23]
        wait_t1 = next_obs.features[23]
        latency_improvement = (wait_t0 - wait_t1) / (wait_t0 + 1.0)
        
        # 2. Qualitative: Reasoning Coherence
        # Use LLM to evaluate if reasoning matches the state
        strategic_alignment = 0.5 # Default
        if self.llm and reasoning:
            strategic_alignment = self._llm_evaluate_reasoning(next_obs, reasoning)

        # 3. Aggregation & Normalization
        # Weights: 60% performance, 40% reasoning alignment
        raw_score = (latency_improvement * 0.6) + (strategic_alignment * 0.4)
        
        # Map [-1, 1] or similar to [0.01, 0.99]
        normalized = 0.5 + (raw_score * 0.4)
        final_score = float(np.clip(normalized, 0.01, 0.99))

        return {
            "score": final_score,
            "feedback": self._generate_feedback(final_score, wait_t1),
            "metrics": {
                "latency_delta": wait_t0 - wait_t1,
                "alignment": strategic_alignment
            }
        }

    def _llm_evaluate_reasoning(self, obs: Observation, reasoning: str) -> float:
        """Heuristic evaluation if no model, LLM evaluation if present."""
        if not self.llm:
            # Heuristic: Penalize empty reasoning or generic strings
            if len(reasoning) < 10 or "nudge" in reasoning.lower():
                return 0.3
            return 0.7

        prompt = (
            f"<|system|>You are a Kernel Expert Auditor. Evaluate if the following "
            f"reasoning makes sense for a system with P99 latency of {obs.features[23]}us.<|end|>\n"
            f"<|user|>Reasoning: {reasoning}\n"
            f"Score from 0.0 to 1.0. Output only the number.<|end|>\n"
            f"<|assistant|>"
        )
        try:
            output = self.llm(prompt, max_tokens=8, stop=["<|end|>"])
            text = output["choices"][0]["text"]
            match = re.search(r"(\d?\.\d+)", text)
            return float(match.group(1)) if match else 0.5
        except:
            return 0.5

    def _generate_feedback(self, score: float, wait_us: float) -> str:
        if score > 0.8: return "Excellent scheduling precision."
        if score > 0.5: return "Acceptable latency management."
        return f"Suboptimal performance. Latency at {wait_us}us is too high."
