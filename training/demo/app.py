"""
KernelX Intelligence Layer — Gradio Demo (Stage 7)

Judge-facing demo that shows:
  1. Baseline SmolLM2-360M output (untrained) on kernel states
  2. World Model predictions vs actual next states
  3. Strategist scheduling actions (action-only, no rationale for speed)
  4. Before/after metrics comparison

Usage:
    python -m training.demo.app \
        --strategist-model training/models/strategist_merged/strategist-q4km.gguf \
        --test-data training/data/test.jsonl

    # Without trained models (shows heuristic baseline):
    python -m training.demo.app --test-data training/data/test.jsonl --no-model
"""

import argparse
import json
import re
import sys
import time

from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.preprocess import (
    FEATURE_NAMES, format_state, load_config,
    IDX_WAIT_US, IDX_CTX_SWITCHES, IDX_EXEC_NS,
)
from training.environment.rewards import RewardComputer

CONFIG = load_config()

# ---------------------------------------------------------------------------
# Heuristic baseline policy (for comparison)
# ---------------------------------------------------------------------------

def heuristic_policy(state: list) -> float:
    """Simple rule-based policy for baseline comparison."""
    wait_us = state[IDX_WAIT_US]
    csw = state[IDX_CTX_SWITCHES]

    if wait_us > 15:
        return -0.6
    elif csw > 10:
        return -0.3
    else:
        return 0.05

# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class StrategistWrapper:
    """Wraps a HF or GGUF model for use in the demo."""

    def __init__(self, model_path: str):
        if model_path.endswith(".gguf"):
            from llama_cpp import Llama
            self.llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)
            self.backend = "gguf"
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="float32", device_map="cpu"
            )
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.backend = "hf"

    def predict(self, state: list, pid: int, cpu: int) -> tuple:
        state_str = format_state(state)
        prompt = (
            "<|system|>You are a Linux kernel scheduling strategist. "
            "Given the current system state, output a scheduling action.<|end|>\n"
            f"<|user|>[STATE] {state_str}\n"
            f"[PID] {pid} [CPU] {cpu}\n"
            "[ACTION]<|end|>\n"
            "<|assistant|>"
        )

        start = time.perf_counter()

        if self.backend == "gguf":
            output = self.llm(prompt, max_tokens=8, temperature=0.2)
            text = output["choices"][0]["text"]
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            out = self.model.generate(
                **inputs, max_new_tokens=8, temperature=0.3,
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id,
            )
            full = self.tokenizer.decode(out[0], skip_special_tokens=False)
            text = full.split("<|assistant|>")[-1] if "<|assistant|>" in full else full

        latency = (time.perf_counter() - start) * 1000

        action_match = re.search(r"([-+]?\d*\.?\d+)", text)
        action = float(action_match.group(1)) if action_match else 0.0
        action = max(-1.0, min(1.0, action))

        return action, latency


class WorldModelWrapper:
    """Wraps a GGUF or HF model for world model predictions."""

    def __init__(self, model_path: str):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)

    def predict_next_state(self, state: list, action: float, pid: int) -> list:
        state_str = format_state(state)
        prompt = (
            "<|system|>You are a Linux kernel simulator. "
            "Predict the next system state.<|end|>\n"
            f"<|user|>[STATE] {state_str}\n"
            f"[ACTION] {action:.4f}\n"
            f"[PID] {pid}\n"
            "Predict [NEXT_STATE]<|end|>\n"
            "<|assistant|>"
        )

        output = self.llm(prompt, max_tokens=128, temperature=0.1)
        text = output["choices"][0]["text"]

        # Parse predicted state
        values = []
        for part in text.split("|"):
            part = part.strip()
            if ":" in part:
                try:
                    values.append(float(part.split(":")[1]))
                except ValueError:
                    pass

        if len(values) == len(FEATURE_NAMES):
            return values
        return state  # fallback: return same state

# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------

def run_comparison(
    record: dict,
    strategist: Optional[StrategistWrapper],
    reward_computer: RewardComputer,
) -> dict:
    """Run baseline vs trained comparison on a single transition."""
    state = record["state"]
    next_state = record["next_state"]
    pid = record["pid"]
    cpu = record["cpu"]

    # Heuristic baseline
    h_action = heuristic_policy(state)
    h_reward = reward_computer.compute_total(
        state=state, action=h_action, prev_action=0.0,
        next_state=next_state,
    )

    result = {
        "state": format_state(state),
        "actual_next_state": format_state(next_state),
        "pid": pid,
        "cpu": cpu,
        "heuristic": {
            "action": h_action,
            "reward": h_reward,
        },
    }

    # Trained strategist
    if strategist:
        s_action, s_latency = strategist.predict(state, pid, cpu)
        s_reward = reward_computer.compute_total(
            state=state, action=s_action, prev_action=0.0,
            next_state=next_state,
        )
        result["strategist"] = {
            "action": s_action,
            "latency_ms": s_latency,
            "reward": s_reward,
        }

    return result

# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_gradio_app(
    test_records: list,
    strategist: Optional[StrategistWrapper],
    world_model: Optional[WorldModelWrapper],
):
    """Build and return the Gradio interface."""
    import gradio as gr

    reward_computer = RewardComputer(alpha=1.0, beta=2.0, gamma=0.5)

    def analyze_state(record_idx: int):
        """Run analysis on a selected test record."""
        idx = int(record_idx) % len(test_records)
        record = test_records[idx]
        result = run_comparison(record, strategist, reward_computer)

        # Format output
        lines = []
        lines.append(f"## State #{idx}")
        lines.append(f"**PID:** {result['pid']}  |  **CPU:** {result['cpu']}")
        lines.append(f"**Current State:** `{result['state']}`")
        lines.append(f"**Actual Next State:** `{result['actual_next_state']}`")
        lines.append("")

        # Heuristic
        h = result["heuristic"]
        lines.append("### Heuristic Baseline")
        lines.append(f"- **Action:** {h['action']:.4f}")
        lines.append(f"- **Total Reward:** {h['reward']['total']:.4f}")
        lines.append(f"  - Throughput: {h['reward']['throughput']:.4f}")
        lines.append(f"  - Latency: {h['reward']['latency']:.4f}")
        lines.append(f"  - Stability: {h['reward']['stability']:.4f}")
        lines.append("")

        # Strategist
        if "strategist" in result:
            s = result["strategist"]
            lines.append("### Trained Strategist")
            lines.append(f"- **Action:** {s['action']:.4f}")
            lines.append(f"- **Inference Latency:** {s['latency_ms']:.1f}ms")
            lines.append(f"- **Total Reward:** {s['reward']['total']:.4f}")
            lines.append(f"  - Throughput: {s['reward']['throughput']:.4f}")
            lines.append(f"  - Latency: {s['reward']['latency']:.4f}")
            lines.append(f"  - Stability: {s['reward']['stability']:.4f}")
            lines.append("")

            # Comparison
            delta = s["reward"]["total"] - h["reward"]["total"]
            direction = "better" if delta > 0 else "worse"
            lines.append(f"### Comparison")
            lines.append(f"Strategist is **{delta:+.4f}** reward ({direction} than heuristic)")
        else:
            lines.append("*No trained model loaded. Run with --strategist-model to compare.*")

        return "\n".join(lines)

    def batch_comparison():
        """Run comparison across all test records and report aggregate stats."""
        h_rewards = []
        s_rewards = []
        s_latencies = []

        n = min(50, len(test_records))  # cap for speed
        for i in range(n):
            result = run_comparison(test_records[i], strategist, reward_computer)
            h_rewards.append(result["heuristic"]["reward"]["total"])
            if "strategist" in result:
                s_rewards.append(result["strategist"]["reward"]["total"])
                s_latencies.append(result["strategist"]["latency_ms"])

        s_mean = f"{np.mean(s_rewards):.4f}" if s_rewards else "N/A"
        s_std = f"{np.std(s_rewards):.4f}" if s_rewards else "N/A"

        lines = [f"## Batch Comparison ({n} samples)\n"]
        lines.append("| Metric | Heuristic | Strategist |")
        lines.append("|--------|-----------|------------|")
        lines.append(f"| Mean Reward | {np.mean(h_rewards):.4f} | {s_mean} |")
        lines.append(f"| Std Reward | {np.std(h_rewards):.4f} | {s_std} |")
        if s_latencies:
            lines.append(f"| Mean Latency | N/A | {np.mean(s_latencies):.1f}ms |")
            lines.append(f"| P95 Latency | N/A | {np.percentile(s_latencies, 95):.1f}ms |")
        if s_rewards:
            win_rate = sum(1 for s, h in zip(s_rewards, h_rewards) if s > h) / len(s_rewards)
            lines.append(f"| Win Rate | — | {win_rate*100:.1f}% |")

        return "\n".join(lines)

    # Build Gradio UI
    with gr.Blocks(title="KernelX Intelligence Layer") as app:
        gr.Markdown("# KernelX Intelligence Layer Demo")
        gr.Markdown("Compare heuristic baseline vs trained Strategist on real kernel states.")

        with gr.Row():
            record_slider = gr.Slider(
                minimum=0, maximum=len(test_records) - 1,
                step=1, value=0, label="Test Record Index"
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

        output_md = gr.Markdown()
        analyze_btn.click(fn=analyze_state, inputs=[record_slider], outputs=[output_md])

        gr.Markdown("---")
        batch_btn = gr.Button("Run Batch Comparison (50 samples)")
        batch_output = gr.Markdown()
        batch_btn.click(fn=batch_comparison, outputs=[batch_output])

    return app

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KernelX Gradio Demo")
    parser.add_argument("--test-data", required=True, help="Path to test.jsonl")
    parser.add_argument("--strategist-model", default=None, help="Path to strategist model (HF dir or GGUF file)")
    parser.add_argument("--world-model", default=None, help="Path to world model (HF dir or GGUF file)")
    parser.add_argument("--no-model", action="store_true", help="Run without trained models")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    # Load test data
    records = [json.loads(l) for l in open(args.test_data) if l.strip()]
    print(f"Loaded {len(records)} test records")

    # Load models
    strategist = None
    world_model = None

    if not args.no_model:
        if args.strategist_model:
            print(f"Loading Strategist: {args.strategist_model}")
            strategist = StrategistWrapper(args.strategist_model)
        if args.world_model:
            print(f"Loading World Model: {args.world_model}")
            world_model = WorldModelWrapper(args.world_model)

    app = build_gradio_app(records, strategist, world_model)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
