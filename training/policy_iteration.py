#!/usr/bin/env python3
"""
KernelX — Automated Policy Iteration Loop

Orchestrates the collect → train → deploy → repeat cycle:
  1. Collect: Run current policy, bridge records trajectories to JSONL
  2. Train:  Preprocess data, run warm-start SFT + GRPO
  3. Deploy: Quantize to GGUF, swap policy in the brain server
  4. Repeat: Go back to step 1 with the improved policy

Usage:
    # Run one full iteration (collect 5min → train → deploy)
    python3 -m training.policy_iteration \
        --trajectories-path /path/to/trajectories.json \
        --model-output training/models/strategist-q4km.gguf \
        --collect-duration 300

    # Skip collection (already have data)
    python3 -m training.policy_iteration \
        --trajectories-path data/trajectories.json \
        --model-output training/models/strategist-q4km.gguf \
        --skip-collect

    # Full automated loop (N iterations)
    python3 -m training.policy_iteration \
        --trajectories-path /path/to/trajectories.json \
        --iterations 3 \
        --collect-duration 300
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def collect_trajectories(
    trajectories_path: str,
    duration_seconds: int = 300,
    brain_url: str = "http://localhost:8000",
):
    """Phase A: Run the current policy and let the bridge collect trajectories.

    The bridge's TrajectoryManager writes transitions to JSONL automatically.
    This function just waits for the specified duration while the system runs.
    """
    print(f"\n{'='*60}")
    print(f" PHASE A: Collecting trajectories ({duration_seconds}s)")
    print(f"{'='*60}")
    print(f"  Trajectories file: {trajectories_path}")
    print(f"  Duration: {duration_seconds}s")

    # Check if bridge is writing to the trajectories file
    if not os.path.exists(trajectories_path):
        print(f"  WARNING: {trajectories_path} does not exist yet.")
        print(f"  Make sure the bridge is running and writing trajectories.")

    initial_size = os.path.getsize(trajectories_path) if os.path.exists(trajectories_path) else 0
    initial_lines = sum(1 for _ in open(trajectories_path)) if os.path.exists(trajectories_path) else 0

    print(f"  Starting collection (existing: {initial_lines} transitions, {initial_size / 1024:.0f}KB)")
    print(f"  Waiting {duration_seconds}s for new data...\n")

    # Wait while the system collects data
    interval = min(30, duration_seconds // 5)
    elapsed = 0
    while elapsed < duration_seconds:
        time.sleep(interval)
        elapsed += interval
        current_size = os.path.getsize(trajectories_path) if os.path.exists(trajectories_path) else 0
        current_lines = sum(1 for _ in open(trajectories_path)) if os.path.exists(trajectories_path) else 0
        new_lines = current_lines - initial_lines
        print(f"  [{elapsed}s/{duration_seconds}s] {new_lines} new transitions collected")

    final_lines = sum(1 for _ in open(trajectories_path)) if os.path.exists(trajectories_path) else 0
    total_new = final_lines - initial_lines
    print(f"\n  Collection complete: {total_new} new transitions ({final_lines} total)")

    if total_new < 100:
        print("  WARNING: Very few new transitions. Training may not improve the model.")

    return final_lines


def train_model(
    trajectories_path: str,
    output_dir: str = "training/data",
    model_output: str = "training/models",
    max_train_samples: int = 100,
    warmstart_examples: int = 100,
):
    """Phase B: Preprocess data and train the strategist.

    Steps:
      1. Preprocess raw JSONL → train/val/test splits
      2. Warm-start SFT (teaches output format)
      3. Export & quantize to GGUF
    """
    print(f"\n{'='*60}")
    print(f" PHASE B: Training model")
    print(f"{'='*60}")

    output_dir = Path(output_dir)
    model_output = Path(model_output)

    # Step 1: Preprocess
    print("\n--- Step 1: Preprocessing ---")
    from training.data.preprocess import run_pipeline
    train_data, val_data, test_data = run_pipeline(
        trajectories_path, str(output_dir), audit=False
    )
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Step 2: Train strategist (warm-start SFT)
    print("\n--- Step 2: Training Strategist ---")
    import numpy as np
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    from training.data.preprocess import (
        FEATURE_NAMES, format_state, load_config,
        IDX_WAIT_US, IDX_CTX_SWITCHES,
    )

    config = load_config()
    MODEL_NAME = config["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Smart sample selection: pick diverse states covering the full distribution
    # Stratify by wait_us (latency) to ensure we see low/medium/high latency cases
    import random
    pool = train_data if len(train_data) <= max_train_samples else random.sample(train_data, min(max_train_samples, len(train_data)))

    # Sort by wait_us and pick evenly spaced samples across the distribution
    pool_sorted = sorted(pool, key=lambda r: r["state"][IDX_WAIT_US])
    n = min(warmstart_examples, len(pool_sorted))
    if n < len(pool_sorted):
        step = len(pool_sorted) / n
        samples = [pool_sorted[int(i * step)] for i in range(n)]
    else:
        samples = pool_sorted
    print(f"  Smart sampling: {n} samples spanning wait_us range "
          f"[{samples[0]['state'][IDX_WAIT_US]:.0f} - {samples[-1]['state'][IDX_WAIT_US]:.0f}]")

    examples = []
    for rec in samples:
        state = rec["state"]
        wait_us = state[IDX_WAIT_US]
        csw = state[IDX_CTX_SWITCHES]

        if wait_us > 15:
            action = -0.6
        elif csw > 10:
            action = -0.3
        elif wait_us < 3:
            action = 0.1
        else:
            action = 0.05

        state_str = format_state(state)
        prompt = (
            "<|system|>You are a Linux kernel scheduling strategist. "
            "Given the current system state, output a scheduling action.<|end|>\n"
            f"<|user|>[STATE] {state_str}\n"
            f"[PID] {rec['pid']} [CPU] {rec['cpu']}\n"
            "[ACTION]<|end|>\n"
            f"<|assistant|>{action:.4f}<|end|>"
        )
        examples.append({"text": prompt})

    dataset = Dataset.from_list(examples)
    print(f"  Warm-start dataset: {len(dataset)} examples")

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    ws_dir = model_output / "iteration_warmstart"
    ws_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(ws_dir / "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,  # MPS compatible
        max_seq_length=512,
        logging_steps=5,
        save_steps=100,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset, peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(ws_dir))
    tokenizer.save_pretrained(str(ws_dir))
    print(f"  Model saved to {ws_dir}")

    return str(ws_dir)


def export_gguf(adapter_path: str, output_path: str):
    """Phase B (cont): Merge LoRA and quantize to GGUF Q4_K_M."""
    print("\n--- Step 3: Export to GGUF ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from training.data.preprocess import load_config

    config = load_config()
    MODEL_NAME = config["model"]["name"]

    # Merge LoRA
    print("  Merging LoRA weights...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="float32", device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()

    merged_dir = Path(adapter_path).parent / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # Convert to GGUF F16
    f16_path = str(Path(output_path).with_suffix(".f16.gguf"))
    print("  Converting to GGUF F16...")
    llama_cpp_dir = Path("/tmp/llama_cpp_tools")
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print("  Cloning llama.cpp...")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggerganov/llama.cpp",
                        str(llama_cpp_dir)], check=True, capture_output=True)

    subprocess.run([
        sys.executable, str(convert_script),
        str(merged_dir), "--outfile", f16_path, "--outtype", "f16"
    ], check=True, capture_output=True)

    # Quantize to Q4_K_M
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print("  Building llama-quantize...")
        build_dir = llama_cpp_dir / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=str(build_dir), check=True, capture_output=True)
        subprocess.run(["cmake", "--build", ".", "--target", "llama-quantize", "-j"],
                       cwd=str(build_dir), check=True, capture_output=True)

    print(f"  Quantizing to Q4_K_M...")
    subprocess.run([str(quantize_bin), f16_path, output_path, "Q4_K_M"],
                   check=True, capture_output=True)

    # Cleanup F16
    os.remove(f16_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  GGUF saved: {output_path} ({size_mb:.0f}MB)")

    return output_path


def deploy_model(gguf_path: str, brain_url: str = "http://localhost:8000"):
    """Phase C: Hot-swap the model in the running brain server.

    Calls the /reload-policy endpoint on the brain server. If the server
    is not running, prints manual instructions instead.
    """
    print(f"\n{'='*60}")
    print(f" PHASE C: Deploying model")
    print(f"{'='*60}")
    print(f"  Model: {gguf_path}")
    print(f"  Size:  {os.path.getsize(gguf_path) / (1024*1024):.0f}MB")

    # Convert to absolute path for the brain server
    abs_path = os.path.abspath(gguf_path)

    # Try to hot-reload via the brain server API
    try:
        import urllib.request
        import urllib.parse
        url = f"{brain_url}/reload-policy?model_path={urllib.parse.quote(abs_path)}"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("status") == "ok":
                print(f"  Hot-reloaded successfully via {brain_url}")
                print(f"  Active policy: {result.get('policy_path')}")
                return True
            else:
                print(f"  Reload returned: {result}")
    except Exception as e:
        print(f"  Brain server not reachable ({e})")

    # Fallback: manual instructions
    print()
    print("  Auto-reload failed. To activate manually:")
    print(f"    Restart brain: cd brain && python3 -m server.app")
    print(f"    Model will auto-load from: {abs_path}")
    return False


def validate_model(gguf_path: str, test_data_path: str, n_samples: int = 50):
    """Quick validation: run the new model on test data and report metrics."""
    print(f"\n--- Validation ({n_samples} samples) ---")
    from llama_cpp import Llama
    import re
    import numpy as np

    llm = Llama(model_path=gguf_path, n_ctx=512, n_threads=4, verbose=False)

    from training.data.preprocess import FEATURE_NAMES, format_state

    records = [json.loads(l) for l in open(test_data_path) if l.strip()][:n_samples]

    latencies = []
    actions = []
    format_ok = 0

    for rec in records:
        state_str = format_state(rec["state"])
        prompt = (
            "<|system|>You are a Linux kernel scheduling strategist. "
            "Given the current system state, output a scheduling action.<|end|>\n"
            f"<|user|>[STATE] {state_str}\n"
            f"[PID] {rec['pid']} [CPU] {rec['cpu']}\n"
            "[ACTION]<|end|>\n"
            "<|assistant|>"
        )

        start = time.perf_counter()
        output = llm(prompt, max_tokens=8, temperature=0.2)
        latencies.append((time.perf_counter() - start) * 1000)

        text = output["choices"][0]["text"]
        match = re.search(r"([-+]?\d*\.?\d+)", text)
        if match:
            val = float(match.group(1))
            if -1.0 <= val <= 1.0:
                format_ok += 1
                actions.append(val)

    latencies_ms = np.array(latencies)
    print(f"  Format OK:    {format_ok}/{len(records)} ({format_ok/len(records)*100:.0f}%)")
    print(f"  Latency mean: {np.mean(latencies_ms):.0f}ms")
    print(f"  Latency P95:  {np.percentile(latencies_ms, 95):.0f}ms")
    if actions:
        print(f"  Action mean:  {np.mean(actions):.4f}")
        print(f"  Action std:   {np.std(actions):.4f}")
        unique = len(set(round(a, 2) for a in actions))
        print(f"  Action diversity: {unique} unique values")


def run_iteration(
    iteration: int,
    trajectories_path: str,
    model_output: str,
    collect_duration: int,
    skip_collect: bool,
    max_train_samples: int,
):
    """Run one full policy iteration cycle."""
    print(f"\n{'#'*60}")
    print(f" POLICY ITERATION #{iteration}")
    print(f"{'#'*60}")

    gguf_path = model_output

    # Phase A: Collect
    if not skip_collect:
        collect_trajectories(trajectories_path, duration_seconds=collect_duration)

    # Phase B: Train
    adapter_path = train_model(
        trajectories_path,
        max_train_samples=max_train_samples,
    )
    export_gguf(adapter_path, gguf_path)

    # Validate
    test_path = "training/data/test.jsonl"
    if os.path.exists(test_path):
        validate_model(gguf_path, test_path)

    # Phase C: Deploy
    deploy_model(gguf_path)

    return gguf_path


def main():
    parser = argparse.ArgumentParser(description="KernelX Policy Iteration Loop")
    parser.add_argument("--trajectories-path", required=True,
                        help="Path to trajectories.json from the bridge")
    parser.add_argument("--model-output",
                        default="training/models/strategist-q4km.gguf",
                        help="Output path for the GGUF model")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of policy iterations to run")
    parser.add_argument("--collect-duration", type=int, default=300,
                        help="Seconds to collect trajectories per iteration")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip data collection (use existing trajectories)")
    parser.add_argument("--max-train-samples", type=int, default=100,
                        help="Max training samples per iteration (100 = fast, 10000 = thorough)")
    args = parser.parse_args()

    for i in range(1, args.iterations + 1):
        run_iteration(
            iteration=i,
            trajectories_path=args.trajectories_path,
            model_output=args.model_output,
            collect_duration=args.collect_duration,
            skip_collect=args.skip_collect,
            max_train_samples=args.max_train_samples,
        )

    print(f"\n{'='*60}")
    print(f" All {args.iterations} iteration(s) complete!")
    print(f"{'='*60}")
    print(f" Model: {args.model_output}")
    print(f" Next: restart brain server to use the updated model")


if __name__ == "__main__":
    main()
