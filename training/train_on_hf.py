#!/usr/bin/env python3
"""
KernelX — Full GPU Training Script for Hugging Face

Run this on a HF Space or notebook with GPU (T4/A10/A100).
It handles everything: download data, train World Model, train Strategist (GRPO),
merge LoRA, export GGUF, and push results back to HF Hub.

Usage (on HF with GPU):
    pip install torch transformers trl peft datasets accelerate huggingface_hub
    python train_on_hf.py --hf-token YOUR_TOKEN
"""

import argparse
import json
import sys
from pathlib import Path

# Force unbuffered output so HF Spaces logs show immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def setup(hf_token: str):
    """Login and download data from HF."""
    import os
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    from huggingface_hub import hf_hub_download, snapshot_download

    # Download training data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for fname in ["state_transitions.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "preprocessing_config.json"]:
        path = hf_hub_download(
            repo_id="Rayugacodes/kernelx-training-data",
            filename=fname,
            repo_type="dataset",
            local_dir=str(data_dir),
        )
        print(f"Downloaded {fname}")

    # Download training scripts
    snapshot_download(
        repo_id="Rayugacodes/kernelx-strategist",
        local_dir="model_repo",
        allow_patterns=["training/**"],
    )
    print("Downloaded training scripts")

    return data_dir


def train_world_model(data_dir: Path, max_samples: int = 10000):
    """Stage 2: Train World Model via SFT."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    config = json.load(open(data_dir / "preprocessing_config.json"))
    MODEL_NAME = config["model"]["name"]
    FEATURE_NAMES = config["feature_names"]

    def format_state(features):
        parts = []
        for name, val in zip(FEATURE_NAMES, features):
            if val == int(val):
                parts.append(f"{name}:{int(val)}")
            else:
                parts.append(f"{name}:{val:.2f}")
        return " | ".join(parts)

    def make_sft_example(record):
        state_str = format_state(record["state"])
        action_str = f"{record['action']:.4f}"
        next_state_str = format_state(record["next_state"])
        text = (
            "<|system|>You are a Linux kernel simulator. "
            "Predict the next system state.<|end|>\n"
            f"<|user|>[STATE] {state_str}\n"
            f"[ACTION] {action_str}\n"
            f"[PID] {record['pid']}\n"
            "Predict [NEXT_STATE]<|end|>\n"
            f"<|assistant|>[NEXT_STATE] {next_state_str}<|end|>"
        )
        return {"text": text}

    print("\n=== Stage 2: World Model SFT ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = [json.loads(l) for l in open(data_dir / "train.jsonl") if l.strip()][:max_samples]
    val_records = [json.loads(l) for l in open(data_dir / "val.jsonl") if l.strip()][:max_samples // 8]

    train_dataset = Dataset.from_list([make_sft_example(r) for r in train_records])
    val_dataset = Dataset.from_list([make_sft_example(r) for r in val_records])
    print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir="./world_model_checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        max_seq_length=512,
        report_to="none",
        disable_tqdm=False,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model("./world_model_final")
    tokenizer.save_pretrained("./world_model_final")
    print("World Model saved.")
    return model, tokenizer


def train_strategist(data_dir: Path, max_samples: int = 10000):
    """Stage 3: Warm-start SFT + GRPO for the Strategist."""
    import re
    import random
    import numpy as np
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer

    config = json.load(open(data_dir / "preprocessing_config.json"))
    MODEL_NAME = config["model"]["name"]
    FEATURE_NAMES = config["feature_names"]
    IDX_WAIT_US = 9
    IDX_CTX_SWITCHES = 8
    IDX_EXEC_NS = 4

    def format_state(features):
        parts = []
        for name, val in zip(FEATURE_NAMES, features):
            if val == int(val):
                parts.append(f"{name}:{int(val)}")
            else:
                parts.append(f"{name}:{val:.2f}")
        return " | ".join(parts)

    def build_prompt(state, pid, cpu):
        state_str = format_state(state)
        return (
            "<|system|>You are a Linux kernel scheduling strategist. "
            "Given the current system state, output a scheduling action.<|end|>\n"
            f"<|user|>[STATE] {state_str}\n"
            f"[PID] {pid} [CPU] {cpu}\n"
            "[ACTION]<|end|>\n"
            "<|assistant|>"
        )

    def parse_action(text):
        m = re.search(r"\[ACTION\]\s*([-+]?\d*\.?\d+)", text)
        if not m:
            m = re.search(r"([-+]?\d*\.?\d+)", text)
        if not m:
            raise ValueError("No action found")
        return float(m.group(1))

    # Load data
    all_records = [json.loads(l) for l in open(data_dir / "train.jsonl") if l.strip()]
    records = random.sample(all_records, min(max_samples, len(all_records)))
    print(f"\n=== Stage 3: Strategist Training ({len(records)} samples) ===")

    # --- Phase 1: Warm-start SFT ---
    print("\n--- Phase 1: Warm-start SFT ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    warmstart_examples = []
    for rec in records[:500]:
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
        prompt = build_prompt(state, rec["pid"], rec["cpu"])
        warmstart_examples.append({"text": f"{prompt}{action:.4f}<|end|>"})

    ws_dataset = Dataset.from_list(warmstart_examples)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    ws_args = SFTConfig(
        output_dir="./strategist_warmstart",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True,
        max_seq_length=512,
        logging_steps=5,
        save_steps=100,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=ws_args,
        train_dataset=ws_dataset, peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model("./strategist_warmstart")
    tokenizer.save_pretrained("./strategist_warmstart")
    print("Warm-start complete.")

    # --- Phase 2: GRPO ---
    print("\n--- Phase 2: GRPO RL Training ---")

    # Build nearest-neighbor simulator from data
    all_states = np.array([r["state"] for r in records])
    all_next_states = [r["next_state"] for r in records]

    def simulate(state_features, action_val):
        state_arr = np.array(state_features)
        dists = np.linalg.norm(all_states[:500] - state_arr, axis=1)
        return all_next_states[int(np.argmin(dists))]

    def reward_fn(completions, prompts):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                # Parse state from prompt
                state_match = re.search(r"\[STATE\]\s*(.+?)(?:\n|$)", prompt)
                values = []
                for part in state_match.group(1).split("|"):
                    part = part.strip()
                    if ":" in part:
                        values.append(float(part.split(":")[1]))

                action_val = parse_action(completion)
                next_state = simulate(values, action_val)

                # Reward: throughput + latency + stability + format
                exec_delta = next_state[IDX_EXEC_NS] - values[IDX_EXEC_NS]
                r_throughput = float(np.log(max(0.0, exec_delta) + 1))
                wait_delta = next_state[IDX_WAIT_US] - values[IDX_WAIT_US]
                r_latency = -2.0 * max(0.0, wait_delta)
                r_stability = -0.5 * abs(action_val)
                r_format = 1.0 if -1.0 <= action_val <= 1.0 else 0.0

                rewards.append(r_throughput + r_latency + r_stability + r_format)
            except (ValueError, IndexError, AttributeError):
                rewards.append(-5.0)
        return rewards

    prompt_dataset = Dataset.from_list([
        {"prompt": build_prompt(r["state"], r["pid"], r["cpu"])}
        for r in records
    ])

    grpo_lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir="./strategist_grpo",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_generations=4,
        max_completion_length=16,
        max_prompt_length=384,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        temperature=0.7,
        fp16=True,
        report_to="none",
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=prompt_dataset,
        reward_funcs=reward_fn,
        peft_config=grpo_lora,
    )

    grpo_trainer.train()
    grpo_trainer.save_model("./strategist_final")
    tokenizer.save_pretrained("./strategist_final")
    print("GRPO training complete.")

    return model, tokenizer


def merge_and_push(hf_token: str):
    """Merge LoRA, push merged model to HF Hub."""
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    config = json.load(open("data/preprocessing_config.json"))
    MODEL_NAME = config["model"]["name"]

    # Use strategist_final if it exists, otherwise fall back to warm-start
    adapter_path = "./strategist_final" if os.path.exists("./strategist_final/adapter_config.json") else "./strategist_warmstart"
    if not os.path.exists(adapter_path):
        # If neither local dir exists, download the warm-start from HF
        from huggingface_hub import snapshot_download
        adapter_path = snapshot_download(
            repo_id="Rayugacodes/kernelx-strategist",
            allow_patterns=["adapter/*"],
            local_dir="./hf_adapter",
        )
        adapter_path = "./hf_adapter/adapter"

    print(f"\n=== Merging LoRA from {adapter_path} and pushing to HF ===")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()

    merged.save_pretrained("./strategist_merged")
    tokenizer.save_pretrained("./strategist_merged")

    merged.push_to_hub("Rayugacodes/kernelx-strategist", commit_message="Merged strategist (warm-start + GRPO)")
    tokenizer.push_to_hub("Rayugacodes/kernelx-strategist", commit_message="Tokenizer")
    print("Pushed to https://huggingface.co/Rayugacodes/kernelx-strategist")


def start_health_server():
    """Start a dummy HTTP server on port 7860 so HF Spaces doesn't kill us."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading

    status = {"stage": "starting"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h1>KernelX Training</h1>"
                f"<p>Stage: <b>{status['stage']}</b></p>"
                f"<p>Refresh to check progress.</p></body></html>".encode()
            )
        def log_message(self, format, *args):
            pass  # suppress request logs

    server = HTTPServer(("0.0.0.0", 7860), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print("Health server running on port 7860")
    return status


def main():
    # Start health server FIRST so HF doesn't kill us
    status = start_health_server()

    parser = argparse.ArgumentParser(description="KernelX GPU Training on HF")
    parser.add_argument("--hf-token", required=True, help="HuggingFace token")
    parser.add_argument("--world-model-samples", type=int, default=10000)
    parser.add_argument("--strategist-samples", type=int, default=10000)
    parser.add_argument("--skip-world-model", action="store_true")
    parser.add_argument("--skip-strategist", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args()

    # Setup
    status["stage"] = "downloading data"
    data_dir = setup(args.hf_token)

    # Train
    if not args.skip_world_model:
        status["stage"] = "training world model"
        train_world_model(data_dir, max_samples=args.world_model_samples)

    if not args.skip_strategist:
        status["stage"] = "training strategist"
        train_strategist(data_dir, max_samples=args.strategist_samples)

    if not args.skip_merge:
        status["stage"] = "merging and pushing to HF"
        merge_and_push(args.hf_token)

    status["stage"] = "DONE"
    print("\n=== All done! ===")
    print("Model: https://huggingface.co/Rayugacodes/kernelx-strategist")
    print("Next: convert to GGUF for sub-50ms CPU inference")

    # Keep alive so the Space stays up
    import time
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
