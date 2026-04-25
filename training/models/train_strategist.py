"""
KernelX Intelligence Layer — Strategist Training (Stage 3)

Two-phase training:
  1. SFT warm-start — teaches the model the output format (action only, no rationale)
  2. GRPO fine-tuning — reinforcement learning with multi-objective rewards

Usage:
    python -m training.models.train_strategist \
        --train-data  training/data/train.jsonl \
        --output-dir  training/models/strategist_final

    # Skip warm-start if already done:
    python -m training.models.train_strategist \
        --train-data  training/data/train.jsonl \
        --output-dir  training/models/strategist_final \
        --skip-warmstart \
        --warmstart-model training/models/strategist_warmstart
"""

import json
import re
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.preprocess import (
    FEATURE_NAMES, format_state, load_config,
    IDX_WAIT_US, IDX_CTX_SWITCHES, IDX_VRUNTIME,
)
from training.environment.rewards import RewardComputer
from training.environment.environment import KernelSchedulerEnv, KernelState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = load_config()
MODEL_NAME = CONFIG["model"]["name"]
MAX_SEQ_LEN = CONFIG["model"]["max_seq_length"]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def build_strategist_prompt(state: KernelState) -> str:
    """Build the input prompt for the Strategist model."""
    state_str = format_state(state.features)
    return (
        "<|system|>You are a Linux kernel scheduling strategist. "
        "Given the current system state, output a scheduling action.<|end|>\n"
        f"<|user|>[STATE] {state_str}\n"
        f"[PID] {state.pid} [CPU] {state.cpu}\n"
        "[ACTION]<|end|>\n"
        "<|assistant|>"
    )

# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_action_from_completion(text: str) -> float:
    """Parse action float from model output.

    Returns:
        action_value: float
    Raises:
        ValueError if parsing fails
    """
    # Try [ACTION] marker first, then fall back to any float
    action_match = re.search(r"\[ACTION\]\s*([-+]?\d*\.?\d+)", text)
    if not action_match:
        action_match = re.search(r"([-+]?\d*\.?\d+)", text)
    if not action_match:
        raise ValueError("No action value found in output")

    return float(action_match.group(1))


def parse_state_from_prompt(prompt: str) -> list:
    """Extract the state feature vector from a strategist prompt."""
    state_match = re.search(r"\[STATE\]\s*(.+?)(?:\n|$)", prompt)
    if not state_match:
        raise ValueError("No [STATE] found in prompt")

    values = []
    for part in state_match.group(1).split("|"):
        part = part.strip()
        if ":" in part:
            values.append(float(part.split(":")[1]))
    return values

# ---------------------------------------------------------------------------
# Phase 1: SFT Warm-Start (teaches output format)
# ---------------------------------------------------------------------------

def generate_warmstart_examples(records: list, n: int = 200) -> List[dict]:
    """Generate heuristic-labeled examples to teach the model output format."""
    examples = []
    for record in records[:n]:
        state = record["state"]
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

        ks = KernelState(
            features=state,
            pid=record["pid"],
            cpu=record["cpu"],
            timestep=0,
            prev_action=0.0,
        )
        prompt = build_strategist_prompt(ks)
        text = f"{prompt}{action:.4f}<|end|>"
        examples.append({"text": text})

    return examples


def run_warmstart(
    records: list,
    output_dir: str,
    num_epochs: int = 2,
    n_examples: int = 200,
    use_wandb: bool = False,
):
    """Phase 1: Brief SFT to teach output format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 1: SFT Warm-Start ({n_examples} examples, {num_epochs} epochs) ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="float32", device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = generate_warmstart_examples(records, n=n_examples)
    dataset = Dataset.from_list(examples)
    print(f"  Warm-start dataset: {len(dataset)} examples")
    print(f"  Sample:\n{dataset[0]['text'][:400]}\n")

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    report_to = "wandb" if use_wandb else "none"
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=100,
        fp16=False,
        max_length=MAX_SEQ_LEN,
        report_to=report_to,
        run_name="kernelx-strategist-warmstart",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Warm-start model saved to {output_dir}")

    return model, tokenizer

# ---------------------------------------------------------------------------
# Phase 2: GRPO Reinforcement Learning
# ---------------------------------------------------------------------------

def build_reward_fn(env: KernelSchedulerEnv, reward_computer: RewardComputer):
    """Build the reward function expected by TRL's GRPOTrainer.

    TRL expects: reward_fn(completions, prompts) -> list[float]
    """

    def reward_fn(completions: list, prompts: list) -> list:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                state_features = parse_state_from_prompt(prompt)
                action_val = parse_action_from_completion(completion)
                next_state = env.simulate(state_features, action_val)

                breakdown = reward_computer.compute_total(
                    state=state_features,
                    action=action_val,
                    prev_action=0.0,
                    next_state=next_state,
                )
                rewards.append(breakdown["total"])
            except (ValueError, IndexError):
                rewards.append(-5.0)  # format failure penalty
        return rewards

    return reward_fn


def build_prompt_dataset(records: list) -> Dataset:
    """Convert training records into a dataset of Strategist prompts."""
    prompts = []
    for rec in records:
        ks = KernelState(
            features=rec["state"],
            pid=rec["pid"],
            cpu=rec["cpu"],
            timestep=0,
            prev_action=0.0,
        )
        prompts.append({"prompt": build_strategist_prompt(ks)})
    return Dataset.from_list(prompts)


def sort_by_difficulty(records: list) -> tuple:
    """Split records into easy/medium/hard by feature variance."""
    scored = [(np.var(r["state"]), r) for r in records]
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    third = n // 3
    easy = [r for _, r in scored[:third]]
    medium = [r for _, r in scored[third : 2 * third]]
    hard = [r for _, r in scored[2 * third :]]
    return easy, medium, hard


def run_grpo(
    model,
    tokenizer,
    train_records: list,
    output_dir: str,
    num_epochs: int = 3,
    num_generations: int = 8,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.5,
    use_curriculum: bool = False,
    max_samples: int = 0,
    use_wandb: bool = False,
):
    """Phase 2: GRPO reinforcement learning."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 2: GRPO Training ({num_epochs} epochs, {num_generations} generations) ===")

    # Limit samples if requested
    if max_samples > 0 and len(train_records) > max_samples:
        import random
        train_records = random.sample(train_records, max_samples)
        print(f"Sampled {max_samples} records from {len(train_records)} total")

    # Build environment and reward
    # Write temporary train data for the env
    tmp_train = output_dir / "_tmp_train.jsonl"
    with open(tmp_train, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    env = KernelSchedulerEnv(
        data_path=str(tmp_train),
        max_steps=10,
        alpha=alpha, beta=beta, gamma=gamma,
    )
    reward_computer = RewardComputer(alpha=alpha, beta=beta, gamma=gamma)

    # Calibrate rewards
    print("\nCalibrating reward function ...")
    reward_computer.calibrate(train_records)

    reward_fn = build_reward_fn(env, reward_computer)

    # Build prompt dataset
    if use_curriculum:
        easy, medium, hard = sort_by_difficulty(train_records)
        print(f"\nCurriculum: easy={len(easy)}, medium={len(medium)}, hard={len(hard)}")
        # Start with easy records
        records_for_prompts = easy
    else:
        records_for_prompts = train_records

    prompts_dataset = build_prompt_dataset(records_for_prompts)
    print(f"Prompt dataset: {len(prompts_dataset)} entries")

    # LoRA config for GRPO
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    report_to = "wandb" if use_wandb else "none"
    grpo_config = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_generations=num_generations,
        max_completion_length=16,
        max_prompt_length=384,
        logging_steps=5,
        save_steps=100,
        save_total_limit=3,
        temperature=0.7,
        fp16=False,
        report_to=report_to,
        run_name="kernelx-strategist-grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=prompts_dataset,
        reward_funcs=reward_fn,
        peft_config=lora_config,
    )

    print("Starting GRPO training ...")
    trainer.train()

    # If curriculum, continue with medium + hard
    if use_curriculum:
        print("\n--- Curriculum Phase 2: adding medium-difficulty states ---")
        medium_prompts = build_prompt_dataset(easy + medium)
        trainer.train_dataset = medium_prompts
        trainer.train()

        print("\n--- Curriculum Phase 3: full dataset ---")
        full_prompts = build_prompt_dataset(train_records)
        trainer.train_dataset = full_prompts
        trainer.train()

    # Save
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Strategist model saved to {output_dir}")

    # Cleanup temp file
    tmp_train.unlink(missing_ok=True)

    return model, tokenizer

# ---------------------------------------------------------------------------
# Manual inspection helper
# ---------------------------------------------------------------------------

def inspect_generations(model, tokenizer, records: list, n: int = 10):
    """Generate and print Strategist outputs for manual review."""
    print(f"\n=== Manual Inspection ({n} samples) ===\n")
    import random
    samples = random.sample(records, min(n, len(records)))

    for i, rec in enumerate(samples):
        ks = KernelState(
            features=rec["state"], pid=rec["pid"],
            cpu=rec["cpu"], timestep=0, prev_action=0.0,
        )
        prompt = build_strategist_prompt(ks)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        assistant_part = generated.split("<|assistant|>")[-1] if "<|assistant|>" in generated else generated

        print(f"--- Sample {i+1} ---")
        print(f"State: {format_state(rec['state'])}")
        print(f"Raw output: {assistant_part.strip()}")
        try:
            action_val = parse_action_from_completion(assistant_part)
            print(f"  Action: {action_val:.4f}  in_range: {-1.0 <= action_val <= 1.0}")
        except ValueError as e:
            print(f"  PARSE FAILURE: {e}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train KernelX Strategist (GRPO)")
    parser.add_argument("--train-data", required=True, help="Path to train.jsonl")
    parser.add_argument("--output-dir", default="training/models/strategist_final")
    parser.add_argument("--warmstart-dir", default="training/models/strategist_warmstart")
    parser.add_argument("--skip-warmstart", action="store_true",
                        help="Skip SFT warm-start (load from --warmstart-dir instead)")
    parser.add_argument("--warmstart-model", default=None,
                        help="Pre-trained warm-start model path (requires --skip-warmstart)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--warmstart-examples", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0, help="Throughput reward weight")
    parser.add_argument("--beta", type=float, default=2.0, help="Latency penalty weight")
    parser.add_argument("--gamma", type=float, default=0.5, help="Stability penalty weight")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit GRPO training samples (0 = use all)")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--inspect", action="store_true",
                        help="Run manual inspection after training")
    args = parser.parse_args()

    # Load training records
    records = [json.loads(l) for l in open(args.train_data) if l.strip()]
    print(f"Loaded {len(records)} training records")

    # Phase 1: Warm-start
    if args.skip_warmstart:
        ws_path = args.warmstart_model or args.warmstart_dir
        print(f"Loading pre-trained warm-start model from {ws_path}")
        tokenizer = AutoTokenizer.from_pretrained(ws_path)
        model = AutoModelForCausalLM.from_pretrained(
            ws_path, torch_dtype="float32", device_map="auto"
        )
    else:
        model, tokenizer = run_warmstart(
            records=records,
            output_dir=args.warmstart_dir,
            n_examples=args.warmstart_examples,
            use_wandb=args.wandb,
        )

    # Phase 2: GRPO
    model, tokenizer = run_grpo(
        model=model,
        tokenizer=tokenizer,
        train_records=records,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        num_generations=args.num_generations,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        use_curriculum=args.curriculum,
        use_wandb=args.wandb,
        max_samples=args.max_samples,
    )

    # Optional inspection
    if args.inspect:
        inspect_generations(model, tokenizer, records, n=10)


if __name__ == "__main__":
    main()
