"""
KernelX Intelligence Layer — World Model Training (Stage 2)

Supervised fine-tuning (SFT) that teaches SmolLM2-360M to predict
S_{t+1} given (S_t, a_t).  The model learns default system dynamics
from baseline data collected by the eBPF sentinel.

Usage:
    python -m training.models.train_world_model \
        --train-data  training/data/train.jsonl \
        --val-data    training/data/val.jsonl \
        --output-dir  training/models/world_model_final
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Add project root so we can import our preprocessing utils
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.preprocess import FEATURE_NAMES, format_state, load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = load_config()
MODEL_NAME = CONFIG["model"]["name"]
MAX_SEQ_LEN = CONFIG["model"]["max_seq_length"]

# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def make_sft_example(record: dict) -> dict:
    """Convert a preprocessed transition record into an SFT text example.

    Format:
        <|system|>...<|end|>
        <|user|>[STATE] ... [ACTION] ... [PID] ... Predict [NEXT_STATE]<|end|>
        <|assistant|>[NEXT_STATE] ...<|end|>
    """
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


def load_split(path: str) -> Dataset:
    """Load a preprocessed JSONL split into a HuggingFace Dataset."""
    records = [json.loads(line) for line in open(path) if line.strip()]
    examples = [make_sft_example(r) for r in records]
    return Dataset.from_list(examples)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def parse_next_state(text: str) -> list:
    """Parse a [NEXT_STATE] line back into a float vector."""
    marker = "[NEXT_STATE]"
    idx = text.rfind(marker)
    if idx == -1:
        raise ValueError("No [NEXT_STATE] marker found in output")
    payload = text[idx + len(marker):].split("<|end|>")[0].strip()
    values = []
    for part in payload.split("|"):
        part = part.strip()
        if ":" in part:
            values.append(float(part.split(":")[1]))
    return values


def evaluate_world_model(model, tokenizer, test_path: str, max_samples: int = 100):
    """Evaluate World Model predictions on a test set via MSE."""
    records = [json.loads(line) for line in open(test_path) if line.strip()]
    records = records[:max_samples]

    mse_scores = []
    per_feature_errors = [[] for _ in range(len(FEATURE_NAMES))]
    format_failures = 0

    for record in records:
        state_str = format_state(record["state"])
        action_str = f"{record['action']:.4f}"

        prompt = (
            "<|system|>You are a Linux kernel simulator. "
            "Predict the next system state.<|end|>\n"
            f"<|user|>[STATE] {state_str}\n"
            f"[ACTION] {action_str}\n"
            f"[PID] {record['pid']}\n"
            "Predict [NEXT_STATE]<|end|>\n"
            "<|assistant|>"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

        try:
            predicted = parse_next_state(generated)
            actual = record["next_state"]
            if len(predicted) != len(actual):
                format_failures += 1
                continue
            errors = [(p - a) ** 2 for p, a in zip(predicted, actual)]
            mse_scores.append(np.mean(errors))
            for i, e in enumerate(errors):
                per_feature_errors[i].append(e)
        except (ValueError, IndexError):
            format_failures += 1

    total = len(records)
    print(f"\n=== World Model Evaluation ({total} samples) ===")
    print(f"Format failures: {format_failures}/{total} ({format_failures/total*100:.1f}%)")
    if mse_scores:
        print(f"Overall MSE: {np.mean(mse_scores):.6f}")
        print(f"\nPer-feature MSE:")
        for i, name in enumerate(FEATURE_NAMES):
            if per_feature_errors[i]:
                print(f"  {name:>10}: {np.mean(per_feature_errors[i]):.6f}")
    else:
        print("No valid predictions to evaluate.")

    return {
        "mse": float(np.mean(mse_scores)) if mse_scores else None,
        "format_failure_rate": format_failures / total,
        "n_samples": total,
    }

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    train_path: str,
    val_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_wandb: bool = False,
    max_samples: int = 0,
):
    """Run World Model SFT training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("Loading datasets ...")
    train_dataset = load_split(train_path)
    val_dataset = load_split(val_path)
    if max_samples > 0:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(max_samples // 8, len(val_dataset))))
    print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # Verify a sample
    print(f"\nSample training example:\n{train_dataset[0]['text'][:500]}\n")

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    report_to = "wandb" if use_wandb else "none"
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        fp16=False,
        max_length=MAX_SEQ_LEN,
        report_to=report_to,
        run_name="kernelx-world-model",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
    )

    print("Starting World Model SFT training ...")
    trainer.train()

    # Save final model
    print(f"Saving model to {output_dir} ...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("World Model training complete.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train KernelX World Model (SFT)")
    parser.add_argument("--train-data", required=True, help="Path to train.jsonl")
    parser.add_argument("--val-data", required=True, help="Path to val.jsonl")
    parser.add_argument("--test-data", default=None, help="Path to test.jsonl (for evaluation)")
    parser.add_argument("--output-dir", default="training/models/world_model_final",
                        help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit training samples (0 = use all)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--eval-only", default=None,
                        help="Skip training; evaluate this checkpoint on --test-data")
    args = parser.parse_args()

    if args.eval_only:
        print(f"Evaluation-only mode: loading {args.eval_only}")
        tokenizer = AutoTokenizer.from_pretrained(args.eval_only)
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_only, dtype="auto", device_map="auto"
        )
        if not args.test_data:
            print("ERROR: --test-data is required for --eval-only")
            sys.exit(1)
        evaluate_world_model(model, tokenizer, args.test_data)
    else:
        model, tokenizer = train(
            train_path=args.train_data,
            val_path=args.val_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            use_wandb=args.wandb,
            max_samples=args.max_samples,
        )
        if args.test_data:
            evaluate_world_model(model, tokenizer, args.test_data)


if __name__ == "__main__":
    main()
