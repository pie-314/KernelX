#!/usr/bin/env python3
"""
KernelX Intelligence Layer — Full Training Pipeline Runner

Runs all stages end-to-end: preprocess -> World Model SFT -> Strategist GRPO -> export.

Usage:
    # Full pipeline
    python training/run_pipeline.py \
        --raw-data data/state_transitions.jsonl \
        --output-root training

    # Resume from a specific stage
    python training/run_pipeline.py \
        --raw-data data/state_transitions.jsonl \
        --output-root training \
        --start-stage 3

Stages:
    1 = Preprocess data
    2 = Train World Model (SFT)
    3 = Train Strategist (warm-start SFT + GRPO)
    4 = Export & quantize to GGUF
    5 = Validate quantized model
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="KernelX full training pipeline")
    parser.add_argument("--raw-data", required=True, help="Path to raw state_transitions.jsonl")
    parser.add_argument("--output-root", default="training", help="Root output directory")
    parser.add_argument("--start-stage", type=int, default=1, help="Stage to start from (1-5)")
    parser.add_argument("--end-stage", type=int, default=5, help="Stage to end at (1-5)")
    parser.add_argument("--epochs-world", type=int, default=3, help="World Model training epochs")
    parser.add_argument("--epochs-strategist", type=int, default=3, help="Strategist GRPO epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--quantize", default="Q4_K_M")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    data_dir = root / "data"
    models_dir = root / "models"

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"

    # ------------------------------------------------------------------
    # Stage 1: Preprocess
    # ------------------------------------------------------------------
    if args.start_stage <= 1 <= args.end_stage:
        print("\n" + "=" * 60)
        print(" STAGE 1: Data Preprocessing")
        print("=" * 60)

        from training.data.preprocess import run_pipeline
        run_pipeline(args.raw_data, str(data_dir), audit=True)

    # ------------------------------------------------------------------
    # Stage 2: World Model SFT
    # ------------------------------------------------------------------
    if args.start_stage <= 2 <= args.end_stage:
        print("\n" + "=" * 60)
        print(" STAGE 2: World Model Training (SFT)")
        print("=" * 60)

        from training.models.train_world_model import train, evaluate_world_model

        wm_dir = models_dir / "world_model_final"
        model, tokenizer = train(
            train_path=str(train_path),
            val_path=str(val_path),
            output_dir=str(wm_dir),
            num_epochs=args.epochs_world,
            batch_size=args.batch_size,
            use_wandb=args.wandb,
        )

        if test_path.exists():
            evaluate_world_model(model, tokenizer, str(test_path))

    # ------------------------------------------------------------------
    # Stage 3: Strategist Training (warm-start + GRPO)
    # ------------------------------------------------------------------
    if args.start_stage <= 3 <= args.end_stage:
        print("\n" + "=" * 60)
        print(" STAGE 3: Strategist Training (GRPO)")
        print("=" * 60)

        from training.models.train_strategist import (
            run_warmstart, run_grpo, inspect_generations,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        records = [json.loads(l) for l in open(train_path) if l.strip()]

        # Phase 1: Warm-start
        ws_dir = models_dir / "strategist_warmstart"
        model, tokenizer = run_warmstart(
            records=records,
            output_dir=str(ws_dir),
            use_wandb=args.wandb,
        )

        # Phase 2: GRPO
        strat_dir = models_dir / "strategist_final"
        model, tokenizer = run_grpo(
            model=model,
            tokenizer=tokenizer,
            train_records=records,
            output_dir=str(strat_dir),
            num_epochs=args.epochs_strategist,
            use_curriculum=args.curriculum,
            use_wandb=args.wandb,
        )

        # Inspect
        inspect_generations(model, tokenizer, records, n=10)

    # ------------------------------------------------------------------
    # Stage 4: Export & Quantize
    # ------------------------------------------------------------------
    if args.start_stage <= 4 <= args.end_stage:
        print("\n" + "=" * 60)
        print(" STAGE 4: Export & Quantize to GGUF")
        print("=" * 60)

        from training.models.export_gguf import merge_lora, convert_to_gguf

        strat_dir = models_dir / "strategist_final"
        merged_dir = models_dir / "strategist_merged"
        merged_path = merge_lora(str(strat_dir), str(merged_dir))
        convert_to_gguf(merged_path, str(merged_dir), args.quantize)

    # ------------------------------------------------------------------
    # Stage 5: Validate
    # ------------------------------------------------------------------
    if args.start_stage <= 5 <= args.end_stage:
        print("\n" + "=" * 60)
        print(" STAGE 5: Validate Quantized Model")
        print("=" * 60)

        from training.models.export_gguf import validate_gguf

        quant_name = f"strategist-{args.quantize.lower().replace('_', '')}.gguf"
        gguf_path = models_dir / "strategist_merged" / quant_name

        if gguf_path.exists():
            validate_gguf(str(gguf_path), str(test_path) if test_path.exists() else None)
        else:
            print(f"GGUF not found at {gguf_path}. Run stage 4 first.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Pipeline complete!")
    print("=" * 60)
    print(f"\nArtifacts in: {root}/")
    print(f"  Data:   {data_dir}/")
    print(f"  Models: {models_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Run demo:     python -m training.demo.app --test-data {test_path}")
    print(f"  2. Run engine:   python -m training.inference.strategy_engine --model <gguf>")
    print(f"  3. Benchmark:    python -m training.inference.benchmark_latency --model <gguf>")


if __name__ == "__main__":
    main()
