"""
KernelX Intelligence Layer — Model Export and Quantization (Stage 5)

Merges LoRA adapters into the base model and converts to GGUF format
for sub-50ms CPU inference via llama.cpp.

Usage:
    # Merge LoRA + export to GGUF
    python -m training.models.export_gguf \
        --adapter-path training/models/strategist_final \
        --output-dir   training/models/strategist_merged \
        --quantize     Q4_K_M

    # Validate quantized model
    python -m training.models.export_gguf \
        --validate training/models/strategist_merged/strategist-q4km.gguf \
        --test-data training/data/test.jsonl
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.preprocess import load_config

CONFIG = load_config()
MODEL_NAME = CONFIG["model"]["name"]

# ---------------------------------------------------------------------------
# Step 1: Merge LoRA into base weights
# ---------------------------------------------------------------------------

def merge_lora(adapter_path: str, output_dir: str):
    """Load base model + LoRA adapter, merge, and save."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype="auto", device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading LoRA adapter from {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA weights into base model ...")
    merged_model = peft_model.merge_and_unload()

    merged_path = output_dir / "merged_hf"
    print(f"Saving merged model to {merged_path}")
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    # Quick sanity check: generate one token
    print("Sanity check: generating one token ...")
    inputs = tokenizer("Hello", return_tensors="pt")
    outputs = merged_model.generate(**inputs, max_new_tokens=5)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Generated: {decoded}")

    return str(merged_path)

# ---------------------------------------------------------------------------
# Step 2: Convert to GGUF via llama.cpp
# ---------------------------------------------------------------------------

def convert_to_gguf(merged_path: str, output_dir: str, quantize: str = "Q4_K_M"):
    """Convert merged HF model to GGUF and quantize."""
    output_dir = Path(output_dir)
    gguf_f32 = output_dir / "strategist-f32.gguf"
    quant_name = f"strategist-{quantize.lower().replace('_', '')}.gguf"
    gguf_quant = output_dir / quant_name

    # Find llama.cpp
    llama_cpp_dir = Path("llama.cpp")
    if not llama_cpp_dir.exists():
        print("llama.cpp not found. Cloning ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp"],
            check=True,
        )

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"

    # Build llama.cpp if needed
    if not quantize_bin.exists():
        print("Building llama.cpp ...")
        build_dir = llama_cpp_dir / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=str(build_dir), check=True)
        subprocess.run(["cmake", "--build", ".", "--config", "Release", "-j"],
                       cwd=str(build_dir), check=True)

    # Convert HF -> GGUF (f32)
    print(f"Converting {merged_path} -> {gguf_f32}")
    subprocess.run(
        [sys.executable, str(convert_script), merged_path, "--outfile", str(gguf_f32)],
        check=True,
    )

    # Quantize
    print(f"Quantizing {gguf_f32} -> {gguf_quant} ({quantize})")
    subprocess.run(
        [str(quantize_bin), str(gguf_f32), str(gguf_quant), quantize],
        check=True,
    )

    # Report sizes
    f32_size = gguf_f32.stat().st_size / (1024 * 1024)
    quant_size = gguf_quant.stat().st_size / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  F32:        {f32_size:.1f} MB")
    print(f"  {quantize}: {quant_size:.1f} MB")
    print(f"  Compression: {f32_size / quant_size:.1f}x")

    return str(gguf_quant)

# ---------------------------------------------------------------------------
# Step 3: Validate quantized model
# ---------------------------------------------------------------------------

def validate_gguf(gguf_path: str, test_data_path: str = None, max_samples: int = 100):
    """Validate the quantized GGUF model for format compliance and latency."""
    import time
    import json
    import numpy as np

    from llama_cpp import Llama

    print(f"\nLoading quantized model: {gguf_path}")
    llm = Llama(model_path=gguf_path, n_ctx=512, n_threads=4, verbose=False)

    from training.data.preprocess import format_state, FEATURE_NAMES
    from training.environment.environment import KernelState
    from training.models.train_strategist import build_strategist_prompt, parse_action_from_completion

    # Load test data if provided
    if test_data_path:
        records = [json.loads(l) for l in open(test_data_path) if l.strip()]
        records = records[:max_samples]
    else:
        # Generate synthetic test states
        records = [{"state": [float(i)] * len(FEATURE_NAMES), "pid": 1000 + i,
                     "cpu": i % 4, "next_state": [float(i)] * len(FEATURE_NAMES)}
                    for i in range(20)]

    latencies = []
    format_ok = 0
    actions_in_range = 0

    print(f"Running inference on {len(records)} samples ...\n")

    for rec in records:
        ks = KernelState(
            features=rec["state"], pid=rec["pid"],
            cpu=rec["cpu"], timestep=0, prev_action=0.0,
        )
        prompt = build_strategist_prompt(ks)

        start = time.perf_counter()
        output = llm(prompt, max_tokens=8, temperature=0.3)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        text = output["choices"][0]["text"]
        try:
            action_val = parse_action_from_completion(text)
            format_ok += 1
            if -1.0 <= action_val <= 1.0:
                actions_in_range += 1
        except ValueError:
            pass

    latencies_ms = [l * 1000 for l in latencies]
    n = len(records)

    print("=== Quantized Model Validation ===")
    print(f"Format compliance: {format_ok}/{n} ({format_ok/n*100:.1f}%)")
    print(f"Actions in range:  {actions_in_range}/{n} ({actions_in_range/n*100:.1f}%)")
    print(f"\nLatency:")
    print(f"  Mean:  {np.mean(latencies_ms):.1f} ms")
    print(f"  P50:   {np.median(latencies_ms):.1f} ms")
    print(f"  P95:   {np.percentile(latencies_ms, 95):.1f} ms")
    print(f"  P99:   {np.percentile(latencies_ms, 99):.1f} ms")
    print(f"  Max:   {max(latencies_ms):.1f} ms")

    target = CONFIG["model"]["target_inference_ms"]
    p95 = np.percentile(latencies_ms, 95)
    if p95 <= target:
        print(f"\n  PASS: P95 ({p95:.1f}ms) <= target ({target}ms)")
    else:
        print(f"\n  FAIL: P95 ({p95:.1f}ms) > target ({target}ms)")

    return {
        "format_compliance": format_ok / n,
        "action_in_range": actions_in_range / n,
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_p95_ms": float(p95),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export KernelX model to GGUF")
    parser.add_argument("--adapter-path", help="Path to LoRA adapter directory")
    parser.add_argument("--output-dir", default="training/models/strategist_merged")
    parser.add_argument("--quantize", default="Q4_K_M",
                        help="GGUF quantization level (Q4_K_M, Q5_K_M, Q8_0, etc.)")
    parser.add_argument("--validate", default=None,
                        help="Path to GGUF file to validate (skip export)")
    parser.add_argument("--test-data", default=None, help="Test data for validation")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip LoRA merge (model already merged at --adapter-path)")
    args = parser.parse_args()

    if args.validate:
        validate_gguf(args.validate, args.test_data)
    else:
        if not args.adapter_path:
            parser.error("--adapter-path is required for export")

        if args.skip_merge:
            merged_path = args.adapter_path
        else:
            merged_path = merge_lora(args.adapter_path, args.output_dir)

        gguf_path = convert_to_gguf(merged_path, args.output_dir, args.quantize)
        print(f"\nExport complete: {gguf_path}")

        if args.test_data:
            validate_gguf(gguf_path, args.test_data)


if __name__ == "__main__":
    main()
