"""
KernelX Intelligence Layer — Latency Benchmark

Measures end-to-end inference latency of the quantized Strategist model
on the target CPU hardware. Reports mean, P50, P95, P99, and max latency.

Usage:
    python -m training.inference.benchmark_latency \
        --model training/models/strategist_merged/strategist-q4km.gguf \
        --samples 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.preprocess import FEATURE_NAMES, format_state, load_config
from training.inference.strategy_engine import build_inference_prompt, parse_output

CONFIG = load_config()


def benchmark(
    model_path: str,
    test_data_path: str = None,
    n_samples: int = 200,
    n_threads: int = 4,
    temperature: float = 0.2,
    max_tokens: int = 64,
    warmup: int = 5,
):
    """Run latency benchmark on the quantized GGUF model."""
    from llama_cpp import Llama

    print(f"Loading model: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=512, n_threads=n_threads, verbose=False)

    # Build test prompts
    if test_data_path:
        records = [json.loads(l) for l in open(test_data_path) if l.strip()]
    else:
        # Synthetic test data
        records = []
        for i in range(n_samples):
            state = [float(i % 16)]  # cpu
            state += [120.0, 120.0, 120.0]  # priorities
            state += [20.0 + i * 0.1, 28.0 + i * 0.01, 8.0 + i * 0.001]  # symlog'd
            state += [16.0, float(i % 50), float(5 + i % 30)]  # cpus, csw, wt_us
            records.append({"state": state, "pid": 1000 + i, "cpu": i % 16})

    records = records[:n_samples]
    prompts = []
    for rec in records:
        prompts.append(build_inference_prompt(rec["state"], rec["pid"], rec["cpu"]))

    # Warmup
    print(f"Warming up ({warmup} iterations) ...")
    for i in range(warmup):
        llm(prompts[i % len(prompts)], max_tokens=max_tokens, temperature=temperature)

    # Benchmark
    print(f"\nBenchmarking {len(prompts)} samples ...")
    latencies = []
    format_ok = 0
    token_counts = []

    for prompt in prompts:
        start = time.perf_counter()
        output = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        text = output["choices"][0]["text"]
        tokens = output["usage"]["completion_tokens"]
        token_counts.append(tokens)

        action_val = parse_output(text)
        if -1.0 <= action_val <= 1.0:
            format_ok += 1

    latencies_ms = np.array(latencies) * 1000

    # Report
    target = CONFIG["model"]["target_inference_ms"]
    print(f"\n{'='*50}")
    print(f" KernelX Latency Benchmark")
    print(f"{'='*50}")
    print(f" Model:    {Path(model_path).name}")
    print(f" Threads:  {n_threads}")
    print(f" Samples:  {len(prompts)}")
    print(f" Target:   <{target}ms")
    print(f"{'='*50}")
    print(f" Mean:     {np.mean(latencies_ms):>8.1f} ms")
    print(f" Median:   {np.median(latencies_ms):>8.1f} ms")
    print(f" P95:      {np.percentile(latencies_ms, 95):>8.1f} ms")
    print(f" P99:      {np.percentile(latencies_ms, 99):>8.1f} ms")
    print(f" Max:      {np.max(latencies_ms):>8.1f} ms")
    print(f" Min:      {np.min(latencies_ms):>8.1f} ms")
    print(f" Std:      {np.std(latencies_ms):>8.1f} ms")
    print(f"{'='*50}")
    print(f" Tokens/s: {np.sum(token_counts) / np.sum(latencies):>8.1f}")
    print(f" Format OK:{format_ok}/{len(prompts)} ({format_ok/len(prompts)*100:.1f}%)")
    print(f"{'='*50}")

    p95 = np.percentile(latencies_ms, 95)
    if p95 <= target:
        print(f" VERDICT:  PASS (P95 {p95:.1f}ms <= {target}ms)")
    else:
        print(f" VERDICT:  FAIL (P95 {p95:.1f}ms > {target}ms)")

    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "p95_ms": float(p95),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "format_ok_pct": format_ok / len(prompts) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark KernelX inference latency")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--test-data", default=None, help="Test JSONL (optional)")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    benchmark(
        model_path=args.model,
        test_data_path=args.test_data,
        n_samples=args.samples,
        n_threads=args.threads,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
