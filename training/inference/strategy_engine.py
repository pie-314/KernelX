"""
KernelX Intelligence Layer — Inference Engine (Stage 6)

Three-thread architecture for real-time kernel scheduling:
  Thread 1 (Telemetry):  reads latest state from shared memory
  Thread 2 (Strategist): runs LLM inference every cycle
  Thread 3 (Updater):    writes action to shared memory command slot

Uses llama.cpp via llama-cpp-python for sub-50ms CPU inference
with the quantized GGUF Strategist model.

Usage:
    python -m training.inference.strategy_engine \
        --model training/models/strategist_merged/strategist-q4km.gguf \
        --shm-path /dev/shm/kernelx_state
"""

import argparse
import json
import mmap
import os
import re
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.preprocess import (
    ACTIVE_FEATURES, FEATURE_NAMES, SYMLOG_FEATURES,
    symmetric_log, format_state, load_config,
)

CONFIG = load_config()

# ---------------------------------------------------------------------------
# Shared memory layout (must match bridge/src/main.rs HUDState)
# ---------------------------------------------------------------------------
# features:           [u64; 24]  = 192 bytes (offset 0)
# current_action:     f32        = 4   bytes (offset 192)
# active_pid:         u32        = 4   bytes (offset 196)
# is_clamped:         u32        = 4   bytes (offset 200)
# reasoning:          [u8; 128]  = 128 bytes (offset 204)
# p99_wait_us:        u64        = 8   bytes (offset 332)
#                                  Total: 340 bytes

SHM_SIZE = 340
FEATURES_OFFSET = 0
FEATURES_SIZE = 192     # 24 * 8 bytes (u64)
ACTION_OFFSET = 192
PID_OFFSET = 196
CLAMPED_OFFSET = 200
REASONING_OFFSET = 204
REASONING_SIZE = 128
P99_OFFSET = 332

# ---------------------------------------------------------------------------
# Prompt building (mirrors train_strategist.py)
# ---------------------------------------------------------------------------

def build_inference_prompt(active_features: list, pid: int, cpu: int) -> str:
    state_str = format_state(active_features)
    return (
        "<|system|>You are a Linux kernel scheduling strategist. "
        "Given the current system state, output a scheduling action.<|end|>\n"
        f"<|user|>[STATE] {state_str}\n"
        f"[PID] {pid} [CPU] {cpu}\n"
        "[ACTION]<|end|>\n"
        "<|assistant|>"
    )


def parse_output(text: str) -> float:
    """Parse action float from model output."""
    action_match = re.search(r"\[ACTION\]\s*([-+]?\d*\.?\d+)", text)
    if not action_match:
        action_match = re.search(r"([-+]?\d*\.?\d+)", text)
    if not action_match:
        return 0.0

    action_val = float(action_match.group(1))
    return max(-1.0, min(1.0, action_val))

# ---------------------------------------------------------------------------
# Shared memory reader/writer
# ---------------------------------------------------------------------------

def read_features_from_shm(shm: mmap.mmap) -> tuple:
    """Read 24D raw features + PID from shared memory.

    Returns:
        (raw_features_24d: list[float], pid: int, cpu: int)
    """
    shm.seek(FEATURES_OFFSET)
    raw_bytes = shm.read(FEATURES_SIZE)
    raw_features = list(np.frombuffer(raw_bytes, dtype=np.uint64).astype(np.float64))

    shm.seek(PID_OFFSET)
    pid = struct.unpack("<I", shm.read(4))[0]

    # CPU is in features[0] (from bpf_get_smp_processor_id)
    cpu = int(raw_features[0]) if raw_features else 0

    return raw_features, pid, cpu


def preprocess_for_inference(raw_features: list) -> list:
    """Apply symlog scaling and extract active features for the LLM."""
    f = list(raw_features)
    for idx in SYMLOG_FEATURES:
        f[idx] = symmetric_log(f[idx])
    return [f[i] for i in ACTIVE_FEATURES]


def write_action_to_shm(shm: mmap.mmap, action: float):
    """Write action value to shared memory."""
    shm.seek(ACTION_OFFSET)
    shm.write(struct.pack("<f", action))

# ---------------------------------------------------------------------------
# Strategy Engine (three-thread architecture)
# ---------------------------------------------------------------------------

class StrategyEngine:
    """Real-time scheduling inference engine.

    Reads kernel state from shared memory, runs the quantized Strategist
    model, and writes the scheduling action back.
    """

    def __init__(
        self,
        model_path: str,
        shm_path: str = "/dev/shm/kernelx_state",
        n_threads: int = 2,
        poll_interval_ms: float = 10.0,
        update_interval_ms: float = 50.0,
        temperature: float = 0.2,
        max_tokens: int = 8,
    ):
        from llama_cpp import Llama

        self.model_path = model_path
        self.shm_path = shm_path
        self.poll_interval = poll_interval_ms / 1000.0
        self.update_interval = update_interval_ms / 1000.0
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load model
        print(f"[StrategyEngine] Loading model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=n_threads,
            verbose=False,
        )

        # Shared state (protected by lock)
        self.lock = threading.Lock()
        self.current_features = [0.0] * len(ACTIVE_FEATURES)
        self.current_pid = 0
        self.current_cpu = 0
        self.latest_action = 0.0

        # Control
        self.running = False
        self.shm = None

        # Metrics
        self.inference_count = 0
        self.inference_latencies = []

    def _open_shm(self) -> mmap.mmap:
        """Open shared memory file for read/write."""
        fd = os.open(self.shm_path, os.O_RDWR)
        return mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def start(self):
        """Start all three threads."""
        if not os.path.exists(self.shm_path):
            print(f"[StrategyEngine] WARNING: SHM {self.shm_path} not found.")
            print("[StrategyEngine] Running in dry-run mode (no SHM I/O).")
            self.shm = None
        else:
            self.shm = self._open_shm()
            print(f"[StrategyEngine] Connected to SHM: {self.shm_path}")

        self.running = True

        threads = [
            threading.Thread(target=self._telemetry_loop, name="telemetry", daemon=True),
            threading.Thread(target=self._strategist_loop, name="strategist", daemon=True),
            threading.Thread(target=self._update_loop, name="updater", daemon=True),
        ]

        for t in threads:
            t.start()
            print(f"[StrategyEngine] Started {t.name} thread")

        print("[StrategyEngine] All threads running. Press Ctrl+C to stop.")

        try:
            while self.running:
                time.sleep(1.0)
                # Periodic stats
                if self.inference_count > 0 and self.inference_count % 20 == 0:
                    recent = self.inference_latencies[-20:]
                    avg_ms = np.mean(recent) * 1000
                    print(f"[StrategyEngine] Inferences: {self.inference_count}, "
                          f"Avg latency: {avg_ms:.1f}ms, "
                          f"Action: {self.latest_action:.4f}")
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop all threads."""
        print("\n[StrategyEngine] Shutting down ...")
        self.running = False
        if self.shm:
            self.shm.close()

    def _telemetry_loop(self):
        """Thread 1: Read latest state from shared memory."""
        while self.running:
            if self.shm:
                try:
                    raw_features, pid, cpu = read_features_from_shm(self.shm)
                    active = preprocess_for_inference(raw_features)
                    with self.lock:
                        self.current_features = active
                        self.current_pid = pid
                        self.current_cpu = cpu
                except Exception:
                    pass
            time.sleep(self.poll_interval)

    def _strategist_loop(self):
        """Thread 2: Run LLM inference every cycle."""
        while self.running:
            with self.lock:
                features = list(self.current_features)
                pid = self.current_pid
                cpu = self.current_cpu

            prompt = build_inference_prompt(features, pid, cpu)

            start = time.perf_counter()
            output = self.llm(prompt, max_tokens=self.max_tokens,
                              temperature=self.temperature)
            elapsed = time.perf_counter() - start

            text = output["choices"][0]["text"]
            action = parse_output(text)

            with self.lock:
                self.latest_action = action

            self.inference_count += 1
            self.inference_latencies.append(elapsed)

    def _update_loop(self):
        """Thread 3: Write action to shared memory command slot."""
        while self.running:
            with self.lock:
                action = self.latest_action

            if self.shm:
                try:
                    write_action_to_shm(self.shm, action)
                except Exception:
                    pass

            time.sleep(self.update_interval)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KernelX Strategy Engine")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--shm-path", default="/dev/shm/kernelx_state")
    parser.add_argument("--threads", type=int, default=2, help="llama.cpp threads")
    parser.add_argument("--poll-ms", type=float, default=10.0)
    parser.add_argument("--update-ms", type=float, default=50.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    engine = StrategyEngine(
        model_path=args.model,
        shm_path=args.shm_path,
        n_threads=args.threads,
        poll_interval_ms=args.poll_ms,
        update_interval_ms=args.update_ms,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    engine.start()


if __name__ == "__main__":
    main()
