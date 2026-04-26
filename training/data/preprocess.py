"""
KernelX Intelligence Layer — Data Ingestion and Preprocessing (Stage 1)

Reads raw trajectories.json from the bridge's TrajectoryManager,
applies feature scaling (symlog for huge counters), drops sparse-zero
features, and produces train/val/test splits for World Model and
Strategist training.

Usage:
    python -m training.data.preprocess --input data/trajectories.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# ---------------------------------------------------------------------------
# Configuration (loaded from preprocessing_config.json)
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "preprocessing_config.json"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

CONFIG = load_config()

SYMLOG_FEATURES = CONFIG["symlog_features"]          # [4, 5, 6]
ACTIVE_FEATURES = CONFIG["active_features"]           # [0,1,2,3,4,5,6,7,12,23]
FEATURE_NAMES   = CONFIG["feature_names"]             # 10 short names
SPARSE_ZERO     = CONFIG["sparse_zero_features"]      # indices to drop

# ---------------------------------------------------------------------------
# Feature index constants (positions within the ACTIVE 10D vector)
# ---------------------------------------------------------------------------

IDX_CPU           = 0   # raw index 0
IDX_PRIO          = 1   # raw index 1
IDX_STATIC_PRIO   = 2   # raw index 2
IDX_NORMAL_PRIO   = 3   # raw index 3
IDX_EXEC_NS       = 4   # raw index 4 (symlog)
IDX_VRUNTIME      = 5   # raw index 5 (symlog)
IDX_MIGRATIONS    = 6   # raw index 6 (symlog)
IDX_CPUS_ALLOWED  = 7   # raw index 7
IDX_CTX_SWITCHES  = 8   # raw index 12
IDX_WAIT_US       = 9   # raw index 23

# ---------------------------------------------------------------------------
# Scaling functions
# ---------------------------------------------------------------------------

def symmetric_log(x: float) -> float:
    """sgn(x) * ln(1 + |x|) — compresses huge values while preserving sign."""
    return float(np.sign(x) * np.log1p(np.abs(x)))


def preprocess_features(raw_features: List[float]) -> List[float]:
    """Transform a raw 24D feature vector: symlog the big counters."""
    f = list(raw_features)  # copy
    for idx in SYMLOG_FEATURES:
        f[idx] = symmetric_log(f[idx])
    return f


def extract_active(scaled_features: List[float]) -> List[float]:
    """Keep only the active (non-zero, information-carrying) features."""
    return [scaled_features[i] for i in ACTIVE_FEATURES]

# ---------------------------------------------------------------------------
# Prompt formatting for LLM consumption
# ---------------------------------------------------------------------------

def format_state(active_vector: List[float]) -> str:
    """Convert a 10D active feature vector into a compact text string."""
    parts = []
    for name, val in zip(FEATURE_NAMES, active_vector):
        if val == int(val):
            parts.append(f"{name}:{int(val)}")
        else:
            parts.append(f"{name}:{val:.2f}")
    return " | ".join(parts)

# ---------------------------------------------------------------------------
# Record processing
# ---------------------------------------------------------------------------

def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single raw JSONL record into training-ready format."""
    s_t_scaled = preprocess_features(record["state_t"]["features"])
    s_t1_scaled = preprocess_features(record["state_t_next"]["features"])

    s_t_active = extract_active(s_t_scaled)
    s_t1_active = extract_active(s_t1_scaled)

    return {
        "state": s_t_active,
        "action": record["action"],
        "reward": record["reward"],
        "next_state": s_t1_active,
        "pid": record["state_t"]["pid"],
        "cpu": record["state_t"]["cpu"],
        "timestamp": record["state_t"]["timestamp"],
    }

# ---------------------------------------------------------------------------
# Dataset audit
# ---------------------------------------------------------------------------

def audit_dataset(records: List[Dict]) -> None:
    """Print per-feature statistics for the raw dataset."""
    all_features = []
    for r in records:
        all_features.append(r["state_t"]["features"])
        all_features.append(r["state_t_next"]["features"])

    arr = np.array(all_features, dtype=np.float64)

    print(f"\nTotal transitions: {len(records)}")
    print(f"Total feature vectors: {len(all_features)}")
    print(f"\n{'Idx':<5} {'Min':<22} {'Max':<22} {'Mean':<22} {'Std':<22} {'Zeros%':<10}")
    print("-" * 103)
    for i in range(24):
        col = arr[:, i]
        zero_pct = (col == 0).sum() / len(col) * 100
        print(f"{i:<5} {col.min():<22.2f} {col.max():<22.2f} {col.mean():<22.2f} {col.std():<22.2f} {zero_pct:<10.1f}")

    print(f"\nNaN count: {np.isnan(arr).sum()}")
    print(f"Inf count: {np.isinf(arr).sum()}")

    actions = [r["action"] for r in records]
    rewards = [r["reward"] for r in records]
    print(f"\nAction — unique values: {sorted(set(actions))}")
    print(f"Reward — min: {min(rewards)}, max: {max(rewards)}, mean: {np.mean(rewards):.2f}, std: {np.std(rewards):.2f}")


# ---------------------------------------------------------------------------
# Train / Val / Test split (chronological)
# ---------------------------------------------------------------------------

def split_chronological(processed: List[Dict], train_ratio=0.8, val_ratio=0.1):
    """Split processed records chronologically (NOT randomly)."""
    processed.sort(key=lambda x: x["timestamp"])
    n = len(processed)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return processed[:train_end], processed[train_end:val_end], processed[val_end:]

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, output_dir: str, audit: bool = True):
    """Full preprocessing pipeline: audit -> scale -> split -> save."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"Loading raw data from {input_path} ...")
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("ERROR: No records found in input file.")
        sys.exit(1)

    print(f"Loaded {len(records)} raw transitions.")

    # Audit
    if audit:
        audit_dataset(records)

    # Preprocess
    print("\nPreprocessing (symlog scaling + active feature extraction) ...")
    processed = [preprocess_record(r) for r in records]

    # Verify transform on first record
    sample = processed[0]
    print(f"\nSample preprocessed state (10D):")
    print(f"  {format_state(sample['state'])}")
    print(f"Sample preprocessed next_state:")
    print(f"  {format_state(sample['next_state'])}")

    # Save full processed dataset
    processed_path = output_dir / "processed_transitions.jsonl"
    with open(processed_path, "w") as f:
        for p in processed:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved {len(processed)} processed records to {processed_path}")

    # Split
    train, val, test = split_chronological(processed)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_path = output_dir / f"{split_name}.jsonl"
        with open(split_path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"{split_name}: {len(split_data)} records -> {split_path}")

    print("\nPreprocessing complete.")
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="KernelX data preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Path to raw trajectories.json")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent), help="Output directory for processed data")
    parser.add_argument("--no-audit", action="store_true", help="Skip the dataset audit step")
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir, audit=not args.no_audit)


if __name__ == "__main__":
    main()
