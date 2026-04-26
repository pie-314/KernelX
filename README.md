# KernelX

KernelX is an OpenEnv-compatible environment and demo stack for the Meta PyTorch OpenEnv Hackathon. It connects Linux scheduler telemetry to a Rust bridge, exposes the environment through FastAPI, runs a simple autonomous policy loop, and ships a rebuilt terminal UI that acts as the final hackathon command console.

## What Ships

- `kernel/`: eBPF scheduler sensor.
- `bridge/`: Rust bridge that exports shared-memory telemetry and records trajectories.
- `brain/`: OpenEnv environment, FastAPI app, and autonomous runner.
- `ui/`: rebuilt `ratatui` console with:
  - live or mock telemetry,
  - April 25-26, 2026 event flow,
  - judging criteria,
  - submission readiness checks,
  - end-to-end operator runbook.

## End-to-End Demo

Run the stack in this order:

```bash
make -C kernel load
cargo run --manifest-path bridge/Cargo.toml --release
cd brain && python3 -m server.app
cd brain/server && python3 run_autonomous.py --steps 50 --verbose
cargo run --manifest-path ui/Cargo.toml
```

If the kernel sensor or bridge is unavailable, the TUI still runs in `MOCK DEMO` mode so the final presentation flow remains usable.

## TUI Controls

- `q`: quit
- `r`: reset history

Screens:

1. `Mission Control Dashboard` (Unified high-density telemetry, AI reasoning, and rewards)

## Data Contract

The UI reads from `/dev/shm/kernelx_state` using this ABI:

```rust
#[repr(C, packed)]
struct HUDState {
    features: [u64; 24],
    current_action: f32,
    active_pid: u32,
    is_clamped: u32,
    reasoning: [u8; 128],
    p99_wait_us: u64,
    core_heat: [f32; 4],
    model_confidence: f32,
    world_model_drift: f32,
    radish_wal_size: u64,
    radish_dirty_pages: u32,
}
```

The brain server uses the same layout with `SHM_SIZE = 376`.

## Submission Notes

Based on the hackathon slides and docs, the final submission on April 26, 2026 requires:

- Hugging Face Space URL
- Colab notebook link
- code repository link
- YouTube video URL or Hugging Face blog URL
- every external URL mirrored in `README.md`

Judging weights:

- Environment innovation: `40%`
- Storytelling and presentation: `30%`
- Observable reward improvement: `20%`
- Reward and training pipeline: `10%`

## Relevant Docs

- [Quickstart](QUICKSTART.md)
- [TUI spec](docs/UI.md)
- [Implementation notes](docs/implementation.md)
