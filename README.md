# KernelX: Cognitive Operating System Layer 🌪️

**KernelX** is a "Wild Card" entry for the **Meta PyTorch OpenEnv Hackathon**. It re-imagines the Linux kernel as a partially observable, dynamic world, replacing static heuristics with a Multi-Agent Reinforcement Learning (MARL) system for real-time hardware optimization.



## 🌌 The Vision
Modern OS schedulers rely on decades-old hardcoded logic. KernelX introduces an intelligence plane that treats CPU, Cache, and I/O as finite resources in a competitive environment. By utilizing eBPF for reflexes and PyTorch for reasoning, KernelX achieves long-horizon optimization that adapts to shifting workloads and adversarial stress.

---

## 👥 The Vayu Trinity (Team Roles)
* **AI Lead (Strategist):** Implements the PPO models in PyTorch and bridges the kernel data into an OpenEnv-compliant Gymnasium environment.
* **Governance/Chaos (Auditor & Adversary):** Manages Scalable Oversight (Auditor) and the Chaos Monkey stressors (Adversary) to ensure system stability.
* **Systems Lead (Sentinel):** Develops the eBPF probes in C and the memory-safe telemetry bridge in Rust using Aya.

---

## 🛠️ Technology Stack
| Layer | Component | Technology |
| :--- | :--- | :--- |
| **Reflexes** | Kernel Space | C (Restricted eBPF), `sched_ext` (SCX) |
| **Nervous System** | Bridge | Rust (Aya), Zero-copy deserialization |
| **Intelligence** | Brain | Python, PyTorch, OpenEnv, Unsloth |
| **Persistence** | Experience Store | Rust (RadishDB - Custom WAL) |
| **HUD** | Visualization | Rust (Ratatui) |

---

## 📐 Technical Architecture & Math

### 1. Multi-Agent Interaction
The swarm consists of three primary agents:
* **The Sentinel (Perception):** Captures 24D state vectors ($S_t$) from the bare metal.
* **The Strategist (Reasoning):** Maps hardware states to optimization actions ($A_t$).
* **The Auditor (Safety):** Explains behavior and maintains OS bounds.

### 2. The Reward Model ($R_t$)
Our objective function balances throughput, tail latency, and compute cost:
$$R_t = \alpha \cdot \log(\text{Throughput}_t) - \beta \cdot \text{Tail\_Latency}_t - \lambda \cdot \text{Cost}(a_t)$$

### 3. World Modeling
The agents maintain an internal representation of "Hardware Physics" by minimizing prediction error:
$$\mathcal{L}_{world} = \mathbb{E} \left[ \| \hat{S}_{t+1} - S_{t+1} \|^2 \right]$$

---

## 📂 Directory Structure
```text
kernelx/
├── kernel/                 # eBPF Probes & SCX Actuators (C)
├── bridge/                 # Aya Loader, Auditor, & WAL (Rust)
├── brain/                  # OpenEnv Wrapper & PyTorch Models (Python)
│   ├── env/                # Vayu-Gym (OpenEnv compliance)
│   └── scripts/            # Unsloth/HF TRL Training
├── ui/                     # Ratatui Dashboard (Rust)
└── adversary/              # Chaos Monkey Stressors (Python)
```

---

## 🚀 Getting Started

### Prerequisites
Install the native toolchain needed by the current `kernel/` and `bridge/` code:

```bash
# Arch Linux
sudo pacman -S base-devel clang llvm libelf libbpf bpftool rust cargo
```

You need:
- A kernel with BTF enabled at `/sys/kernel/btf/vmlinux`
- Root access for loading and attaching eBPF programs

### Run Everything
From the repo root:

1. Generate `vmlinux.h` once for the running kernel.
```bash
cd kernel
make vmlinux
```

2. Build and attach the eBPF scheduler probes.
```bash
make
```

`make` currently does:
- Compile `sentinel.bpf.c` into `sentinel.bpf.o`
- Load all tracepoint programs with `bpftool prog loadall`
- Auto-attach the scheduler hooks

It does not start trace output.

3. In a second terminal, start the Rust Aya bridge.
```bash
cd bridge
cargo run
```

This bridge:
- Loads the same `kernel/sentinel.bpf.o`
- Attaches `sched_wakeup`, `sched_wakeup_new`, and `sched_switch`
- Reads binary `latency_event` records from the ring buffer
- Prints decoded latency telemetry to stdout

4. Generate workload so the scheduler has something to observe.
```bash
yes > /dev/null
```

Stop it later with `Ctrl+C`.

### Optional: Trace Pipe Debugging
If you want kernel trace output for debugging:

```bash
cd kernel
make trace
```

This is optional. The main data path now is the ring buffer consumed by the Rust bridge, not `bpf_printk`.

### Build Only
To only compile the current components:

```bash
cd kernel
make sentinel.bpf.o

cd ../bridge
cargo check
```

### Shutdown / Cleanup
When you are done:

```bash
cd kernel
make unload
make clean
```

This removes the pinned BPF programs from `/sys/fs/bpf/kernelx_sentinel` and clears generated kernel artifacts.

---

## 🎯 Winning Objectives
* **Environment Innovation:** Moving AI training from digital sims to real-world system optimization.
* **Long-Horizon Planning:** Enabling agents to track hardware state over 300+ step trajectories.
* **Measurable Impact:** Targeting a $>15\%$ reduction in P99 latency under adversarial stress.

---
**Created for the Meta PyTorch OpenEnv Hackathon @ Scaler School of Technology.**
