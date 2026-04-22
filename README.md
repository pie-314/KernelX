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

### Prerequisites (Arch Linux)
```bash
# System Toolchain
sudo pacman -S base-devel clang llvm libelf libbpf linux-tools
# Rust BPF Setup
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
cargo install bpf-linker
```

### Installation
1.  **Clone the Repo:** `git clone https://github.com/pie-314/kernelx`
2.  **Generate Headers:** ```bash
    bpftool btf dump file /sys/kernel/btf/vmlinux format c > kernel/include/vmlinux.h
    ```
3.  **Build the Nervous System:**
    ```bash
    cd bridge && cargo build --release
    ```

---

## 🎯 Winning Objectives
* **Environment Innovation:** Moving AI training from digital sims to real-world system optimization.
* **Long-Horizon Planning:** Enabling agents to track hardware state over 300+ step trajectories.
* **Measurable Impact:** Targeting a $>15\%$ reduction in P99 latency under adversarial stress.

---
**Created for the Meta PyTorch OpenEnv Hackathon @ Scaler School of Technology.**
