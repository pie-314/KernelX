# KernelX: Cognitive Operating System Layer 🌪️

**KernelX** is an autonomous intelligence plane for the Linux Kernel. It replaces static scheduling heuristics with a Reinforcement Learning loop that optimizes hardware state (CPU, Cache, I/O) in real-time.

---

## 👥 The KernelX Trinity (Team Leads)
*   **Systems Lead (Sentinel):** The "Metal" layer. eBPF sensors and Rust Bridge.
*   **AI Lead (Strategist):** The "Brain" layer. PPO Agents and Gymnasium Environment.
*   **Governance Lead (Auditor):** The "Safety" layer. Action clipping and TUI Monitoring.

---

## 🚀 Getting Started (The Metal Layer)

This guide helps you build and test the **Perception Loop** (eBPF -> Rust Bridge).

### 1. Prerequisites
You need a Linux machine (Arch/Ubuntu) with BTF enabled.
```bash
# Ubuntu
sudo apt install clang llvm libelf-dev libbpf-dev linux-tools-$(uname -r) cargo
# Arch
sudo pacman -S clang llvm libelf libbpf bpftool cargo
```

### 2. Build the Kernel Sensor (eBPF)
The Sentinel extracts a **24-Dimensional State Vector** directly from the scheduler.
```bash
cd kernel
# Generate kernel headers
make vmlinux
# Compile and load into kernel
make load
```
*Verification:* Run `make trace` to see the raw heartbeat of the kernel.

### 3. Run the Rust Bridge (Telemetry)
The Bridge converts raw kernel bytes into structured telemetry for the AI.
```bash
cd bridge
cargo build
sudo ./target/debug/kernelx-bridge
```
*Expected Output:* You should see a live stream of **24D Vectors** showing PIDs, Wait Latency (us), and VRuntime.

### 4. Cleanup
To safely detach the sensors from your kernel:
```bash
cd kernel
make unload
```

---

## 📂 Project Structure
- `/kernel`: eBPF C code (The "Eyes").
- `/bridge`: Rust Aya application (The "Nervous System").
- `/brain`: Python Gymnasium & PyTorch (The "Intelligence").
- `/ui`: Ratatui TUI dashboard (The "Observability").
- `implementation.md`: Technical deep-dive and math.
- `plan.md`: Current team roadmap and task guide.

---

## 📐 The 24D State Vector
Our agents observe the following micro-metrics per task switch:
- `[0-3]`: CPU Affinity & Priority levels.
- `[4-7]`: Scheduling stats (VRuntime, Exec Runtime).
- `[8-11]`: Run-queue lengths & System Load.
- `[12-15]`: Context Switch frequencies.
- `[23]`: **Ready Queue Latency (μs)** - Our primary reward metric.

---
**Developed for the Meta PyTorch OpenEnv Hackathon.**
