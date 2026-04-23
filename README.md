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

### 3. Run the OpenEnv Server (Brain)
The Brain is a standard **OpenEnv** environment that exposes the kernel telemetry via a FastAPI server.
```bash
cd brain
# Install dependencies
pip install -r server/requirements.txt
# Start the OpenEnv server
python -m server.app
```
*Verification:* Access `http://localhost:8000/docs` to see the OpenEnv API.

### 4. Connect the Agent
You can now connect any AI agent using the `KernelXClient`:
```python
from brain import KernelXClient

client = KernelXClient(url="http://localhost:8000")
obs = client.reset()
# The agent takes actions based on the 24D vector
result = client.step(weights=[0.1, -0.2, 0.5, 0.0])
```

---

## 📂 Project Structure
- `/kernel`: eBPF C code (The "Eyes").
- `/bridge`: Rust Aya application (The "Nervous System").
- `/RadishDB`: **Crash-Safe WAL** (The "Memory").
  - Persistent storage for every kernel trajectory.
  - Supports binary-safe snapshots and JSONL export for AI training.
- `/brain`: **OpenEnv Implementation** (The "Intelligence").
  - `openenv.yaml`: Environment manifest.
  - `models.py`: Pydantic definitions for Action/Observation.
  - `server/`: FastAPI server logic.
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
