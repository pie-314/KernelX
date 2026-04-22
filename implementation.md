# KernelX: Master Implementation Blueprint

This document serves as the sole source of truth for the complete implementation of KernelX. It defines the mathematical models, the architectural flow, component specifications, and task assignments.

## 1. System Architecture & Data Flow

KernelX operates as an ultra-low-latency closed-loop control system interacting directly with the Linux kernel via eBPF.

```text
[ Linux Kernel (CFS) ] <==========================[ Actuation ]========================
       |                                                                            |
       |  (sched_wakeup, sched_switch)                                              |
       v                                                                            |
[ eBPF Sentinel (sentinel.bpf.o) ]                                                  |
       |                                                                            |
       |  (24D State Vector via BPF_MAP_TYPE_RINGBUF)                               |
       v                                                                            |
[ Rust Bridge (kernelx-bridge) ] ===========> [ RadishDB (libradish.a) ]            |
       |                                           (Persistent WAL / Training Data) |
       |  (gRPC / Shared Memory / IPC)                                              |
       v                                                                            |
[ Python Inference Server (env/vayu_gym.py) ]                                       |
       |                                                                            |
       |  (RL Forward Pass: PPO Agent)                                              |
       v                                                                            |
[ Auditor Module (Rust/Safety) ] --(Action Clipping)--> [ eBPF Map (Priority) ] =====
```

## 2. Mathematical Formulation (The "Brain")

### 2.1 State Space ($S \in \mathbb{R}^{24}$)
The environment state $s_t$ is a 24-dimensional vector extracted via eBPF per time-step $t$:
*   `[0-3]`: CPU Load averages (per-core).
*   `[4-7]`: Cache miss rates (LLC).
*   `[8-11]`: Run-queue lengths.
*   `[12-15]`: Context switch frequencies.
*   `[16-19]`: IPC (Instructions Per Cycle) estimates via PMU.
*   `[20-23]`: I/O wait times.

### 2.2 Action Space ($A \in [-1, 1]^N$)
The action $a_t$ represents the "Priority Weight" or "Time Slice" adjustment applied to active thread groups. Continuous action space bounded between -1 (demote) and 1 (promote).

### 2.3 Reward Function ($R$)
The goal is to maximize throughput while penalizing micro-latency.

$$ r_t = \alpha \cdot \text{Throughput}_t - \beta \cdot \text{Latency}_t $$

Where:
*   $\text{Latency} (\Delta t) = t_{switch} - t_{wakeup}$ (measured in $\mu s$ by `sentinel.bpf.c`).
*   $\alpha$ and $\beta$ are hyperparameter weights (e.g., $\alpha = 1.0, \beta = 0.05$).

### 2.4 PPO Objective (Proximal Policy Optimization)
The agent optimizes the clipped surrogate objective to prevent catastrophic policy updates:

$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right] $$
Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ and $\hat{A}_t$ is the Generalized Advantage Estimate.

---

## 3. Team Responsibilities & Task Delegation

### Lead 1: Systems ("The Metal")
*   **eBPF Sensor:** Convert `bpf_printk` to `BPF_MAP_TYPE_RINGBUF` in `kernel/sentinel.bpf.c` for zero-copy state extraction.
*   **Actuator Map:** Create `BPF_MAP_TYPE_HASH` for priority weights. Intercept scheduling decisions and apply these weights.
*   **RadishDB C-Core:** Implement `libradish.a` (a high-speed, crash-safe append-only B-Tree/WAL database) for storing trajectories.
*   **Rust FFI:** Use `bindgen` in the `bridge/` to link `libradish.a` to the Rust bridge.

### Lead 2: AI ("The Brain")
*   **Gym Environment:** Finish `brain/env/vayu_gym.py` conforming to the `gymnasium` API. `step(action)` must send IPC to Rust bridge, wait for execution, and return `(next_state, reward, done, info)`.
*   **Imitation Learning (Shadow Mode):** Train the initial `strategist.pth` using supervised MSE loss against the default Linux CFS decisions to prevent initial system crashes.
*   **PPO Loop:** Implement the RL loop using Ray/RLlib or Stable Baselines 3.

### Lead 3: Governance & UI ("The Safety")
*   **Auditor Module:** Write Rust logic in the bridge that intercepts AI actions. If $a_t$ attempts to starve a critical system process (PID < 1000) or SSH daemon, the action is clamped/clipped.
*   **Ratatui HUD:** Build `ui/main.rs`. Connect via localhost socket to the Rust bridge to visualize the 24D vector, P99 latency, and AI confidence in real-time.
*   **Adversary Suite:** Develop `adversary/chaos_monkey.sh` using `stress-ng` to simulate noisy neighbor workloads, cache thrashing, and fork-bombs to evaluate the AI's response.

---

## 4. Implementation Protocol (Step-by-Step)

### Phase 1: High-Speed Telemetry (Systems)
1.  **Modify BPF (`sentinel.bpf.c`):** Define `struct state_vector { u64 features[24]; u64 timestamp; }`. Push this to a ring buffer on `sched_switch`.
2.  **Bridge (`bridge/src/main.rs`):** Use `aya`'s `RingBuf` API to poll the buffer. Batch reads to reduce overhead.

### Phase 2: Data Persistence & Communication (Systems & AI)
1.  **RadishDB:** Build the static library.
2.  **Bridge FFI:** In Rust, `unsafe { radish_insert(vector.as_ptr(), vector.len()) }`.
3.  **Bridge-to-Python IPC:** Implement a ZeroMQ (ZMQ) PUSH/PULL socket or Unix Domain Socket in `kernelx-bridge` to stream the 24D vectors to Python.

### Phase 3: Shadow Training (AI)
1.  **Run Workloads:** Execute `chaos_monkey.sh`.
2.  **Collect Data:** RadishDB logs state transitions and default CFS scheduling outcomes.
3.  **Train:** Train PyTorch model to mimic CFS. Verify with MSE loss < 0.01.

### Phase 4: Actuation & Governance (Systems & Safety)
1.  **Safety First:** Implement `auditor.rs` in the bridge. Define immutable PIDs.
2.  **Actuator BPF Map:** `BPF_MAP_TYPE_ARRAY` or `HASH`. Key = PID, Value = Weight.
3.  **BPF Injection:** Update `sentinel.bpf.c` to read the map and adjust standard CFS `vruntime`.

### Phase 5: Autonomous RL (All)
1.  **Close the Loop:** Python model infers $a_t$, sends over ZMQ to Bridge. Bridge runs Auditor. Bridge writes to BPF Map.
2.  **Live Monitoring:** Boot `ui/main.rs` to observe system stability and reward optimization.
