# Teaching a Language Model to Schedule a Linux Kernel

*KernelX — Meta PyTorch OpenEnv Hackathon 2026, Theme 3.1: World Modeling*

---

## The premise

Linux makes a scheduling decision every few microseconds. The rules it uses — the Completely Fair Scheduler, CFS — were written in 2007 and haven't fundamentally changed since. They give every process a "fair" turn at the CPU regardless of what each process actually needs.

Fair, but not fast. Your latency-sensitive video call gets the same priority as a cron job checking disk space at 3 AM. The rules are static. They don't adapt.

We wanted to know: can a small language model learn to schedule better than eighteen years of hand-tuned heuristics? And more importantly — can building this environment teach us something useful about training LLMs as agents in *real* systems, not toy MDPs?

The answer to the first question is "yes, on a learned simulator." The answer to the second is what this post is about.

---

## What KernelX actually is

KernelX is an OpenEnv-compliant environment. The agent observes the Linux kernel's scheduling state and produces a single scalar action between -1 and +1 — a priority nudge for the currently-scheduled process. The environment returns a reward and the next state.

That much is standard. What makes it interesting for LLM training is what the state, the dynamics, and the reward actually are.

**The state is real kernel telemetry.** Twenty-four dimensions per observation: CPU ID, three priority fields, virtual runtime, total execution time, migration count, CPU affinity mask size, context switches, wait time in microseconds, and reserved slots for hardware performance counters. These are extracted by an eBPF program hooked into `sched_switch` — the actual context-switch event. We collected 534,134 of these from a real Linux machine running mixed workloads.

**The dynamics are learned.** A SmolLM2-360M model is fine-tuned to predict S<sub>t+1</sub> given (S<sub>t</sub>, a<sub>t</sub>). This is the World Model. The Strategist — the policy we ultimately want — trains by interacting with this World Model, not by replaying recorded data. That distinction matters. In a recorded-replay environment, the agent's action doesn't change anything; you're just optimizing a sequence-prediction loss with extra steps. In a learned-dynamics environment, the agent's action propagates through the simulator and the consequences come back as reward signal. That's the contract that makes this RL.

**The reward decomposes.** Throughput minus latency penalty minus stability penalty plus format reward. Each component is independently inspectable and independently weight-tunable. When the agent learns to game one component, you can see which one in the breakdown — debugging is tractable in a way that monolithic reward functions never are.

---

## Why kernel scheduling is a good world-modeling problem

Theme 3.1 of the hackathon — World Modeling — asks for environments that "require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting short-cuts."

Kernel scheduling has three properties that make it a strong test of an agent's world model:

The state is **partially observable**. The 24D vector is what eBPF can see at one tracepoint. It does not include the full run-queue, all per-CPU state, or the future arrival of new processes. The agent has to reason about what's happening based on a narrow window.

The consequences are **delayed**. A priority nudge applied at time t doesn't fully manifest until several context switches later. Some workload patterns — cache thrashing, lock contention, I/O bursts — only become visible across multi-step trajectories. An agent that only reacts to the immediate observation will lose to one that builds a temporal model.

The trade-offs are **real**. There is no scheduling policy that is universally best. Boosting an interactive process means demoting a batch one. Reducing wait time on critical paths means increasing average wait time elsewhere. The reward function exposes this trade-off explicitly through its three competing components, and the agent has to learn to balance them rather than ride one to a degenerate optimum.

These are properties that show up in any real-world deployment: partial observability, delayed feedback, multi-objective trade-offs. Training on KernelX is practice for the kinds of environments LLM agents will face when they're plugged into actual systems instead of single-turn benchmarks.

---

## How the system fits together

```
Linux kernel (eBPF sentinel)
   ↓ 24D telemetry at every sched_switch
Rust bridge (lockless ring buffer, sub-ms shared memory)
   ↓
Python brain (FastAPI + OpenEnv)
   ↓ Strategist outputs action ∈ [-1, +1]
ZMQ → Bridge → eBPF priority_actions map
   ↓
Kernel applies the nudge at the next context switch
```

Five components, each in the language that fits its job. The eBPF sentinel is C — it has to be, kernel-side. The bridge is Rust on Aya, because it needs lockless ring-buffer reads and sub-millisecond latency. The brain is Python on FastAPI, because OpenEnv is Python-native and llama.cpp's Python bindings are mature. The training pipeline is HuggingFace TRL with LoRA. The HUD is Ratatui.

The data layer worth highlighting: every component reads and writes the same 376-byte shared-memory struct at `/dev/shm/kernelx_state`. The C layout matches byte-for-byte across Rust and Python via `bytemuck::Pod` and `numpy.frombuffer`. This is what lets the brain and the UI run as separate processes without a serialization tax.

---

## Training

Two phases.

**Phase 1: Supervised warm-start.** Before RL, the model needs to learn the output format — that the answer to a kernel-state prompt is a single float between -1 and +1, not English prose. We feed it 200 examples generated by a heuristic policy and run two epochs of SFT with LoRA (r=16, α=32). After warm-start: 100% format compliance, training loss from 2.13 to 0.28.

**Phase 2: GRPO.** Group Relative Policy Optimization, from TRL. The reason for GRPO over PPO is that GRPO doesn't need a separate value function — it compares the rewards of N=8 generations sampled from the same prompt and computes advantages relative to the group mean. For an environment with high reward variance (which kernel scheduling absolutely is), this is a more stable signal than learning a value baseline.

Each GRPO step:
1. Sample a real kernel state from the dataset.
2. Generate 8 candidate actions from the current policy.
3. For each action, ask the World Model what the next state will be.
4. Compute the multi-objective reward against that predicted next state.
5. Update the policy to favor actions that scored above the group mean.

The full training loop runs on a free Colab T4 in a few hours. The trained adapter merges into the base model and quantizes to GGUF Q4_K_M — 258 MB, 44 ms inference on a laptop CPU.

---

## Quantization isn't a footnote

Sub-50ms inference is not a marketing target. It's the requirement for deploying inside an actual scheduling decision loop. A scheduler that takes 2 seconds to decide is a scheduler that *is* the latency it was supposed to fix.

The 1.4GB FP16 model goes through 4-bit quantization (Q4_K_M, llama.cpp's mixed-precision K-quant) and ends up at 258 MB. Generated text quality drops slightly. Inference latency drops from "unusable" to 44 ms. We measured this — see `training/inference/benchmark_latency.py` in the repo.

This is the same trade-off any real-time agent has to make. The interesting part isn't the technique (GGUF is well-trodden); it's that the environment's design constraints — partial observability over 24 dimensions, action space of one float — let us shrink the model enough to actually run in the loop.

---

## What changed when we switched to a learned World Model

We initially built a simpler version of the environment where actions were applied to recorded trajectories — an offline replay setup. Reward curves were beautiful. They were also lying.

The problem: in offline replay, the next state is whatever was recorded next, regardless of what action the agent picked. So the throughput and latency reward components depended only on the state pair, not on the action. The agent's choice could only influence the stability term (jitter) and the format term (in-range output). We were training a policy that had no actual purchase on its environment.

Switching to the learned World Model fixed this. The agent's action now flows through the simulator, the next state actually depends on it, and the reward signal reflects the consequence. Reward curves got messier. They were also honest.

We mention this because we think it's the kind of mistake that's easy to make when you're building an environment in a hurry. Recorded data is right there. It's tempting. But if your "RL training" is just running a sequence model over fixed transitions, you don't have an environment — you have a dataset with extra ceremony. The fix is the World Model. It costs one extra training stage. It's worth it.

---

## Results

The trained Strategist on held-out test states:

- Achieves higher cumulative reward than the untrained SmolLM2-360M baseline.
- Achieves higher cumulative reward than the hand-written heuristic policy.
- Produces a stable distribution of actions, not a single saturated value.
- Stays under the 50 ms decision budget on CPU inference.

Specific numbers are in [`training/PERFORMANCE.md`](https://github.com/pie-314/KernelX/blob/main/training/PERFORMANCE.md) in the repo. The plots:

- World Model SFT loss curve: `training/plots/world_model_training.png`
- Strategist warm-start loss: `training/plots/strategist_warmstart_training.png`
- GRPO reward curves vs baselines: `training/plots/grpo_training.png`

We are deliberately not putting a single headline percentage in this post. The hackathon brief is explicit that judges want to see "observable evidence of training progress" — reward curves and before/after comparisons — not a one-number marketing claim. The plots in the repo are the answer.

---

## What you can do with this

The environment is hosted on Hugging Face Spaces. The OpenEnv interface means any RL training loop you've already written should work against it:

```python
from brain.client import KernelXClient

env = KernelXClient(url="https://your-space.hf.space")
obs = env.reset()
obs = env.step(action=0.5)
score = env.evaluate()
```

Plug in TRL's `GRPOTrainer`, plug in Stable Baselines, plug in your own algorithm. The training notebook (`KernelX_Training.ipynb` in the repo) runs end-to-end on a free T4 — you can fork it, point it at our dataset, and have a trained scheduler in a few hours.

The hot-swap endpoint is the most fun part: `POST /reload-policy?model_path=...` swaps the GGUF model on the running brain server without downtime. Train a new variant, drop it in, watch the reward curve change live in the HUD.

---

## What this taught us about LLM-as-agent training

A few things, in order of how loudly we believe them:

The hardest part of building a good RL environment is not the agent loop, the reward, or even the training. It's making sure the action actually matters. We almost shipped a version where it didn't. If you're building an environment, the first sanity check is: pick a random action, then pick the opposite action, and verify the next state differs. If it doesn't, the agent isn't learning RL — it's learning sequence completion with extra steps.

A 360M parameter model is enough for sub-millisecond control problems if the state and action spaces are tight. We didn't need a 7B model. We didn't need 70B. The constraint was inference latency, and the constraint shaped the architecture down to the smallest model that could format-comply on the output. There's a lesson here about right-sizing.

Decomposed rewards are non-negotiable for debugging. The throughput-latency-stability-format breakdown made every "the agent is doing something weird" moment investigable. Monolithic reward functions are write-once, debug-never.

OpenEnv is a good fit for this kind of environment. The `Environment` base class is small, the FastAPI integration is clean, and the `/tasks` and `/evaluate` endpoints map naturally to hackathon-judging needs. We didn't have to fight the framework.

---

## Links

- **Repo**: [github.com/pie-314/KernelX](https://github.com/pie-314/KernelX)
- **HF Space (live environment)**: [huggingface.co/spaces/Rayugacodes/KernelX](https://huggingface.co/spaces/Rayugacodes/KernelX)
- **Training notebook (Colab T4)**: [`KernelX_Training.ipynb`](https://colab.research.google.com/github/pie-314/KernelX/blob/main/KernelX_Training.ipynb)
- **Trained model**: [Rayugacodes/kernelx-strategist](https://huggingface.co/Rayugacodes/kernelx-strategist)
- **Dataset (534K transitions)**: [Rayugacodes/kernelx-training-data](https://huggingface.co/datasets/Rayugacodes/kernelx-training-data)
- **Demo video**: *[YouTube link]*

---

*Built by Naman Gupta and team for the Meta PyTorch OpenEnv Hackathon 2026.*