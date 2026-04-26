# 🚀 KernelX: Final submission Runbook

Follow these steps in order to launch the full eBPF + AI scheduling stack.

## 🛰️ Step 1: Kernel Sentinel (Sensor)
Loads the eBPF probes into the Linux scheduler to begin 24D telemetry extraction.
```bash
cd kernel
sudo make load
```

## 🛠️ Step 2: Rust Bridge (Data Hub)
Reads eBPF ring buffer, manages Shared Memory (SHM), and saves training data.
```bash
# --record flag enables saving transitions to trajectories.json
cargo run --manifest-path bridge/Cargo.toml --release -- --record
```

## 🧠 Step 3: Strategist Brain (OpenEnv Server)
Starts the FastAPI environment and the LLM-based Grader (OpenEnv Compliance).
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 -m brain.server.app
```

## 🎭 Step 4: Autonomous Loop (The Actor)
The scheduling loop that queries the policy and sends priority nudges to the Bridge via ZMQ.
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 -m brain.server.run_autonomous --verbose --interval 50
```

## 🎨 Step 5: Mission Control HUD (Visualization)
Bt-op inspired TUI for the judges to see live telemetry, AI reasoning, and rewards.
```bash
cargo run --manifest-path ui/Cargo.toml
```

---

### 📈 Training Pipeline
If you have collected enough data in `trajectories.json` and want to retrain:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 training/run_pipeline.py --raw-data trajectories.json --output-root training_results
```
