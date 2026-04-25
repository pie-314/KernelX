"""
KernelX Intelligence Layer — Model Training Pipeline

Stages:
  0. Environment setup (requirements.txt)
  1. Data preprocessing (data/preprocess.py)
  2. World Model SFT training (models/train_world_model.py)
  3. Strategist RL training via GRPO (models/train_strategist.py)
  4. Monitoring (logged during training)
  5. Export & quantization to GGUF (models/export_gguf.py)
  6. Inference engine (inference/strategy_engine.py)
  7. Gradio demo (demo/app.py)
"""
