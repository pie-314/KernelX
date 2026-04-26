import uuid
import time
import os
import numpy as np
import mmap
import zmq
from typing import Tuple, Optional, List, Dict, Any
from openenv_core import Environment
from brain.models import Observation, Action, State
from brain.server.policy import ManualPolicy
from brain.server.trained_policy import TrainedPolicy
from brain.server.grader import LLMGrader


SHM_PATH = "/dev/shm/kernelx_state"
SHM_SIZE = 376
ZMQ_BRIDGE_URL = "tcp://127.0.0.1:5555"

# Default GGUF model path (relative to project root)
DEFAULT_GGUF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "training", "models", "strategist-q4km.gguf"
)

class KernelXEnvironment(Environment[Observation, Action, State]):
    def __init__(self):
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.shm = None
        self.policy_path = None
        
        # OpenEnv Task Definitions
        self.tasks = [
            {"id": "latency_recovery", "description": "Reduce P99 wait time below 500us"},
            {"id": "throughput_max", "description": "Maximize execution runtime across active PIDs"},
            {"id": "safety_alignment", "description": "Issue actions that pass the LLM Grader check"}
        ]

        # Load best available policy: GGUF > ManualPolicy
        self._load_best_policy()
        
        # Initialize LLM Grader (Meta R2 Requirement)
        self.grader = LLMGrader(model_path=DEFAULT_GGUF if os.path.exists(DEFAULT_GGUF) else None)

        # Initialize ZMQ Socket to talk to Rust Bridge
        try:
            self.zmq_ctx = zmq.Context()
            self.zmq_socket = self.zmq_ctx.socket(zmq.PUSH)
            self.zmq_socket.connect(ZMQ_BRIDGE_URL)
            print(f"[KernelX] Connected to Bridge Actuator at {ZMQ_BRIDGE_URL}")
        except Exception as e:
            print(f"[Warn] ZMQ Connection failed: {e}")
        
        # Try to connect to Real Metal (shared memory from kernel)
        if os.path.exists(SHM_PATH):
            try:
                fd = os.open(SHM_PATH, os.O_RDONLY)
                self.shm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
                print(f"[KernelX] Connected to live Telemetry SHM.")
            except Exception as e:
                print(f"[KernelX] SHM Access Failed: {e}. Falling back to Simulator.")
        else:
            print(f"[Warn] SHM file not found. Using simulated data.")
        
        print(f"[KernelX] OpenEnv Server Initialized. Session: {self.episode_id}")

    def _load_best_policy(self):
        """Load the best available policy: trained GGUF > manual heuristic."""
        if os.path.exists(DEFAULT_GGUF):
            try:
                self.policy = TrainedPolicy(DEFAULT_GGUF)
                self.policy_path = DEFAULT_GGUF
                print(f"[KernelX] Loaded trained policy: {DEFAULT_GGUF}")
                return
            except Exception as e:
                print(f"[KernelX] Failed to load GGUF: {e}. Falling back to heuristic.")

        self.policy = ManualPolicy()
        self.policy_path = None
        print("[KernelX] Using ManualPolicy (heuristic)")

    def reload_policy(self, model_path: str = None):
        path = model_path or DEFAULT_GGUF
        if not os.path.exists(path):
            print(f"[KernelX] reload_policy: {path} not found")
            return False

        try:
            new_policy = TrainedPolicy(path)
            self.policy = new_policy
            self.policy_path = path
            print(f"[KernelX] Policy reloaded: {path}")
            return True
        except Exception as e:
            print(f"[KernelX] reload_policy failed: {e}")
            return False

    def reset(self) -> Observation:
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())
        return self._get_observation()

    def step(self, action: Optional[Action] = None) -> Observation:
        # 1. Get current observation
        obs = self._get_observation()
        
        # 2. If no action provided, use policy to generate one
        if action is None:
            action = self.policy.decide(obs)
        
        # 3. Extract reasoning for the Grader
        reasoning = getattr(self.policy, "last_reasoning", "Autonomous policy nudge")

        # 4. Apply Action to the Bridge
        self._apply_action(action)
        
        # 5. Collect next observation
        time.sleep(0.01) 
        next_obs = self._get_observation()
        self.step_count += 1
        
        # 6. Grade the performance (OpenEnv Compliance)
        grading = self.grader.calculate_score(obs, next_obs, reasoning)
        next_obs.reward = grading["score"] # Reward is now normalized 0.01 - 0.99
        
        if self.step_count % 10 == 0:
            print(f"[KernelX] Step {self.step_count}: Score={next_obs.reward:.4f} | Feedback: {grading['feedback']}")
        
        return next_obs

    def evaluate(self) -> Dict[str, Any]:
        """Final evaluation of the current episode."""
        obs = self._get_observation()
        return {
            "episode_id": self.episode_id,
            "total_steps": self.step_count,
            "final_score": obs.reward,
            "status": "completed"
        }

    def get_tasks(self) -> List[Dict[str, str]]:
        """Return the list of tasks for the judges."""
        return self.tasks

    def _get_observation(self) -> Observation:
        if self.shm:
            try:
                self.shm.seek(0)
                data = self.shm.read(SHM_SIZE)
                features = np.frombuffer(data[:192], dtype=np.uint64).astype(float).tolist()
                pid = int.from_bytes(data[196:200], "little")
                return Observation(features=features, timestamp=int(time.time_ns()), pid=pid, cpu=0)
            except Exception as e:
                print(f"[KernelX] SHM Read error: {e}")
        
        return Observation(features=list(np.random.rand(24).astype(float)), timestamp=int(time.time_ns()), pid=0, cpu=0)

    def _apply_action(self, action: Action):
        if self.shm:
            try:
                # Divide by 10 to normalize the policy's [-10, 10] weights to [-1, 1] for the kernel
                weight = float(action.weights[0]) / 10.0
                self.shm.seek(196)
                pid_bytes = self.shm.read(4)
                active_pid = int.from_bytes(pid_bytes, "little")
                target_pid = active_pid if active_pid > 0 else 9999
                
                confidence = 0.95
                drift = 0.01
                reason = getattr(self.policy, "last_reasoning", "Policy nudge")
                
                cmd = f"{target_pid}:{weight}:{confidence}:{drift}:{reason}"
                self.zmq_socket.send_string(cmd, zmq.NOBLOCK)
            except Exception as e:
                pass

    @property
    def state(self) -> State:
        return State(episode_id=self.episode_id, step_count=self.step_count, latency_p99=0.0, cpu_usage=0.0)
