import uuid
import time
import os
import numpy as np
import mmap
from typing import Tuple
from openenv_core import Environment, StepResult
from ..models import Observation, Action, State

SHM_PATH = "/dev/shm/kernelx_state"
# Size of HUDState: 24*8 (features) + 4 (action) + 4 (pid) + 4 (clamped) + 128 (reasoning) + 8 (wait) = 340 bytes
# Actually, the struct is packed, so we should be careful. 
# features(192) + action(4) + pid(4) + clamped(4) + reasoning(128) + wait(8) = 340
SHM_SIZE = 340

class KernelXEnvironment(Environment[Observation, Action, State]):
    def __init__(self):
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.shm = None
        
        if os.path.exists(SHM_PATH):
            fd = os.open(SHM_PATH, os.O_RDONLY)
            self.shm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
            print(f"[KernelX] Connected to live Telemetry SHM.")
        else:
            print(f"[Warn] SHM file not found. AI will use fallback data.")

        print(f"[KernelX] OpenEnv Server Initialized. Session: {self.episode_id}")

    def reset(self) -> Observation:
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())
        return self._get_observation()

    def step(self, action: Action) -> StepResult[Observation]:
        # 1. Apply Action (Placeholder for now)
        self._apply_action(action)
        
        # 2. Collect next observation from SHM
        time.sleep(0.01) 
        obs = self._get_observation()
        self.step_count += 1
        
        return StepResult(
            observation=obs,
            reward=self._calculate_reward(obs),
            done=False,
            info={"step": self.step_count}
        )

    def _get_observation(self) -> Observation:
        if self.shm:
            self.shm.seek(0)
            data = self.shm.read(SHM_SIZE)
            
            # Unpack the 24D features (first 192 bytes)
            features = np.frombuffer(data[:192], dtype=np.uint64).astype(float).tolist()
            
            # Unpack PID (at offset 200) and Wait (at offset 332)
            # (Calculation based on #[repr(C, packed)] in main.rs)
            # features: 0..192
            # action: 192..196
            # pid: 196..200
            # clamped: 200..204
            # reasoning: 204..332
            # wait: 332..340
            pid = int.from_buffer(data[196:200], "little")
            wait = int.from_buffer(data[332:340], "little")
            
            return Observation(
                features=features,
                timestamp=int(time.time_ns()),
                pid=pid,
                cpu=0 # SMP ID is in features[0]
            )
        
        # Fallback to random if SHM is missing
        return Observation(
            features=list(np.random.rand(24).astype(float)),
            timestamp=int(time.time_ns()),
            pid=1234,
            cpu=0
        )

    def _apply_action(self, action: Action):
        # Placeholder: Send to Rust Bridge ZMQ
        pass

    def _calculate_reward(self, obs: Observation) -> float:
        # R = alpha * log(throughput) - beta * latency
        # features[23] is our latency metric
        latency = obs.features[23]
        return -float(latency) # Simplified for prototype
