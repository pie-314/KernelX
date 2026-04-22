from typing import List
from pydantic import BaseModel, Field

class Observation(BaseModel):
    """The 24D state vector extracted from the kernel."""
    features: List[float] = Field(..., min_items=24, max_items=24)
    timestamp: int
    pid: int
    cpu: int

class Action(BaseModel):
    """Priority weights for the process groups."""
    weights: List[float] = Field(..., min_items=4, max_items=4)

class State(BaseModel):
    """Metadata about the current session."""
    episode_id: str
    step_count: int
    latency_p99: float
    cpu_usage: float
