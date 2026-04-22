from openenv_core import EnvClient
from .models import Observation, Action, State

class KernelXClient(EnvClient[Observation, Action, State]):
    """
    Client-side interface for the KernelX environment.
    Use this to connect to a running KernelX OpenEnv server.
    """
    def __init__(self, url: str = "http://localhost:8000"):
        super().__init__(url, Observation, Action, State)
