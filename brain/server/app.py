from openenv.core import create_fastapi_app
from .kernelx_environment import KernelXEnvironment
from ..models import Observation, Action

# Create the FastAPI app with OpenEnv routing
# Pass the class itself, as the server will instantiate it
app = create_fastapi_app(KernelXEnvironment, action_cls=Action, observation_cls=Observation)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
