from openenv_core import create_fastapi_app
from .kernelx_environment import KernelXEnvironment

# Initialize the core environment
env = KernelXEnvironment()

# Create the FastAPI app with OpenEnv routing
app = create_fastapi_app(env)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
