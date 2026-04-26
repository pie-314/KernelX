from openenv_core import create_fastapi_app
from brain.server.kernelx_environment import KernelXEnvironment
from brain.models import Observation, Action

# Create the FastAPI app with OpenEnv routing
# Pass the class itself, as the server will instantiate it
app = create_fastapi_app(KernelXEnvironment, action_cls=Action, observation_cls=Observation)


# --- OpenEnv Compliance Endpoints ---

@app.get("/tasks")
async def get_tasks():
    """Returns official OpenEnv tasks for judges."""
    env = getattr(app.state, "env", None)
    if env:
        return {"tasks": env.get_tasks()}
    return {"error": "Environment not initialized"}


@app.get("/evaluate")
async def evaluate_episode():
    """Provides normalized score [0.01, 0.99] for the current session."""
    env = getattr(app.state, "env", None)
    if env:
        return env.evaluate()
    return {"error": "Environment not initialized"}


# --- Policy Iteration Integration ---

@app.post("/reload-policy")
async def reload_policy(model_path: str = None):
    """Hot-swap the active policy model."""
    env = getattr(app.state, "env", None)
    if env is None:
        return {"status": "error", "message": "Environment not found"}

    success = env.reload_policy(model_path)
    return {
        "status": "ok" if success else "failed",
        "policy_path": env.policy_path,
    }


@app.get("/policy-status")
async def policy_status():
    """Check which policy is currently active."""
    env = getattr(app.state, "env", None)
    if env is None:
        return {"policy": "unknown"}
    return {
        "policy_path": env.policy_path,
        "policy_type": type(env.policy).__name__,
    }


if __name__ == "__main__":
    import uvicorn
    # Store env in app state so endpoints can access it
    # Note: openenv-core usually handles instantiation
    uvicorn.run(app, host="0.0.0.0", port=8000)
