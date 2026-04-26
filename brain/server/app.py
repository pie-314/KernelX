from openenv.core import create_fastapi_app
from .kernelx_environment import KernelXEnvironment
from ..models import Observation, Action

# Create the FastAPI app with OpenEnv routing
# Pass the class itself, as the server will instantiate it
app = create_fastapi_app(KernelXEnvironment, action_cls=Action, observation_cls=Observation)


# --- Policy Iteration Integration ---
# Expose an endpoint for the policy iteration loop to hot-swap models

@app.post("/reload-policy")
async def reload_policy(model_path: str = None):
    """Hot-swap the active policy model. Called by training/policy_iteration.py."""
    # Access the environment instance from the OpenEnv app state
    env = None
    for route in app.routes:
        if hasattr(route, "endpoint") and hasattr(route.endpoint, "__self__"):
            obj = route.endpoint.__self__
            if isinstance(obj, KernelXEnvironment):
                env = obj
                break

    if env is None:
        # Try app.state
        env = getattr(app.state, "env", None)

    if env is None:
        return {"status": "error", "message": "Environment not found in app state"}

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
