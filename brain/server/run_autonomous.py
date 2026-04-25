#!/usr/bin/env python3
"""
Autonomous KernelX environment runner.

This script runs the KernelX environment in autonomous mode, where the manual
policy automatically generates scheduling decisions based on kernel observations.

Usage:
    python run_autonomous.py [--steps N] [--interval MS]

This is useful for testing end-to-end functionality without requiring an external
RL agent client.
"""

import argparse
import time
import sys
from .kernelx_environment import KernelXEnvironment


def main():
    parser = argparse.ArgumentParser(description="Run KernelX autonomously with manual policy")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--interval", type=float, default=100.0, help="Interval between steps (ms)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed decision info")
    args = parser.parse_args()
    
    print("[KernelX] Starting autonomous mode with manual policy...")
    print(f"[KernelX] Will run {args.steps} steps, {args.interval}ms between steps\n")
    
    env = KernelXEnvironment()
    obs = env.reset()
    
    print(f"[KernelX] Episode: {env.episode_id}")
    print(f"[KernelX] Policy: Manual heuristic-based\n")
    
    try:
        for step in range(args.steps):
            # Run step with auto-generated action from policy
            obs = env.step()
            
            if args.verbose or step % 10 == 0:
                print(
                    f"[Step {step:3d}] "
                    f"PID={obs.pid:5d} | "
                    f"CPU load={obs.features[0]:6.1f} | "
                    f"Latency={obs.features[23]:6.1f} | "
                    f"Reward={obs.reward:7.2f}"
                )
            
            # Sleep between steps
            time.sleep(args.interval / 1000.0)
    
    except KeyboardInterrupt:
        print("\n[KernelX] Interrupted by user")
    except Exception as e:
        print(f"\n[KernelX] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n[KernelX] Completed {env.step_count} steps")
    print("[KernelX] Autonomous run finished successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
