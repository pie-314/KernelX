"""
KernelX Intelligence Layer — Multi-Objective Reward Function

Decomposes R_t = alpha * log(throughput + 1) - beta * delta_wait - gamma * |a_t - a_{t-1}|
into independent, inspectable reward components.

Throughput proxy: delta(sum_exec_runtime) since IPC is not yet collected
from PMU sidecars (indices 13-22 are zero in current data).
"""

import numpy as np

# Active feature indices (positions within the 10D active vector)
IDX_CPU           = 0
IDX_PRIO          = 1
IDX_STATIC_PRIO   = 2
IDX_NORMAL_PRIO   = 3
IDX_EXEC_NS       = 4   # symlog-scaled sum_exec_runtime
IDX_VRUNTIME      = 5   # symlog-scaled vruntime
IDX_MIGRATIONS    = 6   # symlog-scaled nr_migrations
IDX_CPUS_ALLOWED  = 7
IDX_CTX_SWITCHES  = 8
IDX_WAIT_US       = 9   # wait time in microseconds


class RewardComputer:
    """Multi-objective reward for kernel scheduling decisions.

    Components:
        throughput  — reward for CPU progress (delta exec_runtime)
        latency     — penalty for increased wait time
        stability   — penalty for jittery action changes
        format      — reward for action value in valid range
    """

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def throughput_reward(self, state: list, next_state: list) -> float:
        """Reward for throughput improvement.

        Uses delta(sum_exec_runtime) as proxy for IPC (not yet available).
        Index 4 = sum_exec_runtime (symlog-scaled).
        Positive delta means the process accumulated CPU time = progress.
        """
        exec_delta = next_state[IDX_EXEC_NS] - state[IDX_EXEC_NS]
        return self.alpha * float(np.log(max(0.0, exec_delta) + 1))

    def latency_reward(self, state: list, next_state: list) -> float:
        """Penalty for increased wait time.

        Index 9 = wait_time in microseconds (raw, not symlog-scaled).
        Positive delta means wait time increased = bad.
        """
        wait_delta = next_state[IDX_WAIT_US] - state[IDX_WAIT_US]
        return -self.beta * max(0.0, wait_delta)

    def stability_reward(self, action: float, prev_action: float) -> float:
        """Penalty for jittery scheduling changes."""
        return -self.gamma * abs(action - prev_action)

    def format_reward(self, action_value: float) -> float:
        """Reward for action value in valid range."""
        return 1.0 if -1.0 <= action_value <= 1.0 else 0.0

    def compute_total(
        self,
        state: list,
        action,  # KernelAction or has .value attribute
        prev_action: float,
        next_state: list,
    ) -> dict:
        """Compute all reward components and return breakdown."""
        action_val = action.value if hasattr(action, "value") else float(action)

        r_throughput = self.throughput_reward(state, next_state)
        r_latency = self.latency_reward(state, next_state)
        r_stability = self.stability_reward(action_val, prev_action)
        r_format = self.format_reward(action_val)

        total = r_throughput + r_latency + r_stability + r_format

        return {
            "total": total,
            "throughput": r_throughput,
            "latency": r_latency,
            "stability": r_stability,
            "format": r_format,
        }

    def calibrate(self, records: list, n_samples: int = 200) -> dict:
        """Run reward function on random actions to verify healthy distribution.

        Returns stats about reward distribution for tuning alpha/beta/gamma.
        """
        import random

        totals = []
        components = {"throughput": [], "latency": [], "stability": [], "format": []}

        samples = random.sample(records, min(n_samples, len(records)))
        for rec in samples:
            fake_action_val = random.uniform(-1.0, 1.0)
            fake_prev = random.uniform(-1.0, 1.0)

            r_t = self.throughput_reward(rec["state"], rec["next_state"])
            r_l = self.latency_reward(rec["state"], rec["next_state"])
            r_s = self.stability_reward(fake_action_val, fake_prev)
            r_f = self.format_reward(fake_action_val)

            total = r_t + r_l + r_s + r_f
            totals.append(total)
            components["throughput"].append(r_t)
            components["latency"].append(r_l)
            components["stability"].append(r_s)
            components["format"].append(r_f)

        stats = {
            "total": {
                "mean": float(np.mean(totals)),
                "std": float(np.std(totals)),
                "min": float(np.min(totals)),
                "max": float(np.max(totals)),
                "zero_rate": float(np.mean([1 if t == 0 else 0 for t in totals])),
            }
        }
        for name, vals in components.items():
            stats[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        print("\n=== Reward Calibration ===")
        for name, s in stats.items():
            print(f"  {name:>12}: mean={s['mean']:+.3f}  std={s['std']:.3f}  "
                  f"range=[{s['min']:+.3f}, {s['max']:+.3f}]")

        return stats
