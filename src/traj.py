"""Shared reference trajectory used by every controller (PID, PPO, Hybrid).

Keeping this in one place guarantees the three controllers are evaluated on an
identical path, which is what makes the comparison in docs/research_plan.md
meaningful. Parameters match gym_pybullet_drones' own `examples/pid.py` demo
(a circular flight at constant altitude around point (0, -R)).
"""
import numpy as np

R = 0.3          # trajectory radius, meters
H = 0.1          # constant altitude, meters
PERIOD_SEC = 10.0  # time to complete one full loop


def build_waypoints(ctrl_freq_hz: int) -> np.ndarray:
    """Return an (N, 3) array of XYZ waypoints, one per control step, covering
    exactly one period of the circular trajectory at the given control rate.
    """
    num_wp = int(ctrl_freq_hz * PERIOD_SEC)
    wp = np.zeros((num_wp, 3))
    for i in range(num_wp):
        angle = (i / num_wp) * 2 * np.pi + np.pi / 2
        wp[i, :] = (R * np.cos(angle), R * np.sin(angle) - R, H)
    return wp
