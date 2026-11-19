"""Gymnasium wrapper that turns a TrajTrackAviary into the "hybrid" controller:
the RL policy only learns a bounded residual correction on top of a PID base
controller, rather than commanding the motors directly.

The wrapped env's action space is unchanged ([-1, 1]^4, one value per motor).
Here that action is reinterpreted as a fractional correction to the PID
controller's commanded RPM for that motor, clipped to
`residual_clip_fraction` of HOVER_RPM, so the policy can only nudge the PID
output rather than override it.
"""
import numpy as np
import gymnasium as gym

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


class HybridResidualWrapper(gym.Wrapper):

    def __init__(self, env, residual_clip_fraction: float = 0.2,
                 drone_model: DroneModel = DroneModel.CF2X):
        super().__init__(env)
        self.residual_clip_fraction = residual_clip_fraction
        self._drone_model = drone_model
        self.pid = DSLPIDControl(drone_model=drone_model)

    def reset(self, **kwargs):
        self.pid.reset()
        return self.env.reset(**kwargs)

    def step(self, residual_action):
        base = self.env.unwrapped
        state = base._getDroneStateVector(0)
        target_pos = base.current_target()

        pid_rpm, _, _ = self.pid.computeControlFromState(
            control_timestep=base.CTRL_TIMESTEP,
            state=state,
            target_pos=target_pos,
        )

        residual = np.clip(np.asarray(residual_action, dtype=np.float32).reshape(4), -1, 1)
        delta_rpm = self.residual_clip_fraction * base.HOVER_RPM * residual
        total_rpm = np.clip(pid_rpm + delta_rpm, 0, None)

        # Invert BaseRLAviary._preprocessAction's RPM mapping
        # (rpm = HOVER_RPM * (1 + 0.05*a)) so the underlying env, which expects
        # a normalized [-1, 1] action, reproduces exactly `total_rpm`.
        norm_action = (total_rpm / base.HOVER_RPM - 1.0) / 0.05
        norm_action = np.clip(norm_action, -1, 1).reshape(1, 4)

        return self.env.step(norm_action)
