"""Single-drone circular-trajectory-tracking RL environment.

Uses the same circular trajectory as gym_pybullet_drones' own `pid.py` example
(see src/traj.py), so the PID baseline, PPO, and the hybrid controller are all
evaluated on an identical reference path.

Supports Domain Randomization over the drone's true mass: at each reset, the
PyBullet rigid body's mass is scaled by a factor drawn uniformly from
`mass_scale_range`, while the controller's internal model of the mass
(`self.M`, used by gym_pybullet_drones to compute HOVER_RPM and by the PID
gains) is left at the nominal value. This reproduces "the physical drone is
heavier/lighter than the controller assumes" without changing what the policy
observes. This is exactly the scenario docs/research_plan.md tests.
"""
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

from src.traj import build_waypoints


class TrajTrackAviary(BaseRLAviary):
    """Single agent RL problem: track a fixed circular trajectory."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 mass_scale_range: tuple = (1.0, 1.0),
                 ):
        self.TARGET_POS = build_waypoints(ctrl_freq)
        self.NUM_WP = self.TARGET_POS.shape[0]
        self.EPISODE_LEN_SEC = self.NUM_WP / ctrl_freq
        self.MASS_SCALE_RANGE = mass_scale_range
        self._mass_scale = 1.0

        initial_xyzs = np.array([self.TARGET_POS[0, :]])

        super().__init__(drone_model=drone_model,
                          num_drones=1,
                          initial_xyzs=initial_xyzs,
                          physics=physics,
                          pyb_freq=pyb_freq,
                          ctrl_freq=ctrl_freq,
                          gui=gui,
                          record=record,
                          obs=obs,
                          act=act,
                          )

    ################################################################################

    def reset(self, seed: int = None, options: dict = None):
        obs, info = super().reset(seed=seed, options=options)
        lo, hi = self.MASS_SCALE_RANGE
        self._mass_scale = float(self.np_random.uniform(lo, hi)) if hi > lo else lo
        if self._mass_scale != 1.0:
            p.changeDynamics(self.DRONE_IDS[0], -1,
                              mass=self.M * self._mass_scale,
                              physicsClientId=self.CLIENT)
        return obs, info

    ################################################################################

    def current_target(self) -> np.ndarray:
        """XYZ position the drone should be at right now, in this control step."""
        wp_idx = int(self.step_counter / self.PYB_STEPS_PER_CTRL) % self.NUM_WP
        return self.TARGET_POS[wp_idx, :]

    ################################################################################

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.current_target() - state[0:3])
        return -float(dist)  # dense: minimizing tracking error maximizes reward

    def _computeTerminated(self):
        return False  # no early-success condition for trajectory tracking

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 2.0 or abs(state[1]) > 2.0 or state[2] > 2.0
                or abs(state[7]) > .6 or abs(state[8]) > .6):  # too far / too tilted
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.current_target() - state[0:3])
        return {"tracking_error": float(dist), "mass_scale": self._mass_scale}
