"""Headless PID baseline on the shared circular trajectory (src/traj.py).

This is controller #1 of the three-way comparison (PID / PPO / Hybrid). It
also doubles as the pattern for the robustness sweep in
scripts/evaluate_robustness.py: `--mass-scale` perturbs the drone's true mass
away from what the PID controller assumes, simulating out-of-distribution
dynamics.

Usage
-----
    python scripts/run_pid_baseline.py
    python scripts/run_pid_baseline.py --mass-scale 1.2 --gui
"""
import os
import sys
import argparse

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.traj import build_waypoints, is_out_of_bounds

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def run_episode(mass_scale: float = 1.0, ctrl_freq: int = 48, pyb_freq: int = 240,
                 gui: bool = False, seed: int = 0) -> dict:
    wp = build_waypoints(ctrl_freq)
    num_wp = wp.shape[0]
    init_xyz = np.array([wp[0, :]])

    env = CtrlAviary(drone_model=DroneModel('cf2x'), num_drones=1,
                      initial_xyzs=init_xyz, physics=Physics('pyb'),
                      pyb_freq=pyb_freq, ctrl_freq=ctrl_freq, gui=gui, obstacles=False)
    ctrl = DSLPIDControl(drone_model=DroneModel('cf2x'))

    obs, info = env.reset(seed=seed)
    if mass_scale != 1.0:
        p.changeDynamics(env.DRONE_IDS[0], -1, mass=env.M * mass_scale,
                          physicsClientId=env.CLIENT)

    errors = np.zeros(num_wp)
    crashed = False
    action = np.zeros((1, 4))
    for i in range(num_wp):
        obs, reward, terminated, truncated, info = env.step(action)
        state = obs[0]
        target = wp[i, :]
        errors[i] = np.linalg.norm(state[0:3] - target)
        crashed = crashed or is_out_of_bounds(state)
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP, state=state, target_pos=target)
    env.close()

    return {
        "mass_scale": mass_scale,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error": float(errors.mean()),
        "max_error": float(errors.max()),
        "episode_length": num_wp,
        "success": not crashed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PID baseline on the circular trajectory")
    parser.add_argument("--mass-scale", type=float, default=1.0,
                         help="True mass / nominal mass (1.0 = no domain shift)")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", type=str, default="results/pid_baseline.csv")
    args = parser.parse_args()

    result = run_episode(mass_scale=args.mass_scale, gui=args.gui)
    print(f"[PID baseline] mass_scale={result['mass_scale']:.2f}  "
          f"RMSE={result['rmse']:.4f} m  mean_error={result['mean_error']:.4f} m  "
          f"max_error={result['max_error']:.4f} m  success={result['success']}  "
          f"episode_length={result['episode_length']}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = not os.path.exists(args.output)
    with open(args.output, "a") as f:
        if write_header:
            f.write("mass_scale,rmse,mean_error,max_error,episode_length,success\n")
        f.write(f"{result['mass_scale']},{result['rmse']},{result['mean_error']},"
                f"{result['max_error']},{result['episode_length']},{result['success']}\n")
