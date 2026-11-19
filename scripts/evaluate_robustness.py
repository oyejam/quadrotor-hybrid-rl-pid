"""Zero-shot robustness sweep (the core experiment, hypothesis H1 in
docs/research_plan.md section 4.4): evaluate PID, PPO, and Hybrid at drone
mass values the policies were not necessarily trained on.

Usage
-----
    python scripts/train_ppo.py      # produces results/ppo/best_model.zip
    python scripts/train_hybrid.py   # produces results/hybrid/best_model.zip
    python scripts/evaluate_robustness.py \
        --mass-scales 0.7 0.8 0.9 1.0 1.1 1.2 1.3 \
        --episodes 10
"""
import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO

from src.envs.traj_track_aviary import TrajTrackAviary
from src.controllers.hybrid_wrapper import HybridResidualWrapper
from src.train_utils import load_config
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from run_pid_baseline import run_episode as run_pid_episode


def run_rl_episode(model, mass_scale, hybrid, env_cfg, residual_clip, seed) -> dict:
    env = TrajTrackAviary(act=ActionType(env_cfg.get("action_type", "rpm")),
                           obs=ObservationType(env_cfg.get("obs_type", "kin")),
                           pyb_freq=env_cfg.get("pyb_freq_hz", 240),
                           ctrl_freq=env_cfg.get("ctrl_freq_hz", 48),
                           mass_scale_range=(mass_scale, mass_scale))
    if hybrid:
        env = HybridResidualWrapper(env, residual_clip_fraction=residual_clip)

    obs, info = env.reset(seed=seed)
    errors = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        errors.append(info["tracking_error"])
    env.close()

    errors = np.array(errors) if errors else np.array([np.nan])
    return {"rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_error": float(errors.mean()),
            "max_error": float(errors.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-model", type=str, default="results/ppo/best_model.zip")
    parser.add_argument("--hybrid-model", type=str, default="results/hybrid/best_model.zip")
    parser.add_argument("--ppo-config", type=str, default="config/ppo.yaml")
    parser.add_argument("--hybrid-config", type=str, default="config/hybrid.yaml")
    parser.add_argument("--mass-scales", type=float, nargs="+",
                         default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/robustness_sweep.csv")
    args = parser.parse_args()

    ppo_cfg = load_config(args.ppo_config)
    hybrid_cfg = load_config(args.hybrid_config)

    ppo_model = PPO.load(args.ppo_model) if os.path.exists(args.ppo_model) else None
    hybrid_model = PPO.load(args.hybrid_model) if os.path.exists(args.hybrid_model) else None
    if ppo_model is None:
        print(f"[WARN] {args.ppo_model} not found, skipping PPO. Run scripts/train_ppo.py first.")
    if hybrid_model is None:
        print(f"[WARN] {args.hybrid_model} not found, skipping Hybrid. Run scripts/train_hybrid.py first.")

    rows = []
    for mass_scale in args.mass_scales:
        for ep in range(args.episodes):
            seed = 1000 + ep

            pid_res = run_pid_episode(mass_scale=mass_scale, seed=seed)
            rows.append({"controller": "pid", "mass_scale": mass_scale, "episode": ep,
                         "rmse": pid_res["rmse"], "mean_error": pid_res["mean_error"],
                         "max_error": pid_res["max_error"]})

            if ppo_model is not None:
                res = run_rl_episode(ppo_model, mass_scale, hybrid=False,
                                     env_cfg=ppo_cfg["env"], residual_clip=None, seed=seed)
                rows.append({"controller": "ppo", "mass_scale": mass_scale, "episode": ep, **res})

            if hybrid_model is not None:
                res = run_rl_episode(hybrid_model, mass_scale, hybrid=True,
                                     env_cfg=hybrid_cfg["env"],
                                     residual_clip=hybrid_cfg["hybrid"]["residual_clip_fraction"],
                                     seed=seed)
                rows.append({"controller": "hybrid", "mass_scale": mass_scale, "episode": ep, **res})

        print(f"[evaluate_robustness] mass_scale={mass_scale:.2f} done "
              f"({args.episodes} episodes/controller)")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["controller", "mass_scale", "episode",
                                               "rmse", "mean_error", "max_error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[evaluate_robustness] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
