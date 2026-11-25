"""Phase 2 zero-shot robustness sweep across multiple training seeds
(docs/gpu_machine_plan.md section 4's seed-sweep request). Extends
evaluate_robustness.py's single-seed-per-controller sweep to the 3 PPO and
3 Hybrid models trained in results/phase2/{ppo,hybrid}_seed{0,1,2}, so the
phase 1 question ("does PID beat PPO/Hybrid outside the training range?")
can be reported as a mean +/- spread across seeds rather than one run.

Reuses run_pid_episode/run_rl_episode from run_pid_baseline.py/
evaluate_robustness.py so the per-episode protocol (success definition,
deterministic actions, episode seeding) is identical to phase 1 and the
single-seed evaluate_robustness.py sweep -- only the model source and the
addition of a `seed` column are new.

Usage
-----
    python scripts/evaluate_seed_sweep.py \
        --mass-scales 0.7 0.8 0.9 1.0 1.1 1.2 1.3 \
        --episodes 30
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO

from src.train_utils import load_config
from run_pid_baseline import run_episode as run_pid_episode
from evaluate_robustness import run_rl_episode

SEEDS = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default="results/phase2")
    parser.add_argument("--ppo-config", type=str, default="config/ppo.yaml")
    parser.add_argument("--hybrid-config", type=str, default="config/hybrid.yaml")
    parser.add_argument("--mass-scales", type=float, nargs="+",
                         default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/phase2/robustness_sweep_seeds.csv")
    args = parser.parse_args()

    ppo_cfg = load_config(args.ppo_config)
    hybrid_cfg = load_config(args.hybrid_config)

    rows = []

    for mass_scale in args.mass_scales:
        for ep in range(args.episodes):
            seed = 1000 + ep
            pid_res = run_pid_episode(mass_scale=mass_scale, seed=seed)
            rows.append({"controller": "pid", "train_seed": "n/a", "mass_scale": mass_scale,
                         "episode": ep, "rmse": pid_res["rmse"],
                         "mean_error": pid_res["mean_error"], "max_error": pid_res["max_error"],
                         "episode_length": pid_res["episode_length"],
                         "success": pid_res["success"]})
        print(f"[evaluate_seed_sweep] pid mass_scale={mass_scale:.2f} done")

    for controller, cfg, hybrid in [("ppo", ppo_cfg, False), ("hybrid", hybrid_cfg, True)]:
        for train_seed in SEEDS:
            model_path = os.path.join(args.models_dir, f"{controller}_seed{train_seed}", "best_model.zip")
            if not os.path.exists(model_path):
                print(f"[WARN] {model_path} not found, skipping.")
                continue
            model = PPO.load(model_path)
            residual_clip = cfg["hybrid"]["residual_clip_fraction"] if hybrid else None

            for mass_scale in args.mass_scales:
                for ep in range(args.episodes):
                    seed = 1000 + ep
                    res = run_rl_episode(model, mass_scale, hybrid=hybrid, env_cfg=cfg["env"],
                                          residual_clip=residual_clip, seed=seed)
                    rows.append({"controller": controller, "train_seed": train_seed,
                                 "mass_scale": mass_scale, "episode": ep, **res})
                print(f"[evaluate_seed_sweep] {controller} train_seed={train_seed} "
                      f"mass_scale={mass_scale:.2f} done")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["controller", "train_seed", "mass_scale", "episode",
                                               "rmse", "mean_error", "max_error",
                                               "episode_length", "success"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[evaluate_seed_sweep] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
