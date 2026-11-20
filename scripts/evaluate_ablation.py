"""Appends one more controller's robustness-sweep results to an existing
results/robustness_sweep.csv, under a custom label. Used for ablations (e.g.
config/hybrid_tight_clip.yaml) without re-running PID/PPO/original-Hybrid,
and without modifying evaluate_robustness.py's three-controller CLI.

Reuses the exact same mass scales, episode count, and seed scheme as
evaluate_robustness.py so the new controller is directly comparable to the
rows already in the CSV. scripts/make_plots.py groups dynamically by
whatever controller labels are present, so a 4th label just adds a 4th line
to the plot automatically.

Usage
-----
    python scripts/evaluate_ablation.py \
        --label hybrid_tight_clip \
        --model results/hybrid_tight_clip/best_model.zip \
        --config config/hybrid_tight_clip.yaml \
        --mass-scales 0.7 0.8 0.9 1.0 1.1 1.2 1.3 \
        --episodes 10
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO

from src.train_utils import load_config
from evaluate_robustness import run_rl_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True,
                         help="Controller name to write into the CSV (e.g. hybrid_tight_clip)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mass-scales", type=float, nargs="+",
                         default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/robustness_sweep.csv")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"{args.model} not found — train it first.")

    cfg = load_config(args.config)
    model = PPO.load(args.model)

    rows = []
    for mass_scale in args.mass_scales:
        for ep in range(args.episodes):
            seed = 1000 + ep  # matches evaluate_robustness.py's seed scheme
            res = run_rl_episode(model, mass_scale, hybrid=True,
                                 env_cfg=cfg["env"],
                                 residual_clip=cfg["hybrid"]["residual_clip_fraction"],
                                 seed=seed)
            rows.append({"controller": args.label, "mass_scale": mass_scale,
                        "episode": ep, **res})
        print(f"[evaluate_ablation] {args.label} mass_scale={mass_scale:.2f} done")

    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["controller", "mass_scale", "episode",
                                               "rmse", "mean_error", "max_error",
                                               "episode_length", "success"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[evaluate_ablation] appended {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
