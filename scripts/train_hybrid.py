"""Train the hybrid (PID + RL-residual) trajectory-tracking controller
(controller #3 of 3). Uses the same env settings and training budget as
scripts/train_ppo.py (config/hybrid.yaml mirrors config/ppo.yaml) so the two
are a fair comparison.

Usage
-----
    python scripts/train_hybrid.py --config config/hybrid.yaml
    python scripts/train_hybrid.py --config config/hybrid.yaml --total-timesteps 2000  # quick smoke test
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.train_utils import load_config, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/hybrid.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None,
                         help="Override config's training.total_timesteps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    final_path = run_training(cfg, hybrid=True, total_timesteps_override=args.total_timesteps)
    print(f"[train_hybrid] done. Model saved to {final_path}")


if __name__ == "__main__":
    main()
