"""Reproduce the core robustness plots (docs/results.md section 3) from
results/robustness_sweep.csv:
  1. Position RMSE vs. test-time mass deviation, one line per controller,
     with error bars across episodes (results/robustness_plot.png).
  2. Success rate (fraction of episodes that never went out of bounds, per
     src/traj.py's is_out_of_bounds) vs. mass deviation, one line per
     controller (results/success_rate_plot.png).

Usage
-----
    python scripts/make_plots.py
    python scripts/make_plots.py --input results/robustness_sweep.csv --output results/robustness_plot.png
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/robustness_sweep.csv")
    parser.add_argument("--output", type=str, default="results/robustness_plot.png")
    parser.add_argument("--success-output", type=str, default="results/success_rate_plot.png")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[make_plots] {args.input} not found. Run scripts/evaluate_robustness.py first.")
        return

    rows = load_rows(args.input)

    # group: controller -> mass_scale -> [rmse, ...]
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r["controller"]][float(r["mass_scale"])].append(float(r["rmse"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"pid": "tab:blue", "ppo": "tab:orange", "hybrid": "tab:green",
              "hybrid_tight_clip": "tab:red"}
    for controller, by_mass in grouped.items():
        mass_scales = sorted(by_mass.keys())
        means = np.array([np.mean(by_mass[m]) for m in mass_scales])
        stds = np.array([np.std(by_mass[m]) for m in mass_scales])
        deviation_pct = [(m - 1.0) * 100 for m in mass_scales]
        ax.errorbar(deviation_pct, means, yerr=stds, marker='o', capsize=3,
                    label=controller.upper(), color=colors.get(controller))

    ax.set_xlabel("Test-time mass deviation from nominal (%)")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Zero-shot robustness to mass perturbation")
    ax.axvspan(-10, 10, alpha=0.1, color='gray', label='Training DR range')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"[make_plots] saved {args.output}")

    # --- Success-rate plot (fraction of episodes that never went out of bounds) ---
    if not all("success" in r for r in rows):
        print("[make_plots] 'success' column missing from CSV (old format), "
              "skipping success-rate plot. Re-run scripts/evaluate_robustness.py.")
        return

    success_grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        success_grouped[r["controller"]][float(r["mass_scale"])].append(
            r["success"].strip().lower() == "true")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for controller, by_mass in success_grouped.items():
        mass_scales = sorted(by_mass.keys())
        rates = [np.mean(by_mass[m]) for m in mass_scales]
        deviation_pct = [(m - 1.0) * 100 for m in mass_scales]
        ax2.plot(deviation_pct, rates, marker='o', label=controller.upper(),
                 color=colors.get(controller))

    ax2.set_xlabel("Test-time mass deviation from nominal (%)")
    ax2.set_ylabel("Success rate (fraction completing the trajectory in-bounds)")
    ax2.set_title("Zero-shot success rate vs. mass perturbation")
    ax2.set_ylim(-0.05, 1.05)
    ax2.axvspan(-10, 10, alpha=0.1, color='gray', label='Training DR range')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    out_dir2 = os.path.dirname(args.success_output)
    if out_dir2:
        os.makedirs(out_dir2, exist_ok=True)
    fig2.savefig(args.success_output, dpi=150, bbox_inches='tight')
    print(f"[make_plots] saved {args.success_output}")


if __name__ == "__main__":
    main()
