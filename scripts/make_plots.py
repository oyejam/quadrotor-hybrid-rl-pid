"""Reproduce the core robustness plot (docs/results.md section 3) from
results/robustness_sweep.csv: position RMSE vs. test-time mass deviation,
one line per controller, with error bars across episodes.

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
    colors = {"pid": "tab:blue", "ppo": "tab:orange", "hybrid": "tab:green"}
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


if __name__ == "__main__":
    main()
