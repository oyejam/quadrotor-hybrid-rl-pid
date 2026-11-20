# Hybrid Reinforcement Learning and Residual PID for Quadrotor Trajectory Tracking

## Participants
- Jamiu Olaide Oyetunde — ISU/ITMO ID: 520194

## Description of the Conducted Research

See [docs/literature_review.md](docs/literature_review.md) for the analytical review of
prior work, [docs/research_plan.md](docs/research_plan.md) for the problem statement,
hypothesis, and methodology, and [docs/results.md](docs/results.md) for experimental
results.

**Short summary:**
This project compares three quadrotor trajectory-tracking controllers — a classical PID
controller, a pure reinforcement-learning policy (PPO), and a hybrid controller where an
RL agent learns a residual correction on top of PID — all trained with Domain
Randomization (DR) over the drone's mass. The hypothesis was that the hybrid controller
would degrade more gracefully than PID or PPO on mass values outside the training range.
**The result is the opposite**: PID completes the reference trajectory at every mass
tested (70%-130% of nominal), while both PPO and the hybrid controller fail to complete
it at all on almost every episode at or below nominal mass, only becoming reliable once
the true mass exceeds nominal by ≥10%. See [docs/results.md](docs/results.md) for the
full analysis, including a follow-up ablation and the methodological correction that
revealed this (an initial RMSE-only evaluation was misleading because a controller that
crashes immediately can show an artificially low error).

## Demonstration (Video)

<video src="results/demo.mp4" controls width="640">
  Your viewer doesn't support inline video — direct link:
  https://raw.githubusercontent.com/oyejam/quadrotor-hybrid-rl-pid/main/results/demo.mp4
</video>

Direct link (mandatory per course requirements):
https://raw.githubusercontent.com/oyejam/quadrotor-hybrid-rl-pid/main/results/demo.mp4

27-second clip recorded headless on this project's own CPU-only machine
(`scripts/record_demo.py`), showing: PID tracking reliably at 70% of nominal mass, PPO
crashing almost immediately at exactly the mass its RMSE looked best at, PPO succeeding
once the drone is 20% heavier, the original hybrid controller crashing at nominal mass
despite the PID floor, and the tightened-clip hybrid (the one case the ablation in
docs/results.md §3.1 fixed) succeeding at nominal mass.

## Installation and Deployment

- Developed and trained on: local machine (CPU only, no GPU) for PID baseline and
  lightweight PPO runs; [Google Colab / other] for longer training runs if needed.
- Python version: 3.12 (required by the current `gym-pybullet-drones` release, which
  depends on `gymnasium` and `torch`, not the older `gym`-based API)

### System dependencies
None beyond Python 3.12 and a C/C++ build toolchain (needed by pybullet's wheel build
on some platforms; on macOS this is provided by Xcode Command Line Tools).

### Environment setup
`gym-pybullet-drones` is not published on PyPI, so it is vendored as a git clone under
`external/` and installed in editable mode.

**Note for Intel macOS (and any platform without a current PyTorch wheel):** the
upstream `gym-pybullet-drones` package pins `torch>=2.13`, but PyTorch stopped
publishing Intel-macOS wheels after `2.2.2`. The simulation code itself does not
import `torch` (only the `stable-baselines3` training backend does), so the package
is installed with `--no-deps` and the real dependencies are installed separately,
pinned to versions that are mutually compatible *and* available for this platform
(`torch==2.2.2` requires `stable-baselines3<=2.4.x`, since 2.5+ requires `torch>=2.3`).
If you're on a platform with current PyTorch wheels (Linux/Windows, or Apple Silicon),
you can likely skip the `--no-deps` step and just `pip install -e external/gym-pybullet-drones`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

git clone --depth 1 https://github.com/utiasDSL/gym-pybullet-drones.git external/gym-pybullet-drones
pip install -e external/gym-pybullet-drones --no-deps   # core env only, skips its strict pins
pip install -r requirements.txt                          # the real, platform-compatible pins
```

## Running and Usage

- Train the PID baseline (no learning, scripted controller): `python scripts/run_pid_baseline.py`
- Train the PPO policy: `python scripts/train_ppo.py --config config/ppo.yaml`
- Train the hybrid residual policy: `python scripts/train_hybrid.py --config config/hybrid.yaml`
- Evaluate robustness under domain shift: `python scripts/evaluate_robustness.py`
- Reproduce all plots: `python scripts/make_plots.py`
- Record the demo video: `python scripts/record_demo.py`

## Description of the Obtained Results

PID reliably completes the reference trajectory at every tested mass (70%-130% of
nominal): 100% success rate, RMSE 0.09-0.11m throughout. Both PPO and the hybrid
controller (PID + RL residual) fail to complete the trajectory on almost every episode
at or below nominal mass, and only become reliable once the drone is at least 10%
heavier than what they were trained around. A follow-up ablation (tightening the
hybrid's residual bound) fixed this specifically at nominal mass but not the broader
pattern. Full tables, plots, and discussion: [docs/results.md](docs/results.md).
Raw data: [results/robustness_sweep.csv](results/robustness_sweep.csv). Plots:
[results/robustness_plot.png](results/robustness_plot.png) and
[results/success_rate_plot.png](results/success_rate_plot.png).

## License
MIT — see [LICENSE](LICENSE).
