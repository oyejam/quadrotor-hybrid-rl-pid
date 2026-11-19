# Hybrid Reinforcement Learning and Residual PID for Quadrotor Trajectory Tracking

## Participants
- [Your full name] — ISU/ITMO ID: [fill in]

## Description of the Conducted Research

See [docs/research_plan.md](docs/research_plan.md) for the full problem statement, hypothesis, and methodology, and [docs/results.md](docs/results.md) for results once experiments are complete.

**Short summary (fill in once work starts):**
This project compares three quadrotor trajectory-tracking controllers — a classical PID
controller, a pure reinforcement-learning policy (PPO), and a hybrid controller where an
RL agent learns a residual correction on top of PID — all trained with Domain
Randomization (DR). The central question is how each approach's tracking accuracy
degrades under out-of-distribution dynamics (e.g. ±20% mass perturbation) that were
never seen during training.

## Demonstration (Video)
- [ ] Link to demo video: TBD

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
- Evaluate robustness under domain shift: `python scripts/evaluate_robustness.py --controller all --mass-range -0.2 0.2`
- Reproduce all plots: `python scripts/make_plots.py`

## Description of the Obtained Results
TBD — see [docs/results.md](docs/results.md).

## License
MIT — see [LICENSE](LICENSE).
