# Hybrid Reinforcement Learning and Residual PID for Quadrotor Trajectory Tracking

[![Docker build](https://github.com/oyejam/quadrotor-hybrid-rl-pid/actions/workflows/docker-build.yml/badge.svg)](https://github.com/oyejam/quadrotor-hybrid-rl-pid/actions/workflows/docker-build.yml)

## Participants
- Jamiu Olaide Oyetunde, ISU/ITMO ID: 520194

## Description of the Conducted Research

See [docs/literature_review.md](docs/literature_review.md) for the literature review,
[docs/research_plan.md](docs/research_plan.md) for the problem statement, hypothesis and
methodology, and [docs/results.md](docs/results.md) for the phase 1 experimental results
(CPU-only Mac, single seed). [docs/gpu_machine_plan.md](docs/gpu_machine_plan.md) and
[docs/results_phase2.md](docs/results_phase2.md) cover the second phase of work on a
second machine with far more CPU cores: a 3-seed sweep, a larger training budget, and
30 (instead of 10) evaluation episodes per point.

**Summary.** The project compares three quadrotor trajectory-tracking controllers: a
classical PID controller, a PPO reinforcement-learning policy, and a hybrid controller
where an RL agent learns a residual correction on top of PID. All three are trained
with Domain Randomization (DR) over the drone's mass. The hypothesis was that the
hybrid controller would degrade more gracefully than PID or PPO outside the training
mass range. It didn't. PID completes the reference trajectory at every mass tested
(70-130% of nominal). PPO and the hybrid controller both fail to complete it on almost
every episode at or below nominal mass. Phase 2's larger seed/episode sweep confirms
this holds up, not as an artifact of a single run or a small CPU-only training budget,
and adds a specific nuance phase 1 could only guess at: the hybrid's reliability right
at nominal mass turned out to be seed-dependent (2 of 3 seeds succeed, 1 doesn't),
while PPO's failure there is consistent across all 3 seeds. Phase 2 also found that the
hybrid structure trains to a more stable, longer-surviving policy than PPO on the
training distribution itself, in every seed — real support for the project's secondary
hypothesis about sample efficiency, even though it doesn't rescue the main hypothesis
about zero-shot robustness. Full analysis, including a follow-up ablation and a
correction to an evaluation bug in an earlier pass, is in
[docs/results.md](docs/results.md) (phase 1) and
[docs/results_phase2.md](docs/results_phase2.md) (phase 2).

## Demonstration (Video)

![Demo: PID vs PPO vs Hybrid under mass perturbation](results/demo.gif)

Full video: https://raw.githubusercontent.com/oyejam/quadrotor-hybrid-rl-pid/main/results/demo.mp4

(GIF above is a preview. GitHub doesn't embed a video player from a plain file link.)

50-second clip (`scripts/record_demo.py`, recorded headless, using the phase 2
seed-swept models so the footage matches the current results rather than the
single-seed phase 1 run): PID tracking reliably at 70% of nominal mass, PPO
(seed 0) crashing almost immediately at nominal mass, PPO still failing at 110%
of nominal (the phase 2 finding that it isn't just a nominal-mass problem),
PPO succeeding once the drone is 20% heavier, the hybrid controller (seed 0)
crashing at nominal mass despite the PID floor, and the same hybrid controller
trained under seed 1 completing the trajectory at that identical mass --
the seed-dependence phase 2 found at that point (section 5 of
[docs/results_phase2.md](docs/results_phase2.md)).

## Installation and Deployment

Developed and trained entirely on a local CPU-only machine (no GPU). PID baseline,
PPO and hybrid training, evaluation, and the demo video all ran there.

Python 3.12, required by the current `gym-pybullet-drones` release (it depends on
`gymnasium` and `torch`, not the older `gym` API).

### System dependencies
None beyond Python 3.12 and a C/C++ build toolchain (needed by pybullet's wheel build
on some platforms; on macOS this is provided by Xcode Command Line Tools).

### Environment setup
`gym-pybullet-drones` isn't published on PyPI, so it's vendored as a git clone under
`external/` and installed in editable mode.

**Note for Intel macOS (and any platform without a current PyTorch wheel):** the
upstream `gym-pybullet-drones` package pins `torch>=2.13`, but PyTorch stopped
publishing Intel-macOS wheels after `2.2.2`. The simulation code itself doesn't import
`torch` (only the `stable-baselines3` training backend does), so the package is
installed with `--no-deps` and the real dependencies come from `requirements.txt`
separately, pinned to versions that are mutually compatible and available for this
platform (`torch==2.2.2` needs `stable-baselines3<=2.4.x`, since 2.5+ needs
`torch>=2.3`).

**On Windows with a CUDA GPU** (the phase 2 machine, docs/gpu_machine_plan.md):
`--no-deps` was still needed, for the opposite reason -- the CUDA 12.8 wheel index's
newest build was `torch==2.11.0+cu128`, still behind `gym-pybullet-drones`'s
`torch>=2.13` pin, so skipping `--no-deps` would have pulled a non-CUDA torch from
plain PyPI over the verified CUDA build. Install torch from
`download.pytorch.org/whl/cu128` first, then the rest of
[requirements-phase2.txt](requirements-phase2.txt) (the versions phase 2 actually
ran, separate from `requirements.txt` above). Don't assume a current-PyTorch-wheel
platform can skip `--no-deps` without checking the wheel index at install time.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

git clone --depth 1 https://github.com/utiasDSL/gym-pybullet-drones.git external/gym-pybullet-drones
pip install -e external/gym-pybullet-drones --no-deps   # core env only, skips its strict pins
pip install -r requirements.txt                          # the real, platform-compatible pins
```

### Docker (recommended if you're not on Intel macOS)

The venv setup above needs several Intel-macOS-only workarounds (Homebrew Python 3.12,
the `torch==2.2.2`/`stable-baselines3==2.4.1` pinning, the `--no-deps` install). None
of that applies on a standard Linux container.

```bash
docker build -t quadrotor-hybrid-rl-pid .
docker run --rm quadrotor-hybrid-rl-pid                                  # PID baseline smoke test
docker run --rm quadrotor-hybrid-rl-pid python scripts/evaluate_robustness.py
docker run --rm -v "$(pwd)/results:/app/results" quadrotor-hybrid-rl-pid python scripts/train_ppo.py
```

Mount `results/` (as in the last example) to keep artifacts written inside the
container on the host afterward.

`pybullet==3.2.7` has no Python 3.12 wheel yet, so the first build compiles it from
source, and that step alone takes a few minutes. Everything else installs from a
prebuilt wheel.

This machine can't run Docker itself: Docker Desktop needs macOS 13+, and this is a
2015 MacBook Air capped by Apple at macOS 12. Colima and Podman have the same floor,
so it isn't a tooling choice. `.github/workflows/docker-build.yml` builds the image
and runs the smoke test on GitHub's Linux runners on every push instead (the badge at
the top tracks this). Its last run gave `RMSE=0.0896m` for the PID baseline at nominal
mass, matching the local result (`0.0904m`) within normal floating-point variation
across CPUs.

The image uses the same pinned versions as `requirements.txt`, so results reproduced
through Docker match `docs/results.md`.

## Running and Usage

- Train the PID baseline (no learning, scripted controller): `python scripts/run_pid_baseline.py`
- Train the PPO policy: `python scripts/train_ppo.py --config config/ppo.yaml`
- Train the hybrid residual policy: `python scripts/train_hybrid.py --config config/hybrid.yaml`
- Train either under a different seed/output dir (phase 2's seed sweep):
  `python scripts/train_ppo.py --seed 1 --output-dir results/phase2/ppo_seed1 --total-timesteps 1000000`
- Evaluate robustness under domain shift: `python scripts/evaluate_robustness.py`
- Evaluate a multi-seed sweep together (phase 2): `python scripts/evaluate_seed_sweep.py`
- Reproduce all plots: `python scripts/make_plots.py`
- Reproduce phase 2's plots instead (pooled across seeds):
  `python scripts/make_plots.py --input results/phase2/robustness_sweep_seeds.csv --output results/phase2/robustness_plot.png --success-output results/phase2/success_rate_plot.png`
- Record the demo video: `python scripts/record_demo.py`

Phase 2 training (`config/*.yaml`'s `n_envs: 16`) spawns that many subprocesses; set
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` in the
environment first, or each subprocess's own BLAS thread pool can exhaust memory (see
[docs/results_phase2.md](docs/results_phase2.md) section 1).

## Description of the Obtained Results

PID completes the reference trajectory at every tested mass (70-130% of nominal):
100% success rate, RMSE 0.09-0.11m throughout. PPO and the hybrid controller (PID + RL
residual) both fail to complete the trajectory on almost every episode at or below
nominal mass, and only become reliable once the drone is at least 10% heavier than
what they were trained on. A follow-up ablation (tightening the hybrid's residual
bound) fixed this at nominal mass specifically, not the broader pattern. Full tables,
plots, and discussion: [docs/results.md](docs/results.md). Raw data:
[results/robustness_sweep.csv](results/robustness_sweep.csv). Plots:
[results/robustness_plot.png](results/robustness_plot.png) and
[results/success_rate_plot.png](results/success_rate_plot.png).

**Phase 2** (3 seeds x 30 episodes/point instead of 1 seed x 10) confirms this
pattern rather than overturning it, and sharpens it: PPO now fails at +10% mass too
(consistently across all 3 seeds, a change from phase 1's single-run result at that
point), and the hybrid's success at nominal mass turns out to depend on the training
seed (2 of 3 seeds succeed, 1 doesn't) — direct confirmation of a limitation phase 1
could only flag as a possibility. Separately, the hybrid trains to a far more stable,
longer-surviving policy than PPO on the training distribution itself, in every seed,
which is real (if partial) support for the project's sample-efficiency hypothesis.
Full analysis: [docs/results_phase2.md](docs/results_phase2.md). Raw data:
[results/phase2/robustness_sweep_seeds.csv](results/phase2/robustness_sweep_seeds.csv).

## License
MIT. See [LICENSE](LICENSE).
