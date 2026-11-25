# Phase 2: Scaling Up on the Second Machine

This document is the plan for continuing the project on a second machine. It was
originally written for an anticipated Lenovo Legion Y7000P (Intel Core Ultra 7 255HX,
20 cores/20 threads, NVIDIA RTX 5060). The machine actually used turned out to be a
different laptop: an Intel Core i7-14650HX (16 cores, 24 threads) with an NVIDIA
GeForce RTX 4050 Laptop GPU (6GB VRAM, Ada architecture, not Blackwell). The setup
steps below did not need to change for this swap, only the specific numbers (core
count for `n_envs`, VRAM headroom). All work described in docs/research_plan.md and
docs/results.md was done on an Intel 2015 MacBook Air with 2 CPU cores and no GPU.
This phase does not replace those results. It extends them, with the version
differences stated plainly so the two are never confused.

**Setup log (2026-11-25):** Python 3.12.10, MSVC Build Tools (C++ workload), and a
project venv were installed on the actual machine above. `torch==2.11.0+cu128`
installed from `download.pytorch.org/whl/cu128` and confirmed working
(`torch.cuda.is_available()` → `True`, device name `NVIDIA GeForce RTX 4050 Laptop
GPU`). `pybullet` built from source (no Python 3.12 wheel yet, same situation as the
Mac). `gym-pybullet-drones` was cloned into `external/` and installed **with**
`--no-deps`, contrary to section 3 step 7 below: its `pyproject.toml` pins
`torch = "^2.13"`, but the CUDA 12.8 wheel index's latest build at install time was
only `2.11.0+cu128`. Installing without `--no-deps` would have had pip satisfy that
pin from the default PyPI index instead, silently replacing the verified CUDA build
with a non-CUDA one. Smoke test (`scripts/run_pid_baseline.py`) passed:
`RMSE=0.0913m, success=True` at nominal mass, consistent with the Mac's Phase 1
result (0.09-0.11m).

## 1. Why this is an extension, not a redo

The original findings stand on their own and are not invalidated by better hardware.
The question this phase answers is different: were the sharp conclusions in
docs/results.md (PID succeeds at every tested mass, PPO and Hybrid fail below
nominal mass) an artifact of a constrained CPU-only training budget, or do they hold
up under a training budget an order of magnitude larger, with multiple seeds and a
much larger evaluation sample. Section 5 of docs/research_plan.md already flagged
the leading hypothesis for the asymmetric failure (the fixed-nominal-mass HOVER_RPM
baseline in the action mapping), and separately flagged two concrete statistical
limitations that only more compute can fix.

## 2. What changes on this machine versus the Mac

The venv instructions in README.md pin torch to 2.2.2 and stable-baselines3 to 2.4.1
specifically because PyPI stopped publishing current PyTorch wheels for Intel macOS.
None of that applies here. This machine should use current package versions that
actually support the RTX 5060 (NVIDIA's Blackwell architecture, compute capability
sm_120), which requires CUDA 12.8 or newer and a PyTorch build from the same era.
Use gym-pybullet-drones' own declared dependency versions directly rather than the
Mac's workaround pins.

## 3. Setup, in order

1. Confirm the GPU and driver are visible: run `nvidia-smi` in a terminal. It should
   report the RTX 5060 and a CUDA version of 12.8 or higher. If the CUDA version
   shown is lower, update the NVIDIA driver first through the NVIDIA app or
   GeForce Experience before continuing.
2. Install Python 3.12 or newer if it is not already present.
3. Install the Microsoft C++ Build Tools (the "Desktop development with C++"
   workload from the Visual Studio Installer). pybullet has historically lagged
   behind the newest Python releases in publishing prebuilt wheels, so a compiler
   may be needed to build it from source, the same situation this project already
   hit on the Mac and in the Docker build.
4. Create a fresh virtual environment and activate it.
5. Install PyTorch using the official selector at pytorch.org/get-started/locally
   for your exact CUDA version, rather than a plain `pip install torch`, to be sure
   the Blackwell-compatible build is the one that gets installed.
6. Verify the GPU is actually reachable from PyTorch before doing anything else:
   `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
   This must print `True` and the GPU's name before moving on.
7. Clone gym-pybullet-drones into `external/` as the README already describes, but
   install it without `--no-deps` this time, letting it pull its own current
   dependencies (gymnasium, numpy, scipy, stable-baselines3) directly, since the
   reason for that workaround on the Mac does not apply here.
8. Run the existing smoke test, `python scripts/run_pid_baseline.py`, to confirm the
   simulator itself works correctly on this machine before changing anything else.

## 4. What to actually change for this hardware

- **n_envs in config/ppo.yaml, config/hybrid.yaml, and config/hybrid_tight_clip.yaml.**
  These are set to 2, matching the Mac's core count. Raise this to somewhere in the
  12 to 16 range, leaving headroom for the OS and other processes, rather than all
  20 cores at once.
- **device.** Add an explicit device setting to src/train_utils.py's PPO
  construction rather than leaving it on the default. Test both `device="cpu"` and
  `device="cuda"` on a short run (a few thousand timesteps) and compare wall clock
  time before committing to one for the full runs, given the MLP-policy caveat in
  section 1 above. Do not assume CUDA is faster here without checking.
- **Seeds.** config/*.yaml currently set seed=0 for a single run per controller.
  Add a seed sweep, for example seeds 0, 1, and 2, training each controller three
  times and reporting the mean and spread across seeds. This directly answers the
  limitation flagged in docs/results.md section 5 about a single policy seed per
  controller.
- **Episode count in the robustness sweep.** scripts/evaluate_robustness.py and
  scripts/evaluate_ablation.py default to 10 episodes per mass value. Raise this to
  30 or 50. This directly answers the other flagged limitation, that a reported
  success rate of 0.0 or 1.0 from only 10 episodes could flip with a different
  random seed at the boundary points.
- **Training budget.** Try at least one run at a much larger timestep count than
  the Mac's 500000, for example 2000000 or more, specifically to test whether the
  asymmetric failure below nominal mass is a training budget problem rather than a
  structural one, as docs/results.md section 4 already proposed as a cause worth
  testing.

## 5. How to report the results

Keep this phase's results in a clearly separate section, for example a new
docs/results_phase2.md file or a clearly labeled new section in docs/results.md,
stating the hardware, the package versions, and the exact changes from phase 1
before presenting any numbers. State plainly whether the original finding held up,
changed, or needs further investigation. Do not edit or soften the original
Phase 1 numbers to match new results; if they differ, say so and explain why.
