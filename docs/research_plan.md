# Research Plan

## 1. Problem Statement

Quadrotor trajectory tracking can be controlled by a classical PID controller, a learned
RL policy, or a hybrid of the two. Each has known trade-offs:
- **PID**: well understood, stable, cheap, but hand-tuned gains degrade under
  parameter changes the designer didn't anticipate (e.g. payload changes, wind, wear).
- **Pure RL (PPO)**: can in principle adapt to complex dynamics, but is sample-hungry,
  less interpretable, and its behavior outside the training distribution is unreliable.
- **Hybrid (RL-residual + PID)**: the RL agent only learns a *correction* added to the
  PID output. The idea is the PID term keeps the system roughly stable and sample
  efficient, while the residual lets it adapt beyond what fixed PID gains can do.

Domain Randomization (DR), training with randomized dynamics parameters (mass,
inertia, drag, etc.), is the standard tool for making a policy robust to real-world
parameter uncertainty. This project tests whether DR benefits all three controller types
equally, or whether the hybrid structure gets disproportionately more robustness per
unit of training.

## 2. Hypothesis

**H1:** Under zero-shot evaluation on out-of-distribution dynamics (quadrotor mass
shifted by up to ±20% beyond the training randomization range), the hybrid
(PID + RL-residual) controller will show a smaller increase in trajectory-tracking error
than both the pure-PID and pure-PPO controllers, because the PID term provides a
stability floor that the RL-residual builds on rather than replaces.

**Secondary hypothesis (H2):** Training the hybrid controller to a given tracking-error
threshold requires fewer environment interactions (timesteps) than training PPO from
scratch to the same threshold, because the PID term reduces the exploration burden.

## 3. Research Questions

1. How does each controller's tracking error (position RMSE) change as test-time mass
   deviates further from the nominal value used at training time?
2. Does Domain Randomization close the robustness gap between PID and PPO, or does a
   gap remain that only the hybrid structure closes?
3. How does sample efficiency (environment steps to reach a target RMSE) compare
   between PPO-from-scratch and the hybrid residual policy?
4. Where does the hybrid controller fail relative to PID? Is the learned residual ever
   actively harmful (makes tracking worse than plain PID) in some part of the test range?

## 4. Methodology

### 4.1 Environment
`gym-pybullet-drones` (CF2X quadrotor model), trajectory-tracking task (e.g. figure-8
or waypoint sequence; pick one fixed reference trajectory used identically across all
three controllers and all experiments).

### 4.2 Controllers under test
| Controller | Description | Training |
|---|---|---|
| PID | Built-in `DSLPIDControl` from gym-pybullet-drones, gains left at library defaults (document them) | No training, scripted baseline |
| PPO | `stable-baselines3` PPO policy mapping state → motor RPM / attitude command directly | Train from scratch with PPO |
| Hybrid | PPO policy outputs a bounded residual added to the PID controller's output | Train from scratch with PPO, PID term held fixed |

### 4.3 Domain Randomization
At the start of each training episode, resample:
- Mass: uniform in nominal ± 10% (training range)
- (Optional, if time allows) inertia and/or max thrust: uniform ± 10%

Keep the DR range identical for PPO and Hybrid so the comparison is fair.

### 4.4 Evaluation protocol (zero-shot robustness)
Evaluate all three trained controllers, without any further training, on mass values
sweeping from nominal −30% to nominal +30% in fixed increments (e.g. every 5%), which
includes both in-distribution (±10%) and out-of-distribution (±10% to ±30%) points.
Run N episodes (e.g. N=10) per mass value per controller to get a mean ± std, since
PyBullet contact/dynamics has some stochasticity and PPO/Hybrid policies are stochastic
at inference unless you force deterministic actions (decide and document which).

**Decision:** evaluation uses deterministic actions
(`model.predict(obs, deterministic=True)`), for both PPO and Hybrid, throughout
`scripts/evaluate_robustness.py` and `scripts/evaluate_ablation.py`. This removes the
policy's own action-sampling noise as a source of run-to-run variance, so the 10
episodes per mass value measure variability from the environment (PyBullet
step-to-step dynamics, reset randomness) rather than conflating it with the policy
randomly sampling different actions for the same state.

**Crash/success definition:** "success" means the episode never triggers
`src/traj.py`'s `is_out_of_bounds(state)` (position more than 2m from origin in x/y,
altitude above 2m, or roll/pitch beyond 0.6 rad), matching the exact bounds
`TrajTrackAviary._computeTruncated` uses to end an RL episode early. The PID baseline
runs in `CtrlAviary`, which has no such built-in truncation, so `run_pid_baseline.py`
checks the identical condition manually at every step so "success" means the same
thing for all three controllers regardless of which environment class produced the
episode (see `scripts/run_pid_baseline.py` and `scripts/evaluate_robustness.py`'s
`run_rl_episode`).

**Important methodological note:** RMSE computed over a crashed/truncated episode
isn't directly comparable to RMSE over a completed one. A policy that crashes almost
immediately near the start of the trajectory can show an artificially low RMSE simply
because it's averaged over very few, easy, near-the-start steps, not because it
tracked well. Concretely: PPO's apparently-best RMSE (0.028m) at nominal mass turned
out to come from episodes that crashed after ~12 of 480 steps. See docs/results.md §3
for the corrected analysis, which reports success rate alongside RMSE and separates
unconditional RMSE (all episodes, comparable to the original write-up) from RMSE
conditional on success (the only version that's actually a fair tracking-quality
comparison).

### 4.5 Metrics
- Position RMSE over the episode (primary metric): **report both unconditional and
  conditional-on-success**, since the two can tell very different stories (see above)
- Success rate (episode completed without crash/instability, under a defined threshold)
- Settling time / max transient error (secondary)
- Training sample efficiency: environment steps to reach a fixed RMSE threshold on the
  training distribution (for H2)

### 4.6 Why this design is justified
- Fixing the reference trajectory and DR range across controllers isolates the
  controller-architecture variable, which is what H1/H2 are about.
- Sweeping mass beyond the training range directly operationalizes "zero-shot
  robustness to out-of-distribution dynamics" rather than just re-testing in-distribution
  performance.
- Multiple episodes per condition gives a defensible mean/variance instead of a single
  noisy run, which matters for justifying conclusions from a CPU-only training budget
  (fewer total timesteps than a GPU setup could afford, so each data point needs to
  carry more statistical weight).

## 5. Implementation Plan (what gets built, in order)

1. Install and smoke-test `gym-pybullet-drones` + `stable-baselines3` (environment runs,
   PID baseline flies a trajectory).
2. Wrap/confirm the trajectory-tracking Gym environment and reward function used for RL
   (reuse the repo's existing RL example as a starting point rather than writing from
   scratch).
3. Implement Domain Randomization hook (randomize mass at `reset()`).
4. Train PPO baseline with DR; log to TensorBoard.
5. Implement the hybrid wrapper: PID computes a base action, policy network outputs a
   residual, env steps with `base + residual` (residual magnitude clipped).
6. Train Hybrid with PPO, same DR range and training budget as step 4.
7. Build the evaluation sweep script (mass sweep × controller × N episodes → CSV).
8. Analysis: plots of RMSE vs. mass deviation per controller, with error bars; sample
   efficiency curve (RMSE vs. training steps) for PPO vs Hybrid.
9. Write up results, discussion, and conclusions.

## 6. Risks and Mitigations (given CPU-only hardware)
- **Training is slow on 2 CPU cores.** Mitigate by vectorizing environments
  (`stable-baselines3` `SubprocVecEnv` with 2 workers), keeping episode length and total
  timesteps modest, and running long training jobs overnight.
- **PPO may not converge well in the time budget.** Keep the task simple (one fixed
  trajectory, not a distribution of trajectories) so a small policy network and modest
  timestep budget (e.g. 200k-1M steps, tuned empirically) is sufficient.
- **Reproducibility.** Fix and record all random seeds; log exact package versions in
  `requirements.txt`.
- **Docker couldn't be verified locally.** This machine (2015 MacBook Air,
  `MacBookAir7,2`) is capped by Apple at macOS 12 (Monterey); Docker Desktop, Colima,
  and Podman all now require macOS 13+. `.github/workflows/docker-build.yml` builds
  the image and runs the smoke test on GitHub's Linux runners on every push instead.
  See README.md's Docker section.
