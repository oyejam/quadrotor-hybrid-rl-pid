# Results

Fill in as experiments complete. Keep this in sync with `README.md`'s "Description of
the Obtained Results" section (link here, don't duplicate).

## 1. Training curves
- PPO: episode reward vs. timesteps (from `results/ppo/evaluations.npz`)
- Hybrid: episode reward vs. timesteps (from `results/hybrid/evaluations.npz`)
- Compare sample efficiency (H2): steps to reach a fixed reward/RMSE threshold.

**PPO run log (2026-11-20):** 500,000 timesteps (503,808 actual, PPO completes full
rollouts), 2 parallel envs, ~1199s wall-clock (~420 steps/sec) on this CPU-only
machine. Final eval: mean reward -75.2 ± 67.5, mean episode length 176/480 steps.
The large variance and short mean episode length (well under the 480-step full
trajectory) suggest the policy frequently triggers the truncation condition (flies
too far off-course or tilts too much) before completing a lap — raw-RPM control is a
genuinely hard RL problem to learn from scratch in this budget. This is itself a
relevant data point for the hybrid comparison: if Hybrid converges to a lower-variance,
longer-surviving policy in the same budget, that is direct evidence for H2 (the PID
term reduces the exploration burden). Needs the Hybrid run to compare against before
drawing conclusions.

## 2. In-distribution tracking accuracy
Position RMSE for PID / PPO / Hybrid at the nominal mass and within the training DR
range (±10%).

## 3. Zero-shot robustness sweep (the core result, H1)
Plot: x-axis = test-time mass deviation from nominal (−30% to +30%), y-axis = mean
position RMSE (with std/error bars across N episodes), one line per controller.

Table: RMSE and success rate at a few key points (nominal, ±10% = edge of training
range, ±20%, ±30%).

## 4. Discussion
- Does the hybrid controller's error grow more slowly than PID/PPO outside the training
  range, as H1 predicted? Where does it break down, if anywhere?
- Any cases where the learned residual made things worse than plain PID?
- Sample efficiency comparison (H2) — did hybrid reach target performance with fewer
  environment steps than PPO-from-scratch?

## 5. Limitations
- Training budget was CPU-only (document actual total timesteps used, wall-clock time,
  number of parallel envs) — note how this may affect how well PPO/Hybrid converged
  compared to a GPU-trained reference.
- Single reference trajectory (document which) — results may not generalize to other
  trajectories.
