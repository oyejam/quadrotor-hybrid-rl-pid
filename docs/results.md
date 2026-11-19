# Results

Fill in as experiments complete. Keep this in sync with `README.md`'s "Description of
the Obtained Results" section (link here, don't duplicate).

## 1. Training curves
- PPO: episode reward vs. timesteps (from `results/ppo/evaluations.npz`)
- Hybrid: episode reward vs. timesteps (from `results/hybrid/evaluations.npz`)
- Compare sample efficiency (H2): steps to reach a fixed reward/RMSE threshold.

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
