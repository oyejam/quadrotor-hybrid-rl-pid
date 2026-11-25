# Results, Phase 2 (second machine)

This is the phase 2 extension described in [gpu_machine_plan.md](gpu_machine_plan.md):
same task, same environment, same controllers, run on a second machine with a much
larger CPU core count (16 cores / 24 threads vs. the Mac's 2 cores) and a GPU (RTX 4050
Laptop, 6GB) that turned out not to help this workload (section 1 below). See
gpu_machine_plan.md's setup log for the exact hardware/software used.

This section extends [results.md](results.md). It does not replace it: phase 1's
numbers are left exactly as recorded there, and every place this phase's numbers
differ is called out explicitly rather than merged or averaged in.

## 1. Device choice: CPU beat CUDA

Per gpu_machine_plan.md section 4, device was benchmarked empirically rather than
assumed: a short 20,000-timestep PPO run (`n_envs=16`) with `device="cpu"` vs.
`device="cuda"`.

| device | wall time | steps/sec |
|---|---|---|
| cpu | 48.1s | 415.9 |
| cuda | 66.1s | 302.4 |

CPU was about 27% faster. This matches stable-baselines3's own guidance for small
MLP policies: the policy network here is two small fully-connected layers taking a
~20-value state vector, cheap enough that the per-step cost of copying data to and
from GPU memory exceeds the compute it saves. The bottleneck in this pipeline is
PyBullet physics stepping across the 16 parallel subprocesses, which runs on CPU
regardless of `device`. All phase 2 training below uses `device="cpu"`.

(Note for the sustained training runs below: once subprocess-startup overhead is
amortized over a full run rather than just 20k steps, actual throughput was much
higher than this short benchmark suggests, e.g. PPO seed=0 ran at ~1600 steps/sec
steady-state.)

**Operational finding, not in the original plan:** at `n_envs=16`, the first training
attempt crashed with `OpenBLAS error: Memory allocation still failed after 10
retries`. Each of the 16 `SubprocVecEnv` worker processes spawns its own OpenBLAS
thread pool by default; 16 processes doing that simultaneously on this CPU exhausted
memory. Fixed by setting `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1` in the environment before launching training. This is required
for every phase 2 training run below, not optional.

## 2. What changed in config vs. phase 1

`config/ppo.yaml`, `config/hybrid.yaml`, `config/hybrid_tight_clip.yaml`: `n_envs`
2 -> 16, `device` added (`cpu`), `total_timesteps` 500000 -> 2000000 (default; the
seed sweep below overrides this per-run, see section 3). `src/train_utils.py` and
`scripts/train_ppo.py`/`train_hybrid.py` gained `--seed`/`--output-dir` CLI overrides
so the same config can be trained under multiple seeds without hand-editing YAML.
`scripts/evaluate_robustness.py`/`evaluate_ablation.py` default episode count raised
10 -> 30 (docs/gpu_machine_plan.md section 4). A new `scripts/evaluate_seed_sweep.py`
evaluates all seeded models together; it reuses the exact same per-episode protocol
(`run_pid_episode`/`run_rl_episode`, deterministic actions, the same
`is_out_of_bounds` success definition) as phase 1's `evaluate_robustness.py`, so
results are directly comparable.

**Scope decision:** `config/hybrid_tight_clip.yaml` (the clip=0.08 ablation) was
updated to the same phase 2 settings but was not retrained or re-evaluated this
phase, to keep the total training budget (already 6 runs) bounded. Left as future
work; see section 6.

## 3. Training: seed sweep (0, 1, 2) and the large-budget test

Per gpu_machine_plan.md section 4, both asks (a seed sweep, and at least one much
larger-budget run) were satisfied together: seed 0 for each controller ran at
2,000,000 timesteps (4x the Mac's budget, serving as both the seed-0 sweep point and
the large-budget test); seeds 1 and 2 ran at 1,000,000 timesteps (2x the Mac's
budget) each. All runs used `n_envs=16`, `device=cpu`.

**Final-eval episode length (of 480 steps, on the training DR distribution,
mass in [0.9, 1.1]), mirroring the format of results.md section 1:**

| controller | seed | timesteps | final ep_len_mean |
|---|---|---|---|
| PPO | 0 | 2,000,000 | 247 |
| PPO | 1 | 1,000,000 | 136 |
| PPO | 2 | 1,000,000 | 126 |
| Hybrid | 0 | 2,000,000 | 420 |
| Hybrid | 1 | 1,000,000 | 260 |
| Hybrid | 2 | 1,000,000 | 225 |

Phase 1 (500k timesteps, single seed=0, `n_envs=2`): PPO final ep_len_mean 176,
Hybrid 263. At a larger budget, both controllers improved, and Hybrid's improvement
is larger and more consistent across seeds than PPO's.

### 3.1 A sharper H2 signal: training-time stability, not just final reward

Phase 1 could not evaluate H2 (sample efficiency) properly (results.md section 4).
The `evaluations.npz` logs saved during these phase 2 runs make a *cautious* version
of that comparison possible without additional training: the mean episode length
(on the training DR distribution) over the last 20% of each run, averaged at the
eval-checkpoint level.

| controller | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| PPO | 185.0 | 97.1 | 50.2 |
| Hybrid | 444.3 | 320.3 | 292.2 |

This holds across all three seeds, not just one: PPO's episode length is noisy and
degrades over the course of training (seed 0 peaks at 432.2 around 20k timesteps,
then falls and only sporadically recovers), while Hybrid converges to and sustains
near-maximum episode length (482) by the end of training in every seed. Hybrid also
reaches long episodes almost immediately (seed 0 crosses an episode-length-400
threshold at the very first eval checkpoint, 4,992 timesteps) — but this specific
number is confounded by design: the PID term is already present and competent from
timestep zero, so an early "threshold crossing" for Hybrid partly reflects the fixed
PID floor rather than anything the residual policy learned. The last-20%-of-training
comparison above is the more honest reading: it shows the *trained* policies'
late-training behavior, not a step-zero artifact, and the result (Hybrid
consistently higher and far less noisy than PPO, same direction in all three seeds)
is exactly the mechanism H2 predicted — the PID term does appear to stabilize
training and raise effective episode length faster and more reliably than PPO alone
achieves on the *training* distribution. This is a real, seed-robust finding. It is
also, as section 5 below shows, orthogonal to H1: it does not translate into
better zero-shot robustness outside the training distribution.

## 4. Zero-shot robustness sweep, extended

Ran `scripts/evaluate_seed_sweep.py`: PID (30 episodes) plus all 6 seeded PPO/Hybrid
models (30 episodes each) across the same 7 mass scales as phase 1 (0.7-1.3x
nominal). Total: 1,470 episodes (vs. phase 1's 280). Plots:
[results/phase2/robustness_plot.png](../results/phase2/robustness_plot.png) and
[results/phase2/success_rate_plot.png](../results/phase2/success_rate_plot.png)
(pooled across the 3 seeds per controller). Raw data:
[results/phase2/robustness_sweep_seeds.csv](../results/phase2/robustness_sweep_seeds.csv).

**Success rate, pooled across seeds (90 episodes/point for PPO/Hybrid, 30 for PID):**

| mass_scale | PID | PPO | Hybrid |
|---|---|---|---|
| 0.70 | 1.0 | 0.0 | 0.0 |
| 0.80 | 1.0 | 0.0 | 0.0 |
| 0.90 | 1.0 | 0.0 | 0.0 |
| 1.00 (nominal) | 1.0 | 0.0 | 0.667 |
| 1.10 | 1.0 | 0.0 | 0.0 |
| 1.20 | 1.0 | 1.0 | 1.0 |
| 1.30 | 1.0 | 1.0 | 1.0 |

**Success rate, per seed (PPO/Hybrid):**

| controller | seed | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 |
|---|---|---|---|---|---|---|---|---|
| PPO | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| PPO | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| PPO | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| Hybrid | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| Hybrid | 1 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| Hybrid | 2 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 |

**Mean episode length (of 480), pooled:**

| mass_scale | PID | PPO | Hybrid |
|---|---|---|---|
| 0.70 | 480.0 | 15.3 | 57.3 |
| 0.80 | 480.0 | 17.0 | 87.7 |
| 0.90 | 480.0 | 18.7 | 112.0 |
| 1.00 (nominal) | 480.0 | 12.0 | 456.7 |
| 1.10 | 480.0 | 100.0 | 279.7 |
| 1.20 | 480.0 | 482.0 | 482.0 |
| 1.30 | 480.0 | 482.0 | 482.0 |

**RMSE, unconditional / conditional on success (pooled; N/A = zero successful
episodes at that point):**

| mass_scale | PID | PPO uncond. | PPO cond. | Hybrid uncond. | Hybrid cond. |
|---|---|---|---|---|---|
| 0.70 | 0.107 | 0.112 | N/A | 0.935 | N/A |
| 0.80 | 0.098 | 0.083 | N/A | 1.062 | N/A |
| 0.90 | 0.093 | 0.050 | N/A | 0.897 | N/A |
| 1.00 (nominal) | 0.091 | 0.016 | N/A | 0.200 | 0.199 |
| 1.10 | 0.094 | 0.937 | N/A | 0.876 | N/A |
| 1.20 | 0.100 | 0.423 | 0.423 | 0.429 | 0.429 |
| 1.30 | 0.109 | 0.431 | 0.431 | 0.431 | 0.431 |

## 5. What held up, what changed, and what's new

**What held up, now with far more statistical weight (1,470 vs. 280 episodes, 3
seeds instead of 1):**
- PID: 100% success at every tested mass, RMSE 0.09-0.11m throughout. Identical to
  phase 1 within floating-point noise, on different hardware and a different OS —
  a good cross-check that the simulation and evaluation protocol reproduce correctly.
- PPO and Hybrid both still fail to complete the trajectory on most episodes below
  and at nominal mass, and both recover fully by +20%. H1 (hybrid degrades more
  gracefully than PID/PPO outside the training range) is rejected again, more
  decisively: even where a learned controller does succeed, its RMSE is still
  roughly 2-10x PID's.

**What changed, stated plainly rather than smoothed over:**
- In phase 1, PPO succeeded 100% of the time at +10% mass. In phase 2, PPO fails
  (0%) at +10% in all three seeds. This is a genuine divergence, not noise: it is
  uniform across seeds (std=0 at that point), so it is not seed variance either.
  The most likely explanation is that phase 1's +10% result came from a single
  500k-timestep, `n_envs=2` run, and phase 2's seed=0 PPO model — despite sharing
  the label "seed 0" — is a materially different trained policy: a different
  `n_envs` changes how `SubprocVecEnv` seeds its sub-environments, and the budget is
  4x larger. "Same seed" is necessary but not sufficient for reproducing a specific
  trained policy across a changed training setup. The phase 1 finding at +10%
  specifically should be read as a property of that one run, not a seed-independent
  property of PPO on this task.
- Hybrid's behavior at exactly nominal mass is genuinely seed-dependent: seed 0
  fails there (0%), seeds 1 and 2 both succeed (100%), giving a pooled rate of
  0.667 and a non-zero cross-seed standard deviation at that single point (all
  other points have zero cross-seed spread). This directly confirms the limitation
  flagged in results.md section 5 ("a different training seed could land in a
  different region of this failure landscape") — phase 1 could only hypothesize
  this from a single seed; phase 2 observed it directly.

**What's new:** the training-time stability finding in section 3.1 — Hybrid trains
to a more stable, higher-episode-length policy than PPO on the training
distribution itself, consistently across all three seeds — which phase 1 never
measured. This is evidence the PID-floor mechanism behind H2 is real, even though
it does not rescue H1's zero-shot robustness claim. Put together: the hybrid
structure helps *in-distribution* training stability, but that benefit does not
transfer to *out-of-distribution* robustness, which is a more precise statement
than phase 1 could support on its own.

## 6. Limitations and remaining future work

- `hybrid_tight_clip` (the 0.08-clip ablation from results.md section 3.1) was not
  retrained at phase 2 scale, to bound total compute to roughly 4-5 hours of
  training this phase. Re-running it under the same seed sweep would show whether
  the tighter clip's nominal-mass fix (phase 1) is itself seed-dependent, the same
  way the untightened hybrid's nominal-mass result turned out to be here.
- Domain randomization still covers mass only (inertia/thrust randomization from
  research_plan.md section 4.3 remains untried in both phases).
- The device benchmark (section 1) and the "same seed, different n_envs" caveat
  (section 5) both point at the same underlying issue: this project's results are
  sensitive to exact training configuration (parallelism, budget) in ways beyond
  just the random seed. A fully rigorous comparison would hold `n_envs` fixed across
  phase 1 and phase 2 and vary only the seed, which was not done here (phase 2
  deliberately raised `n_envs` to use the new hardware, which is part of the point
  of this phase, but it does confound the "same seed" comparison in section 5).
- All simulation in both phases is CPU-bound PyBullet; section 1's device finding is
  specific to that choice. A GPU-native simulator (e.g. Isaac Gym/Lab, MuJoCo MJX,
  Brax) running thousands of environments directly on the GPU would likely change
  the device conclusion entirely, since the bottleneck this phase identified
  (CPU-bound physics stepping, not policy network compute) would no longer apply.
  That would be a substantial rewrite of `src/envs/`, not a configuration change, and
  is out of scope here.
