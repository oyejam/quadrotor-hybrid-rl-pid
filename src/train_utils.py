"""Shared helpers for scripts/train_ppo.py and scripts/train_hybrid.py so the
two controllers are built from the same env/config-loading/training logic and
only differ in whether the hybrid PID+residual wrapper is applied. Keeping
this in one place is what makes the PPO-vs-Hybrid comparison fair: same
training budget, same env settings, same eval protocol.
"""
import os

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from src.envs.traj_track_aviary import TrajTrackAviary
from src.controllers.hybrid_wrapper import HybridResidualWrapper


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env_fn(cfg: dict, hybrid: bool = False, gui: bool = False):
    """Returns a zero-arg factory building one TrajTrackAviary instance
    (optionally wrapped for the hybrid controller), per `cfg`.
    """
    env_cfg = cfg["env"]
    act_type = ActionType(env_cfg.get("action_type", "rpm"))
    obs_type = ObservationType(env_cfg.get("obs_type", "kin"))

    dr_cfg = cfg.get("domain_randomization", {})
    if dr_cfg.get("enabled", False):
        mass_range = tuple(dr_cfg["mass_scale_range"])
    else:
        mass_range = (1.0, 1.0)

    def _init():
        env = TrajTrackAviary(
            act=act_type,
            obs=obs_type,
            pyb_freq=env_cfg.get("pyb_freq_hz", 240),
            ctrl_freq=env_cfg.get("ctrl_freq_hz", 48),
            gui=gui,
            mass_scale_range=mass_range,
        )
        if hybrid:
            residual_clip = cfg["hybrid"]["residual_clip_fraction"]
            env = HybridResidualWrapper(env, residual_clip_fraction=residual_clip)
        return env

    return _init


def build_vec_env(cfg: dict, hybrid: bool, n_envs: int):
    """n_envs>1 uses SubprocVecEnv (one OS process per env, uses multiple
    cores); n_envs==1 uses DummyVecEnv (no subprocess overhead, easier to
    debug). Each sub-env is wrapped in Monitor so SB3 can log episode
    reward/length.
    """
    def make_thunk():
        def _thunk():
            return Monitor(make_env_fn(cfg, hybrid=hybrid)())
        return _thunk

    if n_envs > 1:
        return SubprocVecEnv([make_thunk() for _ in range(n_envs)])
    return DummyVecEnv([make_thunk()])


def run_training(cfg: dict, hybrid: bool, total_timesteps_override: int = None,
                  seed_override: int = None, output_dir_override: str = None):
    """Builds the vectorized training env + a single eval env from `cfg`,
    trains PPO, and saves the final model. Used identically by
    scripts/train_ppo.py and scripts/train_hybrid.py.

    seed_override/output_dir_override exist for the phase-2 seed sweep
    (docs/gpu_machine_plan.md section 4): running the same config under
    seeds 0/1/2 needs a distinct seed and output directory per run without
    hand-editing the YAML each time.
    """
    train_cfg = cfg["training"]
    seed = seed_override if seed_override is not None else train_cfg.get("seed", 0)
    n_envs = train_cfg.get("n_envs", 1)
    total_timesteps = total_timesteps_override or train_cfg["total_timesteps"]

    out_dir = output_dir_override or cfg["logging"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_env = build_vec_env(cfg, hybrid=hybrid, n_envs=n_envs)
    eval_env = Monitor(make_env_fn(cfg, hybrid=hybrid)())

    tb_log = os.path.join(out_dir, "tb") if cfg["logging"].get("tensorboard", False) else None
    device = train_cfg.get("device", "auto")
    model = PPO(train_cfg.get("policy", "MlpPolicy"), train_env, verbose=1,
                seed=seed, tensorboard_log=tb_log, device=device)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=max(train_cfg.get("eval_freq", 5000) // max(n_envs, 1), 1),
        n_eval_episodes=train_cfg.get("n_eval_episodes", 5),
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    final_path = os.path.join(out_dir, "final_model.zip")
    model.save(final_path)
    train_env.close()
    eval_env.close()
    return final_path
