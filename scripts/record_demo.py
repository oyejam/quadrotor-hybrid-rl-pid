"""Records a short, captioned demo video showing the actual phenomenon in
docs/results.md and docs/results_phase2.md: PID completes the trajectory
at every tested mass, while PPO/Hybrid crash almost immediately at or
below nominal mass and only recover once the drone is heavier. Uses the
phase 2 seed-swept models (results/phase2/*) so the clips match the
current, statistically stronger results rather than the single-seed
phase 1 run, and adds the two findings specific to phase 2: PPO's
failure now extends to +10% mass (not just at/below nominal), and the
hybrid's nominal-mass outcome is seed-dependent (shown by playing seed 0
and seed 1 back to back at the same mass).

Runs entirely headless (gui=False) using PyBullet's software TinyRenderer,
so it works on machines with no GPU/display. Uses the `imageio-ffmpeg`
package's bundled static ffmpeg binary to assemble frames into an .mp4
(`pip install imageio-ffmpeg` -- more reliable than a Homebrew ffmpeg
install, which hit an unrelated Homebrew bug on this machine's older
macOS/Homebrew combination).

Usage
-----
    python scripts/record_demo.py
    python scripts/record_demo.py --output results/demo.mp4
"""
import os
import sys
import shutil
import argparse
import subprocess

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO

from src.traj import build_waypoints
from src.envs.traj_track_aviary import TrajTrackAviary
from src.controllers.hybrid_wrapper import HybridResidualWrapper
from src.train_utils import load_config

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

FPS = 24  # matches gym_pybullet_drones' own FRAME_PER_SEC for record=True


def _set_close_camera(env):
    """gym_pybullet_drones' default record camera is 3m from the origin,
    which makes our ~0.3m-radius trajectory an invisible speck (see the
    frame this produced by default). Override it to frame the actual
    trajectory loop (centered at (0, -R, H), R=0.3, H=0.1 -- src/traj.py)
    tightly.
    """
    env.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(
        distance=1.0, yaw=-45, pitch=-35, roll=0,
        cameraTargetPosition=[0, -0.3, 0.15], upAxisIndex=2,
        physicsClientId=env.CLIENT)


def record_pid_clip(frames_dir, mass_scale, max_seconds=10, ctrl_freq=48, pyb_freq=240):
    wp = build_waypoints(ctrl_freq)
    init_xyz = np.array([wp[0, :]])
    env = CtrlAviary(drone_model=DroneModel('cf2x'), num_drones=1,
                      initial_xyzs=init_xyz, physics=Physics('pyb'),
                      pyb_freq=pyb_freq, ctrl_freq=ctrl_freq, gui=False,
                      record=True, obstacles=False, output_folder=frames_dir)
    ctrl = DSLPIDControl(drone_model=DroneModel('cf2x'))
    obs, info = env.reset(seed=0)
    _set_close_camera(env)
    if mass_scale != 1.0:
        p.changeDynamics(env.DRONE_IDS[0], -1, mass=env.M * mass_scale, physicsClientId=env.CLIENT)
    action = np.zeros((1, 4))
    max_steps = min(len(wp), int(max_seconds * ctrl_freq))
    for i in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        state = obs[0]
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP, state=state, target_pos=wp[i, :])
    env.close()
    return _find_recording_dir(frames_dir)


def record_rl_clip(frames_dir, model_path, cfg_path, hybrid, mass_scale, max_seconds=10):
    # max_seconds matches episode_len_sec (10s) in config/*.yaml -- a shorter
    # cap can end before a late failure (e.g. the phase 2 hybrid seed 0
    # model at nominal mass fails at step 406/480, 8.46s in) and make a
    # clip labeled "crashes" look like it's still flying fine.
    # Note: TrajTrackAviary (via BaseRLAviary) does not expose an
    # output_folder constructor argument in this version of
    # gym_pybullet_drones -- it always records under BaseAviary's own
    # default ('results'), unlike CtrlAviary which does accept it. So
    # `frames_dir` is unused here; the caller's _find_recording_dir call
    # must point at 'results' for RL clips specifically.
    cfg = load_config(cfg_path)
    env_cfg = cfg["env"]
    env = TrajTrackAviary(act=ActionType(env_cfg.get("action_type", "rpm")),
                           obs=ObservationType(env_cfg.get("obs_type", "kin")),
                           pyb_freq=env_cfg.get("pyb_freq_hz", 240),
                           ctrl_freq=env_cfg.get("ctrl_freq_hz", 48),
                           gui=False, record=True,
                           mass_scale_range=(mass_scale, mass_scale))
    if hybrid:
        env = HybridResidualWrapper(env, residual_clip_fraction=cfg["hybrid"]["residual_clip_fraction"])
    model = PPO.load(model_path)

    obs, info = env.reset(seed=0)
    _set_close_camera(env.unwrapped)
    terminated = truncated = False
    max_steps = int(max_seconds * env_cfg.get("ctrl_freq_hz", 48))
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    env.close()
    return _find_recording_dir("results")


def _find_recording_dir(frames_dir):
    candidates = [d for d in os.listdir(frames_dir) if d.startswith("recording_")]
    candidates.sort()
    return os.path.join(frames_dir, candidates[-1])


def make_title_frames(label, out_dir, width=640, height=480, seconds=1.5):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(15, 15, 20))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except OSError:
        font = ImageFont.load_default()
    lines = label.split("\n")
    total_h = sum(draw.textbbox((0, 0), l, font=font)[3] for l in lines) + 10 * (len(lines) - 1)
    y = (height - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((width - w) // 2, y), line, fill=(255, 255, 255), font=font)
        y += bbox[3] + 10
    n_frames = int(seconds * FPS)
    for i in range(n_frames):
        img.save(os.path.join(out_dir, f"title_{i:04d}.png"))
    return n_frames


def assemble_clip(title_dir, n_title_frames, flight_dir, combined_dir):
    os.makedirs(combined_dir, exist_ok=True)
    idx = 0
    for i in range(n_title_frames):
        shutil.copy(os.path.join(title_dir, f"title_{i:04d}.png"),
                    os.path.join(combined_dir, f"frame_{idx:05d}.png"))
        idx += 1
    flight_frames = sorted(
        [f for f in os.listdir(flight_dir) if f.startswith("frame_") and f.endswith(".png")],
        key=lambda f: int(f[len("frame_"):-len(".png")])
    )
    for f in flight_frames:
        shutil.copy(os.path.join(flight_dir, f), os.path.join(combined_dir, f"frame_{idx:05d}.png"))
        idx += 1
    return idx


def encode_clip(combined_dir, out_path):
    subprocess.run([
        FFMPEG, "-y", "-framerate", str(FPS),
        "-i", os.path.join(combined_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        out_path
    ], check=True, capture_output=True)


def concat_clips(clip_paths, out_path, work_dir):
    list_path = os.path.join(work_dir, "concat_list.txt")
    with open(list_path, "w") as f:
        for c in clip_paths:
            f.write(f"file '{os.path.abspath(c)}'\n")
    subprocess.run([
        FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", out_path
    ], check=True, capture_output=True)


CLIPS = [
    # (label, kind, controller-specific kwargs)
    ("PID\nmass = 70% of nominal\nstill tracks reliably", "pid", dict(mass_scale=0.7)),
    ("PPO (seed 0)\nmass = nominal\ncrashes almost immediately",
     "rl", dict(model_path="results/phase2/ppo_seed0/best_model.zip", cfg_path="config/ppo.yaml",
                hybrid=False, mass_scale=1.0)),
    ("PPO (seed 0)\nmass = 110% of nominal\nstill fails here -- consistent across all 3 seeds",
     "rl", dict(model_path="results/phase2/ppo_seed0/best_model.zip", cfg_path="config/ppo.yaml",
                hybrid=False, mass_scale=1.1)),
    ("PPO (seed 0)\nmass = 120% of nominal\nsucceeds once heavier",
     "rl", dict(model_path="results/phase2/ppo_seed0/best_model.zip", cfg_path="config/ppo.yaml",
                hybrid=False, mass_scale=1.2)),
    ("Hybrid (seed 0)\nmass = nominal\ncrashes despite the PID floor",
     "rl", dict(model_path="results/phase2/hybrid_seed0/best_model.zip", cfg_path="config/hybrid.yaml",
                hybrid=True, mass_scale=1.0)),
    ("Hybrid (seed 1), same mass\nsucceeds -- nominal-mass outcome is seed-dependent",
     "rl", dict(model_path="results/phase2/hybrid_seed1/best_model.zip", cfg_path="config/hybrid.yaml",
                hybrid=True, mass_scale=1.0)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/demo.mp4")
    parser.add_argument("--work-dir", type=str, default="results/_demo_frames")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    clip_mp4s = []

    for i, (label, kind, kwargs) in enumerate(CLIPS):
        print(f"[record_demo] clip {i+1}/{len(CLIPS)}: {label.splitlines()[0]} ...")
        raw_frames_dir = os.path.join(args.work_dir, f"raw_{i}")
        if kind == "pid":
            flight_dir = record_pid_clip(raw_frames_dir, **kwargs)
        else:
            flight_dir = record_rl_clip(raw_frames_dir, **kwargs)

        title_dir = os.path.join(args.work_dir, f"title_{i}")
        n_title = make_title_frames(label, title_dir)

        combined_dir = os.path.join(args.work_dir, f"combined_{i}")
        assemble_clip(title_dir, n_title, flight_dir, combined_dir)

        clip_path = os.path.join(args.work_dir, f"clip_{i}.mp4")
        encode_clip(combined_dir, clip_path)
        clip_mp4s.append(clip_path)
        print(f"[record_demo] clip {i+1} encoded: {clip_path}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    concat_clips(clip_mp4s, args.output, args.work_dir)
    print(f"[record_demo] final demo video: {args.output}")


if __name__ == "__main__":
    main()
