"""Evaluate one checkpoint on specific stair difficulty levels, with video.

Runs a single robot in a single env on each requested terrain level (default 3, 6, 9),
records video, and reports PASS/FAIL, time-to-goal, and stair-edge contacts.

    ./scripts/eval_levels.sh --checkpoint logs/.../checkpoints/best_agent.pt
    ./scripts/eval_levels.sh --checkpoint <ckpt> --levels 3,6,9 --depth_pov
    ./scripts/eval_levels.sh --checkpoint <ckpt> --depth_pov --no_rgb   # cheapest

Why force the level instead of letting the curriculum pick? The curriculum assigns a
level per env based on past performance, and on solving the last level Isaac Lab
reassigns the env to a RANDOM level. Neither is reproducible. Here the level is pinned
and `_update_terrain_curriculum` is stubbed out, so "level 9" means level 9 every time.

Level -> geometry (difficulty = (row + U[0,1)) / 10, step_height = 0.05 + 0.15*difficulty,
platform height = 6 * step_height):

    level 3  ->  0.095-0.110 m risers,  0.57-0.66 m climb
    level 6  ->  0.140-0.155 m risers,  0.84-0.93 m climb
    level 9  ->  0.185-0.200 m risers,  1.11-1.20 m climb   (max)

PASS means the robot got within `command_target_dist_threshold_m` (0.4 m) of the tile's
centre platform, which is the same latched condition the curriculum uses.

Two video streams, independently toggleable:

  * third-person chase cam (default). Needs the RTX render pipeline, so it implies
    --enable_cameras.
  * --depth_pov: the robot's own depth camera. This is a RayCasterCamera (Warp
    ray-cast, not RTX) that already runs every step to build the policy observation,
    so reading it costs essentially nothing. With --no_rgb the RTX pipeline is never
    booted at all.

The depth POV panel shows the 64x36 raw sensor image AND the 16x16 crop the policy
actually receives, because those are not the same thing and the difference is exactly
where a "why did it miss that step" answer tends to live.

Note on the PLAY config: this script forces viewer.origin_type="world" and drives the
camera itself via sim.set_camera_view. G1StairsEnvCfg_PLAY sets origin_type="asset_root"
/ asset_name="robot", which activates Isaac Lab's viewport camera controller; that
controller expects an env.scene attribute this DirectRLEnv does not expose.
"""

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Evaluate a checkpoint on fixed stair levels with video.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a single .pt checkpoint.")
parser.add_argument("--levels", type=str, default="3,6,9",
                    help="Comma-separated terrain levels to test (0-9). Default 3,6,9.")
parser.add_argument("--trials", type=int, default=1,
                    help="Attempts per level. Reset noise makes runs differ; >1 gives a success rate.")
parser.add_argument("--max_steps", type=int, default=1000,
                    help="Step cap per attempt (episode limit is 1000 = 20 s).")
parser.add_argument("--task", type=str, default="Isaac-G1-AMP-Stairs-Direct-Play-v0")
parser.add_argument("--out_dir", type=str, default=None,
                    help="Output dir. Default: <run_dir>/level_eval/")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--depth_pov", action="store_true",
                    help="Also record the robot's depth-camera POV (raw + policy crop).")
parser.add_argument("--no_rgb", action="store_true",
                    help="Skip the third-person video. Avoids booting the RTX pipeline entirely.")
parser.add_argument("--no_video", action="store_true",
                    help="Metrics only, no video of any kind.")
parser.add_argument("--stochastic", action="store_true",
                    help="Sample from the policy instead of using the mean action.")
parser.add_argument("--cam_offset", type=str, default="3.0,3.0,2.0",
                    help="Chase-cam offset from the robot root, 'x,y,z' in metres.")
parser.add_argument("--fixed_side_cam", action="store_true",
                    help="Use a fixed world-space side-profile camera and also output a 0.5x slow-motion video.")
parser.add_argument("--depth_mode", type=str, choices=["real", "zero", "frozen", "shuffle"], default="real",
                    help="Ablation mode for the depth observation.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

WANT_RGB = not args_cli.no_rgb and not args_cli.no_video
WANT_POV = args_cli.depth_pov and not args_cli.no_video

args_cli.headless = True
# Only the third-person view needs the RTX render pipeline. The depth camera is a
# Warp ray-caster that runs regardless, so a POV-only run can skip RTX completely.
args_cli.enable_cameras = WANT_RGB

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# Imports requiring Isaac Sim
# =============================================================================

import cv2
import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import deterministic_model

import isaaclab_project.g1_stairs  # noqa: F401  (registers the gym tasks)

from isaaclab_project.g1_stairs.agents.wasabi_amp import WasabiAMP
from isaaclab_project.g1_stairs.g1_stairs_env_cfg import (
    STAIRS_GENERATOR_NUM_STEPS,
    STAIRS_STAIR_REGION_HALF_EXTENT_M,
    STAIRS_TREAD_DEPTH_TRUE_M,
)
from isaaclab_project.g1_stairs.models import DepthAmpPolicy, DepthAmpValue, PROPRIO_DIM, DEPTH_DIM, OBS_DIM
from isaaclab_rl.skrl import SkrlVecEnvWrapper


FPS = 50.0  # sim dt 0.005 * decimation 4 = 0.02 s per env step


# =============================================================================
# Helpers
# =============================================================================

def run_dir_of(ckpt_path):
    """<run>/checkpoints/x.pt -> <run>;  <run>/x.pt -> <run>."""
    parent = os.path.dirname(os.path.normpath(ckpt_path))
    return os.path.dirname(parent) if os.path.basename(parent) == "checkpoints" else parent


def logval(log, key, default=float("nan")):
    """Read a scalar out of env.extras['log'], which holds torch tensors."""
    v = log.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def force_level(env_unwrapped, level):
    """Pin every env to `level` and move its origin/goal to that tile.

    terrain_origins is (num_rows, num_cols, 3) and the whole grid already exists in
    the scene, so switching levels is just an index change -- no rebuild needed.
    Must happen BEFORE reset, because _reset_idx computes the spawn pose and the
    goal from terrain.env_origins.
    """
    terrain = env_unwrapped.terrain
    n_rows = terrain.terrain_origins.shape[0]
    lvl = int(np.clip(level, 0, n_rows - 1))
    terrain.terrain_levels[:] = lvl
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    return lvl


def freeze_curriculum(env_unwrapped):
    """Stub out promotion/demotion so a pinned level survives resets.

    The real method rewrites terrain.env_origins on every reset. It also populates
    _curriculum_diag, which _get_rewards later merges into the log, so the stub has
    to keep that attribute alive rather than simply doing nothing.
    """
    def _frozen(env_ids, _u=env_unwrapped):
        _u._track_score_sum[env_ids] = 0.0
        _u._curriculum_diag = {}

    env_unwrapped._update_terrain_curriculum = _frozen


def depth_panel(env_unwrapped, width=640):
    """Render the depth camera as a two-row panel: raw sensor, and policy input.

    Top    : full 64x36 `distance_to_image_plane`, with the crop rectangle drawn on it.
    Bottom : the 16x16 blurred/normalized tensor the policy actually consumes.

    Both use the same 0..DEPTH_RANGE_M[1] scale and colormap, so near/far read the
    same way in each. Nearest-neighbour upscaling is deliberate -- at 16x16 any
    smoothing would invent structure that the policy never saw.
    """
    cfg = env_unwrapped.cfg
    far = cfg.DEPTH_RANGE_M[1]

    raw = env_unwrapped.depth_camera.data.output["distance_to_image_plane"]
    if raw.dim() == 4:
        raw = raw.squeeze(-1)
    raw = torch.nan_to_num(raw, nan=far, posinf=far, neginf=cfg.DEPTH_RANGE_M[0])
    raw_np = raw[0].detach().cpu().numpy()

    proc = env_unwrapped._process_depth_image()          # (N, 16, 16) already in [0, 1]
    proc_np = proc[0].detach().cpu().numpy()

    def colorize(arr01, out_w, out_h):
        u8 = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    raw01 = np.clip(raw_np / far, 0.0, 1.0)
    rh, rw = raw01.shape
    top_h = int(round(width * rh / rw))
    top = colorize(raw01, width, top_h)

    # Crop rectangle, scaled from raw pixels into the upscaled image.
    cx, cy, cw, ch = cfg.DEPTH_CROP_REGION
    sx, sy = width / rw, top_h / rh
    cv2.rectangle(top,
                  (int(cx * sx), int(cy * sy)),
                  (int((cx + cw) * sx), int((cy + ch) * sy)),
                  (255, 255, 255), 2)
    cv2.putText(top, f"RAW {rw}x{rh}  (0-{far:.1f} m)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    bottom = colorize(proc_np, width, width // 2)
    cv2.putText(bottom, f"POLICY INPUT {proc_np.shape[1]}x{proc_np.shape[0]} (cropped, blurred)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([top, bottom])


def annotate(frame, lines):
    """Draw a small HUD, dark text under light text so it reads on any background."""
    y = 26
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return frame


class Writer:
    """Lazy cv2.VideoWriter -- frame size is not known until the first frame."""

    def __init__(self, path):
        self.path = path
        self.w = None

    def write(self, frame_bgr):
        if self.w is None:
            h, w = frame_bgr.shape[:2]
            self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            if not self.w.isOpened():
                raise RuntimeError(f"could not open video writer: {self.path}")
        self.w.write(frame_bgr)

    def close(self):
        if self.w is not None:
            self.w.release()
            self.w = None


# =============================================================================
# Agent construction (mirrors scripts/export_checkpoint_videos.py)
# =============================================================================

def build_agent(env, device, ckpt_path):
    amp_batch = 8
    models = {
        "policy": DepthAmpPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            initial_log_std=0.0,
        ),
        "value": DepthAmpValue(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        ),
        "discriminator": deterministic_model(
            observation_space=env.amp_observation_space,
            action_space=env.action_space,
            device=device,
            network=[{"name": "net", "input": "OBSERVATIONS",
                      "layers": [256, 128, 128], "activations": "elu"}],
            output="ONE",
        ),
    }
    cfg = {
        "amp_batch_size": amp_batch,
        "amp_state_preprocessor": RunningStandardScaler,
        "amp_state_preprocessor_kwargs": {"size": env.amp_observation_space, "device": device},
        "experiment": {"directory": "", "experiment_name": "", "write_interval": 0,
                       "checkpoint_interval": 0, "store_separately": False,
                       "wandb": False, "wandb_kwargs": {}},
    }
    agent = WasabiAMP(
        models=models,
        memory=RandomMemory(memory_size=1, num_envs=env.num_envs, device=device),
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        amp_observation_space=env.amp_observation_space,
        motion_dataset=RandomMemory(memory_size=amp_batch, device=device),
        reply_buffer=RandomMemory(memory_size=amp_batch, device=device),
        collect_reference_motions=lambda n: env.unwrapped.collect_reference_motions(n),
    )
    agent.init(trainer_cfg={"timesteps": 1, "headless": True})
    agent.load(ckpt_path)
    agent.set_running_mode("eval")
    return agent


def choose_action(agent, obs, stochastic):
    """Mean action by default: eval should be reproducible, not a lottery."""
    with torch.no_grad():
        out = agent.act(obs, timestep=0, timesteps=1)
    actions = out[0]
    if not stochastic and len(out) > 2 and isinstance(out[2], dict):
        mean = out[2].get("mean_actions")
        if mean is not None:
            actions = mean
    return actions


# =============================================================================
# One attempt
# =============================================================================

def run_attempt(env, agent, u, level, trial, out_dir, ckpt_name, cam_offset):
    force_level(u, level)
    env._reset_once = True
    obs, _ = env.reset()

    platform_h = float(u.terrain.env_origins[0, 2])
    goal_radius = float(u.cfg.command_target_dist_threshold_m)
    step_dt = float(getattr(u, "step_dt", 0.02))
    margin = float(u.cfg.edge_safety_margin_m)

    tag = f"level{level}_trial{trial}_{ckpt_name}"
    rgb_w = Writer(os.path.join(out_dir, f"{tag}.mp4")) if WANT_RGB else None
    rgb_slow_w = (
        Writer(os.path.join(out_dir, f"{tag}_0.5x.mp4"))
        if (WANT_RGB and args_cli.fixed_side_cam)
        else None
    )
    pov_w = Writer(os.path.join(out_dir, f"{tag}_depthpov.mp4")) if WANT_POV else None

    prev_violating = torch.zeros(1, 2, dtype=torch.bool, device=u.device)
    prev_same_step = False
    edge_hits = 0
    edge_frames = 0
    climb_edge_hits = 0
    climb_edge_frames = 0
    max_pen = 0.0
    max_contact_pen = 0.0
    climb_max_contact_pen = 0.0
    same_step_hits = 0
    same_step_frames = 0
    rgb_frames = 0
    reached_step = None
    max_climb = 0.0
    climb_frac = 0.0
    goal_dist = float("nan")
    term_reason = None
    steps = 0

    off = np.array([float(x) for x in cam_offset.split(",")], dtype=float)
    root0 = u.robot.data.root_pos_w[0].detach().cpu().numpy()
    orig0 = u.terrain.env_origins[0].detach().cpu().numpy()
    d_xy = orig0[:2] - root0[:2]
    d_norm = max(float(np.linalg.norm(d_xy)), 1e-6)
    f_hat = d_xy / d_norm
    s_hat = np.array([f_hat[1], -f_hat[0]], dtype=float)
    target_xy = root0[:2] + 0.52 * d_xy
    target_z = 0.55 * platform_h + 0.45
    fixed_target = (float(target_xy[0]), float(target_xy[1]), float(target_z))
    eye_xy = target_xy + 2.85 * s_hat - 0.15 * f_hat
    eye_z = target_z + 0.18
    fixed_eye = (float(eye_xy[0]), float(eye_xy[1]), float(eye_z))

    try:
        for t in range(args_cli.max_steps):
            steps = t

            # ---- measure on the CURRENT state -------------------------------
            # Isaac Lab resets inside step() when an episode ends, so anything read
            # after step() would describe the fresh episode. Everything derived from
            # live tensors is therefore sampled here, before stepping.
            feet_pos_w = u.robot.data.body_pos_w[:, u.feet_body_indexes]
            forces = u.contact_sensor.data.net_forces_w[:, u.feet_contact_body_indexes]
            in_contact = torch.norm(forces, dim=-1) > 1.0
            edge_dist = u._edge_distance_at(feet_pos_w)
            violating = in_contact & (edge_dist < margin)
            new_hits = int((violating & ~prev_violating).sum())
            new_frames = int(violating.sum())
            edge_hits += new_hits
            edge_frames += new_frames
            prev_violating = violating

            # Geometric 3D volume-point penetration (all phases and stance-gated)
            pts, _ = u._volume_points_world_state()
            pen = u._stair_edge_penetration_depth(pts)
            pen_contact = pen * in_contact.unsqueeze(-1).float()
            step_max_pen = float(pen.max())
            step_contact_pen = float(pen_contact.max())
            max_pen = max(max_pen, step_max_pen)
            max_contact_pen = max(max_contact_pen, step_contact_pen)
            if reached_step is None:
                climb_edge_hits += new_hits
                climb_edge_frames += new_frames
                climb_max_contact_pen = max(climb_max_contact_pen, step_contact_pen)

            # Same-step double-support tracking on stairs (computed directly from live state)
            origin = u.terrain.env_origins
            cheb_feet = torch.maximum(
                torch.abs(feet_pos_w[..., 0] - origin[:, 0].unsqueeze(1)),
                torch.abs(feet_pos_w[..., 1] - origin[:, 1].unsqueeze(1)),
            )
            inward_feet = STAIRS_STAIR_REGION_HALF_EXTENT_M - cheb_feet
            stair_idx_feet = torch.floor(inward_feet / STAIRS_TREAD_DEPTH_TRUE_M).long()
            stairs_max_inward = STAIRS_GENERATOR_NUM_STEPS * STAIRS_TREAD_DEPTH_TRUE_M
            on_stairs = (inward_feet >= 0.0) & (inward_feet < stairs_max_inward)
            same_step_now = bool(
                (on_stairs[:, 0] & on_stairs[:, 1] & in_contact[:, 0] & in_contact[:, 1]
                 & (stair_idx_feet[:, 0] == stair_idx_feet[:, 1])).any()
            )
            if same_step_now:
                same_step_frames += 1
                if not prev_same_step:
                    same_step_hits += 1
            prev_same_step = same_step_now

            root = u.robot.data.root_pos_w[0].detach().cpu().numpy()

            # ---- video ------------------------------------------------------
            if rgb_w is not None:
                if args_cli.fixed_side_cam:
                    u.sim.set_camera_view(eye=fixed_eye, target=fixed_target)
                else:
                    u.sim.set_camera_view(eye=tuple(root + off), target=tuple(root))
                # NOT env.render(): SkrlVecEnvWrapper.render() is a no-op that returns
                # None, so going through the wrapper silently produces no video at all.
                # The frame has to come from the underlying DirectRLEnv.
                frame = u.render()
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR if frame.shape[-1] == 4
                                       else cv2.COLOR_RGB2BGR)
                    annotate(bgr, [
                        f"L{level} t{trial}  step {t}  {t * step_dt:5.2f}s",
                        f"climb {max_climb:.2f}/{platform_h:.2f} m ({climb_frac * 100:4.0f}%)",
                        f"goal {goal_dist:.2f} m (pass <{goal_radius:.1f})",
                        f"climb edge hits {climb_edge_hits}  same-step {same_step_hits}",
                    ])
                    rgb_w.write(bgr)
                    if rgb_slow_w is not None:
                        rgb_slow_w.write(bgr)
                        rgb_slow_w.write(bgr)
                    rgb_frames += 1

            if pov_w is not None:
                panel = depth_panel(u)
                annotate(panel, [f"L{level} t{trial}  step {t}  {t * step_dt:5.2f}s"])
                pov_w.write(panel)

            # ---- act --------------------------------------------------------
            policy_obs = obs.clone()
            if args_cli.depth_mode != "real":
                assert policy_obs.shape[1] == OBS_DIM, f"Expected {OBS_DIM}, got {policy_obs.shape[1]}"
                depth_block = policy_obs[:, PROPRIO_DIM : PROPRIO_DIM + DEPTH_DIM]
                B = depth_block.shape[0]
                depth_block = depth_block.view(B, 8, 16, 16).clone()
                
                if args_cli.depth_mode == "zero":
                    depth_block.zero_()
                elif args_cli.depth_mode == "frozen":
                    depth_block[:] = depth_block[:, 0:1, :, :].clone()
                elif args_cli.depth_mode == "shuffle":
                    idx = torch.randperm(8, device=depth_block.device)
                    depth_block[:] = depth_block[:, idx, :, :].clone()
                
                policy_obs[:, PROPRIO_DIM : PROPRIO_DIM + DEPTH_DIM] = depth_block.view(B, DEPTH_DIM)

            actions = choose_action(agent, policy_obs, args_cli.stochastic)
            obs, _, terminated, truncated, _ = env.step(actions)

            # extras['log'] is filled during _get_rewards, i.e. before any reset.
            log = u.extras.get("log", {})
            max_climb = logval(log, "diag_mean_episode_max_climb_m", max_climb)
            goal_dist = logval(log, "diag_mean_planar_dist_to_goal", goal_dist)
            climb_frac = max_climb / platform_h if platform_h > 1e-6 else 0.0
            if reached_step is None and logval(log, "diag_frac_reached_goal", 0.0) > 0.5:
                reached_step = t + 1

            done = bool(terminated.any()) or bool(truncated.any())
            if done:
                steps = t + 1
                if bool(terminated.any()):
                    for key, name in (("diag_term_height", "fell (height)"),
                                      ("diag_term_tilt", "fell (tilt)"),
                                      ("diag_term_contact", "body contact")):
                        if logval(log, key, 0.0) > 0.5:
                            term_reason = name
                            break
                    term_reason = term_reason or "terminated"
                else:
                    term_reason = "timeout"
                break
        else:
            term_reason = "step cap"
    finally:
        for w in (rgb_w, rgb_slow_w, pov_w):
            if w is not None:
                w.close()
        if rgb_w is not None and rgb_frames == 0:
            print("[WARN] third-person video is empty -- env.render() produced no frames.")

    passed = reached_step is not None
    return {
        "level": level,
        "trial": trial,
        "pass": passed,
        "time_to_goal_s": round(reached_step * step_dt, 2) if passed else None,
        "steps": steps,
        "duration_s": round(steps * step_dt, 2),
        "max_climb_m": round(max_climb, 3),
        "platform_h_m": round(platform_h, 3),
        "climb_frac": round(climb_frac, 3),
        "edge_hits": edge_hits,
        "edge_frames": edge_frames,
        "climb_edge_hits": climb_edge_hits,
        "climb_edge_frames": climb_edge_frames,
        "max_penetration_m": round(max_pen, 4),
        "max_contact_penetration_m": round(max_contact_pen, 4),
        "climb_max_contact_penetration_m": round(climb_max_contact_pen, 4),
        "same_step_hits": same_step_hits,
        "same_step_frames": same_step_frames,
        "outcome": "PASS" if passed else f"FAIL ({term_reason})",
        "video": os.path.join(out_dir, f"{tag}.mp4") if WANT_RGB else None,
        "depth_video": os.path.join(out_dir, f"{tag}_depthpov.mp4") if WANT_POV else None,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ckpt = args_cli.checkpoint
    if not os.path.isfile(ckpt):
        print(f"[ERROR] no such checkpoint: {ckpt}")
        return 1

    levels = [int(x) for x in args_cli.levels.split(",") if x.strip()]
    ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
    out_dir = args_cli.out_dir or os.path.join(run_dir_of(ckpt), "level_eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] checkpoint : {ckpt}")
    print(f"[INFO] levels     : {levels}  x {args_cli.trials} trial(s)")
    print(f"[INFO] video      : rgb={WANT_RGB}  depth_pov={WANT_POV}")
    print(f"[INFO] output     : {out_dir}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    # Shrink 4096-env PhysX capacities for 1-env evaluation so it can run safely alongside training
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    env_cfg.sim.physx.gpu_collision_stack_size = 2**22
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**15
    # Drive the camera manually; see the module docstring for why asset_root is avoided.
    if hasattr(env_cfg, "viewer"):
        try:
            env_cfg.viewer.origin_type = "world"
            env_cfg.viewer.asset_name = None
        except Exception:
            pass

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if WANT_RGB else None)
    env = SkrlVecEnvWrapper(env)
    u = env.unwrapped
    device = u.device

    agent = build_agent(env, device, ckpt)
    freeze_curriculum(u)

    results = []
    any_error = False
    for level in levels:
        for trial in range(args_cli.trials):
            print(f"\n[INFO] === level {level}, trial {trial} ===")
            try:
                r = run_attempt(env, agent, u, level, trial, out_dir, ckpt_name,
                                args_cli.cam_offset)
            except Exception as exc:
                tb = traceback.format_exc()
                print(tb, file=sys.stderr)
                any_error = True
                results.append({
                    "level": level,
                    "trial": trial,
                    "pass": False,
                    "time_to_goal_s": None,
                    "steps": 0,
                    "duration_s": 0.0,
                    "max_climb_m": 0.0,
                    "platform_h_m": 0.0,
                    "climb_frac": 0.0,
                    "edge_hits": 0,
                    "edge_frames": 0,
                    "max_penetration_m": 0.0,
                    "max_contact_penetration_m": 0.0,
                    "same_step_hits": 0,
                    "same_step_frames": 0,
                    "outcome": f"ERROR ({type(exc).__name__})",
                    "error": repr(exc),
                    "traceback": tb,
                    "video": None,
                    "depth_video": None,
                })
                continue
            results.append(r)
            print(f"[INFO] {r['outcome']}  climb {r['max_climb_m']}/{r['platform_h_m']} m  "
                  f"climb_edge_hits {r['climb_edge_hits']} (total {r['edge_hits']})  "
                  f"same_step_hits {r['same_step_hits']}  "
                  f"climb_contact_pen {r['climb_max_contact_penetration_m']:.4f} m")

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 124)
    print(f"LEVEL EVAL  --  {ckpt_name}")
    print("=" * 124)
    hdr = (f"{'LVL':>3s} {'TRIAL':>5s} {'RESULT':>18s} {'t_goal':>7s} {'dur':>6s} "
           f"{'climb':>12s} {'frac':>6s} {'clb_ehit':>8s} {'tot_ehit':>8s} "
           f"{'max_pen':>8s} {'clb_cpen':>8s} {'same_stp':>8s} {'ss_frm':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        tg = f"{r['time_to_goal_s']:.2f}" if r["time_to_goal_s"] is not None else "-"
        print(f"{r['level']:3d} {r['trial']:5d} {r['outcome']:>18s} {tg:>7s} "
              f"{r['duration_s']:6.2f} "
              f"{r['max_climb_m']:6.2f}/{r['platform_h_m']:<5.2f} "
              f"{r['climb_frac']:6.2f} {r.get('climb_edge_hits', 0):8d} {r['edge_hits']:8d} "
              f"{r['max_penetration_m']:8.4f} {r.get('climb_max_contact_penetration_m', 0.0):8.4f} "
              f"{r.get('same_step_hits', 0):8d} {r.get('same_step_frames', 0):6d}")
    print("-" * len(hdr))
    for level in levels:
        sub = [r for r in results if r["level"] == level]
        if sub:
            n_pass = sum(1 for r in sub if r["pass"])
            print(f"  level {level}: {n_pass}/{len(sub)} passed")
    print("=" * 124)

    summary = os.path.join(out_dir, f"eval_{ckpt_name}.json")
    with open(summary, "w") as fh:
        json.dump({"checkpoint": ckpt, "task": args_cli.task, "results": results}, fh, indent=2)
    print(f"[INFO] wrote {summary}")
    if WANT_RGB or WANT_POV:
        print(f"[INFO] videos in {out_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    return 1 if any_error else 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
