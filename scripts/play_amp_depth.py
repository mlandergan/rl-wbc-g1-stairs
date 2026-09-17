"""Play/record a short rollout video from a WasabiAMP + depth-camera checkpoint -- the sibling of
train_amp_depth.py needed for the same reason that script exists at all: Isaac Lab's stock
`scripts/reinforcement_learning/skrl/play.py` resolves the agent/model classes via skrl's
`Runner`, which only knows a hardcoded whitelist (see train_amp_depth.py's own docstring) and
can't construct `WasabiAMP`/`DepthAmpPolicy`/`DepthAmpValue`. This script builds the same
models/agent train_amp_depth.py does, loads a checkpoint into them, and runs a short eval
rollout with RecordVideo instead of calling `trainer.train()`.

Run AFTER training completes, or -- per this project's own documented lesson (see
export_checkpoint_videos.sh's header comment) -- concurrently at real risk of hanging (two
simultaneous Isaac Sim/RTX processes on one GPU have not played well together on this VM before).
If run concurrently anyway, expect a possible stall; kill and retry later if so.

Usage (inside the container):
    ./isaaclab.sh -p /workspace/rl_wbc_g1/scripts/play_amp_depth.py \
        --checkpoint /workspace/isaaclab_root/logs/skrl/g1_stairs/<run>/checkpoints/agent_5000.pt \
        --headless --num_envs 8 --video_length 200
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a WasabiAMP + depth-camera checkpoint and record video.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint (container path).")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-G1-AMP-Stairs-Direct-Play-v0")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument(
    "--camera", type=str, default="follow", choices=("follow", "fixed"),
    help="follow: cam tracks env 0's robot (default). fixed: static shot of env 0's tile.",
)
parser.add_argument(
    "--camera_eye", type=float, nargs=3, default=(5.0, 5.0, 4.0), metavar=("X", "Y", "Z"),
    help="Camera position for --camera fixed, relative to env 0's origin (the centre platform).",
)
parser.add_argument(
    "--camera_lookat", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
    help="Camera target for --camera fixed, relative to the watched env's origin.",
)
parser.add_argument(
    "--camera_env", type=int, default=0,
    help="Which env's tile --camera fixed frames (default 0).",
)
parser.add_argument(
    "--terrain_level", type=int, default=None,
    help="Pin every env to this terrain level (0..num_levels-1) for the whole rollout, "
         "overriding the curriculum. Use to test a specific difficulty -- e.g. 9 for the "
         "hardest stairs, which the mean-terrain-level metric can never show on its own.",
)
parser.add_argument(
    "--video_folder", type=str, default=None,
    help="Defaults to <run_dir>/videos/play_<checkpoint_name> alongside the checkpoint.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym

import isaaclab_project.g1_stairs
from isaaclab_tasks.utils import parse_env_cfg

from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import deterministic_model

from isaaclab_project.g1_stairs.agents.wasabi_amp import WasabiAMP
from isaaclab_project.g1_stairs.models import DepthAmpPolicy, DepthAmpValue


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    if args_cli.camera == "fixed":
        # Static shot: the robot walks through frame instead of the camera chasing it.
        #
        # origin_type="env" does NOT work here. It anchors to the env-spacing grid origin (near
        # the world origin), not to the terrain tile the robot is actually on -- and the stairs
        # generator places tiles tens of metres out (env 0 measured at [-12.48, -7.49]). Using it
        # framed a point ~16 m from the robot. So the camera is placed in WORLD coordinates
        # instead, offset from the real terrain origin, which is only readable after the env
        # exists -- see the update_view_location call below.
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    def _pin_terrain_level():
        """Force every env onto one terrain level, overriding the curriculum.

        Re-applied every step, not just once: the curriculum rewrites terrain_levels AND
        env_origins inside _reset_idx, so an env that falls and resets would otherwise be
        demoted off the level under test partway through the clip.
        """
        if args_cli.terrain_level is None:
            return
        u = env.unwrapped
        u.terrain.terrain_levels[:] = args_cli.terrain_level
        u.terrain.env_origins[:] = u.terrain.terrain_origins[
            u.terrain.terrain_levels, u.terrain.terrain_types
        ]

    # Must run BEFORE the camera is aimed and before the diagnostics dump: pinning moves every
    # env to a different tile, so env_origins changes and both of those read it.
    _pin_terrain_level()

    if args_cli.camera == "fixed":
        _u = env.unwrapped
        _origin = _u.terrain.env_origins[args_cli.camera_env].tolist()
        _eye = [_origin[i] + args_cli.camera_eye[i] for i in range(3)]
        _lookat = [_origin[i] + args_cli.camera_lookat[i] for i in range(3)]
        _u.viewport_camera_controller.update_view_location(eye=_eye, lookat=_lookat)
    device = env.unwrapped.device

    # Diagnostics to a FILE, not stdout: prints from env construction get swallowed inside the
    # container (Isaac Lab's own logger gets through, plain print() does not), so this is the only
    # reliable way to see what the camera and markers actually resolved to.
    try:
        u = env.unwrapped
        diag_path = os.path.join(os.path.dirname(args_cli.checkpoint), "..", "play_diag.txt")
        with open(os.path.abspath(diag_path), "w") as fh:
            fh.write(f"camera_mode={args_cli.camera}\n")
            fh.write(f"viewer.origin_type={env_cfg.viewer.origin_type}\n")
            fh.write(f"viewer.eye={env_cfg.viewer.eye}\n")
            fh.write(f"viewer.lookat={env_cfg.viewer.lookat}\n")
            fh.write(f"viewer.env_index={env_cfg.viewer.env_index}\n")
            fh.write(f"num_envs={u.num_envs}\n")
            fh.write(f"env_origins[0]={u.terrain.env_origins[0].tolist()}\n")
            fh.write(f"env_origins[:4]={u.terrain.env_origins[:4].tolist()}\n")
            fh.write(f"debug_vis_foot_points={u.cfg.debug_vis_foot_points}\n")
            fh.write(f"debug_vis_stair_edges={u.cfg.debug_vis_stair_edges}\n")
            fh.write(f"foot_points_marker={u.foot_points_marker is not None}\n")
            fh.write(f"stair_edge_marker={u.stair_edge_marker is not None}\n")
            fh.write(f"amp_observation_space={u.cfg.amp_observation_space}\n")
            fh.write(f"amp_dof_count={len(u.amp_dof_indexes)}\n")
            t, o, s = u._stair_edge_segments()
            fh.write(f"edge_cylinders={t.shape[0]}\n")
            fh.write(f"terrain_level_arg={args_cli.terrain_level}\n")
            fh.write(f"terrain_levels={u.terrain.terrain_levels[:8].tolist()}\n")
            fh.write(f"platform_heights={[round(v,4) for v in u.terrain.env_origins[:4, 2].tolist()]}\n")
            if u._volume_points_pattern is not None:
                fh.write(f"foot_points_per_foot={u._volume_points_pattern.shape[0]}\n")
        print(f"[INFO] wrote play diagnostics to {os.path.abspath(diag_path)}")
    except Exception as diag_exc:  # diagnostics must never break the render
        print(f"[INFO] play diagnostics failed: {diag_exc}")

    ckpt_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    video_folder = args_cli.video_folder or os.path.join(
        os.path.dirname(os.path.dirname(args_cli.checkpoint)), "videos", f"play_{ckpt_name}"
    )
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        name_prefix=f"play_{ckpt_name}",
    )

    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    env = SkrlVecEnvWrapper(env)

    models = {
        "policy": DepthAmpPolicy(
            observation_space=env.observation_space, action_space=env.action_space, device=device,
            clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, initial_log_std=0.0,
        ),
        "value": DepthAmpValue(observation_space=env.observation_space, action_space=env.action_space, device=device),
        "discriminator": deterministic_model(
            observation_space=env.amp_observation_space, action_space=env.action_space, device=device,
            network=[{"name": "net", "input": "OBSERVATIONS", "layers": [256, 128, 128], "activations": "elu"}],
            output="ONE",
        ),
    }
    memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device)

    _PLAY_AMP_BATCH_SIZE = 8
    cfg = {
        "amp_batch_size": _PLAY_AMP_BATCH_SIZE,
        "amp_state_preprocessor": RunningStandardScaler,
        "amp_state_preprocessor_kwargs": {"size": env.amp_observation_space, "device": device},
        "experiment": {"directory": "", "experiment_name": "", "write_interval": 0, "checkpoint_interval": 0,
                       "store_separately": False, "wandb": False, "wandb_kwargs": {}},
    }
    agent = WasabiAMP(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space, action_space=env.action_space, device=device,
        amp_observation_space=env.amp_observation_space,
        motion_dataset=RandomMemory(memory_size=_PLAY_AMP_BATCH_SIZE, device=device),
        reply_buffer=RandomMemory(memory_size=_PLAY_AMP_BATCH_SIZE, device=device),
        collect_reference_motions=lambda n: env.unwrapped.collect_reference_motions(n),
    )
    agent.init(trainer_cfg={"timesteps": 1, "headless": True})

    # Rolling out only needs the policy. The discriminator is training-only machinery, so a
    # checkpoint whose discriminator no longer matches the current AMP observation size is still
    # perfectly renderable -- notably every checkpoint from before the arms were dropped from the
    # AMP observation (100 -> 72 dims per frame), which makes agent.load() fail on shape. Fall
    # back to loading just the policy in that case rather than refusing to render.
    import torch as _torch
    try:
        agent.load(args_cli.checkpoint)
        print(f"[INFO] Loaded checkpoint (full agent): {args_cli.checkpoint}")
    except (RuntimeError, ValueError, KeyError) as exc:
        checkpoint = _torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "policy" not in checkpoint:
            raise
        agent.models["policy"].load_state_dict(checkpoint["policy"])
        print(
            f"[INFO] Loaded checkpoint (POLICY ONLY): {args_cli.checkpoint}\n"
            f"       Full-agent load failed ({type(exc).__name__}: {exc}).\n"
            f"       This is expected for checkpoints trained before the AMP observation changed;\n"
            f"       the policy is all that is needed to roll out, so the video is still valid."
        )

    agent.set_running_mode("eval")

    _pin_terrain_level()
    obs, _ = env.reset()
    _pin_terrain_level()
    if args_cli.terrain_level is not None:
        print(f"[INFO] terrain pinned to level {args_cli.terrain_level} for all envs")

    for t in range(args_cli.video_length + 5):
        _pin_terrain_level()
        with __import__("torch").no_grad():
            actions = agent.act(obs, timestep=0, timesteps=1)[0]
        obs, reward, terminated, truncated, info = env.step(actions)

    print(f"[INFO] Video written under: {video_folder}")
    try:
        env.close()
    except Exception as e:
        print(f"[INFO] env.close() raised (likely known viewport-teardown issue, not fatal): {e}")
    simulation_app.close()


if __name__ == "__main__":
    main()
