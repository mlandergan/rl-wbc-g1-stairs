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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    device = env.unwrapped.device

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
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval")
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    obs, _ = env.reset()
    for t in range(args_cli.video_length + 5):
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
