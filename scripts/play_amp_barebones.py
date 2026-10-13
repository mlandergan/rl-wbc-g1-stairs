"""Barebones script to load a WasabiAMP checkpoint and visualize it interactively."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Barebones visualization of WasabiAMP checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-G1-AMP-Stairs-Direct-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Ensure camera rendering is active for visualization
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_project.g1_stairs
from isaaclab_tasks.utils import parse_env_cfg

from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import deterministic_model

from isaaclab_project.g1_stairs.agents.wasabi_amp import WasabiAMP
from isaaclab_project.g1_stairs.models import DepthAmpPolicy, DepthAmpValue
from isaaclab_rl.skrl import SkrlVecEnvWrapper


def main():
    # 1. Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Create raw environment with rgb_array / interactive rendering
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    device = env.unwrapped.device

    # 2. Reconstruct models matching training architecture
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

    # 3. Initialize Agent and load checkpoint
    agent = WasabiAMP(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space, action_space=env.action_space, device=device,
        amp_observation_space=env.amp_observation_space,
        motion_dataset=RandomMemory(memory_size=_PLAY_AMP_BATCH_SIZE, device=device),
        reply_buffer=RandomMemory(memory_size=_PLAY_AMP_BATCH_SIZE, device=device),
        collect_reference_motions=lambda n: env.unwrapped.collect_reference_motions(n),
    )
    agent.init(trainer_cfg={"timesteps": 1, "headless": False})
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval")
    print(f"[INFO] Successfully loaded checkpoint: {args_cli.checkpoint}")

    # 4. Simulation loop
    obs, _ = env.reset()
    print("[INFO] Starting simulation loop. Press Ctrl+C in terminal to exit.")

    while simulation_app.is_running():
        with torch.no_grad():
            actions = agent.act(obs, timestep=0, timesteps=1)[0]
        obs, reward, terminated, truncated, info = env.step(actions)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()