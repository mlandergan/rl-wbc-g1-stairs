"""Train G1 stairs-climbing with depth-camera perception + WasabiAMP -- a project-owned variant
of Isaac Lab's stock `scripts/reinforcement_learning/skrl/train.py` (pulled directly from
github.com/isaac-sim/IsaacLab, BSD-3-Clause, and adapted here rather than reimplemented from
memory).

Why this script exists instead of using the stock one: skrl's own `Runner` utility (what the
stock script uses to build the agent/models from YAML) resolves agent and model classes from a
**hardcoded whitelist only** (confirmed by reading `Runner._component()`) -- there's no way to
point it at a custom `WasabiAMP` agent or a custom multi-modal (proprio+depth) policy model
through YAML alone. This script keeps everything about env construction / Hydra config loading
identical to the stock script, and only replaces the `Runner(env, agent_cfg)` line with manual
construction, matching skrl's own documented pre-`Runner` tutorial pattern (see
docs/source/examples/isaaclab/torch_cartpole_direct_box_box_ppo.py in skrl's own repo for the
general shape of this pattern, and tests/agents/torch/test_amp.py for the AMP-specific
model/memory/agent construction this script's models/memory/agent section is based on).

The simple, unchanged-shape discriminator is built via skrl's own `deterministic_model`
instantiator utility function (a plain callable, usable outside the YAML/Runner system) rather
than hand-written -- only the policy and value need a custom Python class (models.py's
DepthAmpPolicy/DepthAmpValue), the discriminator is exactly what it already was.

VERSION NOTE (2026-09-05): a first pass at this script was written against skrl's unreleased
GitHub `main` branch (a `WasabiAMP_CFG` dataclass). A VM smoke test failed immediately
(`ImportError: cannot import name 'compute_gae'`), which led to pulling the actually-installed
skrl version (1.4.3) directly off the Docker image and confirming it's a meaningfully older,
dict-config API. This script's agent-config section below was rewritten to build a plain dict
matching skrl 1.4.3's real `AMP_DEFAULT_CONFIG` keys directly (see wasabi_amp.py's module
docstring for the full story). Re-run the smoke test below after any further change here.

Usage (inside the container, matching this project's other scripts):
    ./isaaclab.sh -p /workspace/rl_wbc_g1/scripts/train_amp_depth.py --headless --num_envs 4096
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 stairs-climbing with WasabiAMP + depth camera.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="Isaac-G1-AMP-Stairs-Direct-v0", help="Name of the registered gym task."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the yaml's trainer.timesteps.")
# Ported from Isaac Lab's stock train.py -- OFF by default (default=False, matching stock), since
# this project's own play_amp_depth.py docstring documents actual hangs from running a second,
# separate Isaac Sim/RTX process concurrently with training on this VM's GPU. Recording in-process
# (this flag) uses the same single Isaac Sim instance/CUDA context training already has, so that
# failure mode does not apply here -- but `--enable_cameras` (forced on below when --video is set)
# has its own, currently-unmeasured GPU memory cost from booting the RTX render pipeline, on top
# of whatever headroom num_envs leaves. Verify with nvidia-smi on the first real use before
# assuming it is free at a given num_envs, especially near 4096.
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video, in steps.")
parser.add_argument(
    "--video_interval", type=int, default=2000, help="Env steps between recordings (gym RecordVideo's step_trigger)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Must happen before AppLauncher(args_cli) below -- Kit boots with the RTX render pipeline on or
# not at all; there is no enabling it after the app has already launched.
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time
from datetime import datetime

import gymnasium as gym

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_project.g1_stairs
from isaaclab_tasks.utils.hydra import hydra_task_config

from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import deterministic_model

from isaaclab_project.g1_stairs.agents.wasabi_amp import WasabiAMP
from isaaclab_project.g1_stairs.models import DepthAmpPolicy, DepthAmpValue


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    env_cfg.seed = seed

    a = agent_cfg["agent"]

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", a["experiment"]["directory"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_wasabi_amp_depth"
    if a["experiment"]["experiment_name"]:
        log_dir += f"_{a['experiment']['experiment_name']}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RecordVideo wraps the raw gym env's rgb_array render, which needs its own viewport camera
    # separate from this task's own ray-cast depth sensor (RayCasterCamera via Warp) that feeds
    # the policy -- the two do not conflict; the ray-cast sensor never touches the RTX renderer
    # this flag turns on.
    env = SkrlVecEnvWrapper(env)
    device = env.device

    print_dict({"task": args_cli.task, "num_envs": env.num_envs, "device": device}, nesting=1)

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
        "value": DepthAmpValue(observation_space=env.observation_space, action_space=env.action_space, device=device),
        "discriminator": deterministic_model(
            observation_space=env.amp_observation_space,
            action_space=env.action_space,
            device=device,
            network=[{"name": "net", "input": "OBSERVATIONS", "layers": [256, 128, 128], "activations": "elu"}],
            output="ONE",
        ),
    }

    memory = RandomMemory(memory_size=a["rollouts"], num_envs=env.num_envs, device=device)
    motion_dataset = RandomMemory(memory_size=agent_cfg["motion_dataset"]["memory_size"], device=device)
    reply_buffer = RandomMemory(memory_size=agent_cfg["reply_buffer"]["memory_size"], device=device)

    cfg = {
        "rollouts": a["rollouts"],
        "learning_epochs": a["learning_epochs"],
        "mini_batches": a["mini_batches"],
        "discount_factor": a["discount_factor"],
        "lambda": a["lambda"],
        "learning_rate": a["learning_rate"],
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": a["learning_rate_scheduler_kwargs"],
        "state_preprocessor": None,
        "value_preprocessor": None,
        "amp_state_preprocessor": RunningStandardScaler,
        "amp_state_preprocessor_kwargs": {"size": env.amp_observation_space, "device": device},
        "random_timesteps": a["random_timesteps"],
        "learning_starts": a["learning_starts"],
        "grad_norm_clip": a["grad_norm_clip"],
        "ratio_clip": a["ratio_clip"],
        "value_clip": a["value_clip"],
        "clip_predicted_values": a["clip_predicted_values"],
        "entropy_loss_scale": a["entropy_loss_scale"],
        "value_loss_scale": a["value_loss_scale"],
        "discriminator_loss_scale": a["discriminator_loss_scale"],
        "amp_batch_size": a["amp_batch_size"],
        "task_reward_weight": a["task_reward_weight"],
        "style_reward_weight": a["style_reward_weight"],
        "discriminator_batch_size": a["discriminator_batch_size"],
        "discriminator_reward_scale": a["discriminator_reward_scale"],
        "discriminator_logit_regularization_scale": a["discriminator_logit_regularization_scale"],
        "discriminator_gradient_penalty_scale": a["discriminator_gradient_penalty_scale"],
        "discriminator_weight_decay_scale": a["discriminator_weight_decay_scale"],
        "discriminator_gradient_tolerance": a["discriminator_gradient_tolerance"],
        "time_limit_bootstrap": a["time_limit_bootstrap"],
        "experiment": {
            "directory": log_root_path,
            "experiment_name": os.path.basename(log_dir),
            "write_interval": a["experiment"]["write_interval"],
            "checkpoint_interval": a["experiment"]["checkpoint_interval"],
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        },
    }

    agent = WasabiAMP(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        amp_observation_space=env.amp_observation_space,
        motion_dataset=motion_dataset,
        reply_buffer=reply_buffer,
        collect_reference_motions=lambda num_samples: env.unwrapped.collect_reference_motions(num_samples),
    )

    if args_cli.checkpoint:
        print(f"[INFO] Loading model checkpoint from: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)

    timesteps = args_cli.max_iterations * a["rollouts"] if args_cli.max_iterations else agent_cfg["trainer"]["timesteps"]
    trainer = SequentialTrainer(
        cfg={
            "timesteps": timesteps,
            "headless": args_cli.headless,
            "close_environment_at_exit": False,
            "environment_info": "log",
        },
        env=env,
        agents=agent,
    )

    start_time = time.time()
    trainer.train()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
