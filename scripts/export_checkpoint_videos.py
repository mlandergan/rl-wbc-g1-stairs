"""Headless batch video exporter for WasabiAMP checkpoints.

Uses Isaac Lab's native environment rendering and avoids the legacy
omni.isaac.core API.

Important:
    This exporter runs with num_envs=1 and headless=True.

    The PLAY config should NOT configure the interactive viewport viewer
    (viewer.asset_name / viewer.origin_type) for this script. Those settings
    can activate Isaac Lab's viewport camera controller, which expects an
    env.scene attribute that this custom DirectRLEnv does not expose.
"""

import argparse
import glob
import os
import sys
import traceback

from isaaclab.app import AppLauncher


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Batch export checkpoint videos headlessly."
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help=(
        "Directory containing checkpoint .pt files "
        "(or parent run directory), or a single .pt file."
    ),
)

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-G1-AMP-Stairs-Direct-Play-v0",
)

parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Number of simulation steps per rollout video.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Pass only Hydra arguments onward.
sys.argv = [sys.argv[0]] + hydra_args

# We explicitly want headless rendering.
args_cli.headless = True
args_cli.enable_cameras = True


# =============================================================================
# Launch Isaac Sim
# =============================================================================

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

import isaaclab_project.g1_stairs

from isaaclab_project.g1_stairs.agents.wasabi_amp import WasabiAMP
from isaaclab_project.g1_stairs.models import (
    DepthAmpPolicy,
    DepthAmpValue,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper


# =============================================================================
# Helpers
# =============================================================================

def find_checkpoints(checkpoint_dir):
    """Resolve a checkpoint file or directory into a sorted checkpoint list."""

    if os.path.isfile(checkpoint_dir):
        return [checkpoint_dir]

    if not os.path.isdir(checkpoint_dir):
        return []

    # Standard Isaac Lab / skrl layout.
    checkpoints = sorted(
        glob.glob(
            os.path.join(
                checkpoint_dir,
                "checkpoints",
                "*.pt",
            )
        )
    )

    # Also support .pt files directly inside the supplied directory.
    if not checkpoints:
        checkpoints = sorted(
            glob.glob(
                os.path.join(
                    checkpoint_dir,
                    "*.pt",
                )
            )
        )

    return checkpoints


def get_video_directory(ckpt_path):
    """Return the exported_videos directory associated with a checkpoint."""

    normalized = os.path.normpath(ckpt_path)

    parent = os.path.dirname(normalized)

    # Typical layout:
    #
    # run/
    #   checkpoints/
    #       agent_1000.pt
    #
    # -> run/exported_videos/
    #
    if os.path.basename(parent) == "checkpoints":
        run_parent_dir = os.path.dirname(parent)
    else:
        run_parent_dir = parent

    video_folder = os.path.join(
        run_parent_dir,
        "exported_videos",
    )

    os.makedirs(
        video_folder,
        exist_ok=True,
    )

    return video_folder


def release_video_writer(video_writer):
    """Safely release an OpenCV video writer."""

    if video_writer is not None:
        try:
            video_writer.release()
        except Exception:
            pass


# =============================================================================
# Main
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Resolve checkpoints
    # -------------------------------------------------------------------------

    checkpoints = find_checkpoints(
        args_cli.checkpoint_dir
    )

    checkpoints = [
    "/workspace/isaaclab_root/logs/skrl/g1_stairs/2026-09-11_01-39-28_wasabi_amp_depth/checkpoints/agent_103200.pt",
    "/workspace/isaaclab_root/logs/skrl/g1_stairs/2026-09-11_01-39-28_wasabi_amp_depth/checkpoints/agent_103200.pt"
    ]

    if not checkpoints:
        print(
            "[ERROR] No checkpoints found in "
            f"{args_cli.checkpoint_dir}"
        )

        simulation_app.close()
        return

    print(
        f"[INFO] Found {len(checkpoints)} checkpoint(s) to process."
    )

    # -------------------------------------------------------------------------
    # Process checkpoints one at a time
    # -------------------------------------------------------------------------

    for ckpt_index, ckpt_path in enumerate(checkpoints):

        ckpt_name = os.path.splitext(
            os.path.basename(ckpt_path)
        )[0]

        print()
        print("=" * 80)
        print(
            f"[INFO] Processing checkpoint "
            f"{ckpt_index + 1}/{len(checkpoints)}"
        )
        print(
            f"[INFO] Checkpoint: {ckpt_name}"
        )
        print("=" * 80)

        env = None
        agent = None
        video_writer = None

        try:

            # =================================================================
            # 1. Environment configuration
            # =================================================================

            print("[INFO] Creating environment...")

            env_cfg = parse_env_cfg(
                args_cli.task,
                device=args_cli.device,
                num_envs=1,
            )

            # -----------------------------------------------------------------
            # Disable interactive viewer behavior.
            #
            # This exporter is headless, so we don't need the interactive
            # viewer following an asset.
            # -----------------------------------------------------------------

            if hasattr(env_cfg, "viewer"):

                try:
                    env_cfg.viewer.origin_type = "world"
                except Exception:
                    pass

                try:
                    env_cfg.viewer.asset_name = None
                except Exception:
                    pass

            # -----------------------------------------------------------------
            # Create environment
            # -----------------------------------------------------------------

            print("[INFO] Calling gym.make()...")

            env = gym.make(
                args_cli.task,
                cfg=env_cfg,
                render_mode="rgb_array",
            )

            print("[INFO] Gym environment created.")

            # -----------------------------------------------------------------
            # Wrap for skrl
            # -----------------------------------------------------------------

            env = SkrlVecEnvWrapper(env)

            device = env.unwrapped.device

            print(
                f"[INFO] Environment device: {device}"
            )

            # =================================================================
            # 2. Reconstruct policy/value/discriminator
            # =================================================================

            print("[INFO] Constructing WasabiAMP models...")

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
                    network=[
                        {
                            "name": "net",
                            "input": "OBSERVATIONS",
                            "layers": [256, 128, 128],
                            "activations": "elu",
                        }
                    ],
                    output="ONE",
                ),
            }

            # -----------------------------------------------------------------
            # Memory
            # -----------------------------------------------------------------

            play_amp_batch_size = 8

            memory = RandomMemory(
                memory_size=1,
                num_envs=env.num_envs,
                device=device,
            )

            # -----------------------------------------------------------------
            # AMP configuration
            # -----------------------------------------------------------------

            cfg = {
                "amp_batch_size": play_amp_batch_size,

                "amp_state_preprocessor": RunningStandardScaler,

                "amp_state_preprocessor_kwargs": {
                    "size": env.amp_observation_space,
                    "device": device,
                },

                "experiment": {
                    "directory": "",
                    "experiment_name": "",
                    "write_interval": 0,
                    "checkpoint_interval": 0,
                    "store_separately": False,
                    "wandb": False,
                    "wandb_kwargs": {},
                },
            }

            # -----------------------------------------------------------------
            # Agent
            # -----------------------------------------------------------------

            agent = WasabiAMP(
                models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                amp_observation_space=env.amp_observation_space,

                motion_dataset=RandomMemory(
                    memory_size=play_amp_batch_size,
                    device=device,
                ),

                reply_buffer=RandomMemory(
                    memory_size=play_amp_batch_size,
                    device=device,
                ),

                collect_reference_motions=lambda n:
                    env.unwrapped.collect_reference_motions(n),
            )

            agent.init(
                trainer_cfg={
                    "timesteps": 1,
                    "headless": True,
                }
            )

            # =================================================================
            # 3. Load checkpoint
            # =================================================================

            print(
                f"[INFO] Loading checkpoint: {ckpt_path}"
            )

            agent.load(
                ckpt_path
            )

            agent.set_running_mode(
                "eval"
            )

            print(
                "[INFO] Checkpoint loaded successfully."
            )

            # =================================================================
            # 4. Video output
            # =================================================================

            video_folder = get_video_directory(
                ckpt_path
            )

            video_path = os.path.join(
                video_folder,
                f"vid_{ckpt_name}.mp4",
            )

            print(
                f"[INFO] Output video: {video_path}"
            )

            # Simulation:
            #
            #     dt = 0.005
            #     decimation = 4
            #
            # Therefore the environment step is:
            #
            #     0.005 * 4 = 0.020 s
            #
            # -> 50 environment frames / second.
            #
            fps = 50.0

            # =================================================================
            # 5. Reset
            # =================================================================

            print("[INFO] Resetting environment...")

            obs, _ = env.reset()

            print(
                "[INFO] Starting rollout for "
                f"{args_cli.video_length} steps."
            )

            # =================================================================
            # 6. Rollout
            # =================================================================

            frames_written = 0

            for t in range(args_cli.video_length):

                # -------------------------------------------------------------
                # Policy inference
                # -------------------------------------------------------------

                with torch.no_grad():

                    actions = agent.act(
                        obs,
                        timestep=0,
                        timesteps=1,
                    )[0]

                # -------------------------------------------------------------
                # Environment step
                # -------------------------------------------------------------

                obs, reward, terminated, truncated, info = env.step(
                    actions
                )

                # -------------------------------------------------------------
                # Render
                # -------------------------------------------------------------

                frame = env.render()

                if frame is None:
                    continue

                if not isinstance(frame, np.ndarray):

                    print(
                        "[WARNING] env.render() returned "
                        f"{type(frame)} instead of numpy array."
                    )

                    continue

                # -------------------------------------------------------------
                # Normalize frame shape
                # -------------------------------------------------------------

                if frame.ndim != 3:

                    print(
                        "[WARNING] Unexpected frame shape: "
                        f"{frame.shape}"
                    )

                    continue

                # RGB -> BGR for OpenCV.
                if frame.shape[-1] == 3:

                    frame_bgr = cv2.cvtColor(
                        frame,
                        cv2.COLOR_RGB2BGR,
                    )

                elif frame.shape[-1] == 4:

                    # RGBA -> BGR.
                    frame_bgr = cv2.cvtColor(
                        frame,
                        cv2.COLOR_RGBA2BGR,
                    )

                else:

                    print(
                        "[WARNING] Unsupported frame channel count: "
                        f"{frame.shape[-1]}"
                    )

                    continue

                # -------------------------------------------------------------
                # Initialize video writer
                # -------------------------------------------------------------

                if video_writer is None:

                    height, width = frame_bgr.shape[:2]

                    print(
                        "[INFO] Video resolution: "
                        f"{width}x{height}"
                    )

                    fourcc = cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )

                    video_writer = cv2.VideoWriter(
                        video_path,
                        fourcc,
                        fps,
                        (width, height),
                    )

                    if not video_writer.isOpened():

                        raise RuntimeError(
                            "OpenCV could not open video writer: "
                            f"{video_path}"
                        )

                # -------------------------------------------------------------
                # Write frame
                # -------------------------------------------------------------

                video_writer.write(
                    frame_bgr
                )

                frames_written += 1

                # -------------------------------------------------------------
                # Progress
                # -------------------------------------------------------------

                if (t + 1) % 50 == 0:

                    print(
                        f"[INFO] Step {t + 1}/"
                        f"{args_cli.video_length} "
                        f"| frames={frames_written}"
                    )

                # -------------------------------------------------------------
                # Stop on episode termination
                # -------------------------------------------------------------

                if np.any(terminated) or np.any(truncated):

                    print(
                        "[INFO] Episode terminated/truncated "
                        f"at step {t + 1}."
                    )

                    break

            # =================================================================
            # 7. Finalize video
            # =================================================================

            if video_writer is not None:

                video_writer.release()
                video_writer = None

            if frames_written > 0:

                print()
                print(
                    "[INFO] Video saved successfully:"
                )
                print(
                    f"       {video_path}"
                )
                print(
                    f"       frames: {frames_written}"
                )
                print(
                    f"       duration: "
                    f"{frames_written / fps:.2f} s"
                )

            else:

                print()
                print(
                    "[WARNING] No frames were captured "
                    f"for {ckpt_name}."
                )

        except Exception as exc:

            # -----------------------------------------------------------------
            # IMPORTANT:
            #
            # The previous version used:
            #
            #     except Exception:
            #
            # and then attempted to reference `exc` below.
            #
            # This caused:
            #
            #     UnboundLocalError:
            #     cannot access local variable 'exc'
            #
            # Catch the exception as `exc` so we can report the real error.
            # -----------------------------------------------------------------

            print()
            print("=" * 80)
            print(
                f"[ERROR] Failed checkpoint: {ckpt_name}"
            )
            print(
                f"[ERROR] {type(exc).__name__}: {exc}"
            )
            print("=" * 80)

            traceback.print_exc()

            release_video_writer(
                video_writer
            )

            video_writer = None

        finally:

            # =================================================================
            # 8. Environment cleanup
            # =================================================================

            if env is not None:

                try:
                    env.close()

                except Exception as exc:

                    print(
                        "[WARNING] Error while closing "
                        f"environment: "
                        f"{type(exc).__name__}: {exc}"
                    )

            # -----------------------------------------------------------------
            # Release references
            # -----------------------------------------------------------------

            agent = None
            env = None

            # -----------------------------------------------------------------
            # CUDA cleanup
            # -----------------------------------------------------------------

            if torch.cuda.is_available():

                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # -----------------------------------------------------------------
            # IMPORTANT:
            #
            # Do NOT call clear_stage() here.
            #
            # The viewport/camera subsystem may still have callbacks registered
            # against USD objects. Clearing the stage between environments can
            # cause callbacks to fire after their owning environment has already
            # been destroyed.
            # -----------------------------------------------------------------

            print(
                f"[INFO] Finished checkpoint: {ckpt_name}"
            )

    # =========================================================================
    # 9. Shutdown Isaac Sim
    # =========================================================================

    print()
    print("=" * 80)
    print("[INFO] Batch video export complete!")
    print("=" * 80)

    simulation_app.close()


if __name__ == "__main__":
    main()
