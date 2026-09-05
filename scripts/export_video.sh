#!/usr/bin/env bash
# Same as eval_policy.sh but with Isaac Lab's built-in video recording flags. Output mp4 lands
# under logs/skrl/<experiment_name>/<run>/videos/play/ (relative to the mounted ./logs volume).
# Uses skrl's play.py -- see eval_policy.sh's header for why (this project has no rsl_rl task).
set -euo pipefail

TASK="${TASK:-Isaac-G1-AMP-Stairs-Direct-Play-v0}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT, e.g. logs/skrl/g1_stairs/<run>/checkpoints/best_agent.pt}"
NUM_ENVS="${NUM_ENVS:-32}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e TASK="${TASK}" -e CHECKPOINT="${CHECKPOINT}" -e NUM_ENVS="${NUM_ENVS}" -e VIDEO_LENGTH="${VIDEO_LENGTH}" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-stairs \
  bash -lc '
    set -euo pipefail
    cd /workspace/isaaclab_root
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
      --task "${TASK}" --num_envs "${NUM_ENVS}" --checkpoint "${CHECKPOINT}" \
      --headless --video --video_length "${VIDEO_LENGTH}" --enable_cameras
  '
