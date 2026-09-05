#!/usr/bin/env bash
# Roll out a trained checkpoint with a small, non-randomized env (the `_PLAY` config variant).
# Uses skrl's play.py, not rsl_rl's -- this project's only registered task (g1_stairs) trains
# via skrl (see train_amp.sh), and rsl_rl doesn't have a hook for it (see
# docker/patch_register_task.py). CHECKPOINT is a path *inside the container*, i.e. under
# /workspace/isaaclab_root/logs/... -- run `ls logs/skrl/<experiment_name>/` on the host first
# to find one (it's the same path, just relative to wherever the ./logs volume is mounted from).
set -euo pipefail

TASK="${TASK:-Isaac-G1-AMP-Stairs-Direct-Play-v0}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a checkpoint path, e.g. logs/skrl/g1_stairs/<run>/checkpoints/best_agent.pt}"
NUM_ENVS="${NUM_ENVS:-32}"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e TASK="${TASK}" -e CHECKPOINT="${CHECKPOINT}" -e NUM_ENVS="${NUM_ENVS}" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-stairs \
  bash -lc '
    set -euo pipefail
    cd /workspace/isaaclab_root
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
      --task "${TASK}" --num_envs "${NUM_ENVS}" --checkpoint "${CHECKPOINT}" --headless
  '
