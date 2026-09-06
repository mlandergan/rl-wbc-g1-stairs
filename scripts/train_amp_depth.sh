#!/usr/bin/env bash
# Launch WasabiAMP + depth-camera training inside the rl-wbc-g1-stairs Docker image -- the
# sibling of train_amp.sh that uses scripts/train_amp_depth.py (a project-owned training entry
# point) instead of Isaac Lab's stock scripts/reinforcement_learning/skrl/train.py, since that
# stock script can't construct a custom agent/model class (see train_amp_depth.py's own
# docstring for why).
#
# Run on the GCP VM after scripts/remote_setup.sh has built `rl-wbc-g1-stairs`. Mounts ./logs
# so checkpoints/TensorBoard logs survive container exit (scripts/sync_results.sh pulls them
# back), and mounts ./scripts so train_amp_depth.py itself is visible inside the container --
# unlike isaaclab_project/, scripts/ isn't baked into the image at build time.
set -euo pipefail

TASK="${TASK:-Isaac-G1-AMP-Stairs-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"

mkdir -p "$(pwd)/logs"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e TASK="${TASK}" -e NUM_ENVS="${NUM_ENVS}" \
  -v "$(pwd)/isaaclab_project:/workspace/rl_wbc_g1/isaaclab_project" \
  -v "$(pwd)/scripts:/workspace/rl_wbc_g1/scripts" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-stairs \
  bash -lc '
    set -euo pipefail
    cd /workspace/isaaclab_root
    ./isaaclab.sh -p /workspace/rl_wbc_g1/scripts/train_amp_depth.py \
      --task "${TASK}" --headless --num_envs "${NUM_ENVS}"
  '
