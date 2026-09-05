#!/usr/bin/env bash
# Launch AMP training (skrl) inside the rl-wbc-g1-stairs Docker image. Run on the GCP VM after
# scripts/remote_setup.sh has built `rl-wbc-g1-stairs`. Mounts ./logs from the current
# directory so checkpoints/TensorBoard logs survive container exit -- that's what
# scripts/sync_results.sh pulls back to your local machine.
#
# TASK options:
#   Isaac-G1-AMP-Stairs-Direct-v0   (this project's task: velocity tracking + AMP style reward
#                                    for stair climbing. Still bootstrapped against rl-wbc-g1-amp's
#                                    Mixamo Strut Walking clip -- stairs-specific reference motion
#                                    is a separate open task, see project_description.md)
set -euo pipefail

TASK="${TASK:-Isaac-G1-AMP-Stairs-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"

mkdir -p "$(pwd)/logs"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -e TASK="${TASK}" -e NUM_ENVS="${NUM_ENVS}" \
  -v "$(pwd)/isaaclab_project:/workspace/rl_wbc_g1/isaaclab_project" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-stairs \
  bash -lc '
    set -euo pipefail
    cd /workspace/isaaclab_root
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
      --task "${TASK}" --headless --num_envs "${NUM_ENVS}"
  '
