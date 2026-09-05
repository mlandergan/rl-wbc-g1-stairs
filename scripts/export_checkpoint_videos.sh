#!/usr/bin/env bash
# Batch-export a short rollout video for every saved checkpoint in a training run, using the
# small-scale _PLAY cfg (num_envs=8, matches this project's export_video.sh pattern but scoped
# down further since this runs once per checkpoint, not once total). Run AFTER training
# completes (or is stopped), not concurrently with it -- concurrent training + video export at
# num_envs=4096 + --enable_cameras is what OOM-killed the training run on 2026-08-12 (see
# real_training.log); this avoids that entirely by never running both at once.
#
# G1StairsEnvCfg_PLAY has debug_vis_goal=True, so every exported video also shows the green
# goal-position marker automatically -- no extra flag needed.
#
# Usage (on the VM, from repo root):
#   ./scripts/export_checkpoint_videos.sh [RUN_DIR]
# RUN_DIR defaults to the most recently modified logs/skrl/g1_stairs/*_ppo_torch directory.
set -euo pipefail

RUN_DIR="${1:-$(find "$(pwd)/logs/skrl/g1_stairs" -maxdepth 1 -type d -name '*_ppo_torch' | sort | tail -1)}"
CKPT_DIR="${RUN_DIR}/checkpoints"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "No checkpoints directory found at ${CKPT_DIR}" >&2
  exit 1
fi

echo "Exporting videos for checkpoints in: ${CKPT_DIR}"
for ckpt in "${CKPT_DIR}"/*.pt; do
  [[ -e "${ckpt}" ]] || continue
  name="$(basename "${ckpt}" .pt)"
  echo "=== ${name} ==="

  # Path as seen INSIDE the container (mounted at /workspace/isaaclab_root/logs)
  rel_path="${ckpt#"$(pwd)"/logs/}"
  container_ckpt="/workspace/isaaclab_root/logs/${rel_path}"

  sudo docker run --rm --gpus all \
    -e ACCEPT_EULA=Y \
    -v "$(pwd)/isaaclab_project:/workspace/rl_wbc_g1/isaaclab_project" \
    -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
    rl-wbc-g1-stairs \
    bash -lc "
      cd /workspace/isaaclab_root
      ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
        --task Isaac-G1-AMP-Stairs-Direct-Play-v0 --num_envs 8 \
        --checkpoint '${container_ckpt}' \
        --headless --video --video_length 200 --enable_cameras
    " > "/tmp/export_${name}.log" 2>&1 \
    && echo "  done -> logs/skrl/g1_stairs/$(basename "${RUN_DIR}")/videos/play/" \
    || echo "  FAILED -- see /tmp/export_${name}.log"
done
