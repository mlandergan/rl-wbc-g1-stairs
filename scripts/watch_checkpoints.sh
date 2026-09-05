#!/usr/bin/env bash
# Poll a training run's checkpoints/ directory and export a short video for each NEW checkpoint
# as it appears, instead of waiting for the whole run to finish (export_checkpoint_videos.sh's
# batch, post-hoc approach). Meant to run CONCURRENTLY with train_amp.sh, on the VM.
#
# KNOWN RISK (2026-08-13): this project previously observed a genuine hang (not just slowness)
# running a small num_envs=8 video-export job concurrently with a full num_envs=4096 training job
# on this GPU -- root cause never isolated (could be VRAM contention, could be two simultaneous
# Vulkan/RTX renderer contexts not coexisting well). This run trains at a reduced num_envs to leave
# headroom, and every export attempt below is wrapped in `timeout` so a hang costs only that one
# checkpoint's video (skipped, logged, move on) rather than blocking this watcher indefinitely or
# risking the training container. If every attempt times out, fall back to
# export_checkpoint_videos.sh's post-hoc batch export after training completes.
#
# Usage (on the VM, from repo root, typically via nohup so it survives SSH disconnect):
#   nohup ./scripts/watch_checkpoints.sh [RUN_DIR] > /tmp/watch_checkpoints.log 2>&1 &
# RUN_DIR defaults to the most recently modified logs/skrl/g1_stairs/*_ppo_torch directory --
# resolved ONCE at startup (not re-resolved per loop iteration), since a fresh training run
# creates its RUN_DIR near-immediately and we want to watch that one, not whatever's newest later.
set -uo pipefail  # no -e: a single failed/timed-out export must not kill the watch loop

POLL_INTERVAL_S="${POLL_INTERVAL_S:-30}"
EXPORT_TIMEOUT_S="${EXPORT_TIMEOUT_S:-240}"
NUM_ENVS="${NUM_ENVS:-8}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

RUN_DIR="${1:-$(find "$(pwd)/logs/skrl/g1_stairs" -maxdepth 1 -type d -name '*_ppo_torch' | sort | tail -1)}"
CKPT_DIR="${RUN_DIR}/checkpoints"
echo "[watch_checkpoints] watching ${CKPT_DIR} (poll every ${POLL_INTERVAL_S}s, export timeout ${EXPORT_TIMEOUT_S}s, num_envs=${NUM_ENVS})"

mkdir -p "${CKPT_DIR}"  # in case training hasn't written its first checkpoint yet
declare -A seen

export_one() {
  local ckpt="$1"
  local name
  name="$(basename "${ckpt}" .pt)"
  local rel_path="${ckpt#"$(pwd)"/logs/}"
  local container_ckpt="/workspace/isaaclab_root/logs/${rel_path}"

  echo "[watch_checkpoints] $(date -u +%H:%M:%S) exporting ${name}..."
  timeout "${EXPORT_TIMEOUT_S}" sudo docker run --rm --gpus all \
    -e ACCEPT_EULA=Y \
    -v "$(pwd)/isaaclab_project:/workspace/rl_wbc_g1/isaaclab_project" \
    -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
    rl-wbc-g1-stairs \
    bash -lc "
      cd /workspace/isaaclab_root
      ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
        --task Isaac-G1-AMP-Stairs-Direct-Play-v0 --num_envs ${NUM_ENVS} \
        --checkpoint '${container_ckpt}' \
        --headless --video --video_length ${VIDEO_LENGTH} --enable_cameras
    " > "/tmp/watch_export_${name}.log" 2>&1
  local status=$?
  if [[ ${status} -eq 124 ]]; then
    echo "[watch_checkpoints] $(date -u +%H:%M:%S) ${name} TIMED OUT after ${EXPORT_TIMEOUT_S}s -- skipped, see /tmp/watch_export_${name}.log"
  elif [[ ${status} -ne 0 ]]; then
    echo "[watch_checkpoints] $(date -u +%H:%M:%S) ${name} FAILED (exit ${status}) -- see /tmp/watch_export_${name}.log"
  else
    echo "[watch_checkpoints] $(date -u +%H:%M:%S) ${name} done -> logs/skrl/g1_stairs/$(basename "${RUN_DIR}")/videos/play/"
  fi
}

while true; do
  for ckpt in "${CKPT_DIR}"/*.pt; do
    [[ -e "${ckpt}" ]] || continue
    if [[ -z "${seen[${ckpt}]+x}" ]]; then
      seen["${ckpt}"]=1
      export_one "${ckpt}"
    fi
  done
  sleep "${POLL_INTERVAL_S}"
done
