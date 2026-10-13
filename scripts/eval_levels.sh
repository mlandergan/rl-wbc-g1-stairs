#!/usr/bin/env bash
# Run scripts/eval_levels.py inside the rl-wbc-g1-stairs image.
#
# Mounts match train_amp_depth.sh: isaaclab_project/ and scripts/ so local edits are
# live (the image's baked copies are stale), and logs/ so checkpoints are readable and
# videos land on the host. eval_policy.sh and export_video.sh mount ONLY logs/, which
# means they silently run the image's old config -- this script deliberately does not.
#
# No sudo: fizzgig is in the docker group, and `sudo docker` would block on a password
# prompt when run non-interactively.
#
#   ./scripts/eval_levels.sh --checkpoint logs/skrl/g1_stairs/<run>/checkpoints/best_agent.pt
#   ./scripts/eval_levels.sh --checkpoint <ckpt> --levels 3,6,9 --depth_pov
#   ./scripts/eval_levels.sh --checkpoint <ckpt> --depth_pov --no_rgb   # no RTX at all
set -euo pipefail

TASK="${TASK:-Isaac-G1-AMP-Stairs-Direct-Play-v0}"

# Host paths under logs/ have to be rewritten to where that volume is mounted inside
# the container, so callers can just paste the path they see in `ls`.
ARGS=()
for a in "$@"; do
  case "$a" in
    logs/*)   ARGS+=("/workspace/isaaclab_root/${a}") ;;
    ./logs/*) ARGS+=("/workspace/isaaclab_root/${a#./}") ;;
    *)        ARGS+=("$a") ;;
  esac
done

mkdir -p "$(pwd)/logs"

docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -v "$(pwd)/isaaclab_project:/workspace/rl_wbc_g1/isaaclab_project" \
  -v "$(pwd)/scripts:/workspace/rl_wbc_g1/scripts" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-stairs \
  bash -lc '
    set -euo pipefail
    cd /workspace/isaaclab_root
    ./isaaclab.sh -p /workspace/rl_wbc_g1/scripts/eval_levels.py --task "'"${TASK}"'" "$@"
  ' _ "${ARGS[@]}"
