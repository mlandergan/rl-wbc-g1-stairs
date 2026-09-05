#!/usr/bin/env bash
# Run on your LOCAL machine (not the VM) to pull logs/checkpoints/videos down before tearing
# down the VM with scripts/gcp_teardown_vm.sh.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project id}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-g1-stairs-l4}"
REMOTE_LOGS_DIR="${REMOTE_LOGS_DIR:-~/rl-wbc-g1-stairs/logs}"
LOCAL_DEST="${LOCAL_DEST:-./logs}"

mkdir -p "${LOCAL_DEST}"

gcloud compute scp --recurse --tunnel-through-iap \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  "${INSTANCE_NAME}:${REMOTE_LOGS_DIR}" "${LOCAL_DEST}/.."

echo "Synced to ${LOCAL_DEST}"
