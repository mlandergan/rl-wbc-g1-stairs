#!/usr/bin/env bash
# Deletes the training VM (and its boot disk). DESTRUCTIVE -- make sure
# scripts/sync_results.sh already ran, or you lose checkpoints/logs/videos.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project id}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-g1-stairs-l4}"

echo "About to DELETE instance '${INSTANCE_NAME}' in ${ZONE} (project ${PROJECT_ID})."
echo "This also deletes its boot disk. Results not already synced locally will be lost."
read -r -p "Type the instance name to confirm deletion: " confirm
if [[ "${confirm}" != "${INSTANCE_NAME}" ]]; then
  echo "Confirmation did not match. Aborting, nothing deleted."
  exit 1
fi

gcloud compute instances delete "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --quiet
