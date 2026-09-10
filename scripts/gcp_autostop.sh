#!/usr/bin/env bash
# Deterministic, unattended VM shutoff -- a hard cost backstop that does not depend on this
# session, Claude, or any monitoring loop staying alive. Run this and log off; it fires on its
# own via a detached local background process (not on the VM itself, and not a guest-OS
# `shutdown`, which GCE auto-restarts from by default unless automatic-restart is explicitly
# disabled -- that would silently keep billing you). This calls the Compute Engine API's own
# `stop`, which is a real STOP with no auto-restart.
#
# This project's own history is exactly why this exists: an unnoticed 12.5-day VM run was ~half
# of a $103.73 August bill (see project memory / feedback_gcp_vm_cost_discipline).
#
# All config is substituted into the backgrounded command as literal values, not passed through
# variable inheritance -- `nohup bash -c '...'` runs in a FRESH shell that does not see this
# script's un-exported variables, which silently broke an earlier version of this script (every
# path resolved empty; caught by checking the log immediately after arming, before trusting it
# and logging off -- verify, don't assume, for exactly this reason).
#
# Usage:
#   ./scripts/gcp_autostop.sh                       # stop g1-stairs-l4 in 30 min
#   DELAY_MINUTES=10 ./scripts/gcp_autostop.sh       # override the delay
#   INSTANCE_NAME=foo ZONE=us-central1-a ./scripts/gcp_autostop.sh
#
# Verify it's alive at any point:
#   tail -f /tmp/gcp_autostop.log
# Cancel before it fires:
#   kill $(cat /tmp/gcp_autostop.pid)
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-g1-rl-training}"
ZONE="${ZONE:-us-east1-b}"
INSTANCE_NAME="${INSTANCE_NAME:-g1-stairs-l4}"
DELAY_MINUTES="${DELAY_MINUTES:-30}"
LOG_FILE="${LOG_FILE:-/tmp/gcp_autostop.log}"
PID_FILE="${PID_FILE:-/tmp/gcp_autostop.pid}"
DELAY_SECONDS=$(( DELAY_MINUTES * 60 ))
FIRE_AT="$(date -v+"${DELAY_MINUTES}"M '+%Y-%m-%d %H:%M:%S')"

CMD="
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] armed: will stop ${INSTANCE_NAME} (${ZONE}) at ~${FIRE_AT}\" >> '${LOG_FILE}'
sleep ${DELAY_SECONDS}
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] firing: issuing stop for ${INSTANCE_NAME}\" >> '${LOG_FILE}'
if gcloud compute instances stop '${INSTANCE_NAME}' --zone='${ZONE}' --project='${PROJECT_ID}' >> '${LOG_FILE}' 2>&1; then
  echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] stop command accepted\" >> '${LOG_FILE}'
else
  echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] stop command FAILED -- check auth / instance name\" >> '${LOG_FILE}'
fi
status=\$(gcloud compute instances describe '${INSTANCE_NAME}' --zone='${ZONE}' --project='${PROJECT_ID}' --format='value(status)' 2>>'${LOG_FILE}' || echo UNKNOWN)
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] confirmed status: \${status}\" >> '${LOG_FILE}'
rm -f '${PID_FILE}'
"

: > "${LOG_FILE}"
nohup bash -c "${CMD}" < /dev/null >> "${LOG_FILE}" 2>&1 &
disown
echo $! > "${PID_FILE}"

sleep 1  # let the child write its first log line before we report success
echo "Armed. PID $(cat "${PID_FILE}") will stop ${INSTANCE_NAME} at ~${FIRE_AT} (in ${DELAY_MINUTES} min)."
echo "Log:    ${LOG_FILE}"
echo "Cancel: kill \$(cat ${PID_FILE})"
echo
echo "--- verifying it actually armed (not just launched) ---"
if grep -q "^\[.*armed:" "${LOG_FILE}"; then
  cat "${LOG_FILE}"
else
  echo "WARNING: no 'armed' line in the log yet -- do not trust this run. Log contents:"
  cat "${LOG_FILE}"
  exit 1
fi
