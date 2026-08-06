#!/bin/bash

set -euo pipefail

: "${SLURM_JOB_ID:?Campaign tasks must run inside a Slurm allocation}"
: "${SLURMD_NODENAME:?Campaign tasks must run on a Slurm compute node}"
: "${MODSSC_ROOT:?MODSSC_ROOT is required}"
: "${MODSSC_CAMPAIGN_MANIFEST:?MODSSC_CAMPAIGN_MANIFEST is required}"
: "${MODSSC_CAMPAIGN_META:?MODSSC_CAMPAIGN_META is required}"
: "${MODSSC_ARRAY_INDEX_FILE:?MODSSC_ARRAY_INDEX_FILE is required}"
: "${MODSSC_ARRAY_INDEX_SHA256:?MODSSC_ARRAY_INDEX_SHA256 is required}"
: "${MODSSC_CAMPAIGN_MANIFEST_SHA256:?MODSSC_CAMPAIGN_MANIFEST_SHA256 is required}"
: "${MODSSC_CAMPAIGN_RESULT_ROOT:?MODSSC_CAMPAIGN_RESULT_ROOT is required}"
: "${MODSSC_CAMPAIGN_SITE_ID:?MODSSC_CAMPAIGN_SITE_ID is required}"
: "${MODSSC_PREFLIGHT_REPORT:?MODSSC_PREFLIGHT_REPORT is required}"
: "${MODSSC_PYTHON:?MODSSC_PYTHON is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

source "$MODSSC_ROOT/tools/hpc/slurm/runtime-context.sh"

if [[ ! -x "$MODSSC_PYTHON" ]]; then
  echo "MODSSC_PYTHON is not executable: $MODSSC_PYTHON" >&2
  exit 66
fi

if [[ "$(hostname -s)" != "$SLURMD_NODENAME" ]]; then
  echo "Campaign tasks refuse login-node execution" >&2
  exit 64
fi

if [[ -n "${MODSSC_ENVIRONMENT_LOCK_SHA256:-}" \
  && "$MODSSC_ENVIRONMENT_LOCK_SHA256" != "unlocked" \
  && -z "${MODSSC_ENVIRONMENT_MANIFEST:-}" ]]; then
  echo "Pinned Slurm execution requires MODSSC_ENVIRONMENT_MANIFEST" >&2
  exit 64
fi

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    echo "No SHA-256 utility is available" >&2
    return 69
  fi
}

for EXPECTED_DIGEST in "$MODSSC_ARRAY_INDEX_SHA256" "$MODSSC_CAMPAIGN_MANIFEST_SHA256"; do
  if [[ "${#EXPECTED_DIGEST}" -ne 64 || "$EXPECTED_DIGEST" == *[!0-9a-f]* ]]; then
    echo "Invalid expected SHA-256 digest in generated wrapper" >&2
    exit 64
  fi
done

ACTUAL_MANIFEST_SHA256="$(sha256_file "$MODSSC_CAMPAIGN_MANIFEST")"
if [[ "$ACTUAL_MANIFEST_SHA256" != "$MODSSC_CAMPAIGN_MANIFEST_SHA256" ]]; then
  echo "Campaign manifest SHA-256 mismatch" >&2
  exit 65
fi

ACTUAL_INDEX_SHA256="$(sha256_file "$MODSSC_ARRAY_INDEX_FILE")"
if [[ "$ACTUAL_INDEX_SHA256" != "$MODSSC_ARRAY_INDEX_SHA256" ]]; then
  echo "Array index SHA-256 mismatch" >&2
  exit 65
fi

GLOBAL_INDEX="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MODSSC_ARRAY_INDEX_FILE")"
if [[ ! "$GLOBAL_INDEX" =~ ^[0-9]+$ ]]; then
  echo "Invalid or missing global task index for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 64
fi

TASK_WORK_BASE="${JOBSCRATCH:-${TMPDIR:?TMPDIR or JOBSCRATCH is required}}"
WORK_ROOT="$TASK_WORK_BASE/modssc-campaign/${MODSSC_CAMPAIGN_ID:?MODSSC_CAMPAIGN_ID is required}"
cd "$MODSSC_ROOT"

RUN_TASK_ARGS=(
  -m bench.campaign run-task
  --manifest "$MODSSC_CAMPAIGN_MANIFEST"
  --meta "$MODSSC_CAMPAIGN_META"
  --index "$GLOBAL_INDEX"
  --repo-root "$MODSSC_ROOT"
  --result-root "$MODSSC_CAMPAIGN_RESULT_ROOT"
  --work-root "$WORK_ROOT"
  --site-id "$MODSSC_CAMPAIGN_SITE_ID"
  --environment-lock-sha256 "${MODSSC_ENVIRONMENT_LOCK_SHA256:-unlocked}"
  --preflight-report "$MODSSC_PREFLIGHT_REPORT"
)
if [[ -n "${MODSSC_ENVIRONMENT_MANIFEST:-}" ]]; then
  RUN_TASK_ARGS+=(--environment-manifest "$MODSSC_ENVIRONMENT_MANIFEST")
fi
if [[ -n "${MODSSC_RECLAIM_STALE_LOCK_AFTER_HOURS:-}" ]]; then
  RUN_TASK_ARGS+=(--reclaim-stale-lock-after-hours "$MODSSC_RECLAIM_STALE_LOCK_AFTER_HOURS")
fi
if [[ -n "${MODSSC_CAMPAIGN_CHECKPOINT_ROOT:-}" ]]; then
  RUN_TASK_ARGS+=(--checkpoint-root "$MODSSC_CAMPAIGN_CHECKPOINT_ROOT")
fi

record_scheduler_failure() {
  local failure_class="$1"
  local scheduler_state="$2"
  local exit_code="$3"
  if ! "$MODSSC_PYTHON" -m tools.hpc.scheduler_failure \
    --manifest "$MODSSC_CAMPAIGN_MANIFEST" \
    --meta "$MODSSC_CAMPAIGN_META" \
    --index "$GLOBAL_INDEX" \
    --result-root "$MODSSC_CAMPAIGN_RESULT_ROOT" \
    --site-id "$MODSSC_CAMPAIGN_SITE_ID" \
    --failure-class "$failure_class" \
    --scheduler-state "$scheduler_state" \
    --exit-code "$exit_code"; then
    echo "Unable to persist Slurm failure $failure_class for task index $GLOBAL_INDEX" >&2
    return 1
  fi
}

CHILD_PID=""
SEGMENT_TIMER_PID=""
TERM_RECEIVED=0
USR1_RECEIVED=0
handle_usr1() {
  USR1_RECEIVED=1
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -USR1 "$CHILD_PID" 2>/dev/null || true
  fi
}
handle_term() {
  if (( TERM_RECEIVED != 0 )); then
    return
  fi
  TERM_RECEIVED=1
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
  fi
  record_scheduler_failure resource_timeout TERM 143 || true
}
trap handle_usr1 USR1
trap handle_term TERM

if [[ -n "${MODSSC_PLANNED_SEGMENT_SECONDS:-}" ]]; then
  if [[ ! "$MODSSC_PLANNED_SEGMENT_SECONDS" =~ ^[0-9]+$ ]] \
    || (( MODSSC_PLANNED_SEGMENT_SECONDS <= 300 )); then
    echo "MODSSC_PLANNED_SEGMENT_SECONDS must be an integer greater than 300" >&2
    exit 64
  fi
fi

set +e
"$MODSSC_PYTHON" "${RUN_TASK_ARGS[@]}" &
CHILD_PID="$!"
if [[ -n "${MODSSC_PLANNED_SEGMENT_SECONDS:-}" ]]; then
  PARENT_PID="$$"
  (
    sleep "$MODSSC_PLANNED_SEGMENT_SECONDS"
    kill -USR1 "$PARENT_PID" 2>/dev/null || true
  ) &
  SEGMENT_TIMER_PID="$!"
fi
while true; do
  wait "$CHILD_PID"
  CHILD_STATUS="$?"
  if ! kill -0 "$CHILD_PID" 2>/dev/null; then
    break
  fi
done
if [[ -n "$SEGMENT_TIMER_PID" ]]; then
  kill "$SEGMENT_TIMER_PID" 2>/dev/null || true
  wait "$SEGMENT_TIMER_PID" 2>/dev/null || true
fi
set -e

# Exit 85 is emitted only after the campaign executor has authenticated and
# published a durable CONTINUE marker.  It may originate from Slurm USR1 or
# from the deterministic diagnostic-only forced-resume hook.
if (( CHILD_STATUS == 85 && TERM_RECEIVED == 0 )); then
  exit 0
fi

if (( CHILD_STATUS != 0 || TERM_RECEIVED != 0 )); then
  FAILURE_CLASS=""
  SCHEDULER_STATE=""
  if (( TERM_RECEIVED != 0 )); then
    FAILURE_CLASS="resource_timeout"
    SCHEDULER_STATE="TERM"
  elif command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SACCT_STATES="$(
      sacct --jobs "$SLURM_JOB_ID" --noheader --parsable2 --format=State%32 2>/dev/null || true
    )"
    UPPER_STATES="$(printf '%s' "$SACCT_STATES" | tr '[:lower:]' '[:upper:]')"
    if [[ "$UPPER_STATES" == *"OUT_OF_MEMORY"* ]]; then
      FAILURE_CLASS="resource_oom"
      SCHEDULER_STATE="$SACCT_STATES"
    elif [[ "$UPPER_STATES" == *"TIMEOUT"* ]]; then
      FAILURE_CLASS="resource_timeout"
      SCHEDULER_STATE="$SACCT_STATES"
    fi
  fi
  if [[ -n "$FAILURE_CLASS" ]]; then
    record_scheduler_failure "$FAILURE_CLASS" "$SCHEDULER_STATE" "$CHILD_STATUS" || true
  fi
fi

if (( TERM_RECEIVED != 0 && CHILD_STATUS == 0 )); then
  CHILD_STATUS=124
fi
exit "$CHILD_STATUS"
