#!/bin/bash

# Generic compute-node entry point for campaign administration that must not
# run on a login node.  Site overlays only provide scheduler directives and
# environment activation; this file owns no account, partition, QoS or path.

set -euo pipefail

: "${SLURM_JOB_ID:?Campaign operations must run inside a Slurm allocation}"
: "${SLURMD_NODENAME:?Campaign operations must run on a Slurm compute node}"
: "${MODSSC_ROOT:?MODSSC_ROOT is required}"

source "$MODSSC_ROOT/tools/hpc/slurm/runtime-context.sh"

if [[ -n "${MODSSC_SITE_OVERLAY:-}" ]]; then
  if [[ ! -f "$MODSSC_SITE_OVERLAY" ]]; then
    echo "MODSSC_SITE_OVERLAY is not a readable file" >&2
    exit 66
  fi
  # The overlay path is supplied by the operator and intentionally lives
  # outside the public repository.
  source "$MODSSC_SITE_OVERLAY"
fi

: "${MODSSC_PYTHON:?MODSSC_PYTHON must name the pre-existing pinned interpreter}"
if [[ ! -x "$MODSSC_PYTHON" ]]; then
  echo "MODSSC_PYTHON is not executable: $MODSSC_PYTHON" >&2
  exit 66
fi

current_node="$(hostname -s)"
if [[ "$current_node" != "$SLURMD_NODENAME" ]]; then
  echo "Refusing operation on $current_node; allocated node is $SLURMD_NODENAME" >&2
  exit 64
fi

operation="${1:?usage: run-operation.sh OPERATION [arguments ...]}"
shift

case "$operation" in
  preflight)
    exec "$MODSSC_PYTHON" -m tools.hpc preflight "$@"
    ;;
  stage-validation)
    exec "$MODSSC_PYTHON" -m pytest "$@"
    ;;
  dataset-lock)
    exec "$MODSSC_PYTHON" -m bench.campaign lock-datasets "$@"
    ;;
  reconcile)
    exec "$MODSSC_PYTHON" -m bench.campaign reconcile "$@"
    ;;
  daily-report)
    exec "$MODSSC_PYTHON" -m tools.hpc daily-report "$@"
    ;;
  evaluate-paper)
    exec "$MODSSC_PYTHON" -m bench.campaign evaluate-paper "$@"
    ;;
  continuation)
    exec "$MODSSC_PYTHON" -m tools.hpc.match_continuation_controller run "$@"
    ;;
  calder-prepare)
    exec "$MODSSC_PYTHON" -m tools.replication_audit.calder.artifacts prepare "$@"
    ;;
  calder-verify)
    exec "$MODSSC_PYTHON" -m tools.replication_audit.calder.artifacts verify "$@"
    ;;
  calder-materialize)
    exec "$MODSSC_PYTHON" -m tools.replication_audit.calder.artifacts materialize "$@"
    ;;
  *)
    echo "Unsupported campaign operation: $operation" >&2
    exit 64
    ;;
esac
