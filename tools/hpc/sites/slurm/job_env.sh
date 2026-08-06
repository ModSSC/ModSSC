#!/bin/bash

# Portable Slurm runtime contract.  Site-specific module loading, accounts,
# partitions and filesystem paths belong in an untracked site overlay.

set -euo pipefail

modssc_slurm_env() {
  : "${SLURM_JOB_ID:?ModSSC workloads must run inside a Slurm allocation}"
  : "${SLURMD_NODENAME:?ModSSC workloads must execute on a Slurm compute node}"
  : "${MODSSC_ROOT:?MODSSC_ROOT is required}"
  : "${MODSSC_SCRATCH:?MODSSC_SCRATCH must point to persistent storage}"

  source "$MODSSC_ROOT/tools/hpc/slurm/runtime-context.sh"

  local current_node
  current_node="$(hostname -s)"
  if [[ "$current_node" != "$SLURMD_NODENAME" ]]; then
    echo "Refusing workload on $current_node; allocated node is $SLURMD_NODENAME" >&2
    return 64
  fi

  local architecture="${1:-cpu}"
  if [[ ! "$architecture" =~ ^[a-z0-9._-]+$ ]]; then
    echo "Invalid accelerator architecture: $architecture" >&2
    return 64
  fi
  export MODSSC_ACCELERATOR_ARCH="$architecture"

  # A private site overlay may prepare modules/containers/environments before
  # calling this function.  The public layer only verifies the resulting
  # interpreter and never mutates it.
  : "${MODSSC_PYTHON:?MODSSC_PYTHON must name the pre-existing pinned interpreter}"
  if [[ ! -x "$MODSSC_PYTHON" ]]; then
    echo "MODSSC_PYTHON is not executable: $MODSSC_PYTHON" >&2
    return 66
  fi
  export PYTHONPATH="$MODSSC_ROOT/src:$MODSSC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONNOUSERSITE=1

  export MODSSC_CACHE_ROOT="${MODSSC_CACHE_ROOT:-$MODSSC_SCRATCH/modssc_cache}"
  export MODSSC_OUTPUT_DIR="${MODSSC_OUTPUT_DIR:-$MODSSC_SCRATCH/outputs}"
  export MODSSC_DATASET_CACHE_DIR="${MODSSC_DATASET_CACHE_DIR:-$MODSSC_CACHE_ROOT/datasets}"
  export MODSSC_PREPROCESS_CACHE_DIR="${MODSSC_PREPROCESS_CACHE_DIR:-$MODSSC_CACHE_ROOT/preprocess}"
  export MODSSC_SPLIT_CACHE_DIR="${MODSSC_SPLIT_CACHE_DIR:-$MODSSC_CACHE_ROOT/splits}"
  export MODSSC_GRAPH_CACHE_DIR="${MODSSC_GRAPH_CACHE_DIR:-$MODSSC_CACHE_ROOT/graph}"
  export MODSSC_GRAPH_VIEWS_CACHE_DIR="${MODSSC_GRAPH_VIEWS_CACHE_DIR:-$MODSSC_CACHE_ROOT/graph_views}"
  export MODSSC_CAMPAIGN_CHECKPOINTS="${MODSSC_CAMPAIGN_CHECKPOINTS:-$MODSSC_SCRATCH/modssc_checkpoints}"

  mkdir -p \
    "$MODSSC_OUTPUT_DIR" \
    "$MODSSC_DATASET_CACHE_DIR" \
    "$MODSSC_PREPROCESS_CACHE_DIR" \
    "$MODSSC_SPLIT_CACHE_DIR" \
    "$MODSSC_GRAPH_CACHE_DIR" \
    "$MODSSC_GRAPH_VIEWS_CACHE_DIR" \
    "$MODSSC_CAMPAIGN_CHECKPOINTS"

  if [[ -n "${JOBSCRATCH:-}" ]]; then
    export TMPDIR="$JOBSCRATCH"
    mkdir -p "$TMPDIR"
  fi
}
