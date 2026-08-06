#!/bin/bash

# Translate Slurm's process environment once, at the scheduler boundary. Code
# under bench/ consumes only the neutral MODSSC_EXECUTION_* contract.

: "${SLURM_JOB_ID:?ModSSC workloads must run inside a Slurm allocation}"
: "${SLURMD_NODENAME:?ModSSC workloads must execute on a Slurm compute node}"

export MODSSC_EXECUTION_SCHEDULER=slurm
export MODSSC_EXECUTION_JOB_ID="$SLURM_JOB_ID"
export MODSSC_EXECUTION_JOB_NAME="${SLURM_JOB_NAME:-}"
export MODSSC_EXECUTION_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}"
export MODSSC_EXECUTION_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
export MODSSC_EXECUTION_CLUSTER="${SLURM_CLUSTER_NAME:-}"
export MODSSC_EXECUTION_CONSTRAINTS="${SLURM_JOB_CONSTRAINTS:-}"
export MODSSC_EXECUTION_PARTITION="${SLURM_JOB_PARTITION:-}"
export MODSSC_EXECUTION_QOS="${SLURM_JOB_QOS:-}"
export MODSSC_EXECUTION_NODE="$SLURMD_NODENAME"
