# Running ModSSC on Jean Zay or another scheduler

ModSSC has one scientific execution interface on a workstation and in an HPC
allocation:

```bash
modssc-bench --config CARD.yaml --seed-index N
```

There is no repository campaign engine or scheduler-specific scientific path.
The runner reads the YAML, validates the composed capabilities, calls native
`src/modssc` components, and writes one result. A Slurm array only maps its task
index to `--seed-index`. The rejected root `tools/` and `provenance/` trees and
their legacy tests have been removed; they are not dependencies or supported
interfaces for this execution path.

## Private deployment boundary

Keep the following outside the repository:

- Jean Zay account, QoS, partition, reservation, and module commands;
- environment activation and local wheel/snapshot path;
- dataset, preprocessing, graph, checkpoint, output, and log roots;
- array throttling, wall time, memory, CPU, and accelerator requests;
- resubmission and monitoring policy.

Those values are operational state, not a scientific protocol. The tracked YAML
continues to declare device/backend requirements so the generic capability gate
can reject an incompatible allocation.

## Minimal Slurm adapter

A private job script can be as small as:

```bash
#!/usr/bin/env bash
#SBATCH --array=0-2

set -euo pipefail

source /private/path/to/environment/bin/activate
export MODSSC_OUTPUT_DIR=/private/path/to/results
export MODSSC_DATASET_CACHE_DIR=/private/path/to/cache/datasets
export MODSSC_PREPROCESS_CACHE_DIR=/private/path/to/cache/preprocess
export MODSSC_GRAPH_CACHE_DIR=/private/path/to/cache/graph

cd /private/path/to/pinned/modssc
modssc-bench --config "$CARD" --seed-index "$SLURM_ARRAY_TASK_ID"
```

The array range must match the zero-based positions in the card's `run.seeds`.
The private submission environment supplies `CARD`; the job must preserve the
same immutable source snapshot and environment for every seed.

## Resume and publication

Cards that support continuation declare `run.resume_policy`. The runner passes
an explicit native execution context to the method, and checkpoints are bound
to a portable scientific-protocol hash, the seed, and a software digest. Output,
cache, and checkpoint paths are deliberately excluded from the protocol hash,
so the same checkpoint identity can move between a workstation and Jean Zay;
scientific parameter or code/dependency changes invalidate it. No scheduler job
identifier or campaign name participates in scientific identity. The full
effective configuration hash remains recorded separately in each result.

Each task writes into its normal ModSSC result directory. Treat a task as
successful only when `run.json` records success and the expected identity.
Aggregate only the complete declared seed set; missing and failed tasks remain
visible and make the sweep non-certifiable. Do not overwrite an earlier result
to hide a retry.

## Validation gate before a costly launch

Before submitting a long array:

1. validate every YAML card and its capability contract without training;
2. authenticate prepared datasets and confirm all cache/output paths are
   writable from a compute node;
3. run the relevant short unit and integration tests in the pinned environment;
4. execute one bounded canary for dataset loading, native sampling,
   preprocessing/graph construction, method startup, checkpointing, and result
   publication;
5. review the canary and obtain explicit approval for the full run.

A configuration load is not evidence that a costly method will converge, and a
single canary is not a paper result. No long benchmark is part of the migration
itself.

## Reconciliation

Reconciliation is data analysis, not a second execution framework. Read the
normal per-seed `run.json` files, verify their effective config and seed
identities, and aggregate with the native evaluation utilities. Report:

- expected, successful, failed, and missing seeds;
- every declared metric with count, mean, sample/population standard deviation,
  extrema, and confidence interval as applicable;
- any non-convergence, fallback, or `not_evaluable` outcome;
- whether the complete sweep is eligible for a scientific comparison;
- for a card with `acceptance`, the native `assessment_status`, independent
  `fidelity_status`, reasons, and `acceptance_sha256` from
  `aggregate.json.acceptance`.

The scheduler wrapper must not reproduce numerical gates. The generic runner
passes the card's acceptance block and authenticated seed reports to
`modssc.evaluation.evaluate_acceptance` and only serializes its canonical
result. Missing or non-successful repetitions remain `not_evaluable`.

Historical external backups can be consulted to interpret prior runs, but they
must never be mounted as a dependency of a new Jean Zay task.
