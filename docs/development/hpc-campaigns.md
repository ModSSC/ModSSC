# Portable HPC campaigns

The public HPC layer is scheduler-generic. It contains no centre name, user or
project account, queue, partition, module stack, host name, or physical storage
path. Operators provide those values through a private site overlay outside the
repository.

The campaign layer turns benchmark or paper-reproduction YAML into an immutable
one-seed-per-task manifest. `bench.main` still owns data preparation, training,
evaluation, and `run.json`; the campaign layer owns task identity, scheduling,
atomic publication, attempts, continuation, reconciliation, and scientific
gates.

## Public and private boundaries

The tracked public interface is:

- `tools/hpc/config/profiles/slurm.example.yaml`: portable Slurm site template;
- `tools/hpc/config/allocations/slurm.example.yaml`: allocation template;
- `tools/hpc/slurm_renderer.py`: renderer from a neutral campaign to Slurm wrappers;
- `tools/hpc/sites/slurm/job_env.sh`: compute-node environment contract;
- `tools/hpc/slurm/runtime-context.sh`: scheduler-to-ModSSC environment adapter;
- `tools/hpc/slurm/array-task.sh`: one generated array-task payload;
- `tools/hpc/slurm/run-operation.sh`: compute-node administrative dispatcher;
- `tools/hpc/submit_chained_arrays.py`: validated array-chain submission.

A private overlay may define scheduler directives, activate an existing Python
environment, and export storage/cache paths. Keep it outside the repository,
for example under `.modssc-private/sites/<logical-site>/`. Never add a real
account, queue, host, module command, allocation identifier, or physical result
path to a tracked file.

No dependency is installed by this workflow. The overlay must expose an
existing executable as `MODSSC_PYTHON`. Both generic payloads reject execution
when that interpreter is missing or non-executable.

## Execution rule

Training, dataset preparation or hashing, full preflight, test suites,
reconciliation, acceptance, and scientific sealing run in a Slurm allocation.
The login node is limited to lightweight inspection, manifest administration,
submission, and queue monitoring.

`job_env.sh`, `array-task.sh`, and `run-operation.sh` all require a Slurm job and
verify that the current host is the allocated compute node. This makes the
boundary executable rather than documentary. At that boundary,
`runtime-context.sh` translates scheduler variables into the neutral
`MODSSC_EXECUTION_*` contract; no module under `bench/` reads scheduler process
variables directly.

## Private overlay contract

Before invoking a public payload, a private batch script sets at least:

```bash
export MODSSC_ROOT=/path/to/read-only/repository-snapshot
export MODSSC_SCRATCH=/path/to/persistent/project-storage
export MODSSC_PYTHON=/path/to/existing/python
export MODSSC_CAMPAIGN_RESULT_ROOT=/path/to/persistent/results
```

It may also set:

```bash
export MODSSC_CACHE_ROOT=/path/to/cache
export MODSSC_ENVIRONMENT_MANIFEST=/path/to/immutable/environment.json
export MODSSC_CAMPAIGN_CHECKPOINTS=/path/to/checkpoints
```

Then it sources the portable contract and selects the logical architecture
declared by its private profile:

```bash
source "$MODSSC_ROOT/tools/hpc/sites/slurm/job_env.sh"
modssc_slurm_env "$MODSSC_EXPECTED_ACCELERATOR_ARCH"
```

The public script derives cache subdirectories from `MODSSC_SCRATCH` when they
are not supplied. It never chooses an account, partition, accelerator model, or
filesystem mount.

## Scientific scope and immutable identity

Every campaign spec declares `scientific_scope` independently of its human
campaign name:

```yaml
scientific_scope:
  claim_scope_id: article10
  campaign_stage: production
  claim_eligible: true
  gate_policy_id: modssc-scientific-gates-v2
  gate_policy_sha256: from_registry
```

Diagnostic and canary campaigns set `claim_eligible: false`. A campaign name is
never a gate exemption. Generation binds the tracked policy identifier and
SHA-256 into every task row and the manifest metadata. Preflight and execution
verify the same policy identity.

Manifest schema v4 also records the logical site, resource profile, expected
accelerator architecture, environment SHA-256, dataset/split identities, and
sampling seeds. It records no physical site path.

## Generate

Copy a relevant example spec and the Slurm profile template into private
operator storage, replace placeholders, and generate from a clean pinned
snapshot:

```bash
"$MODSSC_PYTHON" -m bench.campaign generate \
  --spec /private/specs/campaign.yaml \
  --repo-root "$MODSSC_ROOT" \
  --output-dir /private/campaigns/campaign-id
```

Generation creates:

- `manifest.jsonl` and `manifest.meta.json`;
- no scheduler or allocation files.

It deliberately creates no scheduler script. Render the operational Slurm
layer afterwards:

```bash
"$MODSSC_PYTHON" -m tools.hpc.slurm_renderer \
  --campaign-dir /private/campaigns/campaign-id \
  --site /private/sites/slurm.yaml
```

This produces one or more wrappers under `submit/<logical-site>/`. The same
two-step flow applies to neutral retry and continuation campaigns emitted by
reconciliation.

Each wrapper contains at most 500 homogeneous tasks. A local CPU profile does
not produce Slurm scripts.

## Compute-node operations

The generic dispatcher accepts only known operations. Submit it from a private
batch script with scheduler directives supplied by that script.

Dataset lock or paper observations:

```bash
"$MODSSC_ROOT/tools/hpc/slurm/run-operation.sh" dataset-lock \
  --spec /private/specs/campaign.yaml \
  --repo-root "$MODSSC_ROOT" \
  --dataset-cache-dir "$MODSSC_DATASET_CACHE_DIR" \
  --output /private/campaigns/campaign-id/dataset-observations.yaml
```

Review generated observations outside the immutable snapshot. If one must
become a tracked protocol lock, add it in a later source commit; never mutate
the deployed snapshot from a compute job.

Preflight:

```bash
"$MODSSC_ROOT/tools/hpc/slurm/run-operation.sh" preflight \
  --manifest /private/campaigns/campaign-id/manifest.jsonl \
  --allocation /private/sites/allocation.yaml \
  --site /private/sites/slurm.yaml \
  --repo-root "$MODSSC_ROOT" \
  --require-architecture "$MODSSC_EXPECTED_ACCELERATOR_ARCH" \
  --output /private/campaigns/campaign-id/preflight.json
```

Stage validation:

```bash
"$MODSSC_ROOT/tools/hpc/slurm/run-operation.sh" stage-validation -q
```

Reconciliation and paper evaluation:

```bash
"$MODSSC_ROOT/tools/hpc/slurm/run-operation.sh" reconcile \
  --manifest /private/campaigns/campaign-id/manifest.jsonl \
  --result-root "$MODSSC_CAMPAIGN_RESULT_ROOT" \
  --output-dir /private/reconciliations/campaign-id/run-001

"$MODSSC_ROOT/tools/hpc/slurm/run-operation.sh" evaluate-paper \
  --manifest /private/campaigns/campaign-id/manifest.jsonl \
  --reconcile /private/reconciliations/campaign-id/run-001/reconcile.json \
  --acceptance "$MODSSC_ROOT/bench/campaigns/article10-paper-acceptance.yaml" \
  --scientific-gates "$MODSSC_ROOT/bench/campaigns/scientific-gates.yaml" \
  --output-dir /private/acceptance/campaign-id/run-001
```

The exact command options remain defined by `python -m bench.campaign --help`;
the dispatcher only enforces the allocation boundary and interpreter identity.

## Submit arrays

First create a passing architecture-specific preflight report in an allocation.
On the login node, validate and submit generated wrappers in explicit order:

```bash
"$MODSSC_PYTHON" -m tools.hpc.submit_chained_arrays \
  --throttle 5 \
  --time 80:00:00 \
  --preflight-report /private/campaigns/campaign-id/preflight.json \
  /private/campaigns/campaign-id/submit/slurm-gpu/gpu_long.block000.slurm \
  /private/campaigns/campaign-id/submit/slurm-gpu/gpu_long.block001.slurm
```

The helper authenticates the manifest, resource catalogue, array indices,
profile limits, and preflight at submission. Blocks use `afterok`, so a failed
block does not release later work. Crucially, `array-task.sh` checks preflight
freshness again when each task actually starts. A report that expires while
queued produces a separate `authorization_expired` event; it is not a
scientific, deterministic, or infrastructure attempt. Run a fresh preflight and
resubmit that manifest row.

## Array-task guarantees

`array-task.sh` verifies before Python starts:

- Slurm allocation and compute-node identity;
- executable `MODSSC_PYTHON`;
- manifest and array-index SHA-256;
- a valid global task index;
- presence of the preflight and pinned environment manifest when required.

It invokes `bench.campaign run-task` and the optional
`tools.hpc.scheduler_failure` adapter with the same pinned `MODSSC_PYTHON`.
Scheduler OOM and timeout events are authenticated separately.
No public script infers a centre from a hostname.

## Attempts, continuation, and reconciliation

One task is one configuration and one seed. Successful results publish
atomically under the task output path. Authenticated failure records permit only
the registered classes and matching retry policy:

- `deterministic`: blocked, no automatic retry;
- `infrastructure`: at most three retries;
- `resource_oom` or `resource_timeout`: complete-cell reprofiling required;
- planned continuation: same scientific task, not a retry.

Attempt records contain logical checkpoint references and no physical work
directory. Reconciliation verifies record digests, de-duplicates identical
mirrors, and treats divergent records sharing an identifier as conflicts.

Reconciliation output is a new immutable bundle. It refuses overwrite and any
symbolic link, inventories each file by SHA-256, and exposes only `bundle://`
and `result://root-NNN/` references. Physical root mappings remain private
operator state.

## Checkpoints

Long profiles may request a scheduler signal before walltime. On that signal,
the executor atomically stores model, optimizer, scheduler, EMA, RNG, sampler,
step, evaluation history, and method-adaptive state under the private checkpoint
root. `CONTINUE.json` remains bound to the same task, commit, environment,
profile, and partition. A continuation never creates a new scientific seed.

## Operational checklist

1. Prepare a clean tagged repository snapshot.
2. Prepare and hash datasets and models inside an allocation.
3. Build the immutable environment manifest without installing dependencies.
4. Generate the immutable scientific campaign.
5. Build the private resource plan and render Slurm wrappers with
   `python -m tools.hpc.slurm_renderer`.
6. Run tests/oracles and preflight in an allocation.
7. Submit generated arrays through the validated chain helper.
8. Monitor success, walltime, memory, accelerator memory, and failure classes.
9. Run reconciliation in an allocation into a new destination.
10. Run scientific acceptance from the sealed reconciliation evidence.
11. Keep physical site mappings and scheduler details outside the public tree.

Never edit a generated manifest, overwrite a sealed bundle, rewrite historical
evidence to disguise its origin, or relax a scientific protocol to fit a queue.
