# Match continuations on configured Slurm site

The Match controller turns each planned checkpoint exit into a new,
authenticated Slurm segment. Reconciliation and all controller decisions run
inside a compute allocation. The login node performs only the initial
validation, immutable state creation, and `sbatch` call.

After submitting the first generated H100 array, bootstrap the chain once:

```bash
python -m tools.hpc.match_continuation_controller bootstrap \
  --repo-root "$MODSSC_ROOT" \
  --campaign-dir "$MODSSC_CAMPAIGN_DIR" \
  --result-root "$MODSSC_CAMPAIGN_RESULTS/$MODSSC_CAMPAIGN_ID" \
  --state-dir "$MODSSC_SCRATCH/modssc_controllers/$MODSSC_CAMPAIGN_ID" \
  --site "$MODSSC_PRIVATE_SITE_ROOT/campaign-profiles/site.yaml" \
  --allocation /private/path/slurm-gpu-allocation-current.yaml \
  --environment-manifest "$MODSSC_ENVIRONMENT_MANIFEST" \
  --checkpoint-base "$MODSSC_CAMPAIGN_CHECKPOINTS" \
  --max-segments 8 \
  --after-job-id "$LAST_ARRAY_JOB_ID"
```

`--max-segments` is mandatory and includes the initial segment. Repeating the
same bootstrap command is idempotent. A different initial job, campaign
manifest, controller configuration, task identity, seed, architecture, or
checkpoint root is rejected.

For every planned continuation, the controller:

1. reconciles the original immutable manifest against the persistent results;
2. renders a continuation-only manifest containing the unchanged task rows;
3. submits a fresh H100 preflight bound to that continuation manifest;
4. submits the generated H100 wrapper with an `afterok` dependency;
5. submits the next controller with an `afterany` dependency on the wrapper.

The authoritative state and journal share one atomic, SHA-256-authenticated
snapshot under the state directory. A non-blocking file lock prevents
concurrent controllers. Deterministic Slurm job names allow an interrupted
controller to recover a submitted job instead of duplicating it. If a segment
does not create a new planned-continuation attempt or a valid success bundle,
the chain fails closed.

Refresh the private allocation snapshot at its configured path before a long
segment ends. A stale allocation makes the next preflight fail; it never
weakens the reserve or launches an unauthenticated continuation.
