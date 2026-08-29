# Native Match checkpoint continuation

FixMatch, FlexMatch, FreeMatch, and SoftMatch use the same native runtime
contract as other ModSSC methods. There is no separate continuation controller.

Cards that permit continuation declare:

```yaml
run:
  resume_policy: auto
```

The runner creates an execution identity from the scientific protocol hash,
software digest, and seed, passes it explicitly to the registered method, and
stores content-addressed checkpoints through `modssc.runtime`. Operational
paths, cache locations, logging, and the resume policy are deliberately outside
the scientific protocol hash. Match-specific state is serialized by the Match
implementation, while identity, integrity, atomic publication, and generation
retention remain generic runtime services.

`resume_policy` has three meanings:

- `never`: ignore prior checkpoints and start a new training trajectory;
- `auto`: resume the newest valid checkpoint for the same execution identity,
  or start fresh when none exists;
- `required`: fail if no valid checkpoint exists.

A checkpoint with a different scientific protocol, software digest, seed, or
payload identity is rejected. Scheduler job IDs, array IDs, host names,
operational paths, and human task names are not scientific identity.

On a scheduler, resubmit the same public command after an interruption:

```bash
modssc-bench --config "$CARD" --seed-index "$SLURM_ARRAY_TASK_ID"
```

Use the same persistent checkpoint root and immutable source/environment. A
retry is successful only when the ordinary `run.json` is complete; planned
checkpoint exits and infrastructure failures must remain visible. Bound the
number of resubmissions in the private scheduler policy, and never promote an
incomplete trajectory to a paper result.
