# Replication publication schema v1

!!! note "Native builder and verifier enabled"

    The pure in-memory builder and verifier are implemented under
    `src/modssc/evaluation/publication.py`. No filesystem or runner publication
    command is exposed: raw evidence remains sealed separately and an explicit
    adapter may only persist bytes already produced and verified by this native
    contract.

This schema is an allow-list projection from authenticated native objects. It
must never serialize `SeedReconciliation.runs` or `aggregate.json` directly:
those structures intentionally retain operational paths and raw errors for
local audit.

## `manifest.json`

The canonical manifest has this shape. Every `*_sha256` is a lowercase digest;
maps keyed by seed use canonical decimal strings in ascending numeric order.

```json
{
  "schema_version": 1,
  "release_id": "YYYY-MM-DD-cohort-source_sha12",
  "track": "paper",
  "created_at": "ISO-8601 timestamp with timezone",
  "supersedes": null,
  "source": {
    "git_commit": "full commit id",
    "git_tree": "full tree id",
    "clean": true,
    "distribution_sha256": "...",
    "environment_manifest_sha256": "..."
  },
  "raw_archive": {
    "archive_id": "stable logical identifier",
    "archive_ref": "stable retrievable URI or repository record",
    "format_version": 1,
    "manifest_sha256": "...",
    "archive_sha256": "...",
    "bytes": 1024,
    "verified_after_transfer": true
  },
  "cards": [
    {
      "card_id": "stable release-local identifier",
      "card_path": "bench/configs/reproductions/...yaml",
      "card_sha256": "...",
      "method_id": "native registry identifier",
      "dataset": {"id": "dataset id", "fingerprint": "..."},
      "requested_seeds": [0],
      "effective_config_sha256_by_seed": {"0": "..."},
      "protocol_sha256_by_seed": {"0": "..."},
      "software_sha256_by_seed": {"0": null},
      "execution_identity_sha256_by_seed": {"0": null}
    }
  ]
}
```

Execution and protocol identities are per seed, not per card. An HPO-enabled
card may also resolve a different effective configuration per seed. A missing
run therefore has no observed software or execution identity, and its manifest
values are `null`; its expected protocol identity remains present when it can
be derived from the frozen card. `archive_ref` must remain resolvable without a
user home, scratch, work, or temporary path.

## `results.json`

`results.json` contains one entry per manifest card, in manifest order:

```json
{
  "schema_version": 1,
  "release_id": "same as manifest",
  "cards": [
    {
      "card_id": "same as manifest",
      "reconciliation": {
        "status": "success | partial_failure | not_evaluable | failed",
        "certifiable": false,
        "execution_identity_complete": false,
        "requested_seeds": [0],
        "categories": {
          "success": [],
          "failed": [],
          "not_evaluable": [],
          "missing": [0]
        }
      },
      "metrics": {},
      "acceptance": {"...": "complete native AcceptanceReport payload"}
    }
  ]
}
```

Every paper card requires acceptance. `acceptance` is the complete native
`AcceptanceReport.to_dict()` payload, including `acceptance_sha256`. The
verifier removes that digest, canonicalizes the remaining report, recomputes
the digest, and rejects a mismatch. It never promotes a failed,
`not_evaluable`, non-paper-matched, or conformity-limited assessment.

## `observations.jsonl`

There is exactly one line for every requested `(card_id, seed)`, ordered by
manifest card and then numeric seed. Its allow-list is:

```json
{
  "card_id": "...",
  "seed": 0,
  "status": "success | failed | not_evaluable | missing",
  "run_id": null,
  "error_code": null,
  "metrics": null,
  "run_time_seconds": null,
  "protocol_sha256": "...",
  "software_sha256": null,
  "execution_identity_sha256": null,
  "source_run_sha256": null
}
```

For an observed run, all identities and `source_run_sha256` are required. For
`missing`, run-specific fields and `source_run_sha256` are `null`. Raw errors,
tracebacks, paths, hostnames, node names, account and job identifiers are never
accepted. `error_code` is a bounded stable public code, not an arbitrary error
message.

## Native and runner boundary

The pure native module implements the allow-list builder and in-memory verifier
under `src/modssc/evaluation/publication.py`. It receives native reconciliation
and acceptance objects plus explicit source, card, dataset, raw-archive, and
source-report digests. It performs no path, Git, YAML, scheduler, network, or
repository write.

A later explicit `modssc-bench publish` adapter may parse one frozen YAML card,
recompute native reconciliation and acceptance, and invoke that builder. It
must require a destination, refuse an existing destination, and never be called
by `run` or `reconcile`. It must not look for `docs/` or choose a method,
article, dataset, or scheduler branch.

The renderer produces UTF-8/LF canonical bytes for `manifest.json`,
`results.json`, `observations.jsonl`, an already reviewed `index.md`, and a
sorted `SHA256SUMS` covering all other release files. Input order, host paths,
and machine names cannot alter those bytes.

## Certification invariants

A card is certifiable only when:

- the four seed categories are disjoint and exactly cover the requested set;
- every requested seed is successful;
- every observed report has a valid portable execution identity and source
  digest;
- mandatory native acceptance is `passed` and `paper_matched`;
- the acceptance report's expected and observed runs correspond exactly to the
  non-missing observations and its digest recomputes successfully;
- the executed source is clean and all card, data, software, and environment
  identities match the manifest;
- the raw archive was verified after transfer.

The bundle verifier rejects any missing, added, mutated, duplicate, oversized,
or symlinked file. A negative or partial bundle may still be published, but its
reconciliation and acceptance values stay negative and `certifiable` stays
false.
