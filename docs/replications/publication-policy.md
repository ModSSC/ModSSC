# Replication publication policy

!!! warning "No current native publication"

    The policy below defines the format for future results. At the current
    `HOLD` gate there is no new bundle to publish.

## Purpose

A public bundle is a compact, immutable index into authenticated evidence. It
is not the execution directory and it is not a replacement for the raw archive.
The format must let a reader identify the source, protocol, software, data,
seed set, result, and external evidence without exposing Jean Zay credentials or
depending on paths that will disappear.

## Publication namespaces

New bundles live under the article-replication root:

```text
docs/replications/results/paper/<YYYY-MM-DD>-<cohort>-<source-sha12>/
```

The namespace contains only article-specific claims. Generic benchmark
comparisons do not enter this registry. The readable cohort name is
descriptive; hashes in the manifest provide identity.

Published directories are append-only. If an error is discovered, create a new
bundle whose manifest names the replaced bundle in `supersedes`; do not rewrite
or delete the earlier evidence. A mutable convenience page may link to the
latest bundle, but it cannot be the sole record of a result.

## Required bundle content

Each publication directory contains:

- `index.md`, a negative-first human summary with limitations, failures, and
  the exact scientific claim;
- `manifest.json`, the canonical machine-readable inventory and identity;
- `SHA256SUMS`, covering every other file in the directory;
- `results.json`, compact per-card reconciliation and acceptance outcomes;
- `observations.jsonl`, compact per-seed observations and the SHA-256 of each
  source `run.json`.

The manifest records at least:

- schema version, publication track, immutable bundle identifier, creation
  timestamp, and optional `supersedes` identifier;
- Git commit and clean-tree assertion for the executed source;
- environment lock or installed-distribution digest;
- reproduction-card path and file SHA-256, data fingerprint, declared seed set,
  and effective-config, protocol, software, and execution-identity SHA-256 maps
  by seed;
- expected, successful, failed, `not_evaluable`, and missing run counts;
- reconciliation status, acceptance status, fidelity ceiling or status,
  reasons, and acceptance SHA-256 when an acceptance contract exists;
- external raw-archive reference, archive-format version, content-manifest
  SHA-256, archive SHA-256, and byte size;
- the digest algorithm and canonical serialization rules used by the bundle.

Files use UTF-8, LF line endings, deterministic key ordering, finite JSON
numbers, and no implicit `NaN` values. `SHA256SUMS` is regenerated only while a
new bundle is staged, then verified before the bundle enters Git.

## Compact observations

The public per-seed record contains scientific identity and outcomes, not the
full runner report. It may include the method, card, seed, status, metrics,
diagnostic reason codes, duration, resource class, and source `run.json`
SHA-256. Absolute paths, command lines containing private locations, account,
project, hostname, node, scheduler job identifier, reservation, and environment
secrets are prohibited.

Errors are normalized to bounded stable public reason codes. The original
message, traceback, and logs remain in the raw archive. This redaction must not
hide failure: a failed or `not_evaluable` observation stays failed or
`not_evaluable` in both the compact bundle and its summary.

There is exactly one compact observation for every declared seed. A missing
seed is represented by `status: "missing"` and `source_run_sha256: null`; it is
not omitted. Observed seeds require the SHA-256 of their source `run.json`.
The complete field-level contract is defined by
[publication schema v1](publication-schema-v1.md).

## Raw evidence outside Git

The sealed external archive preserves the material needed for a forensic
recheck, subject to dataset licences:

- original per-seed `run.json` files and aggregate reports;
- effective configuration and source-card snapshots;
- environment inventory and immutable source identifier;
- checkpoints required by the declared continuation policy;
- scheduler stdout/stderr and accounting exports;
- preprocessing and graph manifests, with the content digests of large cached
  objects;
- a sorted content manifest containing size and SHA-256 for every archived
  file.

Datasets and restricted artefacts are never made public merely because an
archive is sealed. The manifest records their authenticated identities and
licensing boundary. The public `archive_ref` must be a stable logical URI or
repository record, never a user home, scratch, work, or temporary path.

The archive is sealed before public publication. Its digest is computed from
the final bytes, copied into `manifest.json`, and independently verified after
transfer. Retries are preserved as separate attempts rather than overwritten.

## Eligibility and workflow

A publication is eligible only when all of these checks pass:

1. the executed commit and environment are immutable and authenticated;
2. every card, dataset, split, preprocessing object, graph, and seed has the
   expected identity;
3. reconciliation accounts for the complete declared seed set;
4. the native acceptance evaluator has produced its hashed report where the
   card declares an acceptance contract;
5. raw evidence has been sealed, transferred, and verified;
6. public observations have passed schema, redaction, size, and hash checks;
7. the human summary states failures and limitations before positive claims;
8. an independent review confirms that the bundle is an article replication
   from the active card inventory.

Missing or failed runs do not prohibit publishing a negative or incomplete
outcome. They prohibit a certifiable complete-run claim. A canary or
configuration validation must be labelled as such and cannot enter a paper
result count.

Publication is an explicit post-run action performed from a clean publication
checkout after evidence has been copied off Jean Zay. Compute jobs never commit
to `docs/`, and the generic benchmark runner never contains documentation or
repository-write logic.

## Repository guardrails

Targeted ignore rules reject `raw/`, `.staging/`, common checkpoint and array
formats, archives, and scheduler logs below `docs/replications`. Reviewers must
also reject oversized or binary evidence even when its extension is unknown.
Only the compact text bundle and documentation belong in Git.

The current CI check covers only filesystem layout, text portability, size,
and checksums. It is deliberately not called a scientific verifier. Result
directories remain forbidden until the native v1 builder/verifier is
implemented and the gate is replaced by end-to-end schema, reconciliation,
acceptance, identity, and certification checks.
