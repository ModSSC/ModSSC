# Architecture

This page describes the current package layout of ModSSC and the stability boundaries used in the codebase. It is intentionally about structure, not about the algorithms themselves.


## Public vs internal modules
The public Python API is organized around the top-level bricks:
- `modssc.data_loader`
- `modssc.data_augmentation`
- `modssc.sampling`
- `modssc.preprocess`
- `modssc.views`
- `modssc.graph`
- `modssc.evaluation`
- `modssc.hpo`
- `modssc.inductive`
- `modssc.transductive`
- `modssc.supervised`
- `modssc.runtime`

Those package-level entrypoints are what the reference pages document and what user code should import by default.


## Runtime and support packages
Several support packages were moved out of the historical package root and now have explicit homes:
- `modssc.runtime`: runtime-facing utilities such as device resolution, logging, and local path discovery.
- `modssc.cache`: model and backend cache resolution helpers.
- `modssc.dependencies`: shared optional-dependency helpers and dependency-specific errors.
- `modssc.utils`: generic internal utilities used across bricks.
- `modssc.backends`: backend-wide shared helpers that are not specific to a single brick.

These modules are real parts of the package structure, but only `modssc.runtime` is treated as user-facing in the reference docs today.


## Package layout
At a high level, the source tree is organized like this:

```text
src/modssc/
  runtime/          # runtime-facing utilities
  cache/            # model/backend cache helpers
  dependencies/     # optional-dependency loading and dependency errors
  utils/            # shared internal helpers
  cli/              # Typer CLI entrypoints
  data_loader/      # dataset catalogs, providers, storage, public loading API
  data_augmentation/# training-time augmentation plans and registries
  preprocess/       # deterministic preprocessing plans and model backends
  sampling/         # split plans, storage, and reproducible sampling
  views/            # feature-view planning and generation
  graph/            # graph construction, graph featurization, graph artifacts
  supervised/       # baseline classifiers and backend registry
  inductive/        # inductive SSL methods, adapters, deep bundles
  transductive/     # graph-based SSL methods, operators, adapters, solvers
  evaluation/       # metrics and reports
  hpo/              # search space and samplers
```


## Public facades and internal implementation
Several bricks expose a stable package API while delegating implementation to internal modules:
- `modssc.data_loader` exports its public functions from the package while provider resolution, storage, and internal orchestration live in submodules such as `catalog/`, `providers/`, `storage/`, and `services/`.
- `modssc.preprocess`, `modssc.sampling`, `modssc.views`, and `modssc.graph` follow the same pattern: the package-level API is public, while internal orchestration may live in `services/` or other subpackages.
- `modssc.inductive` contains additional internal support layers such as `helpers/`, `adapters/`, and `deep/bundle_factories/`.

The existence of an `api.py` file does not mean that the whole implementation lives there. In several bricks, `api.py` is now primarily a public facade or compatibility entrypoint.


## What is documented in the API reference
The API reference focuses on:
- package-level public imports;
- stable public types and functions;
- runtime utilities that are intentionally imported directly.

The API reference does not try to document every internal support directory as public surface area. Internal folders such as `services/`, `helpers/`, `bundle_factories/`, `adapters/`, and backend-specific implementation packages are described here as architecture, not as stable user API.


## Autonomous paper replication
The scientific implementation belongs to `src/modssc`: a supported article
strategy must be implemented with ModSSC's own method and backend interfaces.
Running a replication must never clone, import, or execute an upstream research
repository. Third-party source snapshots are not part of the runtime tree.

The repository is the complete execution unit for supported paper results:

- ModSSC owns the algorithms and reusable backends;
- `bench/` owns the validated runner, protocol cards, fixed protocol resources,
  and scientific campaign orchestration;
- datasets are resolved and prepared through ModSSC providers rather than by a
  manually seeded cache;
- optional Python functionality is declared through project extras;
- no external JAR, hand-installed Weka runtime, or undeclared executable is a
  valid prerequisite for an autonomous protocol.

Article identities are also a boundary. A reproduction card may use a
`method.profile` such as `paper:…` to identify and authenticate the campaign,
but that value is not copied into a `modssc` method specification. The card
must express executable behaviour through generic fields such as
`training_mode`, `protocol`, `solver`, sampler policy, and explicit
hyperparameters. Library code validates those mechanics without maintaining a
catalogue of article-profile identifiers.

Large datasets do not need to be committed to Git. Their provider identity,
options, expected fingerprint, and preparation rule form part of the protocol.
The runner may download them through the declared provider and must fail
explicitly if their identity cannot be verified.

## Bench repository layout
`bench/` contains only scientific benchmark and replication concerns:

- the end-to-end runner, schemas, and stage orchestrators;
- `bench/configs/experiments/`: authored examples and small templates;
- `bench/configs/best/`: the standardized benchmark matrix and manifests;
- `bench/configs/reproductions/`: immutable article protocol cards;
- compact fixed resources that are direct inputs to a protocol;
- campaign generation, execution, reconciliation, and scientific acceptance.

Campaign generation and reconciliation may group tasks by logical resource
profile, but their outputs remain scheduler-neutral. Rendering submission
scripts and translating scheduler environment variables are exclusively
`tools/hpc/` responsibilities.

Operational and archival concerns do not belong there:

- `tools/hpc/` contains generic Slurm launchers and operational helpers;
- `provenance/article10/` contains completed evidence, historical audit records,
  and audit-only source manifests; checksums, notices, and licences required for
  direct protocol inputs remain beside those inputs under `bench/assets/`;
- site names, accounts, partitions, QoS settings, credentials, deployment
  paths, caches, checkpoints, logs, and result directories are runtime state;
- copies of external research source trees are not retained.

This separation keeps the scientific path runnable on a workstation or through
any scheduler, while preventing one cluster's policy from becoming part of the
benchmark contract.

The dependency direction is:

```text
bench runner / protocols ──> src/modssc scientific implementations
tools/hpc ─────────────────> bench public runner and campaign interfaces
provenance/article10 <────── immutable manifests and completed evidence
```

Package code under `src/modssc` must never import `bench`, `tools/hpc`, or a
provenance bundle. Scientific execution may authenticate archived evidence, but
must never import or execute code from provenance or from a third-party source
snapshot.


## Documentation policy
When a page needs to explain user code, it should point to package imports such as `modssc.preprocess` or `modssc.runtime.logging`, not to historical root modules or internal implementation files.

When a page needs to explain repository structure, it should explicitly say whether a path is:
- public API,
- internal implementation,
- benchmark or replication config,
- scientific campaign orchestration,
- operational HPC tooling,
- or immutable provenance.

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc"><code>src/modssc/</code></a></li>
  <li><a href="https://github.com/ModSSC/ModSSC/tree/main/bench"><code>bench/</code></a></li>
  <li><a href="https://github.com/ModSSC/ModSSC/blob/main/mkdocs.yml"><code>mkdocs.yml</code></a></li>
</ol>
</details>
