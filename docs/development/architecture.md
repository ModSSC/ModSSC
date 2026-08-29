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
  evaluation/       # metrics, reconciliation, and scientific acceptance
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
The scientific implementation belongs to `src/modssc`. A paper card composes
the same sampling, preprocessing, views, augmentation, graph, method,
evaluation, and runtime APIs as any other experiment. It may retain a
`method.profile` as a human-readable citation, but that label never selects
code. Executable differences are expressed by typed parameters and registered
components.

Each method declares its first-order modality, representation, graph/view,
backend, device, target, and output requirements. The runner materializes
upstream artifacts, then the native runtime routes the exact regime-specific
input. The execution brick derives capabilities from the object the method will
actually consume and rejects known incompatible compositions immediately before
`fit`. Graph-mask conversion is rejected by default and must be declared by the
YAML sampling policy. No method-name switch is permitted in `bench/`.

This gate is deliberately not presented as a proof that every registered
method can run on every raw modality. `modalities=None` means that a method is
independent of the source modality *after* compatible preprocessing, graph
construction, views, and model binding; it does not mean that arbitrary raw
objects are accepted. The implemented promise is that every compatible
declared composition is explicitly verified, while known incompatibilities and
strict-mode missing proofs are rejected before `fit`. The current registry has
51 methods, while the inventory of 5,305 statically audited benchmark and
reproduction YAML cards covers 49 of them. This inventory is not a success
count: all 111 SimCLRv2 cards that
request contrastive pretraining are currently rejected because their native
classifier bundles do not expose a model-owned projection head. Cardless
methods and unavailable optional backends remain explicit coverage limits.

Component registries also own their optional-dependency declarations. The
runner passes dataset, step, method, classifier, and graph identifiers to the
native resolver, then only expands the returned extras through package metadata.
The exact selected distribution versions enter checkpoint identity; unrelated
installed extras do not.

Supported reproduction code is autonomous. It must not clone or execute an
upstream research repository, import code from an evidence archive, require an
external JAR, or depend on a manually seeded cache. Large datasets remain
external, but their provider identity and expected fingerprint are part of the
card and are verified after explicit preparation.

Scientific acceptance is also a native evaluation concern. A reproduction card
may contain a declarative top-level `acceptance` block beside the protocol it
assesses. `modssc.evaluation.acceptance` parses that contract and evaluates a
cohort of already authenticated seed reports without filesystem, scheduler,
article, or method-specific dispatch. It returns a canonical report with a
three-state assessment (`passed`, `failed`, or `not_evaluable`), a fidelity
classification (`paper_matched`, `paper_approx`, or `not_claimable`), and its
own SHA-256 digest. `bench` only passes inputs to this API and serializes the
result under `aggregate.json.acceptance`.

## Bench repository layout
`bench/` contains only declarative benchmark concerns:

- the end-to-end runner, schemas, and stage orchestrators;
- `bench/configs/experiments/`: authored examples and small templates;
- `bench/configs/best/`: the standardized benchmark matrix and manifests;
- `bench/configs/reproductions/`: article protocol cards.

The runtime has no campaign subsystem. Multi-seed execution is the generic
runner applied to `run.seeds`; one task is addressed with `--seed-index`.
Schedulers call that interface directly. Retry policy, array sizing, accounts,
partitions, modules, physical paths, caches, checkpoints, logs, and result
storage are deployment state outside the scientific contract. The rejected
root `tools/` and `provenance/` trees and their legacy tests have been removed;
no native execution or acceptance path may depend on them.

Replication documentation keeps source-only article notes, not recovered run
bundles. Package and benchmark code must not import audit material. When the
construction rule is known, a native component recomputes splits, features, or
graphs instead of loading a paper-specific artifact.

The dependency direction is:

```text
YAML card ──> generic bench runner ──> native resolution and input routing
                                           |
                                           v
                sampling / preprocess / graph / method execution
                                           |
                                           v
                         consumed-input capability validation
                                           |
                                           v
                                    per-seed run.json
                                           |
                                           v
                  native reconciliation and optional acceptance
                                           |
                                           v
                                      aggregate.json
```

Package code under `src/modssc` must never import `bench` or an audit bundle.
`bench` may parse the YAML, orchestrate public `modssc` APIs, reconcile
authenticated reports, and serialize their native results, but must never
contain method-specific scientific logic or its own acceptance mathematics.
Article and official-code evidence can inform a YAML card and its tests; it
cannot be an executable dependency.


## Documentation policy
When a page needs to explain user code, it should point to package imports such as `modssc.preprocess` or `modssc.runtime.logging`, not to historical root modules or internal implementation files.

When a page needs to explain repository structure, it should explicitly say whether a path is:
- public API,
- internal implementation,
- benchmark or replication config,
- private deployment state,
- or source evidence kept outside runtime.

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc"><code>src/modssc/</code></a></li>
  <li><a href="https://github.com/ModSSC/ModSSC/tree/main/bench"><code>bench/</code></a></li>
  <li><a href="https://github.com/ModSSC/ModSSC/blob/main/mkdocs.yml"><code>mkdocs.yml</code></a></li>
</ol>
</details>
