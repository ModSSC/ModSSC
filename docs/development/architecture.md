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


## Bench repository layout
The `bench/` tree has three different roles:
- `bench/configs/experiments/`: authored examples, tutorial configs, and smaller runnable templates.
- `bench/configs/best/`: curated benchmark configuration sets and generated command listings used for larger runs.
- `bench/slurm/jean_zay/`: cluster launchers and deployment-oriented job structure for Jean Zay runs.

The docs reference each of these for different purposes and should not present them as one undifferentiated config directory.


## Documentation policy
When a page needs to explain user code, it should point to package imports such as `modssc.preprocess` or `modssc.runtime.logging`, not to historical root modules or internal implementation files.

When a page needs to explain repository structure, it should explicitly say whether a path is:
- public API,
- internal implementation,
- benchmark config,
- or cluster/runtime tooling.

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc"><code>src/modssc/</code></a></li>
  <li><a href="https://github.com/ModSSC/ModSSC/tree/main/bench"><code>bench/</code></a></li>
  <li><a href="https://github.com/ModSSC/ModSSC/blob/main/mkdocs.yml"><code>mkdocs.yml</code></a></li>
</ol>
</details>
