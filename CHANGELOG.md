# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog", and this project adheres to Semantic Versioning.

## Unreleased

### Added
- Added native method capability contracts covering data modality,
  representation, graph/view/augmentation inputs, classifier outputs, backend,
  device, dtype, and checkpoint support.
- Added explicit native execution contexts, content-addressed atomic
  checkpoints, seed-index execution, and honest multi-seed aggregation.
- Added native sampling controls required by the published protocols, including
  exact holdout sizes, class-balanced streams, legacy RNG compatibility, and
  inclusive unlabeled pools.
- Added paper-faithful method parameters and declarative reproduction cards for
  classic, Match-family, Calder graph-learning, and GRAND methods.
- Added a composed pre-fit execution contract that verifies exact input roles,
  model outputs, optimizers, EMA objects, schedulers, and component relations,
  with a canonical report and SHA-256 in every benchmark result.
- Added method-agnostic scientific acceptance in `modssc.evaluation`, with
  declarative targets and diagnostics, three-state assessment, independent
  fidelity classification, and a canonical SHA-256 report.

### Changed
- Reduced `bench` to one responsibility: validate a YAML experiment, orchestrate
  registered ModSSC bricks, and report results. Method protocols and article
  identities no longer select hidden runner branches.
- Calder cards now recompute their VAE representation and exact FAISS kNN graph
  through native preprocessing and graph-construction bricks.
- Match-family and Democratic Co-Learning cards now construct their partitions
  from native sampling declarations instead of bundled replay files.
- Reproduction claims distinguish historical frozen evidence from new
  statistical replications when a bit-identical source sequence is unavailable.
- Moved each numerical acceptance specification into the reproduction YAML card
  it assesses. `bench` parses, orchestrates, and serializes the native result;
  it contains no article-specific acceptance mathematics.

### Removed
- Removed the rejected benchmark and root campaign frameworks and bundled
  runtime paper artefacts; scientific behaviour now uses native registered
  components. The residual root `tools/`, `provenance/`, `tests/tools/`, and
  legacy HPC/continuation tests were removed after their recovery archive was
  checksummed and verified.
- Removed the separate `modssc-reproduce` execution path; reproduction cards use
  the same `modssc-bench` runner as every other experiment.

### Fixed
- Made offline dataset loads and content verification inspect existing cache
  layouts without creating administrative entries, purging corrupt cache data,
  or backfilling missing content manifests.
- Make cache promotion portable and resumable on filesystems that do not
  support `RENAME_NOREPLACE`, while preserving exclusive publication and
  fail-closed recovery semantics.
- Removed the Weka/Java runtime and all vendored GraphLearning, FixMatch,
  TorchSSL, and USB source dependencies; ModSSC executes its own scientific
  implementations.
- Preserved historical public constructor signatures, standardized-method
  numerics and index spaces, cache read-only behavior, and graph precision while
  adding native replication contracts.
- Removed hidden campaign/environment identity from Match checkpoints; resume
  behavior is now explicit in the run YAML and verified against run identity and
  payload integrity.
- Re-hash declared dataset content both immediately after loading and immediately
  before writing a result, so same-size mutations and mid-run input changes fail
  with a typed integrity error.
- Require SciPy for exact Student-t confidence intervals instead of silently
  substituting a normal approximation when the dependency is unavailable.
- Exclude local cache contents and developer lock files from source
  distributions, with release-audit checks preventing either from being shipped.
- Classify a declared non-convergence or insufficient pseudo-label outcome as
  `not_evaluable`, preserving native diagnostics instead of publishing a
  successful replication result.
- Require DASO and TriNet to consume declared encoder/shared features, and
  require SimCLRv2 contrastive pretraining to consume a model-owned, optimized
  projection head. Classifier logits and undeclared feature aliases now fail
  closed instead of passing by shape.
- Bind preprocess, graph, graph-view, and VAE cache keys to exact input content,
  implementation and software identity; publish authenticated entries
  atomically and reject legacy, partial, or modified cache data before reuse.

## 1.2.2
### Fixed
- Stabilized Dynamic Label Propagation by renormalizing dynamic transition matrices after each update and failing explicitly on invalid transition weights instead of returning non-finite scores.
- Replaced torch sparse tensor coalescing in DGI and GNN graph helpers with deterministic duplicate-edge aggregation to avoid PyTorch sparse invariant warnings without disabling checks.

## 1.2.1
### Changed
- Added benchmark seed-section control for reproducible ISO-style sweeps.
- Aligned experiment templates and GCN smoke configs with paper-style defaults.
- Improved FashionMNIST catalog coverage and AET precomputed feature caching.

## 1.2.0
### Added
- Added the `core.vae` preprocessing step with Poisson-style vision presets, cache-backed checkpoints, runtime metadata, and latent feature output through `features.vae`.
- Added the `vision.aet` preprocessing step for CIFAR Auto-Encoding Transformations features, including checkpoint loading and precomputed feature alignment.
- Added a torch graph construction backend, local-scale `knn_gaussian` weights, and `mean` symmetrization for directed KNN graphs.

### Changed
- Updated transductive vision benchmark configs to use the new VAE/AET feature pipelines and local-scale graph construction for non-GNN methods.
- Aligned `PoissonLearning`, `PoissonMBO`, and `LaplaceLearning` with the reference paper pipeline, including normalized-Laplacian solving with the final `D^-1/2` transform and stricter solver edge-case handling.
- Reworked torch sparse matrix multiplication to use scatter-add, avoiding `torch.sparse.mm` so transductive methods can run on CPU, CUDA, and Apple MPS.
- Propagated runtime artifacts from preprocessing steps so training outputs include the metadata used to build cached VAE/AET features.

### Removed
- Removed transductive benchmark configs and generated command references for `graph_mincuts` and `lazy_random_walk`.

### Fixed
- Fixed Poisson and Laplace edge cases around zero-degree graphs, inactive right-hand sides, non-finite solver outputs, CG failures, and custom Poisson MBO class priors.
- Expanded tests for VAE/AET preprocessing, graph construction, Laplacian operators, torch transductive backends, and Poisson/Laplace methods; the full `pytest -q tests/` suite now reaches 100% coverage.

## 1.1
- Reorganized internal module boundaries around services, helpers, runtime utilities, and registry data while preserving public entry points.
- Aligned lazy registries, optional dependency handling, tooling, and the source/test mirror, with the full suite back at 100% coverage.

## 1.0.1
- Fixed the torchvision image encoder so batched grayscale arrays shaped `(N, H, W)` are handled as batches instead of being misread as a single sample.

## 1.0.0
- Marked ModSSC as a stable `1.0.0` release.
- Standardized benchmark configurations across vision, text, audio, tabular, and graph modalities to improve fairness, comparability, and resource discipline.
- Unified benchmark backbones, feature extraction pipelines, input resolutions, and augmentation policies where technically possible, and documented explicit exceptions where standardization was not feasible.
- Added a torchvision-based image embedding backend for standardized vision preprocessing and transductive feature extraction.
- Added token-level text augmentation operators for post-tokenization inductive pipelines.
- Updated benchmark templates, minimal examples, and benchmark documentation to match the new standardization policies.
- Expanded backend, cache, and augmentation test coverage and brought the full test suite to 100% coverage.
- Fixed the torchvision image feature hook implementation to satisfy linting and avoid loop-capture issues.
- Hardened benchmark path and cache directory handling.

## 0.3.0
- Refactored modules to reduce redundancy, centralize shared helpers, and improve maintainability and test coverage.

## 0.2.5
- Added env var passthrough for dataset, preprocess, and runs cache directories in benchmarks.

## 0.2.4
- Improved torch container and device handling in inductive benchmarks.
- Added LSTM embedding support and hidden_size alias in inductive bundles.
- Added preprocess cache_dir passthrough for benchmarks.
- Added VAT embedding perturbation support and related tests.

## 0.2.3
- Fixed bugs in Trinet inductive pipeline.

## 0.2.2
- Added `activation` support for inductive GraphSAGE.
- Added `core.to_torch` in TriNet vision configs (best + smoke).
- Updated GraphSAGE tests.

## 0.2.1
- Bumped version metadata.

## 0.2.0
- Improved inductive pipeline performance and critical paths.
- Strengthened test coverage to secure the new optimizations.

## 0.1.2
- Fixed miscellaneous bugs.

## 0.1.1
- Bump version metadata and docs.
- Allow docs/release workflows to run on numeric tags.

## 0.1.0
- Contains all desired transductive methods and marks a stable benchmark release.

## 0.0.4
- Added multiple semi-supervised methods.

## 0.0.3
- Fixed bugs and improved logging for inductive methods.

## 0.0.2
- Updated GitHub workflows.

## 0.0.1
- Initial public release.
