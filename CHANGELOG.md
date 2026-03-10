# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog", and this project adheres to Semantic Versioning.

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
