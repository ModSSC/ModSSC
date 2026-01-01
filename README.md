# ModSSC

[![Ruff](https://img.shields.io/github/actions/workflow/status/ModSSC/ModSSC/ruff.yml?branch=main&style=for-the-badge&label=ruff&color=0B3D91&labelColor=0B2D5E)](https://github.com/ModSSC/ModSSC/actions/workflows/ruff.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/ModSSC/ModSSC/tests.yml?branch=main&style=for-the-badge&label=tests&color=0B3D91&labelColor=0B2D5E)](https://github.com/ModSSC/ModSSC/actions/workflows/tests.yml)
[![codecov](https://img.shields.io/codecov/c/github/ModSSC/ModSSC?token=5S1R9H5L8G&style=for-the-badge&label=coverage&color=0B3D91&labelColor=0B2D5E)](https://codecov.io/gh/ModSSC/ModSSC)
[![Downloads](https://img.shields.io/endpoint?url=https%3A%2F%2Fmichaelthomasletts.github.io%2Fpepy-stats%2Fmodssc.json&style=for-the-badge&logo=python&labelColor=0B2D5E&color=0B3D91)](https://pepy.tech/project/modssc)
[![Stars](https://img.shields.io/github/stars/ModSSC/ModSSC?style=for-the-badge&label=stars&color=0B3D91&labelColor=0B2D5E)](https://github.com/ModSSC/ModSSC/stargazers)

ModSSC is a modular framework for semi-supervised classification across heterogeneous
modalities (text, vision, tabular, graph). It is designed for academic research: clear
abstractions, reproducible pipelines, and extensible method registries.

## Research goals

- Provide composable "bricks" for data loading, sampling, preprocessing, graphs, and SSL methods.
- Make experiments reproducible via declarative configs and deterministic seeds.
- Support both inductive and transductive SSL with lightweight baselines and extensible APIs.

## Repository map

- `src/modssc`: core library + CLI tools
- `bench/`: end-to-end benchmark runner (GitHub-only, not shipped to PyPI)
- `docs/`: MkDocs site (concepts, CLI, API)
- `examples/`, `notebooks/`: demos and exploratory workflows

## Install

PyPI (library + CLI tools):

```bash
python -m pip install modssc
```

From source (recommended for benchmark runs):

```bash
git clone https://github.com/ModSSC/ModSSC
cd ModSSC
python -m pip install -e "."
```

Python 3.11+ is required. For modality-specific extras, see `bench/README.md`.

## Quickstart (library + CLI)

CLI:

```bash
modssc-datasets list
modssc-sampling --help
modssc-preprocess steps list
modssc-graph build --help
modssc-inductive methods list
modssc-transductive methods list
```

Python:

```python
from modssc.data_loader import load_dataset
from modssc.sampling import sample
from modssc.preprocess import preprocess
```

## Benchmark quickstart (GitHub-only)

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
python -m bench.main --config bench/configs/experiments/toy_transductive.yaml
```

Artifacts land in `runs/<name-timestamp>/` with:
- `config.yaml` (immutable config snapshot)
- `run.json` (metrics + metadata)
- `error.txt` (full traceback on failure)

## Reproducibility notes

- Every run derives a timestamped run directory and a per-stage seed from a master seed.
- Preprocess and graph caches are fingerprinted by dataset + plan + seed.
- Configs are declarative YAML; they are copied into the run directory for auditability.

## Citation

If you use ModSSC in research, please cite `CITATION.cff`.

## License

MIT. See `LICENSE`.
