# ModSSC Benchmark Runner (GitHub-only)

This folder contains the end-to-end benchmark runner used from the repo. It is not
shipped in the PyPI wheel.

## Install (core)

```bash
python -m pip install -e "."
# or: python -m pip install -e ".[dev]"
```

## Install extras by scenario

```bash
# Text datasets + sentence-transformer features
python -m pip install -e ".[text,preprocess-text,sklearn]"

# Vision datasets + OpenCLIP features
python -m pip install -e ".[vision,preprocess-vision]"

# Graph datasets (PyG)
python -m pip install -e ".[graph]"

# Torch-based inductive methods
python -m pip install -e ".[inductive-torch,supervised-torch]"
```

## Run

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
python -m bench.main --config bench/configs/experiments/toy_transductive.yaml
```

Artifacts:
- runs/<name-timestamp>/config.yaml
- runs/<name-timestamp>/run.json
- runs/<name-timestamp>/error.txt (if failed)

Notes:
- Templates in bench/configs/experiments/ download data and may require extras.
