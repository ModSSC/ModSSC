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
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml --num-runs 5
```

To run the same config on multiple seeds, set `run.seeds` in YAML:

```yaml
run:
  name: toy_pseudo_label
  seed: 0
  seeds: [1, 2, 3, 4, 5]
```

Or generate the same sweep pattern from CLI without editing YAML:

```bash
# Equivalent to run.seeds: [run.seed, run.seed+1, ..., run.seed+4]
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml --num-runs 5
```

When `--num-runs` is provided, it overrides `run.seeds`.

## Cache behavior in multi-seed

Recommended setup:

```bash
export MODSSC_CACHE_ROOT=/tmp/modssc_cache
```

Keep one shared cache root for speed. Fingerprints isolate seed-dependent artifacts automatically.

- Dataset cache: reused across seeds when dataset identity/options are unchanged.
- Sampling splits: seed-dependent fingerprints (one per seed), but bench currently computes splits in-memory (`save=False`) and does not persist split cache entries.
- Preprocess cache: seed-dependent (one entry per seed for the same plan).
- Graph cache: seed-dependent (one entry per seed for the same graph spec and preprocess fingerprint).
- Method fit/eval: always recomputed.

Use a different `MODSSC_CACHE_ROOT` only when you need strict clean-room runs (for example branch/commit comparisons from scratch).

## Memory limits
Use `limits` in the YAML to cap batch sizes and graph chunking when GPUs are tight on memory:

```yaml
limits:
  profile: auto
  max_method_batch_size: 128
  max_graph_chunk_size: 512
```

Profiles apply default caps (`auto` resolves to `v100` or `h100`); explicit `max_*` overrides them.

Artifacts:
- runs/<name-timestamp>/config.yaml
- runs/<name-timestamp>/run.json
- runs/<name-timestamp>/error.txt (if failed)

Notes:
- Templates in bench/configs/experiments/ download data and may require extras.
