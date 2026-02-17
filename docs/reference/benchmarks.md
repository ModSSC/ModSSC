# Benchmarks

This page explains how to run the benchmark runner and interpret its outputs. For config structure, see the [Configuration reference](configuration.md).

Use the bench runner when you want end-to-end, reproducible experiments. If you only need one brick, the [CLI reference](cli.md) and the how-to guides may be a faster starting point.


## How to run bench
Use the benchmark runner module with an experiment config:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
python -m bench.main --config bench/configs/experiments/toy_transductive.yaml
```

If you are using environment variable placeholders in configs (like `${MODSSC_OUTPUT_DIR}`), set them once before running:

```bash
export MODSSC_OUTPUT_DIR=/tmp/modssc_runs
export MODSSC_DATASET_CACHE_DIR=/tmp/modssc_cache/datasets
export MODSSC_PREPROCESS_CACHE_DIR=/tmp/modssc_cache/preprocess
```

If one of these placeholders is missing at runtime, config loading fails fast with an explicit error.

You can also set one global cache root for runtime caches (dataset, preprocess, split, graph):

```bash
export MODSSC_CACHE_ROOT=/tmp/modssc_cache
```

Optional graph-specific overrides:

```bash
export MODSSC_GRAPH_CACHE_DIR=/tmp/modssc_cache/graphs
export MODSSC_GRAPH_VIEWS_CACHE_DIR=/tmp/modssc_cache/graph_views
```

Enable verbose logging for a run:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml --log-level detailed
```

Run the same config on multiple seeds:

```yaml
run:
  name: toy_pseudo_label
  seed: 0
  seeds: [1, 2, 3, 4, 5]
```

In sweep mode, the runner executes one run per seed and auto-suffixes `run.name` with `-seed<N>`. For each run, `run.seed` and section seeds (`sampling`, `preprocess`, `views`, `graph`, `augmentation`, `search`) are aligned to that seed.

Or sweep by run count from the CLI:

```bash
# Uses run.seed as base and runs seeds [seed, seed+1, ..., seed+N-1]
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml --num-runs 5
```

`--num-runs` follows the same seed-sweep logic and overrides `run.seeds` when both are present.

The `--log-level` flag is defined on the bench CLI entry point. <sup class="cite"><a href="#source-1">[1]</a></sup>


The bench entry point and example configs are in [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py) and [`bench/configs/experiments/`](https://github.com/ModSSC/ModSSC/tree/main/bench/configs/experiments). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Cache behavior in multi-seed runs
For multi-seed sweeps, keep one shared cache root and let fingerprints isolate seed-dependent artifacts:

```bash
export MODSSC_CACHE_ROOT=/tmp/modssc_cache
```

Use separate `MODSSC_CACHE_ROOT` values only when you explicitly need hard isolation (for example strict clean-room comparisons across branches/commits).

When you run 5 seeds, you should expect 5 distinct split/preprocess/graph fingerprints (one per seed), while dataset artifacts are shared.

- Dataset cache (`datasets/`):
Reused across seeds when dataset identity is unchanged (provider, dataset id, version, resolved options). Usually do not recompute.

- Sampling split cache (`splits/`):
Conceptually seed-dependent (different split fingerprint per seed), so each seed maps to a different split entry.
Current bench orchestration computes sampling in-memory (`save=False`) and does not persist split cache entries.

- Preprocess cache (`preprocess/`):
Seed-dependent. Fingerprint includes dataset fingerprint, resolved plan fingerprint, fit fingerprint, and seed.
In a 5-seed sweep, expect 5 preprocess entries. Reuse only happens when rerunning the same seed + same plan + same fit inputs.

- Graph cache (`graphs/`):
Seed-dependent. Fingerprint includes dataset fingerprint, preprocess fingerprint, graph spec, and seed.
In a 5-seed sweep, expect 5 graph entries (unless graph is dataset-provided and not rebuilt).

- Method training/inference:
Not cached by the benchmark runner. Model fit/eval is recomputed every run.

Practical rules:
- Keep the same cache root for speed when rerunning identical experiments.
- Do not reuse cache for "strict from-scratch" comparisons; use a new cache root.
- If you change dataset options, preprocessing steps/params, `fit_on`, graph spec, or seeds, expect recomputation for impacted stages.
- If preprocessing or graph implementation changes in code while config stays the same, clear impacted cache folders to avoid stale artifacts.


## How outputs are stored
Each run writes a timestamped directory under [`runs/`](https://github.com/ModSSC/ModSSC/tree/main/runs) with:
- `config.yaml` (copied config)
- `run.json` (metrics + metadata)
- `error.txt` (only on failure)

These outputs are created by the run context and reporting orchestrator. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-4">[4]</a><a href="#source-5">[5]</a></sup>


## How to interpret results
`run.json` includes:
- run metadata (name, seed, status)
- resolved config blocks
- artifacts and metrics
- HPO summary when search is enabled

This structure is written in [`bench/orchestrators/reporting.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/reporting.py). <sup class="cite"><a href="#source-4">[4]</a></sup>


## Reproducibility tips
- Fix `run.seed` to make sampling, preprocessing, and method seeds deterministic. <sup class="cite"><a href="#source-6">[6]</a><a href="#source-3">[3]</a></sup>

- Keep the copied `config.yaml` alongside results for auditability. <sup class="cite"><a href="#source-3">[3]</a></sup>

- Caches for datasets, graphs, and views reduce re-downloads and make runs faster. <sup class="cite"><a href="#source-7">[7]</a><a href="#source-8">[8]</a></sup>


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/main.py"><code>bench/main.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/tree/main/bench/configs/experiments"><code>bench/configs/experiments/</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/context.py"><code>bench/context.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/reporting.py"><code>bench/orchestrators/reporting.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/cache.py"><code>src/modssc/data_loader/cache.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/cache.py"><code>src/modssc/graph/cache.py</code></a></li>
</ol>
</details>
