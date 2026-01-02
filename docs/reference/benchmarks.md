# Benchmarks

This page explains how to run the benchmark runner and interpret its outputs. For config structure, see the [Configuration reference](configuration.md).

Use the bench runner when you want end-to-end, reproducible experiments. If you only need one brick, the [CLI reference](cli.md) and the how-to guides may be a faster starting point.


## How to run bench
Use the benchmark runner module with an experiment config:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
python -m bench.main --config bench/configs/experiments/toy_transductive.yaml
```

Enable verbose logging for a run:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml --log-level detailed
```

The `--log-level` flag is defined on the bench CLI entry point. <sup class="cite"><a href="#source-1">[1]</a></sup>


The bench entry point and example configs are in [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py) and [`bench/configs/experiments/`](https://github.com/ModSSC/ModSSC/tree/main/bench/configs/experiments). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


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
