# Benchmarks

Use this reference when you want to run the benchmark runner end to end and understand the artifacts it writes. For the YAML contract itself, continue with the [Configuration reference](configuration.md).

!!! warning
    Bench configs and caches are trusted local inputs. In particular, `method.model.factory` is disabled unless you explicitly opt in with `run.allow_custom_factories: true`, and that mode should only be used for configs you control.


## What it is for
The benchmark runner orchestrates dataset loading, sampling, preprocess, optional graph and views stages, method execution, evaluation, and reporting from one validated config. It is the highest-level execution path in the repository. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-5">[5]</a><a href="#source-6">[6]</a></sup>


## When to use
- Use the benchmark runner when you want a reproducible experiment with saved configs and result artifacts.
- Use it when you need seed sweeps, output folders, or HPO orchestration from YAML.
- Use CLI bricks or Python APIs instead when you only need one stage in isolation.


## Minimal examples
Run the toy configs:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
python -m bench.main --config bench/configs/experiments/toy_transductive.yaml
```

Enable verbose logs:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml --log-level detailed
```

Run a simple seed sweep from the CLI:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml --num-runs 5
```


## Repository layout
- [`bench/configs/experiments/`](https://github.com/ModSSC/ModSSC/tree/main/bench/configs/experiments): authored examples, tutorial configs, and runnable templates
- [`bench/configs/best/`](https://github.com/ModSSC/ModSSC/tree/main/bench/configs/best): curated benchmark suites and command listings
- local `bench/` deployment helpers such as `bench/slurm/`: cluster launchers and job structure for internal or site-specific environments

These directories serve different audiences. Do not treat them as one undifferentiated config bucket.


## How to run bench
If your config uses environment placeholders such as `${MODSSC_OUTPUT_DIR}`, export them before the run:

```bash
export MODSSC_OUTPUT_DIR=/tmp/modssc_runs
export MODSSC_DATASET_CACHE_DIR=/tmp/modssc_cache/datasets
export MODSSC_PREPROCESS_CACHE_DIR=/tmp/modssc_cache/preprocess
```

If one of these placeholders is missing at runtime, config loading fails fast with an explicit error.

You can also set one global cache root for runtime caches:

```bash
export MODSSC_CACHE_ROOT=/tmp/modssc_cache
```

Optional graph-specific overrides:

```bash
export MODSSC_GRAPH_CACHE_DIR=/tmp/modssc_cache/graphs
export MODSSC_GRAPH_VIEWS_CACHE_DIR=/tmp/modssc_cache/graph_views
```

Run the same config on specific seeds:

```yaml
run:
  name: toy_pseudo_label
  seed: 0
  seeds: [1, 2, 3, 4, 5]
```

In sweep mode, the runner executes one run per seed and auto-suffixes `run.name` with `-seed<N>`. For each run, `run.seed` and section seeds (`sampling`, `preprocess`, `views`, `graph`, `augmentation`, `search`) are aligned to that seed. `--num-runs` follows the same seed-sweep logic and overrides `run.seeds` when both are present. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-6">[6]</a></sup>


## Cache behavior in multi-seed runs
For multi-seed sweeps, keep one shared cache root and let fingerprints isolate seed-dependent artifacts:

```bash
export MODSSC_CACHE_ROOT=/tmp/modssc_cache
```

Use separate cache roots only when you need hard isolation, for example a strict clean-room comparison across branches or commits.

When you run five seeds, expect five distinct split, preprocess, and graph fingerprints, while dataset artifacts remain shared.

- Dataset cache (`datasets/`): reused across seeds when dataset identity is unchanged
- Sampling split cache (`splits/`): conceptually seed-dependent, but current bench orchestration computes sampling in memory and does not persist split cache entries
- Preprocess cache (`preprocess/`): seed-dependent and keyed by dataset fingerprint, plan fingerprint, fit fingerprint, and seed
- Graph cache (`graphs/`): seed-dependent and keyed by dataset fingerprint, preprocess fingerprint, graph spec, and seed
- Method training and inference: not cached by the benchmark runner

Practical rules:
- keep the same cache root for speed when rerunning the same experiment
- use a new cache root for strict from-scratch comparisons
- expect recomputation when dataset options, preprocess steps or params, `fit_on`, graph spec, or seeds change
- clear impacted cache folders if implementation code changes while the config stays the same


## How outputs are stored
Each run writes a timestamped directory under the configured `run.output_dir` (commonly `runs/`) with:
- `config.yaml`
- `run.json`
- `error.txt` on failure only

For multi-seed sweeps, the configured `run.output_dir` becomes a container of sweep folders. Each sweep writes:
- `<run.output_dir>/<run.name>-sweep-<timestamp>/aggregate.json`
- `<run.output_dir>/<run.name>-sweep-<timestamp>/<seed-run-dir>/run.json`

This keeps the aggregate plus all child runs together in one folder tree. These outputs are created by the run context and reporting orchestrator. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-4">[4]</a><a href="#source-5">[5]</a></sup>


## How to interpret results
`run.json` includes:
- run metadata such as name, seed, and status
- resolved config blocks
- artifacts and metrics
- HPO summary when search is enabled

`aggregate.json` includes:
- sweep metadata such as requested seeds and success or failure counts
- aggregated metrics across successful seeds
- references to child run directories and their `run.json` files

This structure is written in [`bench/orchestrators/reporting.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/reporting.py). <sup class="cite"><a href="#source-4">[4]</a></sup>


## Common mistakes
- Running bench from a PyPI-only install and expecting `bench/` configs to be present locally.
- Forgetting to export environment variables used by the YAML before launching the runner.
- Reusing old caches during a “from scratch” comparison and then attributing the speedup or output drift to the method itself.
- Expecting the runner to persist split cache entries during bench orchestration the same way the standalone sampling CLI can.
- Treating `run.json` as the only artifact in multi-seed mode and overlooking `aggregate.json`.


## Related links
- [Configuration reference](configuration.md)
- [Bench config cookbook](../how-to/bench-cookbook.md)
- [Reproducibility](../how-to/reproducibility.md)
- [Common errors and where to go](../how-to/common-errors.md)
- [Troubleshooting](../how-to/troubleshooting.md)
- [Optional extras and platform support](../getting-started/extras-and-platforms.md)


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
