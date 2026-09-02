# Benchmarks

Use this reference when you want to run the benchmark runner end to end and
understand the artifacts it writes. For the YAML contract itself, continue with
the [Configuration reference](configuration.md). For an exact published result,
select a frozen card from `bench/configs/reproductions/` rather than adapting a
standardized benchmark config.

!!! warning
    Bench configs and caches are trusted local inputs. In particular, `method.model.factory` is disabled unless you explicitly opt in with `run.allow_custom_factories: true`, and that mode should only be used for configs you control.


## What it is for
The repository-level benchmark runner orchestrates dataset loading, sampling,
preprocess, optional graph and views stages, method execution, evaluation, and
reporting from one validated config. Method implementations and reusable
backends come from `src/modssc`; the scientific experiment contract stays in
the YAML. Seed sweeps, capability validation, resume identity, and aggregation
use generic native services; there is no article-specific execution path.
<sup class="cite"><a href="#source-1">[1]</a><a href="#source-5">[5]</a><a href="#source-6">[6]</a></sup>

The two config families have different contracts:

- `bench/configs/experiments/` and `bench/configs/best/` define examples and the
  standardized benchmark;
- `bench/configs/reproductions/` freezes article-specific datasets, splits,
  hyperparameters, repetitions, metrics, and expected protocol metadata. A card
  that makes a numerical replication claim also contains its own declarative
  `acceptance` block. The native evaluation API interprets that block; the
  runner only parses, passes, and serializes it.


## When to use
- Use the benchmark runner when you want a reproducible experiment with saved configs and result artifacts.
- Use it when you need seed sweeps, output folders, or HPO orchestration from YAML.
- Use a reproduction card when the objective is to reproduce a published
  result. This avoids silently replacing a fixed split, graph, metric, or
  aggregation rule with a convenient local choice.
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
- `bench/main.py`, `bench/schema.py`, and `bench/orchestrators/`: validated
  YAML parser, generic stage orchestration, and report serialization
- [`bench/configs/experiments/`](https://github.com/ModSSC/ModSSC/tree/main/bench/configs/experiments): authored examples, tutorial configs, and runnable templates
- [`bench/configs/best/`](https://github.com/ModSSC/ModSSC/tree/main/bench/configs/best): curated benchmark suites and manifests
- `bench/configs/reproductions/`: frozen article protocol cards

Private scheduler wrappers and historical evidence archives are outside the
runtime contract. Copies of third-party research source trees are not retained.
Accounts, partitions, QoS names, filesystem roots, caches, logs, and results
remain deployment state. The rejected root `tools/`, `provenance/`, and legacy
test paths have been removed; they are neither dependencies nor supported
interfaces of the runner.


## How to run bench
If your config uses environment placeholders such as `${MODSSC_OUTPUT_DIR}`, export them before the run:

```bash
export MODSSC_OUTPUT_DIR=/tmp/modssc_runs
export MODSSC_DATASET_CACHE_DIR=/tmp/modssc_cache/datasets
export MODSSC_PREPROCESS_CACHE_DIR=/tmp/modssc_cache/preprocess
export MODSSC_GRAPH_CACHE_DIR=/tmp/modssc_cache/graph
```

If one of these placeholders is missing at runtime, config loading fails fast with an explicit error.

You can also set one global cache root for runtime caches:

```bash
export MODSSC_CACHE_ROOT=/tmp/modssc_cache
```

Optional graph-view override:

```bash
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

For a local or scheduler task that must execute exactly one declared seed, use
its zero-based position:

```bash
modssc-bench --config CARD.yaml --seed-index 0
```

`--seed-index`, `--seed`, and `--num-runs` are mutually exclusive. A Slurm
array can pass `$SLURM_ARRAY_TASK_ID` directly as the seed index.


## Paper replication without external source code
A supported reproduction uses the same runner with a frozen card:

```bash
modssc datasets download --dataset DATASET_ID --cache-dir "$MODSSC_DATASET_CACHE_DIR"
modssc-bench --config bench/configs/reproductions/METHOD/PROTOCOL.yaml
```

The checkout contains the ModSSC strategy implementation and declarative
protocol metadata. It does not contain a paper-specific runtime bundle and must
not clone or import the authors' code, invoke an external JAR, rely on Weka
installed outside the project, or require a manually seeded dataset cache.
Datasets are downloaded or materialized through ModSSC providers using the
identity recorded by the protocol. Dataset preparation and fail-closed
execution are separate steps; the runner verifies the prepared identity.
Optional Python backends are installed through declared project extras.

A private Slurm wrapper calls this exact interface. Scheduler resources and
paths may change, but the scientific config may not.

The 20 active article-reproduction cards carry an `acceptance` block in their
own YAML. Five bounded canary cards live separately under
`bench/configs/diagnostics/paper_canaries/`; they are diagnostic-only and never
enter paper seed totals or acceptance aggregation. An acceptance card must use
`run.benchmark_mode: true`,
bind `acceptance.method_id` to `method.id`, and declare exactly as many
`run.seeds` as `acceptance.repetitions`. This keeps protocol parameters,
scientific conformity evidence, numerical targets, diagnostics, deviations,
unknowns, and the fidelity ceiling together. There is no separate acceptance
registry or campaign-specific evaluator.


## Cache behavior in multi-seed runs
For multi-seed sweeps, keep one shared cache root and let fingerprints isolate seed-dependent artifacts:

```bash
export MODSSC_CACHE_ROOT=/tmp/modssc_cache
```

Use separate cache roots only when you need hard isolation, for example a strict clean-room comparison across branches or commits.

Only stages whose effective inputs depend on the seed receive distinct cache
keys. Dataset artifacts normally remain shared; preprocessing or graph entries
may also be shared when their exact inputs and full producer identity are the
same.

- Dataset cache (`datasets/`): reused across seeds when dataset identity is unchanged
- Sampling split cache (`splits/`): conceptually seed-dependent, but current bench orchestration computes sampling in memory and does not persist split cache entries
- Preprocess cache (`preprocess/`): schema v2 keys commit to exact in-memory dataset content, resolved plan, fit indices, seed, preprocessing source, Python, and selected distribution versions
- Graph cache (`graph/`): schema v2 keys commit to the exact feature bytes, graph spec, resolved backend, seed, graph source, and selected backend versions
- Graph-view cache (`graph_views/`): commits to exact graph content, feature content, view spec, seed, and producer identity
- Method training and inference: not cached by the benchmark runner

Preprocessing entries are published as immutable generations through an atomic
pointer. Graph and graph-view entries are staged, fully authenticated, and then
published atomically. Every load verifies the manifest, file SHA-256, logical
array dtype/shape, and content commitment. Legacy or damaged entries are cache
misses and are rebuilt; a component configured with `require_cache_hit: true`
fails instead of rebuilding.

Practical rules:
- keep the same cache root for speed when rerunning the same experiment
- use a new cache root for strict from-scratch comparisons
- expect recomputation when dataset options, preprocess steps or params, `fit_on`, graph spec, or seeds change
- expect automatic cache invalidation when relevant implementation code, backend, Python, or dependency versions change; manual purging is not required for correctness


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
- `acceptance`: either `null` for a card without a numerical acceptance
  contract, or the canonical native acceptance report

The native report distinguishes `assessment_status` (`passed`, `failed`, or
`not_evaluable`) from `fidelity_status` (`paper_matched`, `paper_approx`, or
`not_claimable`). It includes target and diagnostic details, reasons, the
declared fidelity ceiling, and `acceptance_sha256`, computed over the canonical
report payload. Missing or non-successful repetitions and unresolved scientific
conformity stay `not_evaluable`; they are not silently treated as a failed
numerical replication. A complete evaluated cohort can fail its gates, and a
passing numerical assessment can still be capped at `paper_approx`.

Reconciliation and acceptance mathematics come from `modssc.evaluation`.
[`bench/orchestrators/reporting.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/reporting.py)
only validates reports, calls those APIs, and writes the resulting JSON.
<sup class="cite"><a href="#source-4">[4]</a><a href="#source-9">[9]</a></sup>


## Common mistakes
- Enabling downloads inside a frozen article task instead of preparing and
  authenticating the dataset before execution.
- Treating a standardized `best/` config as an article reproduction protocol.
- Installing an author's repository or external Weka JAR to make a protocol
  pass instead of implementing the required strategy through ModSSC.
- Preparing a dataset cache manually instead of using the provider and identity
  declared by the protocol.
- Forgetting to export the cache and output variables declared by a card.
- Reusing old caches during a “from scratch” comparison and then attributing the speedup or output drift to the method itself.
- Expecting the runner to persist split cache entries during bench orchestration the same way the standalone sampling CLI can.
- Treating `run.json` as the only artifact in multi-seed mode and overlooking `aggregate.json`.
- Treating `assessment_status: passed` as automatically `paper_matched`, or
  treating `not_evaluable` as a numerical failure.


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
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/acceptance.py"><code>src/modssc/evaluation/acceptance.py</code></a></li>
</ol>
</details>
