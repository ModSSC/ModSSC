# Transductive tutorial: toy label propagation

This is an end-to-end transductive walkthrough with graph construction and label propagation. If you want to assemble it brick-by-brick, start with [datasets](../how-to/datasets.md), [sampling](../how-to/sampling.md), [preprocess](../how-to/preprocess.md), and [graph](../how-to/graph.md).


## Goal
Run a full transductive SSL experiment on the toy dataset using a graph construction spec and label propagation. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>

## Why this tutorial
Use this tutorial when your method expects a graph and node masks (`NodeDatasetLike`) and you plan to run a graph construction step. If you only need feature matrices without a graph, use the [inductive tutorial](inductive-toy.md) instead. <sup class="cite"><a href="#source-14">[14]</a></sup>

This walkthrough uses the bench runner because it validates a single YAML config and orchestrates dataset, sampling, preprocess, graph build, and method execution. For individual bricks, start with the [dataset](../how-to/datasets.md), [sampling](../how-to/sampling.md), [preprocess](../how-to/preprocess.md), and [graph](../how-to/graph.md) how-to guides instead. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-10">[10]</a></sup>


## Prerequisites
- Python 3.11+ with ModSSC installed from source (bench runner is in the repo). <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a></sup>

- No extra dependencies are required for the toy dataset and numpy graph backend. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-6">[6]</a></sup>


## Files used
- Benchmark entry point: [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py)
- Experiment config: [`bench/configs/experiments/toy_transductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_transductive.yaml)
- Graph spec schema: [`src/modssc/graph/specs.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py)

## Step by step commands
1) Install the repo in editable mode:

```bash
python -m pip install -e "."
```

2) Run the transductive toy experiment:

```bash
python -m bench.main --config bench/configs/experiments/toy_transductive.yaml
```

The benchmark runner and config paths are in [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py) and [`bench/configs/experiments/toy_transductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_transductive.yaml). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Full YAML config used
This is the full config file from [`bench/configs/experiments/toy_transductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_transductive.yaml):

```yaml
run:
  name: "toy_label_propagation_knn"
  seed: 7
  output_dir: "runs"
  fail_fast: true

dataset:
  id: "toy"

sampling:
  seed: 7
  plan:
    split:
      kind: "holdout"
      test_fraction: 0.0
      val_fraction: 0.2
      stratify: true
      shuffle: true
    labeling:
      mode: "fraction"
      value: 0.1
      strategy: "balanced"
      min_per_class: 1
    imbalance:
      kind: "none"
    policy:
      respect_official_test: true
      allow_override_official: false

preprocess:
  seed: 7
  fit_on: "train_labeled"
  cache: true
  plan:
    output_key: "features.X"
    steps:
      - id: "core.ensure_2d"
      - id: "core.to_numpy"

graph:
  enabled: true
  seed: 7
  cache: true
  spec:
    scheme: "knn"
    metric: "euclidean"
    k: 8
    symmetrize: "mutual"
    weights:
      kind: "heat"
      sigma: 1.0
    normalize: "rw"
    self_loops: true
    backend: "numpy"
    chunk_size: 128
    feature_field: "features.X"

method:
  kind: "transductive"
  id: "label_propagation"
  device:
    device: "auto"
    dtype: "float32"
  params:
    max_iter: 50
    tol: 1.0e-4
    normalize_rows: true

evaluation:
  report_splits: ["val", "test"]
  metrics: ["accuracy", "macro_f1"]
```

## Expected outputs and where they appear
A run directory is created under [`runs/`](https://github.com/ModSSC/ModSSC/tree/main/runs) with the config snapshot and the `run.json` summary. <sup class="cite"><a href="#source-7">[7]</a><a href="#source-8">[8]</a></sup>


Graph artifacts are cached when `graph.cache: true` is set. The cache layout is managed by `modssc.graph.cache.GraphCache`. <sup class="cite"><a href="#source-9">[9]</a><a href="#source-2">[2]</a></sup>


## How it works
- The bench runner validates the config and orchestrates dataset, sampling, preprocess, graph build, and method execution. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-10">[10]</a></sup>

- The graph is constructed using the `GraphBuilderSpec` fields in the config. <sup class="cite"><a href="#source-11">[11]</a><a href="#source-12">[12]</a></sup>

- Label propagation runs with hard clamping over the graph. <sup class="cite"><a href="#source-3">[3]</a></sup>


## Common pitfalls and troubleshooting
!!! warning
    Transductive methods require a graph; if `graph.enabled` is false and the dataset is not a graph dataset, the bench runner raises a config error. <sup class="cite"><a href="#source-1">[1]</a></sup>


!!! tip
    Use `modssc graph build --help` to see graph options and validate the spec. <sup class="cite"><a href="#source-13">[13]</a></sup>

## Related links
- [Concepts](../getting-started/concepts.md)
- [Inductive tutorial](inductive-toy.md)
- [Configuration reference](../reference/configuration.md)
- [Catalogs and registries](../reference/catalogs.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/main.py"><code>bench/main.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_transductive.yaml"><code>bench/configs/experiments/toy_transductive.yaml</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/methods/classic/label_propagation.py"><code>src/modssc/transductive/methods/classic/label_propagation.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/construction/backends/numpy_backend.py"><code>src/modssc/graph/construction/backends/numpy_backend.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/context.py"><code>bench/context.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/reporting.py"><code>bench/orchestrators/reporting.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/cache.py"><code>src/modssc/graph/cache.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py"><code>src/modssc/graph/specs.py</code></a></li>
  <li id="source-12"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/construction/api.py"><code>src/modssc/graph/construction/api.py</code></a></li>
  <li id="source-13"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/graph.py"><code>src/modssc/cli/graph.py</code></a></li>
  <li id="source-14"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/base.py"><code>src/modssc/transductive/base.py</code></a></li>
</ol>
</details>
