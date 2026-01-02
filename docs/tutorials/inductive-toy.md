# Inductive tutorial: toy pseudo-label run

This is an end-to-end inductive walkthrough using the toy benchmark config. If you prefer step-by-step brick workflows, use the [datasets](../how-to/datasets.md), [sampling](../how-to/sampling.md), [preprocess](../how-to/preprocess.md), and [evaluation](../how-to/evaluation.md) guides.


## Goal
Run a full inductive SSL experiment on the built-in toy dataset using the benchmark runner and a YAML config. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>

## Why this tutorial
Use this tutorial when your method consumes feature matrices and labeled/unlabeled splits (`InductiveDataset`) and you do not need an explicit graph. If your method expects a graph and node masks, use the [transductive tutorial](transductive-toy.md) instead. <sup class="cite"><a href="#source-19">[19]</a><a href="#source-20">[20]</a></sup>

This walkthrough uses the bench runner because it validates a single YAML config and orchestrates the full pipeline (dataset, sampling, preprocess, method, evaluation). For individual bricks, start with the [dataset](../how-to/datasets.md), [sampling](../how-to/sampling.md), [preprocess](../how-to/preprocess.md), and [evaluation](../how-to/evaluation.md) how-to guides instead. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-9">[9]</a></sup>


## Prerequisites
- Python 3.11+ with ModSSC installed from source (bench runner is in the repo). <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a></sup>

- No extra dependencies are required for the toy dataset and numpy backends used here. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-6">[6]</a></sup>


## Files used
- Benchmark entry point: [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py)
- Experiment config: [`bench/configs/experiments/toy_inductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml)
- Toy dataset definition: [`src/modssc/data_loader/catalog/toy.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/catalog/toy.py)

## Step by step commands
1) Install the repo in editable mode:

```bash
python -m pip install -e "."
```

2) Run the inductive toy experiment:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
```

The bench runner and the example config are in [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py) and [`bench/configs/experiments/toy_inductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Full YAML config used
This is the full config file from [`bench/configs/experiments/toy_inductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml):

```yaml
run:
  name: "toy_pseudo_label_numpy"
  seed: 42
  output_dir: "runs"
  fail_fast: true

dataset:
  id: "toy"

sampling:
  seed: 42
  plan:
    split:
      kind: "holdout"
      test_fraction: 0.0
      val_fraction: 0.2
      stratify: true
      shuffle: true
    labeling:
      mode: "fraction"
      value: 0.2
      strategy: "balanced"
      min_per_class: 1
    imbalance:
      kind: "none"
    policy:
      respect_official_test: true
      allow_override_official: false

preprocess:
  seed: 42
  fit_on: "train_labeled"
  cache: true
  plan:
    output_key: "features.X"
    steps:
      - id: "core.ensure_2d"
      - id: "core.to_numpy"

method:
  kind: "inductive"
  id: "pseudo_label"
  device:
    device: "auto"
    dtype: "float32"
  params:
    classifier_id: "knn"
    classifier_backend: "numpy"
    max_iter: 5
    confidence_threshold: 0.8

evaluation:
  split_for_model_selection: "val"
  report_splits: ["val", "test"]
  metrics: ["accuracy", "macro_f1"]
```

## Expected outputs and where they appear
A new run directory is created under [`runs/`](https://github.com/ModSSC/ModSSC/tree/main/runs) with:
- `config.yaml` (config snapshot)
- `run.json` (metrics and metadata)
- `error.txt` (if the run fails)

These outputs are written by the bench context and reporting orchestrator. <sup class="cite"><a href="#source-7">[7]</a><a href="#source-8">[8]</a></sup>


## How it works
- [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py) loads the YAML, validates it against the schema, and orchestrates each stage. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-9">[9]</a></sup>

- The toy dataset is loaded via the data loader and cached. <sup class="cite"><a href="#source-10">[10]</a><a href="#source-3">[3]</a></sup>

- Sampling produces labeled/unlabeled splits using the sampling plan. <sup class="cite"><a href="#source-11">[11]</a><a href="#source-12">[12]</a></sup>

- Preprocess steps convert raw features into 2D numpy arrays. <sup class="cite"><a href="#source-13">[13]</a><a href="#source-14">[14]</a><a href="#source-15">[15]</a></sup>

- The pseudo-label method runs with a numpy kNN classifier. <sup class="cite"><a href="#source-16">[16]</a><a href="#source-6">[6]</a></sup>


## Common pitfalls and troubleshooting
!!! warning
    If the run fails because [`runs/`](https://github.com/ModSSC/ModSSC/tree/main/runs) already contains a folder with the same name and timestamp collision, delete the old folder and rerun. The run directory is created with `exist_ok=False`. <sup class="cite"><a href="#source-7">[7]</a></sup>


!!! tip
    Use `modssc --log-level detailed` to increase logging detail if a stage fails. <sup class="cite"><a href="#source-17">[17]</a><a href="#source-18">[18]</a></sup>

## Related links
- [Concepts](../getting-started/concepts.md)
- [Transductive tutorial](transductive-toy.md)
- [Configuration reference](../reference/configuration.md)
- [Catalogs and registries](../reference/catalogs.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/main.py"><code>bench/main.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml"><code>bench/configs/experiments/toy_inductive.yaml</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/catalog/toy.py"><code>src/modssc/data_loader/catalog/toy.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/supervised/backends/numpy/knn.py"><code>src/modssc/supervised/backends/numpy/knn.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/context.py"><code>bench/context.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/reporting.py"><code>bench/orchestrators/reporting.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/api.py"><code>src/modssc/data_loader/api.py</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/api.py"><code>src/modssc/sampling/api.py</code></a></li>
  <li id="source-12"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-13"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/plan.py"><code>src/modssc/preprocess/plan.py</code></a></li>
  <li id="source-14"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/steps/core/ensure_2d.py"><code>src/modssc/preprocess/steps/core/ensure_2d.py</code></a></li>
  <li id="source-15"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/steps/core/to_numpy.py"><code>src/modssc/preprocess/steps/core/to_numpy.py</code></a></li>
  <li id="source-16"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/methods/pseudo_label.py"><code>src/modssc/inductive/methods/pseudo_label.py</code></a></li>
  <li id="source-17"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/logging.py"><code>src/modssc/logging.py</code></a></li>
  <li id="source-18"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/app.py"><code>src/modssc/cli/app.py</code></a></li>
  <li id="source-19"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/types.py"><code>src/modssc/inductive/types.py</code></a></li>
  <li id="source-20"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/base.py"><code>src/modssc/transductive/base.py</code></a></li>
</ol>
</details>
