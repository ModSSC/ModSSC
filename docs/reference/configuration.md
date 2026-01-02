# Configuration reference

This reference documents the YAML schemas used by the benchmark runner and CLI plans. For runnable configs, see [Benchmarks](benchmarks.md).

Start with where files live, then how they are loaded, then the schema blocks you can edit. The full examples at the end are complete configs you can copy and adapt.


## Where configs live
- Benchmark experiment configs: [`bench/configs/experiments/`](https://github.com/ModSSC/ModSSC/tree/main/bench/configs/experiments) (templates, toy configs, best/smoke configs). <sup class="cite"><a href="#source-1">[1]</a></sup>

- Graph specs and preprocessing plans are embedded inside experiment configs when enabled. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## How configs are loaded
- Bench configs are read from YAML with `bench.utils.io.load_yaml` and validated via `ExperimentConfig.from_dict`. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-2">[2]</a></sup>

- CLI plan/spec files use `load_yaml_or_json` and are validated with their respective `from_dict` methods. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a><a href="#source-6">[6]</a></sup>

- Preprocess plans are loaded from YAML with `modssc.preprocess.plan.load_plan`. <sup class="cite"><a href="#source-7">[7]</a></sup>


## Examples
Use the CLI run when you want to execute a config with the benchmark runner, and use the Python snippet when you want to load and inspect the config object in code. <sup class="cite"><a href="#source-8">[8]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>

Run a bench config:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
```

Load and inspect a config in Python:

```python
from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml

cfg = ExperimentConfig.from_dict(load_yaml("bench/configs/experiments/toy_inductive.yaml"))
print(cfg.method.kind, cfg.method.method_id)
```

The bench entry point, schema, and YAML loader are in [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py), [`bench/schema.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py), and [`bench/utils/io.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/utils/io.py). <sup class="cite"><a href="#source-8">[8]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>


## Experiment config schema (bench)
Top-level keys and their required fields are defined in [`bench/schema.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py):

- `run`: `name`, `seed`, `output_dir`, `fail_fast`, optional `log_level`. <sup class="cite"><a href="#source-2">[2]</a></sup>

- `dataset`: `id`, optional `options`, `download`, `cache_dir`. <sup class="cite"><a href="#source-2">[2]</a></sup>

- `sampling`: `seed`, `plan` (sampling plan mapping). <sup class="cite"><a href="#source-2">[2]</a></sup>

- `preprocess`: `seed`, `fit_on`, [`cache`](https://github.com/ModSSC/ModSSC/tree/main/cache), `plan` (preprocess plan mapping). <sup class="cite"><a href="#source-2">[2]</a></sup>

- `method`: `kind` (inductive|transductive), `id`, `device`, `params`, optional `model`. <sup class="cite"><a href="#source-2">[2]</a></sup>

- `evaluation`: `report_splits`, `metrics`, optional `split_for_model_selection`. <sup class="cite"><a href="#source-2">[2]</a></sup>

- Optional blocks: `graph`, `views`, `augmentation`, `search`. <sup class="cite"><a href="#source-2">[2]</a></sup>


## Sampling plan schema
Sampling plans are validated by `SamplingPlan.from_dict` and include:
- `split`: `kind` (holdout|kfold) plus parameters <sup class="cite"><a href="#source-9">[9]</a><a href="#source-10">[10]</a><a href="#source-11">[11]</a><a href="#source-12">[12]</a><a href="#source-13">[13]</a><a href="#source-14">[14]</a><a href="#source-5">[5]</a></sup>

- `labeling`: `mode` (fraction|count|per_class), `value`, `strategy`, `min_per_class`, optional `fixed_indices`. <sup class="cite"><a href="#source-5">[5]</a></sup>

- `imbalance`: `kind` (none|subsample_max_per_class|long_tail) and its parameters. <sup class="cite"><a href="#source-5">[5]</a></sup>

- `policy`: `respect_official_test`, `use_official_graph_masks`, `allow_override_official`. <sup class="cite"><a href="#source-5">[5]</a></sup>


## Preprocess plan schema
A preprocess plan YAML contains:
- `output_key` <sup class="cite"><a href="#source-15">[15]</a><a href="#source-7">[7]</a></sup>

- `steps`: list of step mappings with `id`, `params`, optional `modalities`, `requires_fields`, `enabled`. <sup class="cite"><a href="#source-7">[7]</a></sup>


Available step IDs and their metadata are listed in the built-in catalog. <sup class="cite"><a href="#source-16">[16]</a></sup>


## Views plan schema
Views plans define multiple feature views:
- `views`: list of view definitions with `name`, optional `preprocess`, `columns`, and `meta`. <sup class="cite"><a href="#source-17">[17]</a></sup>

- `columns` supports modes `all`, `indices`, `random`, `complement`. <sup class="cite"><a href="#source-17">[17]</a></sup>


## Graph builder spec schema
Graph specs match `GraphBuilderSpec`:
- `scheme`: knn|epsilon|anchor
- `metric`: cosine|euclidean
- `k` or `radius` depending on scheme
- `symmetrize`, `weights`, `normalize`, `self_loops`
- `backend`: auto|numpy|sklearn|faiss
- anchor and faiss parameters

All validation rules are defined in [`src/modssc/graph/specs.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py). <sup class="cite"><a href="#source-6">[6]</a></sup>


## Augmentation plan schema
Augmentation plans include:
- `steps`: list of ops with `id` and `params` <sup class="cite"><a href="#source-18">[18]</a><a href="#source-19">[19]</a><a href="#source-2">[2]</a><a href="#source-20">[20]</a></sup>

- `modality`: optional modality hint

Augmentation ops are registered in [`src/modssc/data_augmentation/ops/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_augmentation/ops). <sup class="cite"><a href="#source-21">[21]</a><a href="#source-22">[22]</a></sup>


## Search (HPO) schema
The bench search block includes:
- `enabled`, `kind` (grid|random), `seed`, `n_trials`, `repeats`.
- `objective`: `split`, `metric`, `direction`, `aggregate`.
- `space`: nested mapping of `method.params.*` to lists or distributions.

Validation rules are enforced by [`bench/schema.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py), and distributions are defined in [`src/modssc/hpo/samplers.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/hpo/samplers.py). <sup class="cite"><a href="#source-2">[2]</a><a href="#source-23">[23]</a></sup>


## Complete example configs
Toy inductive experiment : <sup class="cite"><a href="#source-24">[24]</a></sup>


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

Toy transductive experiment : <sup class="cite"><a href="#source-25">[25]</a></sup>


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

Example with augmentation : <sup class="cite"><a href="#source-26">[26]</a></sup>


```yaml
run:
  name: best_text_inductive_softmatch_ag_news
  seed: 2
  output_dir: runs/inductive/softmatch/text/ag_news
  log_level: detailed
  fail_fast: true
dataset:
  id: ag_news
  download: true
  options:
    text_column: text
    label_column: label
    prefer_test_split: true
sampling:
  seed: 2
  plan:
    split:
      kind: holdout
      test_fraction: 0.2
      val_fraction: 0.1
      stratify: true
      shuffle: true
    labeling:
      mode: fraction
      value: 0.2
      strategy: balanced
      min_per_class: 1
    imbalance:
      kind: none
    policy:
      respect_official_test: true
      use_official_graph_masks: true
      allow_override_official: false
preprocess:
  seed: 2
  fit_on: train_labeled
  cache: true
  plan:
    output_key: features.X
    steps:
    - id: labels.encode
    - id: text.ensure_strings
    - id: text.sentence_transformer
      params:
        batch_size: 64
    - id: core.pca
      params:
        n_components: 128
    - id: core.to_torch
      params:
        device: "auto"
        dtype: float32
augmentation:
  enabled: true
  seed: 2
  mode: fixed
  modality: tabular
  weak:
    steps:
    - id: tabular.gaussian_noise
      params:
        std: 0.01
  strong:
    steps:
    - id: tabular.feature_dropout
      params:
        p: 0.2
method:
  kind: inductive
  id: softmatch
  device:
    device: "auto"
    dtype: float32
  params:
    lambda_u: 1.0
    temperature: 0.5
    ema_p: 0.999
    n_sigma: 2.0
    per_class: false
    dist_align: true
    dist_uniform: true
    hard_label: true
    use_cat: false
    batch_size: 128
    max_epochs: 50
    detach_target: true
  model:
    classifier_id: mlp
    classifier_backend: torch
    classifier_params:
      hidden_sizes:
      - 128
      activation: relu
      dropout: 0.1
      lr: 0.001
      weight_decay: 0.0
      batch_size: 256
      max_epochs: 50
    ema: false
evaluation:
  split_for_model_selection: val
  report_splits:
  - val
  - test
  metrics:
  - accuracy
  - macro_f1
```

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/tree/main/bench/configs/experiments"><code>bench/configs/experiments/</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/utils/io.py"><code>bench/utils/io.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/_utils.py"><code>src/modssc/cli/_utils.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py"><code>src/modssc/graph/specs.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/plan.py"><code>src/modssc/preprocess/plan.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/main.py"><code>bench/main.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/tree/main/test_fraction"><code>test_fraction</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/tree/main/val_fraction"><code>val_fraction</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/tree/main/k"><code>k</code></a></li>
  <li id="source-12"><a href="https://github.com/ModSSC/ModSSC/tree/main/fold"><code>fold</code></a></li>
  <li id="source-13"><a href="https://github.com/ModSSC/ModSSC/tree/main/stratify"><code>stratify</code></a></li>
  <li id="source-14"><a href="https://github.com/ModSSC/ModSSC/tree/main/shuffle"><code>shuffle</code></a></li>
  <li id="source-15"><a href="https://github.com/ModSSC/ModSSC/blob/main/features.X"><code>features.X</code></a></li>
  <li id="source-16"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/catalog.py"><code>src/modssc/preprocess/catalog.py</code></a></li>
  <li id="source-17"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/plan.py"><code>src/modssc/views/plan.py</code></a></li>
  <li id="source-18"><a href="https://github.com/ModSSC/ModSSC/tree/main/id"><code>id</code></a></li>
  <li id="source-19"><a href="https://github.com/ModSSC/ModSSC/tree/main/op_id"><code>op_id</code></a></li>
  <li id="source-20"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/plan.py"><code>src/modssc/data_augmentation/plan.py</code></a></li>
  <li id="source-21"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/registry.py"><code>src/modssc/data_augmentation/registry.py</code></a></li>
  <li id="source-22"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_augmentation/ops"><code>src/modssc/data_augmentation/ops/</code></a></li>
  <li id="source-23"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/hpo/samplers.py"><code>src/modssc/hpo/samplers.py</code></a></li>
  <li id="source-24"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml"><code>bench/configs/experiments/toy_inductive.yaml</code></a></li>
  <li id="source-25"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_transductive.yaml"><code>bench/configs/experiments/toy_transductive.yaml</code></a></li>
  <li id="source-26"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/best/inductive/softmatch/text/ag_news.yaml"><code>bench/configs/experiments/best/inductive/softmatch/text/ag_news.yaml</code></a></li>
</ol>
</details>
