# Configuration reference

Use this reference when you need the structure of benchmark YAML files or brick-level plans. For runnable examples and execution advice, continue with [Benchmarks](benchmarks.md) and the [Bench config cookbook](../how-to/bench-cookbook.md).


## What it is for
ModSSC uses YAML configs for the benchmark runner and structured plan/spec files for sampling, preprocess, views, graph construction, augmentation, and HPO. These files are validated before execution, so this page is the place to confirm field names, top-level blocks, and the behavior of advanced switches. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-3">[3]</a><a href="#source-5">[5]</a><a href="#source-6">[6]</a><a href="#source-7">[7]</a></sup>


## When to use
- Use this page when editing a benchmark YAML and you need to confirm the allowed shape of a block.
- Use this page when debugging config-loading failures, unresolved environment variables, or schema errors.
- Use the how-to guides instead when you want workflow advice rather than field-level reference.


## Minimal examples
Run a benchmark config:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
```

Run a config that uses environment placeholders:

```bash
MODSSC_OUTPUT_DIR=/tmp/modssc_runs \
MODSSC_DATASET_CACHE_DIR=/tmp/modssc_cache/datasets \
MODSSC_PREPROCESS_CACHE_DIR=/tmp/modssc_cache/preprocess \
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
```

Load and inspect a config in Python:

```python
from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml

cfg = ExperimentConfig.from_dict(load_yaml("bench/configs/experiments/toy_inductive.yaml"))
print(cfg.method.kind, cfg.method.method_id)
```

The bench entry point, schema, and YAML loader are implemented in [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py), [`bench/schema.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py), and [`bench/utils/io.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/utils/io.py). <sup class="cite"><a href="#source-8">[8]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>


## Where configs live
- Authored benchmark examples and templates: [`bench/configs/experiments/`](https://github.com/ModSSC/ModSSC/tree/main/bench/configs/experiments). <sup class="cite"><a href="#source-1">[1]</a></sup>
- Curated benchmark suites and command listings: [`bench/configs/best/`](https://github.com/ModSSC/ModSSC/tree/main/bench/configs/best).
- Cluster launchers and Jean Zay deployment helpers: [`bench/slurm/`](https://github.com/ModSSC/ModSSC/tree/main/bench/slurm).
- Brick-level plan files can also live outside `bench/`, but they must still satisfy the same schema objects described below.


## How configs are loaded
- Bench configs are read from YAML with `bench.utils.io.load_yaml` and validated via `ExperimentConfig.from_dict`. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-2">[2]</a></sup>
- Environment variables are expanded inside string values, so placeholders like `${MODSSC_OUTPUT_DIR}` or `$MODSSC_OUTPUT_DIR` are valid in YAML. Unresolved placeholders fail fast with an explicit error instead of being left as literal strings. <sup class="cite"><a href="#source-3">[3]</a></sup>
- `method.model.factory` is an advanced extension hook for trusted configs only. It is rejected unless `run.allow_custom_factories: true` is set explicitly. <sup class="cite"><a href="#source-2">[2]</a></sup>
- Runtime cache environment variables outside YAML placeholders are `MODSSC_CACHE_ROOT`, `MODSSC_CACHE_DIR`, `MODSSC_PREPROCESS_CACHE_DIR`, `MODSSC_SPLIT_CACHE_DIR`, `MODSSC_GRAPH_CACHE_DIR`, and `MODSSC_GRAPH_VIEWS_CACHE_DIR`.
- CLI plan/spec files use `load_yaml_or_json` and their own `from_dict` validators. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a><a href="#source-6">[6]</a></sup>


## Config blocks at a glance

| Block | What it controls | Required core fields |
| --- | --- | --- |
| `run` | run identity, output, logging, sweep behavior | `name`, `seed`, `output_dir`, `fail_fast` |
| `dataset` | dataset selection and provider options | `id` |
| `sampling` | split and labeling plan | `seed`, `plan` |
| `preprocess` | feature pipeline before training | `seed`, `fit_on`, `cache`, `plan` |
| `method` | inductive or transductive method selection | `kind`, `id`, `device`, `params` |
| `evaluation` | metrics and report splits | `report_splits`, `metrics` |
| `graph` | graph construction for transductive workflows | optional block |
| `views` | multi-view feature generation | optional block |
| `augmentation` | weak/strong training-time augmentation | optional block |
| `search` | HPO search procedure and space | optional block |


## Experiment config schema (bench)
Top-level keys and their required fields are defined in [`bench/schema.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py):

- `run`: `name`, `seed`, `output_dir`, `fail_fast`, optional `log_level`, optional `seeds`, optional `benchmark_mode`, optional `allow_custom_factories`. <sup class="cite"><a href="#source-2">[2]</a></sup>
- `dataset`: `id`, optional `options`, `download`, `cache_dir`. <sup class="cite"><a href="#source-2">[2]</a></sup>
- `sampling`: `seed`, `plan`. <sup class="cite"><a href="#source-2">[2]</a></sup>
- `preprocess`: `seed`, `fit_on`, `cache`, `plan`. <sup class="cite"><a href="#source-2">[2]</a></sup>
- `method`: `kind` (`inductive` or `transductive`), `id`, `device`, `params`, optional `model`. `model` accepts classifier settings or an advanced `factory` + `params` pair when `run.allow_custom_factories=true`. <sup class="cite"><a href="#source-2">[2]</a></sup>
- `evaluation`: `report_splits`, `metrics`, optional `split_for_model_selection`. <sup class="cite"><a href="#source-2">[2]</a></sup>
- Optional blocks: `graph`, `views`, `augmentation`, `search`. <sup class="cite"><a href="#source-2">[2]</a></sup>


## Sampling plan schema
Sampling plans are validated by `SamplingPlan.from_dict` and include:
- `split`: `kind` (`holdout` or `kfold`) plus its parameters <sup class="cite"><a href="#source-9">[9]</a><a href="#source-10">[10]</a><a href="#source-11">[11]</a><a href="#source-12">[12]</a><a href="#source-13">[13]</a><a href="#source-14">[14]</a><a href="#source-5">[5]</a></sup>
- `labeling`: `mode` (`fraction`, `count`, `per_class`), `value`, `strategy`, `min_per_class`, optional `fixed_indices` <sup class="cite"><a href="#source-5">[5]</a></sup>
- `imbalance`: `kind` (`none`, `subsample_max_per_class`, `long_tail`) and its parameters <sup class="cite"><a href="#source-5">[5]</a></sup>
- `policy`: `respect_official_test`, `use_official_graph_masks`, `allow_override_official` <sup class="cite"><a href="#source-5">[5]</a></sup>

When `allow_override_official=true` on inductive datasets, provider test partitions are ignored and the user split is applied to `dataset.train`. See the [glossary](../getting-started/glossary.md) if these policy names are still unfamiliar.


## Preprocess plan schema
A preprocess plan YAML contains:
- `output_key` <sup class="cite"><a href="#source-15">[15]</a><a href="#source-7">[7]</a></sup>
- `steps`: list of step mappings with `id`, `params`, optional `modalities`, `requires_fields`, `enabled` <sup class="cite"><a href="#source-7">[7]</a></sup>

Available step IDs and their metadata are listed in the built-in catalog. <sup class="cite"><a href="#source-16">[16]</a></sup>


## Views plan schema
Views plans define multiple feature views:
- `views`: list of view definitions with `name`, optional `preprocess`, `columns`, and `meta` <sup class="cite"><a href="#source-17">[17]</a></sup>
- `columns` supports `all`, `indices`, `random`, and `complement` <sup class="cite"><a href="#source-17">[17]</a></sup>


## Graph builder spec schema
Graph specs match `GraphBuilderSpec`:
- `scheme`: `knn`, `epsilon`, or `anchor`
- `metric`: `cosine` or `euclidean`
- `k` or `radius` depending on scheme
- `symmetrize`, `weights`, `normalize`, `self_loops`
- `backend`: `auto`, `numpy`, `sklearn`, or `faiss`
- anchor-specific and faiss-specific parameters

All validation rules are defined in [`src/modssc/graph/specs.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py). <sup class="cite"><a href="#source-6">[6]</a></sup>


## Augmentation plan schema
Augmentation plans include:
- `steps`: list of ops with `id` and `params` <sup class="cite"><a href="#source-18">[18]</a><a href="#source-19">[19]</a><a href="#source-2">[2]</a><a href="#source-20">[20]</a></sup>
- `modality`: optional modality hint

Augmentation ops are registered in [`src/modssc/data_augmentation/ops/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_augmentation/ops). <sup class="cite"><a href="#source-21">[21]</a><a href="#source-22">[22]</a></sup>


## Search (HPO) schema
The bench search block includes:
- `enabled`, `kind` (`grid` or `random`), `seed`, `n_trials`, `repeats`
- `objective`: `split`, `metric`, `direction`, `aggregate`
- `space`: nested mapping of `method.params.*` to lists or distributions

Validation rules are enforced by [`bench/schema.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py), and distributions are defined in [`src/modssc/hpo/samplers.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/hpo/samplers.py). <sup class="cite"><a href="#source-2">[2]</a><a href="#source-23">[23]</a></sup>


## Common mistakes
- Using a field name that exists in Python objects but not in YAML. The schema names here are the contract for configs.
- Leaving `${VAR_NAME}` unresolved in a config and expecting the loader to ignore it.
- Treating `method.model.factory` as a normal extension point for shared configs. It is intentionally gated behind `run.allow_custom_factories: true`.
- Forgetting that `fit_on` changes the fitted preprocess state and therefore the cache fingerprint.
- Mixing “official test split” semantics with “custom split” semantics without checking the sampling policy flags.


## Complete example configs
Toy inductive experiment: <sup class="cite"><a href="#source-24">[24]</a></sup>

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

Toy transductive experiment: <sup class="cite"><a href="#source-25">[25]</a></sup>

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

Example with augmentation: <sup class="cite"><a href="#source-26">[26]</a></sup>

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


## Related links
- [Benchmarks](benchmarks.md)
- [Bench config cookbook](../how-to/bench-cookbook.md)
- [Glossary](../getting-started/glossary.md)
- [Common errors and where to go](../how-to/common-errors.md)
- [Troubleshooting](../how-to/troubleshooting.md)

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
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-12"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-13"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-14"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-15"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/plan.py"><code>src/modssc/preprocess/plan.py</code></a></li>
  <li id="source-16"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/catalog.py"><code>src/modssc/preprocess/catalog.py</code></a></li>
  <li id="source-17"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/plan.py"><code>src/modssc/views/plan.py</code></a></li>
  <li id="source-18"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/plan.py"><code>src/modssc/data_augmentation/plan.py</code></a></li>
  <li id="source-19"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/registry.py"><code>src/modssc/data_augmentation/registry.py</code></a></li>
  <li id="source-20"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_augmentation/ops"><code>src/modssc/data_augmentation/ops/</code></a></li>
  <li id="source-21"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/registry.py"><code>src/modssc/data_augmentation/registry.py</code></a></li>
  <li id="source-22"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_augmentation/ops"><code>src/modssc/data_augmentation/ops/</code></a></li>
  <li id="source-23"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/hpo/samplers.py"><code>src/modssc/hpo/samplers.py</code></a></li>
  <li id="source-24"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml"><code>bench/configs/experiments/toy_inductive.yaml</code></a></li>
  <li id="source-25"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_transductive.yaml"><code>bench/configs/experiments/toy_transductive.yaml</code></a></li>
  <li id="source-26"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/inductive/pseudo_label/text/imdb.yaml"><code>bench/configs/experiments/minimal/inductive/pseudo_label/text/imdb.yaml</code></a></li>
</ol>
</details>
