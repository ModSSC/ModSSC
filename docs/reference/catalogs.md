# Catalogs and registries

This reference collects registry-backed catalogs (datasets, steps, methods, metrics) and shows how to query them.


## What this page is
ModSSC exposes registry-backed lists for datasets, preprocess steps/models, augmentation ops, methods, and metrics through CLI commands and Python APIs. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a><a href="#source-4">[4]</a><a href="#source-5">[5]</a><a href="#source-6">[6]</a></sup>

Use this page when you need IDs for configs or when you want to check optional dependencies. Dataset specs and preprocess registries expose `required_extra`, and `modssc doctor` reports missing CLI bricks. <sup class="cite"><a href="#source-7">[7]</a><a href="#source-8">[8]</a><a href="#source-9">[9]</a><a href="#source-10">[10]</a></sup>

Use the CLI blocks for quick terminal inspection, and use the Python blocks when you want registry metadata inside a script.

## Datasets and providers
Use providers to understand which backends are available, and use dataset keys when wiring configs or CLI commands. Dataset info includes modality and `required_extra`. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-7">[7]</a><a href="#source-11">[11]</a></sup>

CLI: `modssc datasets` in [`src/modssc/cli/datasets.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/datasets.py). Python: data loader helpers in [`src/modssc/data_loader/api.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/api.py). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-11">[11]</a></sup>

CLI:

```bash
modssc datasets providers
modssc datasets list --modalities text
modssc datasets info --dataset toy
```

Python:

```python
from modssc.data_loader import available_datasets, dataset_info, provider_names

print(provider_names())
print(available_datasets())
print(dataset_info("toy").as_dict())
```

## Preprocess steps
Steps are registered in the preprocess catalog and surfaced through the CLI and registry helpers. Use `step_info` to check `required_extra` before you add a step to a plan. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-8">[8]</a><a href="#source-12">[12]</a></sup>

CLI: `modssc preprocess` in [`src/modssc/cli/preprocess.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/preprocess.py). Python: preprocess registry in [`src/modssc/preprocess/registry.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/registry.py). <sup class="cite"><a href="#source-2">[2]</a><a href="#source-12">[12]</a></sup>

CLI:

```bash
modssc preprocess steps list
modssc preprocess steps info core.ensure_2d
```

Python:

```python
from modssc.preprocess import available_steps, step_info

print(available_steps())
print(step_info("core.ensure_2d"))
```

## Pretrained models
Pretrained encoder models are listed by the preprocess model registry. Use the CLI for quick inspection or the Python helpers when you need the metadata in code. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-9">[9]</a></sup>

CLI: `modssc preprocess` in [`src/modssc/cli/preprocess.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/preprocess.py). Python: model registry in [`src/modssc/preprocess/models.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/models.py). <sup class="cite"><a href="#source-2">[2]</a><a href="#source-9">[9]</a></sup>

CLI:

```bash
modssc preprocess models list --modality text
modssc preprocess models info stub:text
```

Python:

```python
from modssc.preprocess import available_models, model_info

print(available_models(modality="text"))
print(model_info("stub:text"))
```

## Augmentation ops
Augmentation operations are registered in the augmentation registry and can be listed or inspected from the CLI. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-13">[13]</a></sup>

CLI: `modssc augmentation` in [`src/modssc/cli/augmentation.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/augmentation.py). Python: augmentation registry in [`src/modssc/data_augmentation/registry.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/registry.py). <sup class="cite"><a href="#source-3">[3]</a><a href="#source-13">[13]</a></sup>

CLI:

```bash
modssc augmentation list --modality text
modssc augmentation info text.word_dropout --as-json
```

Python:

```python
from modssc.data_augmentation.registry import available_ops, op_info

print(available_ops(modality="text"))
print(op_info("text.word_dropout"))
```

## Methods
Inductive and transductive registries expose method IDs. Use `--available-only` if you want to exclude planned or unresolvable methods. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a><a href="#source-14">[14]</a><a href="#source-15">[15]</a></sup>

CLI: inductive/transductive CLIs in [`src/modssc/cli/inductive.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/inductive.py) and [`src/modssc/cli/transductive.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/transductive.py). Python: registries in [`src/modssc/inductive/registry.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/registry.py) and [`src/modssc/transductive/registry.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/registry.py). <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a><a href="#source-14">[14]</a><a href="#source-15">[15]</a></sup>

CLI:

```bash
modssc inductive methods list --available-only
modssc transductive methods list --available-only
```

Python:

```python
from modssc.inductive import registry as inductive_registry
from modssc.transductive import registry as transductive_registry

print(inductive_registry.available_methods())
print(transductive_registry.available_methods())
```

## Evaluation metrics
Metric names are listed by the evaluation module and exposed in the CLI. <sup class="cite"><a href="#source-6">[6]</a><a href="#source-16">[16]</a></sup>

CLI: `modssc evaluation` in [`src/modssc/cli/evaluation.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/evaluation.py). Python: metric helpers in [`src/modssc/evaluation/metrics.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/metrics.py). <sup class="cite"><a href="#source-6">[6]</a><a href="#source-16">[16]</a></sup>

CLI:

```bash
modssc evaluation list
```

Python:

```python
from modssc.evaluation import list_metrics

print(list_metrics())
```

## Related links
- [Dataset how-to](../how-to/datasets.md)
- [Preprocess how-to](../how-to/preprocess.md)
- [Data augmentation how-to](../how-to/augmentation.md)
- [Evaluation how-to](../how-to/evaluation.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/datasets.py"><code>src/modssc/cli/datasets.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/preprocess.py"><code>src/modssc/cli/preprocess.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/augmentation.py"><code>src/modssc/cli/augmentation.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/inductive.py"><code>src/modssc/cli/inductive.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/transductive.py"><code>src/modssc/cli/transductive.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/evaluation.py"><code>src/modssc/cli/evaluation.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/types.py"><code>src/modssc/data_loader/types.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/catalog.py"><code>src/modssc/preprocess/catalog.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/models.py"><code>src/modssc/preprocess/models.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/app.py"><code>src/modssc/cli/app.py</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/api.py"><code>src/modssc/data_loader/api.py</code></a></li>
  <li id="source-12"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/registry.py"><code>src/modssc/preprocess/registry.py</code></a></li>
  <li id="source-13"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_augmentation/registry.py"><code>src/modssc/data_augmentation/registry.py</code></a></li>
  <li id="source-14"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/registry.py"><code>src/modssc/inductive/registry.py</code></a></li>
  <li id="source-15"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/registry.py"><code>src/modssc/transductive/registry.py</code></a></li>
  <li id="source-16"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/metrics.py"><code>src/modssc/evaluation/metrics.py</code></a></li>
</ol>
</details>
