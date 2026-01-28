# How to run preprocessing plans

Treat this as a recipe for building preprocessing plans and running them from the CLI or Python. The steps mirror the plan structure used in configs, and the examples show the same flow in code; for step and model IDs, see [Catalogs and registries](../reference/catalogs.md).


## Problem statement
You want to turn raw datasets into feature matrices using a deterministic, cacheable preprocessing plan. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup> Plans make it easy to reuse the same feature pipeline across experiments.


## When to use
Use preprocessing to standardize inputs, build embeddings, or convert raw data to numeric features before modeling. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-4">[4]</a></sup>


## Steps
1) Inspect available steps and their metadata. <sup class="cite"><a href="#source-5">[5]</a><a href="#source-6">[6]</a></sup>

2) Write a preprocessing plan (YAML). <sup class="cite"><a href="#source-2">[2]</a></sup>

3) Run the plan with the CLI or the Python API. <sup class="cite"><a href="#source-5">[5]</a><a href="#source-1">[1]</a></sup>

For the full list of step IDs and pretrained model IDs, see [Catalogs and registries](../reference/catalogs.md).

Common step IDs (examples):
- `tabular.standard_scaler` (extra: `preprocess-sklearn`)
- `tabular.impute` (extra: `preprocess-sklearn`)
- `text.vocab_tokenizer`
- `audio.log_mel_spectrogram` (extra: `preprocess-audio`)
- `graph.sparse_adjacency`


## Copy-paste example
Use the CLI when you want to run plan files from the terminal (`modssc preprocess` in [`src/modssc/cli/preprocess.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/preprocess.py)), and use Python when you want to build and run plans in code (API in [`src/modssc/preprocess/api.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/api.py)). <sup class="cite"><a href="#source-5">[5]</a><a href="#source-1">[1]</a></sup>

Plan YAML (from [`bench/configs/experiments/toy_inductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml)): <sup class="cite"><a href="#source-7">[7]</a></sup>


```yaml
output_key: "features.X"
steps:
  - id: "core.ensure_2d"
  - id: "core.to_numpy"
```

CLI:

```bash
modssc preprocess steps list
modssc preprocess run --plan preprocess_plan.yaml --dataset toy --seed 0
```

Python:

```python
from modssc.data_loader import load_dataset
from modssc.preprocess import PreprocessPlan, StepConfig, preprocess

plan = PreprocessPlan(steps=(StepConfig(step_id="core.ensure_2d"), StepConfig(step_id="core.to_numpy")))
ds = load_dataset("toy", download=True)
result = preprocess(ds, plan, seed=0, fit_indices=list(range(len(ds.train.y))))
print(result.preprocess_fingerprint)
```

## Pitfalls
!!! warning
    Some steps require optional extras (for example, `text.tfidf`, `vision.openclip`). Use `modssc preprocess steps info <id>` or `step_info()` to check `required_extra`. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-6">[6]</a><a href="#source-5">[5]</a><a href="#source-8">[8]</a></sup>


!!! tip
    If your plan includes fittable steps (PCA, TF-IDF, ZCA), pass `fit_indices` in Python or set `fit_on` in [bench configs](../reference/configuration.md) to control the fit subset. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-9">[9]</a></sup>


## Related links
- [Configuration reference](../reference/configuration.md)
- [Data augmentation how-to](augmentation.md)
- [Graph how-to](graph.md)
- [Catalogs and registries](../reference/catalogs.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/api.py"><code>src/modssc/preprocess/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/plan.py"><code>src/modssc/preprocess/plan.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/catalog.py"><code>src/modssc/preprocess/catalog.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/preprocess/steps"><code>src/modssc/preprocess/steps/</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/preprocess.py"><code>src/modssc/cli/preprocess.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/registry.py"><code>src/modssc/preprocess/registry.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml"><code>bench/configs/experiments/toy_inductive.yaml</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/__init__.py"><code>src/modssc/preprocess/__init__.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
</ol>
</details>
