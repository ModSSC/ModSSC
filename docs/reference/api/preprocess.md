# Preprocess API

This page documents the preprocess API. For workflows, see [Preprocess how-to](../../how-to/preprocess.md).


## What it is for
The preprocess brick runs deterministic, cacheable transformation pipelines to produce feature matrices. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
Build and run a plan:

```python
from modssc.data_loader import load_dataset
from modssc.preprocess import PreprocessPlan, StepConfig, preprocess

ds = load_dataset("toy", download=True)
plan = PreprocessPlan(steps=(StepConfig(step_id="core.ensure_2d"), StepConfig(step_id="core.to_numpy")))
res = preprocess(ds, plan, seed=0, fit_indices=list(range(len(ds.train.y))))
print(res.preprocess_fingerprint)
```

Inspect available steps:

```python
from modssc.preprocess import available_steps, step_info

print(available_steps())
print(step_info("core.ensure_2d"))
```

The step catalog is defined in [`src/modssc/preprocess/catalog.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/catalog.py). <sup class="cite"><a href="#source-3">[3]</a></sup>


## API reference

::: modssc.preprocess

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/api.py"><code>src/modssc/preprocess/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/plan.py"><code>src/modssc/preprocess/plan.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/catalog.py"><code>src/modssc/preprocess/catalog.py</code></a></li>
</ol>
</details>
