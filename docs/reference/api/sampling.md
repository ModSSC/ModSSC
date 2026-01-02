# Sampling API

This page documents the sampling API. For workflows, see [Sampling how-to](../../how-to/sampling.md).


## What it is for
The sampling brick builds deterministic labeled/unlabeled splits and stores them on disk. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
Create a sampling plan and sample a dataset:

```python
from modssc.data_loader import load_dataset
from modssc.sampling import HoldoutSplitSpec, LabelingSpec, SamplingPlan, sample

ds = load_dataset("toy", download=True)
plan = SamplingPlan(split=HoldoutSplitSpec(test_fraction=0.0, val_fraction=0.2), labeling=LabelingSpec())
res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint=str(ds.meta["dataset_fingerprint"]))
print(res.stats)
```

Save and load a split:

```python
from modssc.sampling import load_split, save_split

out_dir = save_split(res, out_dir="splits/toy", overwrite=True)
loaded = load_split(out_dir)
print(loaded.split_fingerprint)
```

Plan and storage helpers are defined in [`src/modssc/sampling/plan.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py) and [`src/modssc/sampling/storage.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/storage.py). <sup class="cite"><a href="#source-3">[3]</a><a href="#source-2">[2]</a></sup>


## API reference

::: modssc.sampling

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/api.py"><code>src/modssc/sampling/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/storage.py"><code>src/modssc/sampling/storage.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
</ol>
</details>
