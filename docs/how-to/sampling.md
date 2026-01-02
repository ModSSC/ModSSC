# How to create and reuse sampling splits

Here you will build reproducible SSL splits that you can reuse across methods and experiments. The steps map to the sampling plan fields and the examples show both CLI and Python forms. For end-to-end usage, see the [inductive tutorial](../tutorials/inductive-toy.md) or [transductive tutorial](../tutorials/transductive-toy.md).


## Problem statement
You need a deterministic train/val/test split with labeled vs unlabeled partitions, and you want to reuse it across runs. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup> Saving the split keeps experiments comparable when you switch methods or rerun trials.


## When to use
Use this when you want reproducible SSL splits across methods or when sharing splits with collaborators. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-3">[3]</a></sup>


## Steps
1) Define a sampling plan (holdout or k-fold, plus labeling strategy). <sup class="cite"><a href="#source-2">[2]</a></sup>

2) Create the split from a dataset (CLI or Python). <sup class="cite"><a href="#source-5">[5]</a><a href="#source-1">[1]</a></sup>

3) Save and load the split for reuse. <sup class="cite"><a href="#source-3">[3]</a></sup>

If you are running the bench runner, embed the plan in the experiment config described in the [configuration reference](../reference/configuration.md).


## Copy-paste example
Use the CLI when you want terminal-driven split artifacts (`modssc sampling` in [`src/modssc/cli/sampling.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/sampling.py)), and use Python when you want to generate splits inside a pipeline (API in [`src/modssc/sampling/api.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/api.py)). <sup class="cite"><a href="#source-5">[5]</a><a href="#source-1">[1]</a></sup>

Sample plan (from [`bench/configs/experiments/toy_inductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml)): <sup class="cite"><a href="#source-6">[6]</a></sup>


```yaml
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
```

CLI:

```bash
modssc sampling create --dataset toy --plan sampling_plan.yaml --seed 42 --out splits/toy
modssc sampling show splits/toy
modssc sampling validate splits/toy --dataset toy
```

Python:

```python
from modssc.data_loader import load_dataset
from modssc.sampling import HoldoutSplitSpec, LabelingSpec, SamplingPlan, sample, save_split

ds = load_dataset("toy", download=True)
plan = SamplingPlan(
    split=HoldoutSplitSpec(test_fraction=0.0, val_fraction=0.2, stratify=True),
    labeling=LabelingSpec(mode="fraction", value=0.2, strategy="balanced"),
)
res, _ = sample(ds, plan=plan, seed=42, dataset_fingerprint=str(ds.meta["dataset_fingerprint"]))
_ = save_split(res, out_dir="splits/toy", overwrite=True)
```

## Pitfalls
!!! warning
    `modssc sampling create` expects the dataset fingerprint in `dataset.meta["dataset_fingerprint"]`. This is injected by the data loader when datasets are cached. <sup class="cite"><a href="#source-7">[7]</a><a href="#source-1">[1]</a></sup>


!!! tip
    Split artifacts include `split.json` and `arrays.npz` in the output directory, which is what `load_split` expects. <sup class="cite"><a href="#source-3">[3]</a></sup>


## Related links
- [Configuration reference](../reference/configuration.md)
- [Preprocess how-to](preprocess.md)
- [Benchmarks](../reference/benchmarks.md)
- [Catalogs and registries](../reference/catalogs.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/api.py"><code>src/modssc/sampling/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/storage.py"><code>src/modssc/sampling/storage.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/fingerprint.py"><code>src/modssc/sampling/fingerprint.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/sampling.py"><code>src/modssc/cli/sampling.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml"><code>bench/configs/experiments/toy_inductive.yaml</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/api.py"><code>src/modssc/data_loader/api.py</code></a></li>
</ol>
</details>
