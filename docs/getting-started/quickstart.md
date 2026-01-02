# Quickstart

This quickstart gives the smallest runnable CLI and Python examples and tells you what output to expect. For background and terminology, see [Concepts](concepts.md).


## One minimal run
Use the benchmark runner when you want a full pipeline driven by a YAML config, and use the Python snippet when you want to exercise the sampling API directly in code. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a><a href="#source-4">[4]</a></sup>

Run the benchmark runner with the toy inductive config:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
```

Run a quick Python API pass with the dataset loader and sampling plan:

```python
from modssc.data_loader import load_dataset
from modssc.sampling import HoldoutSplitSpec, LabelingSpec, SamplingPlan, sample

ds = load_dataset("toy", download=True)
plan = SamplingPlan(
    split=HoldoutSplitSpec(test_fraction=0.0, val_fraction=0.2, stratify=True),
    labeling=LabelingSpec(mode="fraction", value=0.2, strategy="balanced"),
)
res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint=str(ds.meta["dataset_fingerprint"]))
print(res.stats)
```

The benchmark command and the toy config live in [`bench/main.py`](https://github.com/ModSSC/ModSSC/blob/main/bench/main.py) and [`bench/configs/experiments/toy_inductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml). The sampling API is defined in [`src/modssc/sampling/api.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/api.py) and [`src/modssc/sampling/plan.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a><a href="#source-4">[4]</a></sup>


## What you should see
For the benchmark run, a timestamped folder is created under [`runs/`](https://github.com/ModSSC/ModSSC/tree/main/runs) containing:
- `config.yaml` (copied config)
- `run.json` (metrics + metadata)
- `error.txt` (only if failed)

These outputs are written by the bench runner and context utilities. <sup class="cite"><a href="#source-5">[5]</a><a href="#source-6">[6]</a><a href="#source-7">[7]</a><a href="#source-8">[8]</a></sup>


For the Python snippet, you should see a stats dictionary printed to stdout. The stats structure is produced by `modssc.sampling.stats.build_inductive_stats`. <sup class="cite"><a href="#source-9">[9]</a></sup>


## Next steps
- [Inductive tutorial](../tutorials/inductive-toy.md)
- [Transductive tutorial](../tutorials/transductive-toy.md)
- [Sampling how-to](../how-to/sampling.md)
- [Preprocess how-to](../how-to/preprocess.md)

## Troubleshooting
!!! warning
    If a dataset provider is missing, the loader raises an optional dependency error with a suggested `pip install "modssc[extra]"` command. <sup class="cite"><a href="#source-10">[10]</a></sup>


!!! tip
    Use `modssc doctor` to see which CLI bricks are available and which extras are missing. <sup class="cite"><a href="#source-11">[11]</a></sup>


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/main.py"><code>bench/main.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml"><code>bench/configs/experiments/toy_inductive.yaml</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/api.py"><code>src/modssc/sampling/api.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/README.md"><code>README.md</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/context.py"><code>bench/context.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/reporting.py"><code>bench/orchestrators/reporting.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/stats.py"><code>src/modssc/sampling/stats.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/errors.py"><code>src/modssc/data_loader/errors.py</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/app.py"><code>src/modssc/cli/app.py</code></a></li>
</ol>
</details>
