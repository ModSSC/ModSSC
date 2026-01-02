# How to run hyperparameter search

This guide focuses on the benchmark runner search workflow so you can sweep method parameters in a controlled way. It keeps the rest of the experiment fixed and points back to the config schema when you need field definitions. For config fields and schema details, see the [Configuration reference](../reference/configuration.md).


## Problem statement
You want to run grid or random search over method parameters using the benchmark runner. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup> The search space only touches `method.params.*`, so data, sampling, and preprocessing stay fixed while you compare trials. <sup class="cite"><a href="#source-1">[1]</a></sup>


## When to use
Use this to tune `method.params.*` for an inductive or transductive method while keeping the rest of the experiment fixed. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Steps
1) Create a bench config with a `search` block. <sup class="cite"><a href="#source-1">[1]</a></sup>

2) Run the benchmark runner with that config. <sup class="cite"><a href="#source-3">[3]</a></sup>

3) Inspect `runs/<run>/hpo/trials.jsonl` and the best patch in `run.json`. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-4">[4]</a></sup>


## Copy-paste example
Use the CLI command when you want to execute a full search from a YAML config. Use the Python example when you want to build or inspect a search space in code. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-5">[5]</a><a href="#source-6">[6]</a></sup>

CLI:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive_hpo.yaml
```

Python:

```python
from modssc.hpo import Space

space = Space.from_dict({"method": {"params": {"max_iter": [5, 10], "confidence_threshold": [0.7, 0.9]}}})
for trial in space.iter_grid():
    print(trial.index, trial.params)
```

The full HPO example config is in [`bench/configs/experiments/toy_inductive_hpo.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive_hpo.yaml). The space primitives are in [`src/modssc/hpo/space.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/hpo/space.py). <sup class="cite"><a href="#source-5">[5]</a><a href="#source-6">[6]</a></sup>


## Pitfalls
!!! warning
    The bench schema restricts `search.space` to `method.params.*` paths in v1. <sup class="cite"><a href="#source-1">[1]</a></sup>


!!! tip
    Random search requires both `search.seed` and `search.n_trials`. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Related links
- [Configuration reference](../reference/configuration.md)
- [Benchmarks](../reference/benchmarks.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/hpo.py"><code>bench/orchestrators/hpo.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/main.py"><code>bench/main.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/reporting.py"><code>bench/orchestrators/reporting.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive_hpo.yaml"><code>bench/configs/experiments/toy_inductive_hpo.yaml</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/hpo/space.py"><code>src/modssc/hpo/space.py</code></a></li>
</ol>
</details>
