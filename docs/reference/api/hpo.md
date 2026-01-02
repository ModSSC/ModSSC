# HPO API

This page documents HPO APIs. For bench search workflows, see [HPO how-to](../../how-to/hpo.md).


## What it is for
The HPO utilities provide search space definitions and sampling for grid or random search. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
Grid search over a parameter space:

```python
from modssc.hpo import Space

space = Space.from_dict({"method": {"params": {"max_iter": [5, 10, 20]}}})
for trial in space.iter_grid():
    print(trial.index, trial.params)
```

Random search with distributions:

```python
from modssc.hpo import Space

space = Space.from_dict({"method": {"params": {"confidence_threshold": {"dist": "uniform", "low": 0.7, "high": 0.95}}}})
for trial in space.iter_random(seed=0, n_trials=2):
    print(trial.params)
```

Distribution validation is implemented in [`src/modssc/hpo/samplers.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/hpo/samplers.py). <sup class="cite"><a href="#source-2">[2]</a></sup>


## API reference

::: modssc.hpo

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/hpo/space.py"><code>src/modssc/hpo/space.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/hpo/samplers.py"><code>src/modssc/hpo/samplers.py</code></a></li>
</ol>
</details>
