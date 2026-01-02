# How to compute evaluation metrics

Use this recipe to list metrics and compute them from predictions or score matrices. It shows both CLI and Python options, with [Catalogs and registries](../reference/catalogs.md) linked for the full metric list.


## Problem statement
You need consistent accuracy and F1 metrics for model outputs. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup> Use the metric IDs from this page in configs or scripts so results are comparable.


## When to use
Use these helpers when scoring predictions produced by inductive or transductive methods. <sup class="cite"><a href="#source-3">[3]</a></sup>


## Steps
1) Inspect available metrics. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>

2) Compute metrics from label arrays or score matrices. <sup class="cite"><a href="#source-1">[1]</a></sup>

3) (Optional) Use the CLI for quick checks on `.npy` files. <sup class="cite"><a href="#source-2">[2]</a></sup>

For the full list of metric IDs, see [Catalogs and registries](../reference/catalogs.md).


## Copy-paste example
Use the CLI when you want file-based evaluation from `.npy` arrays, and use the Python API when you already have arrays in memory. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-1">[1]</a></sup>

CLI:

```bash
modssc evaluation list
modssc evaluation compute --y-true y_true.npy --y-pred y_pred.npy --metric accuracy --metric macro_f1
```

Python:

```python
import numpy as np
from modssc.evaluation import evaluate

y_true = np.array([0, 1, 2, 1, 0])
scores = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.3, 0.5],
    [0.2, 0.6, 0.2],
    [0.6, 0.3, 0.1],
])
print(evaluate(y_true, scores, ["accuracy", "macro_f1"]))
```

## Pitfalls
!!! warning
    `modssc evaluation compute` only accepts `.npy` files and will reject other formats. <sup class="cite"><a href="#source-2">[2]</a></sup>


!!! tip
    `evaluate` accepts one-hot labels or class scores; it converts them internally. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Related links
- [CLI reference](../reference/cli.md)
- [API reference: evaluation](../reference/api/evaluation.md)
- [Catalogs and registries](../reference/catalogs.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/metrics.py"><code>src/modssc/evaluation/metrics.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/evaluation.py"><code>src/modssc/cli/evaluation.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/__init__.py"><code>src/modssc/evaluation/__init__.py</code></a></li>
</ol>
</details>
