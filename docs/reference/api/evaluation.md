# Evaluation API

This page documents the evaluation API. For workflows, see [Evaluation how-to](../../how-to/evaluation.md).


## What it is for
The evaluation brick provides metric implementations and helpers for labels or score matrices. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Examples
List metrics:

```python
from modssc.evaluation import list_metrics

print(list_metrics())
```

Evaluate accuracy and macro F1:

```python
import numpy as np
from modssc.evaluation import evaluate

y_true = np.array([0, 1, 1])
y_pred = np.array([0, 1, 0])
print(evaluate(y_true, y_pred, ["accuracy", "macro_f1"]))
```

Metrics are implemented in [`src/modssc/evaluation/metrics.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/metrics.py). <sup class="cite"><a href="#source-1">[1]</a></sup>


## API reference

::: modssc.evaluation

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/metrics.py"><code>src/modssc/evaluation/metrics.py</code></a></li>
</ol>
</details>
