# Inductive API

This page documents the inductive API. For a full run, see the [inductive tutorial](../../tutorials/inductive-toy.md).


## What it is for
The inductive brick defines method registries and datasets for SSL methods that do not require a graph. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
Instantiate a method by ID:

```python
import numpy as np
from modssc.inductive import DeviceSpec, InductiveDataset, get_method_class
from modssc.inductive.methods.pseudo_label import PseudoLabelSpec

X_l = np.random.randn(5, 4)
y_l = np.array([0, 1, 0, 1, 0])
X_u = np.random.randn(20, 4)

spec = PseudoLabelSpec(classifier_id="knn", classifier_backend="numpy")
method = get_method_class("pseudo_label")(spec=spec)
method.fit(InductiveDataset(X_l=X_l, y_l=y_l, X_u=X_u), device=DeviceSpec(device="cpu"), seed=0)
```

List available method IDs:

```python
from modssc.inductive import available_methods

print(available_methods())
```

The registry and dataset types are in [`src/modssc/inductive/registry.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/registry.py) and [`src/modssc/inductive/types.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/types.py). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## API reference

::: modssc.inductive

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/registry.py"><code>src/modssc/inductive/registry.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/types.py"><code>src/modssc/inductive/types.py</code></a></li>
</ol>
</details>
