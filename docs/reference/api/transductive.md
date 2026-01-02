# Transductive API

This page documents the transductive API. For a full run, see the [transductive tutorial](../../tutorials/transductive-toy.md).


## What it is for
The transductive brick provides registries and utilities for graph-based SSL methods. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
Run label propagation on a small graph:

```python
import numpy as np
from modssc.graph import GraphBuilderSpec, build_graph
from modssc.graph.artifacts import NodeDataset
from modssc.transductive import get_method_class
from modssc.transductive.methods.classic.label_propagation import LabelPropagationSpec

X = np.random.randn(10, 4).astype(np.float32)
G = build_graph(X, spec=GraphBuilderSpec(scheme="knn", metric="cosine", k=3), seed=0, cache=False)
train_mask = np.zeros((10,), dtype=bool)
train_mask[:2] = True

node_ds = NodeDataset(X=X, y=np.zeros((10,), dtype=np.int64), graph=G, masks={"train_mask": train_mask})
method = get_method_class("label_propagation")(spec=LabelPropagationSpec())
method.fit(node_ds)
proba = method.predict_proba(node_ds)
print(proba.shape)
```

List available method IDs:

```python
from modssc.transductive import available_methods

print(available_methods())
```

The method registry is defined in [`src/modssc/transductive/registry.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/registry.py). <sup class="cite"><a href="#source-1">[1]</a></sup>


## API reference

::: modssc.transductive

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/registry.py"><code>src/modssc/transductive/registry.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/base.py"><code>src/modssc/transductive/base.py</code></a></li>
</ol>
</details>
