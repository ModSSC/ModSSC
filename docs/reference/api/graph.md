# Graph API

This page documents the graph API. For workflows, see [Graph how-to](../../how-to/graph.md).


## What it is for
The graph brick constructs similarity graphs and derives graph-based feature views. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
Build a kNN graph:

```python
import numpy as np
from modssc.graph import GraphBuilderSpec, build_graph

X = np.random.randn(20, 8).astype(np.float32)
spec = GraphBuilderSpec(scheme="knn", metric="cosine", k=3)
G = build_graph(X, spec=spec, seed=0, cache=False)
print(G.n_nodes, G.n_edges)
```

Generate graph views:

```python
from modssc.graph import GraphFeaturizerSpec, graph_to_views
from modssc.graph.artifacts import NodeDataset

node_ds = NodeDataset(X=X, y=np.zeros((20,), dtype=np.int64), graph=G, masks={})
fspec = GraphFeaturizerSpec(views=("attr", "diffusion"))
views = graph_to_views(node_ds, spec=fspec, seed=0, cache=False)
print(list(views.views.keys()))
```

Specs and artifacts are defined in [`src/modssc/graph/specs.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py) and [`src/modssc/graph/artifacts.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/artifacts.py). <sup class="cite"><a href="#source-3">[3]</a><a href="#source-4">[4]</a></sup>


## API reference

::: modssc.graph

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/construction/api.py"><code>src/modssc/graph/construction/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/featurization/api.py"><code>src/modssc/graph/featurization/api.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py"><code>src/modssc/graph/specs.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/artifacts.py"><code>src/modssc/graph/artifacts.py</code></a></li>
</ol>
</details>
