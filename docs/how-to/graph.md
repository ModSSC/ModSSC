# How to build graphs and graph views

Use this recipe when you need a graph for transductive methods or want graph-derived feature views. The steps match the CLI commands and the Python snippet uses the same specs. For a full run, see the [transductive tutorial](../tutorials/transductive-toy.md).


## Problem statement
You want to build a similarity graph from feature vectors and optionally derive graph-based views (attribute, diffusion, structural). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup> Start with a simple kNN spec and refine it as you evaluate methods.


## When to use
Use this for [transductive methods](../tutorials/transductive-toy.md) that require a graph, or for graph-derived feature views. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-2">[2]</a></sup>


## Steps
1) Define a `GraphBuilderSpec` (scheme, metric, weights, backend). <sup class="cite"><a href="#source-1">[1]</a></sup>

2) Build the graph with CLI or Python. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a></sup>

3) (Optional) Generate views using `GraphFeaturizerSpec`. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-1">[1]</a></sup>


## Copy-paste example
Use the CLI when you want to build graphs and views from the terminal (`modssc graph` in [`src/modssc/cli/graph.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/graph.py)), and use Python when you want to embed graph construction in code (helpers in [`src/modssc/graph/construction/api.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/construction/api.py)). <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a></sup>

CLI (graph build):

```bash
modssc graph build --dataset toy --scheme knn --metric euclidean --k 8 --backend numpy
```

CLI (graph views):

```bash
modssc graph views build --dataset toy --views attr diffusion --diffusion-steps 5
```

Python:

```python
import numpy as np
from modssc.graph import GraphBuilderSpec, GraphFeaturizerSpec, build_graph, graph_to_views
from modssc.graph.artifacts import NodeDataset

X = np.random.randn(50, 8).astype(np.float32)

gspec = GraphBuilderSpec(scheme="knn", metric="cosine", k=5)
G = build_graph(X, spec=gspec, seed=0, cache=False)

fspec = GraphFeaturizerSpec(views=("attr", "diffusion"), diffusion_steps=3, diffusion_alpha=0.1)
node_ds = NodeDataset(X=X, y=np.zeros((50,), dtype=np.int64), graph=G, masks={})
views = graph_to_views(node_ds, spec=fspec, seed=0, cache=False)
print(list(views.views.keys()))
```

Graph CLI options and specs are defined in [`src/modssc/cli/graph.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/graph.py) and [`src/modssc/graph/specs.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py). <sup class="cite"><a href="#source-4">[4]</a><a href="#source-1">[1]</a></sup>


## Pitfalls
!!! warning
    `GraphBuilderSpec` validation is strict; unsupported combinations (for example, `backend=faiss` with `scheme=epsilon`) raise `GraphValidationError`. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-6">[6]</a></sup>


!!! tip
    The graph and views caches are managed by `GraphCache` and `ViewsCache`. Use the CLI cache commands to inspect or purge them. <sup class="cite"><a href="#source-7">[7]</a><a href="#source-4">[4]</a></sup>


## Related links
- [Configuration reference](../reference/configuration.md)
- [Transductive tutorial](../tutorials/transductive-toy.md)
- [Catalogs and registries](../reference/catalogs.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py"><code>src/modssc/graph/specs.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/featurization/api.py"><code>src/modssc/graph/featurization/api.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/registry.py"><code>src/modssc/transductive/registry.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/graph.py"><code>src/modssc/cli/graph.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/construction/api.py"><code>src/modssc/graph/construction/api.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/construction/builder.py"><code>src/modssc/graph/construction/builder.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/cache.py"><code>src/modssc/graph/cache.py</code></a></li>
</ol>
</details>
