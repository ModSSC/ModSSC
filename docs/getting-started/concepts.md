# Concepts

This page introduces the key ideas and vocabulary used throughout the docs. For runnable examples, go to the [inductive tutorial](../tutorials/inductive-toy.md) or [transductive tutorial](../tutorials/transductive-toy.md).


## Problem framing
ModSSC targets semi-supervised classification, where a small labeled set and a larger unlabeled set are used together. This framing is reflected in the inductive and transductive bricks and their datasets. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>


## Inductive vs transductive in this project
Inductive methods operate on feature matrices and labeled/unlabeled splits, without requiring a graph. The inductive brick lives in [`src/modssc/inductive/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc/inductive) and validates `InductiveDataset` inputs. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a><a href="#source-1">[1]</a></sup>


Transductive methods operate on a fixed graph over all nodes and accept `NodeDataset`-like objects with a graph and optional masks. Sampling outputs for graph datasets use masks like train/val/test/labeled/unlabeled. The transductive brick lives in [`src/modssc/transductive/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc/transductive), and graph utilities are in [`src/modssc/graph/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc/graph). <sup class="cite"><a href="#source-2">[2]</a><a href="#source-6">[6]</a><a href="#source-7">[7]</a><a href="#source-8">[8]</a></sup>


## Key abstractions in this codebase
- **Dataset catalog and providers:** curated dataset keys and provider URIs for downloading and caching. <sup class="cite"><a href="#source-9">[9]</a><a href="#source-10">[10]</a><a href="#source-11">[11]</a></sup>

- **Sampling plans:** deterministic split + labeling specs that produce reproducible indices/masks. <sup class="cite"><a href="#source-12">[12]</a><a href="#source-13">[13]</a></sup>

- **Preprocess plans:** ordered steps that transform raw datasets into feature representations. <sup class="cite"><a href="#source-14">[14]</a><a href="#source-15">[15]</a></sup>

- **Graph specs and views:** graph construction specs and view generation (attr/diffusion/struct). <sup class="cite"><a href="#source-16">[16]</a><a href="#source-17">[17]</a></sup>

- **View plans:** multi-view feature generation for methods like co-training. <sup class="cite"><a href="#source-18">[18]</a><a href="#source-19">[19]</a></sup>

- **Registries:** method registries for inductive and transductive algorithms. <sup class="cite"><a href="#source-20">[20]</a><a href="#source-21">[21]</a></sup>

- **Benchmark configs:** end-to-end experiment configuration for reproducible runs. <sup class="cite"><a href="#source-22">[22]</a><a href="#source-23">[23]</a></sup>


## Small illustrative examples
Inductive dataset payload (labeled + unlabeled):

```python
import numpy as np
from modssc.inductive import InductiveDataset

X_l = np.random.randn(10, 4)
y_l = np.random.randint(0, 3, size=(10,))
X_u = np.random.randn(50, 4)

payload = InductiveDataset(X_l=X_l, y_l=y_l, X_u=X_u)
```

Transductive dataset payload (graph + masks):

```python
import numpy as np
from modssc.graph import GraphBuilderSpec, build_graph
from modssc.graph.artifacts import NodeDataset

X = np.random.randn(20, 8).astype(np.float32)
edge_spec = GraphBuilderSpec(scheme="knn", metric="cosine", k=3)
graph = build_graph(X, spec=edge_spec, seed=0, cache=False)

train_mask = np.zeros((20,), dtype=bool)
train_mask[:3] = True
node_data = NodeDataset(X=X, y=np.zeros((20,), dtype=np.int64), graph=graph, masks={"train_mask": train_mask})
```

The inductive and transductive dataset types are defined in [`src/modssc/inductive/types.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/types.py) and [`src/modssc/graph/artifacts.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/artifacts.py), and graph construction is implemented in [`src/modssc/graph/construction/api.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/construction/api.py). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-6">[6]</a><a href="#source-24">[24]</a></sup>


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/types.py"><code>src/modssc/inductive/types.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/base.py"><code>src/modssc/transductive/base.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/README.md"><code>README.md</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/inductive"><code>src/modssc/inductive/</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/validation.py"><code>src/modssc/inductive/validation.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/artifacts.py"><code>src/modssc/graph/artifacts.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/result.py"><code>src/modssc/sampling/result.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/graph"><code>src/modssc/graph/</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_loader/catalog"><code>src/modssc/data_loader/catalog/</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_loader/providers"><code>src/modssc/data_loader/providers/</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/api.py"><code>src/modssc/data_loader/api.py</code></a></li>
  <li id="source-12"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-13"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/api.py"><code>src/modssc/sampling/api.py</code></a></li>
  <li id="source-14"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/plan.py"><code>src/modssc/preprocess/plan.py</code></a></li>
  <li id="source-15"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/catalog.py"><code>src/modssc/preprocess/catalog.py</code></a></li>
  <li id="source-16"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py"><code>src/modssc/graph/specs.py</code></a></li>
  <li id="source-17"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/featurization/api.py"><code>src/modssc/graph/featurization/api.py</code></a></li>
  <li id="source-18"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/plan.py"><code>src/modssc/views/plan.py</code></a></li>
  <li id="source-19"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/views/api.py"><code>src/modssc/views/api.py</code></a></li>
  <li id="source-20"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/registry.py"><code>src/modssc/inductive/registry.py</code></a></li>
  <li id="source-21"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/registry.py"><code>src/modssc/transductive/registry.py</code></a></li>
  <li id="source-22"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
  <li id="source-23"><a href="https://github.com/ModSSC/ModSSC/tree/main/bench/configs/experiments"><code>bench/configs/experiments/</code></a></li>
  <li id="source-24"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/construction/api.py"><code>src/modssc/graph/construction/api.py</code></a></li>
</ol>
</details>
