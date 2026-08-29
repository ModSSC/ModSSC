# How to build graphs and views

Use these guidelines when you need a graph for transductive methods or want graph-derived feature views. The steps match the CLI commands and the Python snippet uses the same specifications. For a full run, see the [transductive tutorial](../tutorials/transductive-toy.md).


## Problem statement
You want to build a similarity graph from feature vectors and optionally derive graph-based views (attribute, diffusion, structural). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup> Start with a simple kNN specification and refine it as you evaluate methods.


## When to use
Use this for [transductive methods](../tutorials/transductive-toy.md) that require a graph, or for graph-derived feature views. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-2">[2]</a></sup>


## Steps
1) Define a `GraphBuilderSpec` (scheme, metric, weights, backend). <sup class="cite"><a href="#source-1">[1]</a></sup>

2) Build the graph with CLI or Python. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a></sup>

3) (Optional) Generate views using `GraphFeaturizerSpec`. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-1">[1]</a></sup>


## Copy-paste example
Use the CLI when you want to build graphs and views from the terminal (`modssc graph` in [`src/modssc/cli/graph.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/graph.py)), and use Python when you want to embed graph construction in code through the public `modssc.graph` package API. Graph construction and graph featurization are implemented in dedicated subpackages behind that facade. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a></sup>

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


## VAE and AET features for non-graph datasets
For graph-learning methods on non-graph datasets, keep the no-VAE baseline by building the graph from `features.X`:

```yaml
graph:
  enabled: true
  spec:
    scheme: knn
    metric: cosine
    k: 15
    feature_field: features.X
```

For the Poisson-learning MNIST VAE branch, use raw 28x28 pixels, flatten them to 784 dimensions, add `core.vae` with the paper preset, and point graph construction at `features.vae`:

```yaml
preprocess:
  fit_on: train
  plan:
    output_key: features.vae
    steps:
    - id: labels.encode
    - id: vision.ensure_num_channels
      params:
        num_channels: 1
    - id: vision.resize
      params:
        height: 28
        width: 28
    - id: core.ensure_2d
    - id: core.vae
      params:
        preset: poisson_mnist
        cache_key: poisson-mnist-raw784
        model_seed: 0
        device: auto
graph:
  enabled: true
  spec:
    scheme: knn
    metric: euclidean
    k: 10
    symmetrize: mean
    weights:
      kind: knn_gaussian
    normalize: none
    self_loops: false
    feature_field: features.vae
```

This keeps `features.X` available for the existing baseline while exposing the VAE latent means through `features.vae`.
VAE checkpoints are cached outside git under `modssc_cache/preprocess/vae_models/` by default, or under `${MODSSC_PREPROCESS_CACHE_DIR}/vae_models` when that environment variable is set.
Each benchmark run logs the VAE fingerprint, cache path, cache hit status, fit sample count, input hash, and `uses_labels: false` under `artifacts.preprocess.logged_artifacts["features.vae.info"]` in `run.json`; split-specific metadata is also available under `artifacts.preprocess.logged_artifacts.by_split`.
`preset: poisson_mnist` resolves to the Poisson-style VAE: global min-max scaling, `784 -> 400 -> 20 -> 400 -> 784`, sigmoid decoder, BCE+KL loss, Adam `lr=1e-3`, batch size 128, 100 epochs.
`preset: poisson_fashionmnist` uses the same settings with latent dimension 30.
The Poisson paper does not use this VAE for CIFAR-10; it uses an AutoEncodingTransformations embedding instead. <sup class="cite"><a href="#source-8">[8]</a></sup>
For the CIFAR-10 branch, prefer the precomputed AET artifact for Poisson-style graph experiments.
Store `cifar_aet.npz` and `cifar_labels.npz` outside git under `modssc_cache/preprocess/pretrained_features/aet/`.
The first run extracts the compressed `cifar_aet.npz` into `cifar_aet.npy`, then memory maps that `.npy` so subsequent runs only read the rows needed by the current train/test subset:

```yaml
preprocess:
  plan:
    output_key: features.aet
    steps:
    - id: labels.encode
    - id: vision.aet
      params:
        source: precomputed
        preset: poisson_cifar10_projective
        features_path: modssc_cache/preprocess/pretrained_features/aet/cifar_aet.npz
        labels_path: modssc_cache/preprocess/pretrained_features/aet/cifar_labels.npz
        unit_normalize: true
graph:
  enabled: true
  spec:
    scheme: knn
    metric: euclidean
    k: 10
    symmetrize: mean
    weights:
      kind: knn_gaussian
    normalize: none
    self_loops: false
    feature_field: features.aet
```

`knn_gaussian` uses a local-scale Gaussian kernel, `exp(-4*d_ij^2/d_k(x_i)^2)`, and `symmetrize: mean` applies `(W + W.T) / 2`.
Each benchmark run logs the AET artifact path/hash, row offset, unit-normalization flag, and `uses_labels: false` under `artifacts.preprocess.logged_artifacts["features.aet.info"]`; train/test offsets are available under `artifacts.preprocess.logged_artifacts.by_split`.
The mode uses labels only to align rows with the expected train/test order; it does not use labels to learn or alter the embedding.
The pipeline also supports a checkpoint-backed `vision.aet` mode for PyTorch AET checkpoints, but the linked AET README documents CIFAR-10 training rather than publishing a CIFAR-10 checkpoint. <sup class="cite"><a href="#source-9">[9]</a></sup>


## Annoy candidate search

For a seeded Annoy candidate search, set `backend="annoy"` with a Euclidean
kNN graph. `annoy_n_trees` controls index construction, `annoy_query_k` controls
how many candidates are retrieved before retaining the final `k`, and
`annoy_search_k` controls Annoy's search effort. Set `annoy_rerank=true` to
reorder those candidates using exact Euclidean distances; its default is false.
The `seed` passed to `build_graph` builds the index and enters the graph cache key.


## Pitfalls
!!! warning
    `GraphBuilderSpec` validation is strict; unsupported combinations (for example, `backend=faiss` with `scheme=epsilon`) raise `GraphValidationError`. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-6">[6]</a></sup>


!!! tip
    The graph and views caches are managed by `GraphCache` and `ViewsCache`. Use the CLI cache commands to inspect or purge them. <sup class="cite"><a href="#source-7">[7]</a><a href="#source-4">[4]</a></sup>


## Related links
- [Configuration reference](../reference/configuration.md)
- [Transductive tutorial: toy label propagation](../tutorials/transductive-toy.md)
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
  <li id="source-8"><a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_AET_vs._AED_Unsupervised_Representation_Learning_by_Auto-Encoding_Transformations_Rather_CVPR_2019_paper.html">Zhang et al. 2019 AET vs. AED</a></li>
  <li id="source-9"><a href="https://github.com/maple-research-lab/AET">Official AET code</a></li>
</ol>
</details>
