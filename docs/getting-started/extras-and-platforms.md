# Optional extras and platform support

Use this guide to choose the smallest dependency set that matches your workflow. Extras in ModSSC are grouped by provider family, preprocess backends, training backends, and meta bundles.


## Quick recommendations

| Goal | Install |
| --- | --- |
| Core package API and CLI only | `python -m pip install modssc` |
| Repository workflows with benchmarks, examples, and notebooks | `python -m pip install -e "."` |
| Most dataset providers in one shot | `python -m pip install "modssc[datasets]"` |
| Most preprocessing backends in one shot | `python -m pip install "modssc[preprocess]"` |
| Full local feature set | `python -m pip install "modssc[full]"` |
| Docs authoring | `python -m pip install "modssc[docs]"` |
| Development and tests | `python -m pip install "modssc[dev]"` |


## Provider and dataset extras

| Extra | Unlocks | Typical use |
| --- | --- | --- |
| `openml` | OpenML-backed tabular datasets | `iris`, `adult`, `breast_cancer` |
| `hf` | Hugging Face datasets provider | `ag_news`, `imdb`, `amazon_polarity` |
| `text` | Alias used for text dataset installs | Text datasets backed by `datasets` |
| `tfds` | TensorFlow Datasets provider | TFDS-backed datasets |
| `vision` | torchvision-backed datasets | `mnist`, `cifar10`, `svhn`, `stl10` |
| `audio` | torchaudio-backed datasets | `speechcommands`, `yesno` |
| `graph` | PyG-backed graph datasets and graph-oriented tooling | `cora`, `citeseer`, `pubmed` |
| `datasets` / `data` | Meta bundle for common dataset providers | Broad data-loading setup |


## Method and model extras

| Extra | Unlocks | Typical use |
| --- | --- | --- |
| `inductive-torch` | Torch-based inductive methods | deep inductive SSL workflows |
| `inductive-tf` | TensorFlow-based inductive methods | TF inductive experiments |
| `supervised-torch` | Torch supervised baselines | MLP/CNN/LSTM-style baselines |
| `supervised-torch-geometric` | torch-geometric supervised baselines | GraphSAGE inductive baseline |
| `transductive-torch` | Torch-based transductive methods | GCN, GAT, APPNP and similar |
| `transductive-pyg` | PyG-backed transductive support | graph-native torch workflows |
| `transductive-advanced` | Advanced transductive torch/PyG stack | larger graph method surface |
| `graph-faiss` | FAISS graph construction backend | fast approximate kNN on supported platforms |


## Preprocess extras

| Extra | Unlocks | Typical use |
| --- | --- | --- |
| `preprocess-sklearn` | scikit-learn preprocess steps | imputation, scaling, PCA |
| `preprocess-text` | sentence-transformers and transformers | text embeddings |
| `preprocess-vision` | torchvision, Pillow, OpenCLIP | image transforms and pretrained encoders |
| `preprocess-audio` | torchaudio preprocess stack | waveform/audio features |
| `preprocess-graph` | scipy-based graph preprocess steps | sparse graph features |
| `preprocess` | Meta bundle for preprocess extras | mixed preprocess workflows |


## Platform notes
- The benchmark runner and authored benchmark configs live in `bench/` and are repository assets, not PyPI package assets.
- `graph-faiss` is guarded by a platform marker and does not install on Darwin in `pyproject.toml`.
- `full` is the easiest way to get a rich local environment, but it is also the heaviest dependency profile.
- If you only need one workflow, prefer the narrow extra over `full` so environments stay smaller and easier to maintain.


## Practical install recipes
CPU-friendly local benchmark setup:

```bash
python -m pip install -e "."
python -m pip install "modssc[datasets,preprocess,inductive-torch,transductive-torch]"
```

Text-oriented inductive workflow:

```bash
python -m pip install "modssc[hf,preprocess-text,inductive-torch]"
```

Graph-oriented transductive workflow:

```bash
python -m pip install "modssc[graph,transductive-torch]"
```


## Related links
- [Manage datasets](../how-to/datasets.md)
- [Run preprocessing plans](../how-to/preprocess.md)
- [Benchmarks](../reference/benchmarks.md)
- [Troubleshooting](../how-to/troubleshooting.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_loader/providers"><code>src/modssc/data_loader/providers/</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/preprocess/steps"><code>src/modssc/preprocess/steps/</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/supervised/registry_data.py"><code>src/modssc/supervised/registry_data.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
</ol>
</details>
