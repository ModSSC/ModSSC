# ModSSC documentation

Start here for a quick tour of ModSSC, its core capabilities, and the recommended next steps. If you're new, begin with [Installation](getting-started/installation.md) or [Quickstart](getting-started/quickstart.md).


## What is ModSSC
ModSSC is a modular framework for semi-supervised classification across heterogeneous modalities (audio, text, vision, tabular, graph) with research-focused abstractions and reproducible pipelines. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Key features
- Dataset catalog with curated keys for tabular, text, vision, audio, and graph datasets. <sup class="cite"><a href="#source-7">[7]</a></sup>

- Optional provider backends for OpenML, Hugging Face datasets, TFDS, torchvision, torchaudio, and PyG. <sup class="cite"><a href="#source-8">[8]</a></sup>

- Deterministic sampling plans (holdout/k-fold + labeling strategies) with cached split artifacts. <sup class="cite"><a href="#source-5">[5]</a><a href="#source-9">[9]</a><a href="#source-10">[10]</a></sup>

- Deterministic preprocessing plans with step registry and optional pretrained encoders. <sup class="cite"><a href="#source-6">[6]</a><a href="#source-11">[11]</a><a href="#source-12">[12]</a></sup>

- Graph construction (kNN/epsilon/anchor) and graph-derived views (attribute, diffusion, structural). <sup class="cite"><a href="#source-13">[13]</a><a href="#source-14">[14]</a></sup>

- Inductive and transductive method registries with method IDs. <sup class="cite"><a href="#source-15">[15]</a><a href="#source-16">[16]</a></sup>

- CLI tools for datasets, sampling, preprocessing, graphs, augmentation, and evaluation. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>

- Benchmark runner with YAML experiment configs (GitHub-only, not shipped to PyPI). <sup class="cite"><a href="#source-17">[17]</a><a href="#source-2">[2]</a></sup>


## Quickstart links
- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Concepts](getting-started/concepts.md)
- [Inductive tutorial](tutorials/inductive-toy.md)
- [Transductive tutorial](tutorials/transductive-toy.md)

## Version
Current version is `0.2.2`, sourced from [`src/modssc/__about__.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/__about__.py) and referenced by the Hatch version config in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml). <sup class="cite"><a href="#source-18">[18]</a><a href="#source-2">[2]</a></sup>


## Project status and support
- Status is alpha (Development Status 3) in project metadata. <sup class="cite"><a href="#source-2">[2]</a></sup>

- Report issues via the GitHub tracker. <sup class="cite"><a href="#source-2">[2]</a></sup>

- Contributor guidance lives in the docs. <sup class="cite"><a href="#source-19">[19]</a></sup>

- Citation metadata is provided in [`CITATION.cff`](https://github.com/ModSSC/ModSSC/blob/main/CITATION.cff). <sup class="cite"><a href="#source-20">[20]</a></sup>


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/README.md"><code>README.md</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/cli"><code>src/modssc/cli/</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_loader"><code>src/modssc/data_loader/</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/sampling"><code>src/modssc/sampling/</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/preprocess"><code>src/modssc/preprocess/</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_loader/catalog"><code>src/modssc/data_loader/catalog/</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_loader/providers"><code>src/modssc/data_loader/providers/</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/storage.py"><code>src/modssc/sampling/storage.py</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/catalog.py"><code>src/modssc/preprocess/catalog.py</code></a></li>
  <li id="source-12"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/models.py"><code>src/modssc/preprocess/models.py</code></a></li>
  <li id="source-13"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py"><code>src/modssc/graph/specs.py</code></a></li>
  <li id="source-14"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/featurization/api.py"><code>src/modssc/graph/featurization/api.py</code></a></li>
  <li id="source-15"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/registry.py"><code>src/modssc/inductive/registry.py</code></a></li>
  <li id="source-16"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/registry.py"><code>src/modssc/transductive/registry.py</code></a></li>
  <li id="source-17"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
  <li id="source-18"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/__about__.py"><code>src/modssc/__about__.py</code></a></li>
  <li id="source-19"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/development/contributing.md"><code>docs/development/contributing.md</code></a></li>
  <li id="source-20"><a href="https://github.com/ModSSC/ModSSC/blob/main/CITATION.cff"><code>CITATION.cff</code></a></li>
</ol>
</details>
