# Glossary

Use this page when a term appears in the docs and you want the shortest stable definition before going deeper.


## Core workflow terms
- **benchmark runner**: the repository-level execution path driven by `python -m bench.main --config ...`. It orchestrates dataset loading, sampling, preprocess, optional graph and views stages, method execution, and reporting.
- **catalog**: a curated list of IDs and metadata, such as datasets or preprocess steps, exposed through CLI and Python helpers.
- **provider**: the backend that knows how to fetch or materialize a dataset, such as OpenML, Hugging Face, TFDS, torchvision, torchaudio, or PyG.
- **modality**: the data family a dataset or step primarily targets, such as tabular, text, vision, audio, or graph.
- **method ID**: the string used to select a learning method in a registry or config, for example `pseudo_label` or `label_propagation`.
- **classifier ID**: the string used to select a supervised baseline or model backend inside an inductive method configuration.


## Learning setting terms
- **inductive**: a workflow where the method learns from labeled and unlabeled examples and must generalize to unseen samples without requiring an explicit graph.
- **transductive**: a workflow where the method operates on a fixed graph over all nodes and predicts labels inside that graph.
- **split**: the partition of data into subsets such as train, validation, and test.
- **labeling**: the rule that decides how many train examples remain labeled versus unlabeled inside a semi-supervised split.
- **view**: an alternative feature representation of the same examples, often used for multi-view methods such as co-training.
- **graph spec**: the structured description of how to build a graph, including scheme, metric, backend, and weighting choices.


## Reproducibility and artifact terms
- **fingerprint**: the stable hash-like identity ModSSC derives from inputs such as dataset content, config blocks, seeds, and selected fields to name cache artifacts deterministically.
- **cache**: the on-disk storage for reusable artifacts such as downloaded datasets, preprocess outputs, graphs, and graph views.
- **fit_on**: the subset used to fit preprocess steps that learn statistics, such as scaling or PCA.
- **official split**: a provider-defined train or test partition that comes from the dataset source rather than from a user-defined split plan.


## Dependency and availability terms
- **optional extra**: an install group exposed by Python packaging, such as `graph`, `preprocess-text`, or `transductive-torch`.
- **required_extra**: the metadata field that tells you which optional extra must be installed for a dataset, step, model, or method to be usable.
- **available-only**: a registry filter that hides entries whose optional dependencies are currently missing.


## Sampling policy terms
- **respect_official_test**: sampling policy flag that keeps the provider test split when the dataset ships one.
- **use_official_graph_masks**: sampling policy flag that preserves graph-native train, validation, and test masks when the dataset already defines them.
- **allow_override_official**: sampling policy flag that lets an inductive custom split override the provider test partition and instead resplit `dataset.train`.


## Related links
- [Concepts](concepts.md)
- [Configuration reference](../reference/configuration.md)
- [Reproducibility](../how-to/reproducibility.md)
- [Common errors and where to go](../how-to/common-errors.md)
- [Troubleshooting](../how-to/troubleshooting.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/getting-started/concepts.md"><code>docs/getting-started/concepts.md</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/reference/configuration.md"><code>docs/reference/configuration.md</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/how-to/reproducibility.md"><code>docs/how-to/reproducibility.md</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
</ol>
</details>
