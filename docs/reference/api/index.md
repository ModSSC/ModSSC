# API reference

This page links to the Python API modules and helps you navigate the reference. For workflows, start with [Quickstart](../../getting-started/quickstart.md).


## What it is for
This section documents the public Python API across ModSSC bricks. Each page includes a short overview, runnable examples, and mkdocstrings output. The reference targets stable package-level imports, while internal implementation packages are described separately in the [Architecture](../../development/architecture.md) page. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Examples
Load a dataset and inspect its spec:

```python
from modssc.data_loader import load_dataset, dataset_info

ds = load_dataset("toy", download=True)
print(dataset_info("toy").as_dict())
```

Run a simple evaluation:

```python
import numpy as np
from modssc.evaluation import evaluate

print(evaluate(np.array([0, 1]), np.array([0, 1]), ["accuracy"]))
```

These APIs are exported from the package-level bricks under [`src/modssc/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc). Some bricks delegate orchestration to internal `services/`, `helpers/`, or `bundle_factories/` subpackages, but those internal directories are not treated as the primary public surface. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>


## Modules
- [Data loader](data-loader.md)
- [Data augmentation](data-augmentation.md)
- [Sampling](sampling.md)
- [Preprocess](preprocess.md)
- [Views](views.md)
- [Graph](graph.md)
- [Evaluation](evaluation.md)
- [Inductive](inductive.md)
- [Transductive](transductive.md)
- [Supervised](supervised.md)
- [HPO](hpo.md)
- [Runtime logging](logging.md)
- [Runtime device](device.md)


## Structure notes
- Package pages such as `modssc.preprocess` and `modssc.sampling` document the public facade imported by user code.
- Internal support packages such as `services/`, `helpers/`, `bundle_factories/`, `adapters/`, and backend-specific implementation packages are intentionally treated as implementation details.
- Runtime support utilities now live under `modssc.runtime`, not at the package root.
- Cache resolution helpers live under `modssc.cache`, and shared optional-dependency helpers live under `modssc.dependencies`.

For the current repository layout, see [Architecture](../../development/architecture.md).

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc"><code>src/modssc/</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_loader"><code>src/modssc/data_loader/</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/evaluation"><code>src/modssc/evaluation/</code></a></li>
</ol>
</details>
