# Supervised API

This page documents supervised baseline APIs. For CLI usage, see the [CLI reference](../cli.md).


## What it is for
The supervised brick exposes baseline classifiers with multiple backends (numpy, sklearn, torch). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
List classifiers and backends:

```python
from modssc.supervised import available_classifiers

print(available_classifiers())
```

Create a classifier (auto backend):

```python
from modssc.supervised import create_classifier

clf = create_classifier("knn", backend="numpy", params={"k": 3})
```

Backends and metadata are defined in [`src/modssc/supervised/registry.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/supervised/registry.py). <sup class="cite"><a href="#source-2">[2]</a></sup>

## Common classifier IDs
- `lstm_scratch` (Torch LSTM for text sequences, extra: `supervised-torch`)
- `audio_cnn_scratch` (Torch CNN for spectrograms, extra: `supervised-torch`)
- `graphsage_inductive` (Torch Geometric GraphSAGE, extra: `supervised-torch-geometric`)

Use `modssc supervised info <classifier_id>` or `classifier_info()` to inspect required extras.


## API reference

::: modssc.supervised

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/supervised/api.py"><code>src/modssc/supervised/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/supervised/registry.py"><code>src/modssc/supervised/registry.py</code></a></li>
</ol>
</details>
