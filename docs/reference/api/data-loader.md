# Data loader API

This page documents the data loader API used to list, inspect, and load datasets. For usage examples, see the [dataset how-to](../../how-to/datasets.md).


## What it is for
The data loader brick resolves dataset identifiers, downloads raw data, caches processed datasets, and returns canonical splits. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Examples
Load a curated dataset:

```python
from modssc.data_loader import load_dataset

ds = load_dataset("toy", download=True)
print(ds.train.X.shape, ds.train.y.shape)
```

Inspect the catalog and provider list:

```python
from modssc.data_loader import available_datasets, available_providers, dataset_info

print(available_datasets())
print(available_providers())
print(dataset_info("toy").as_dict())
```

The public API is exported from [`src/modssc/data_loader/__init__.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/__init__.py). <sup class="cite"><a href="#source-3">[3]</a></sup>


## API reference

::: modssc.data_loader

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/api.py"><code>src/modssc/data_loader/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/types.py"><code>src/modssc/data_loader/types.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/__init__.py"><code>src/modssc/data_loader/__init__.py</code></a></li>
</ol>
</details>
