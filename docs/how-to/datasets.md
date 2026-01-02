# How to manage datasets

Need to figure out which dataset IDs exist and how to manage their caches? This recipe walks you through discovery, metadata inspection, and cache management with CLI and Python examples side by side. If ModSSC is not installed yet, start with [Installation](../getting-started/installation.md).


## Problem statement
You want to list, inspect, and download datasets that ModSSC can load, and understand where they are cached. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup> Once you pick a dataset key, continue with [sampling](sampling.md) and [preprocess](preprocess.md) to shape the data for a run.


## When to use
Use these steps when you are starting a new experiment, switching modalities, or pre-downloading datasets before a large run. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a></sup>

Use `providers` when you need to know which backends are wired up, and `list` when you need curated dataset keys for [configs](../reference/configuration.md) and CLI commands. Use `info` to check the `required_extra` field before downloading a dataset. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-6">[6]</a></sup>


## Steps
1) List providers and dataset keys. <sup class="cite"><a href="#source-2">[2]</a></sup>

2) Inspect a dataset spec (modality, provider, required extra). <sup class="cite"><a href="#source-6">[6]</a><a href="#source-2">[2]</a></sup>

3) Download a dataset into the local cache. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>

   Use `--all` when you want an offline cache for a full modality, and `--dataset` when you only need a single dataset key. <sup class="cite"><a href="#source-2">[2]</a></sup>

4) Inspect or clean the cache index. <sup class="cite"><a href="#source-3">[3]</a><a href="#source-2">[2]</a></sup>


## Copy-paste example
Use the CLI for quick inspection in the terminal (`modssc datasets` in [`src/modssc/cli/datasets.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/datasets.py)), and use Python when you want dataset access inside a script (helpers in [`src/modssc/data_loader/api.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/api.py)). <sup class="cite"><a href="#source-2">[2]</a><a href="#source-1">[1]</a></sup>

CLI:

```bash
modssc datasets providers
modssc datasets list --modalities text
modssc datasets info --dataset ag_news
modssc datasets download --dataset ag_news
modssc datasets cache ls
```

Python:

```python
from modssc.data_loader import available_datasets, dataset_info, download_dataset

print(available_datasets())
print(dataset_info("toy").as_dict())
_ = download_dataset("toy")
```

## Pitfalls
!!! warning
    If a dataset requires optional dependencies, download will fail with an actionable error message. Install the suggested extra from [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml). <sup class="cite"><a href="#source-7">[7]</a><a href="#source-8">[8]</a></sup>


!!! tip
    Override the dataset cache directory with `MODSSC_CACHE_DIR` if you want to store datasets outside the repo or default user cache. <sup class="cite"><a href="#source-3">[3]</a></sup>


## Related links
- [Configuration reference](../reference/configuration.md)
- [Sampling how-to](sampling.md)
- [Preprocess how-to](preprocess.md)
- [Catalogs and registries](../reference/catalogs.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/api.py"><code>src/modssc/data_loader/api.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/datasets.py"><code>src/modssc/cli/datasets.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/cache.py"><code>src/modssc/data_loader/cache.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_loader/catalog"><code>src/modssc/data_loader/catalog/</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/types.py"><code>src/modssc/data_loader/types.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/errors.py"><code>src/modssc/data_loader/errors.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
</ol>
</details>
