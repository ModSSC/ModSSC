# Contributing and development

This page explains how to set up a dev environment, run tests, and contribute changes. If you are preparing a release, see [Release process](release-process.md).


## Dev setup
Use the Makefile targets or install dependencies directly.

Use `make install-dev` if you want a single command that matches the repo defaults. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-1">[1]</a></sup>

```bash
make install-dev
```

Use the editable `pip install -e` form when you want explicit control over installs and extras. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-1">[1]</a></sup>

```bash
python -m pip install -e "." && python -m pip install -e ".[dev]"
```

The development extras and Makefile targets are defined in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml) and [`Makefile`](https://github.com/ModSSC/ModSSC/blob/main/Makefile). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Running tests
Tests are organized under [`tests/`](https://github.com/ModSSC/ModSSC/tree/main/tests) and use pytest:

Use `make test` for the default suite. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-1">[1]</a></sup>

```bash
make test
```

Use `python -m pytest` when you need custom flags or subset selection. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-1">[1]</a></sup>

```bash
python -m pytest
```

Pytest options and markers are configured in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-3">[3]</a></sup>


## Style and linting
The project uses Ruff for linting and formatting:

```bash
make lint
make format
```

Ruff configuration is in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## Project structure explanation
- [`src/modssc/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc): core library and CLI implementations. <sup class="cite"><a href="#source-4">[4]</a></sup>

- [`bench/`](https://github.com/ModSSC/ModSSC/tree/main/bench): benchmark runner and experiment configs (GitHub-only). <sup class="cite"><a href="#source-5">[5]</a></sup>

- [`examples/`](https://github.com/ModSSC/ModSSC/tree/main/examples) and [`notebooks/`](https://github.com/ModSSC/ModSSC/tree/main/notebooks): runnable demos and exploratory workflows. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-6">[6]</a></sup>

- [`docs/`](https://github.com/ModSSC/ModSSC/tree/main/docs): MkDocs site sources. <sup class="cite"><a href="#source-4">[4]</a></sup>


## Adding a new algorithm or dataset
Inductive methods:
- Implement the `InductiveMethod` protocol and define a `MethodInfo` object. <sup class="cite"><a href="#source-7">[7]</a></sup>

- Register the method ID in `register_builtin_methods`. <sup class="cite"><a href="#source-8">[8]</a></sup>


Transductive methods:
- Implement the `TransductiveMethod` protocol and define `MethodInfo`. <sup class="cite"><a href="#source-9">[9]</a></sup>

- Register the method ID in `register_builtin_methods`. <sup class="cite"><a href="#source-10">[10]</a></sup>


Datasets:
- Add curated datasets by extending `DATASET_CATALOG` in the relevant modality file. <sup class="cite"><a href="#source-11">[11]</a></sup>

- Implement new providers by subclassing `BaseProvider` and registering it. <sup class="cite"><a href="#source-12">[12]</a><a href="#source-13">[13]</a></sup>

Use a catalog entry when the dataset already fits an existing provider. Add a new provider when you need a new backend or authentication flow. Catalog entries reference providers by name. <sup class="cite"><a href="#source-11">[11]</a><a href="#source-12">[12]</a><a href="#source-13">[13]</a></sup>

If you are preparing a tag or release artifacts, follow the [release process](release-process.md).

## Related links
- [Release process](release-process.md)
- [Catalogs and registries](../reference/catalogs.md)
- [Dataset how-to](../how-to/datasets.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/Makefile"><code>Makefile</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/tree/main/tests"><code>tests/</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/README.md"><code>README.md</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/tree/main/examples"><code>examples/</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/base.py"><code>src/modssc/inductive/base.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/registry.py"><code>src/modssc/inductive/registry.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/base.py"><code>src/modssc/transductive/base.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/registry.py"><code>src/modssc/transductive/registry.py</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/data_loader/catalog"><code>src/modssc/data_loader/catalog/</code></a></li>
  <li id="source-12"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/providers/base.py"><code>src/modssc/data_loader/providers/base.py</code></a></li>
  <li id="source-13"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/providers/__init__.py"><code>src/modssc/data_loader/providers/__init__.py</code></a></li>
</ol>
</details>
