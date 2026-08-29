# Contributing and development

This page explains how to set up a development environment, run tests, and contribute to the project. If you are preparing a release, see [Release process](release-process.md).


## Dev setup
Install dependencies directly from the development extras declared in the project metadata. Use the editable `pip install -e` form when you want local code changes to be reflected immediately. <sup class="cite"><a href="#source-1">[1]</a></sup>

```bash
python -m pip install -e "." && python -m pip install -e ".[dev]"
```

The development extras are defined in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml). <sup class="cite"><a href="#source-1">[1]</a></sup>


## Running tests
Tests are organized under [`tests/`](https://github.com/ModSSC/ModSSC/tree/main/tests) and use pytest:

Use `python -m pytest` for the default suite. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>

```bash
python -m pytest
```

Add flags or subset selection when needed. If you are only checking documentation contracts, prefer `--no-cov` so the repository-wide coverage gate does not fail on a tiny targeted subset. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>

```bash
python -m pytest
python -m pytest --no-cov tests/contracts/test_docs_contracts.py
```

Pytest options and markers are configured in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-3">[3]</a></sup>


## Style and linting
The project uses Ruff for linting and formatting. <sup class="cite"><a href="#source-1">[1]</a></sup>

```bash
ruff check .
ruff format .
```

Ruff configuration is in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml). <sup class="cite"><a href="#source-1">[1]</a></sup>


## Project structure explanation
- [`src/modssc/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc): core
  library, method implementations, reusable backends, and installed CLIs.
  <sup class="cite"><a href="#source-4">[4]</a></sup>

- [`bench/`](https://github.com/ModSSC/ModSSC/tree/main/bench): separately
  packaged benchmark and paper-replication runner, validated configs, schemas,
  and scientific orchestration. <sup class="cite"><a href="#source-5">[5]</a></sup>

- Private scheduler/deployment wrappers and historical evidence archives remain
  outside the runtime repository. External research source snapshots are not
  retained. Site-specific accounts, partitions, paths, credentials, caches,
  checkpoints, and results never belong in the public repository.

- [`examples/`](https://github.com/ModSSC/ModSSC/tree/main/examples) and [`notebooks/`](https://github.com/ModSSC/ModSSC/tree/main/notebooks): runnable demos and exploratory workflows. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-6">[6]</a></sup>

- [`docs/`](https://github.com/ModSSC/ModSSC/tree/main/docs): MkDocs site sources. <sup class="cite"><a href="#source-4">[4]</a></sup>


## Adding a new algorithm or dataset
Inductive methods:
- Implement the `InductiveMethod` protocol and define a `MethodInfo` object,
  including its explicit capability contract. <sup class="cite"><a href="#source-7">[7]</a></sup>

- When requirements depend on the resolved spec, implement the method class's
  native `execution_contract` hook. Declare the exact labeled, unlabeled,
  weak/strong, named-view, graph, and prediction roles that `fit` consumes;
  declare every required model output, optimizer, EMA model, scheduler, and
  component relation. Do not reproduce this logic in `bench`.

- A native Torch bundle factory must attach a static `ModelContract` describing
  its accepted representations, dtype kinds, ranks, and real callable outputs.
  An external bundle without that contract is deliberately `unverified` and is
  rejected by strict benchmark execution.

- Register the method ID in `register_builtin_methods`. <sup class="cite"><a href="#source-8">[8]</a></sup>

- Add a negative test that supplies one incompatible input or component and
  asserts that the native execution gate rejects it before the first `fit`
  call. Also exercise every spec-dependent branch of the contract hook.


Transductive methods:
- Implement the `TransductiveMethod` protocol and define `MethodInfo`, including
  its explicit capability contract. <sup class="cite"><a href="#source-9">[9]</a></sup>

- Keep input requirements in the method's native execution contract. A source
  modality is not an execution representation: declare the materialized
  feature, graph, target, backend, dtype, rank, and row relations that the
  solver actually reads.

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
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/tree/main/tests"><code>tests/</code></a></li>
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
