# Installation

Use this page to install ModSSC, verify the setup, and choose the right install path. After installation, continue with [Quickstart](quickstart.md).


## Supported Python versions
ModSSC requires Python 3.11 or newer. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Install from PyPI
Install the core library and CLI tools from PyPI.

Choose this if you only need the packaged library and CLI. The benchmark runner lives in `bench/` and is not shipped to PyPI. <sup class="cite"><a href="#source-2">[2]</a></sup>

This package name and entry points are declared in project metadata. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>

```bash
python -m pip install modssc
```


## Install from source
Use this for development, [benchmarks](../reference/benchmarks.md), or when you need the latest main branch.

This path is recommended for benchmark runs because the runner and configs are in the repository. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>

```bash
git clone https://github.com/ModSSC/ModSSC
cd ModSSC
python -m pip install -e "."
```


## Verify installation
Use the CLI check if you plan to run `modssc` commands, and use the import check if you embed ModSSC inside your own Python scripts. See the [CLI reference](../reference/cli.md) for command details. <sup class="cite"><a href="#source-4">[4]</a><a href="#source-6">[6]</a></sup>

CLI version (from the main Typer app):

```bash
modssc --version
```

Python import smoke test:

```bash
python -c "import modssc; print(modssc.__version__)"
```

The CLI version flag is implemented in [`src/modssc/cli/app.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/app.py), and `modssc.__version__` is defined in [`src/modssc/__about__.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/__about__.py). <sup class="cite"><a href="#source-4">[4]</a><a href="#source-5">[5]</a><a href="#source-6">[6]</a></sup>


## Common pitfalls
!!! warning
    Some datasets and methods require optional extras (for example, `hf`, `openml`, `graph`, `inductive-torch`). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-7">[7]</a></sup>


!!! tip
    For a full-feature installation (datasets, preprocess encoders, torch backends), use the `full` extra: `python -m pip install "modssc[full]"`. <sup class="cite"><a href="#source-1">[1]</a></sup>

## Related links
- [Quickstart](quickstart.md)
- [Concepts](concepts.md)
- [Catalogs and registries](../reference/catalogs.md)
- [Contributing and development](../development/contributing.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/README.md"><code>README.md</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/app.py"><code>src/modssc/cli/app.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/__about__.py"><code>src/modssc/__about__.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/__init__.py"><code>src/modssc/__init__.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/errors.py"><code>src/modssc/data_loader/errors.py</code></a></li>
</ol>
</details>
