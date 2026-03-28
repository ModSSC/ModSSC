# Choose your path

Use this guide to pick the fastest entrypoint for your goal. ModSSC supports several workflows: package-level Python APIs, CLI bricks, end-to-end benchmark configs, example scripts, and interactive notebooks.


## Fastest routes by goal

| Goal | Start here | Then continue with |
| --- | --- | --- |
| First successful install and run | [Installation](installation.md) and [Quickstart](quickstart.md) | [Inductive tutorial](../tutorials/inductive-toy.md) or [Transductive tutorial](../tutorials/transductive-toy.md) |
| Learn the concepts before running code | [Concepts](concepts.md) | [Example scripts](../examples/index.md) |
| Use the Python API directly | [Quickstart](quickstart.md) | [Example scripts](../examples/index.md), [Notebook tour](../notebooks/index.md), [API reference](../reference/api/index.md) |
| Run reproducible benchmarks from YAML configs | [Benchmarks](../reference/benchmarks.md) | [Configuration reference](../reference/configuration.md), [Bench config cookbook](../how-to/bench-cookbook.md), [Reproducibility](../how-to/reproducibility.md) |
| Set up modality-specific dependencies | [Optional extras and platform support](extras-and-platforms.md) | [Manage datasets](../how-to/datasets.md), [Run preprocessing plans](../how-to/preprocess.md), [Build graphs and views](../how-to/graph.md) |
| Work interactively in notebooks | [Notebook tour](../notebooks/index.md) | The matching how-to page for the same brick |
| Extend ModSSC with new datasets or methods | [Architecture](../development/architecture.md) | [Contributing and development](../development/contributing.md), [Catalogs and registries](../reference/catalogs.md) |


## Recommended first commands
If you cloned the repository and want the broadest surface area available locally:

```bash
python -m pip install -e "."
python -m modssc --help
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
```

If you prefer a lighter Python-first path:

```bash
python -m pip install modssc
python examples/00_inductive_toy_pseudo_label.py
```

Use a source install when you need the benchmark runner, authored benchmark configs, notebooks, or example scripts from the repository. Use the PyPI install when you only need the packaged library and CLI.


## Which learning format should you choose
- Use the [tutorials](../tutorials/inductive-toy.md) when you want a full end-to-end run with one opinionated config.
- Use the [how-to guides](../how-to/datasets.md) when you want to understand one brick in isolation.
- Use the [example scripts](../examples/index.md) when you want copy-pasteable Python files.
- Use the [notebooks](../notebooks/index.md) when you want an interactive, exploratory workflow.
- Use the [reference](../reference/cli.md) when you already know what you want and only need precise command or schema details.


## Related links
- [Installation](installation.md)
- [Glossary](glossary.md)
- [Optional extras and platform support](extras-and-platforms.md)
- [Example scripts](../examples/index.md)
- [Notebook tour](../notebooks/index.md)
- [Bench config cookbook](../how-to/bench-cookbook.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/README.md"><code>README.md</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/mkdocs.yml"><code>mkdocs.yml</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/tree/main/examples"><code>examples/</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/tree/main/notebooks"><code>notebooks/</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/README.md"><code>bench/configs/experiments/minimal/README.md</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml"><code>bench/configs/experiments/toy_inductive.yaml</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_transductive.yaml"><code>bench/configs/experiments/toy_transductive.yaml</code></a></li>
</ol>
</details>
