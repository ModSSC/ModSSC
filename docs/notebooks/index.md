# Notebook tour

Use this index when you want an interactive workflow for inspecting intermediate objects, trying short code changes, and explaining one brick at a time.


## How to use them
- Clone the repository and install ModSSC locally.
- Open the notebooks with your preferred notebook environment.
- Run them from the repository root so relative paths like `bench/configs/...` and `runs/` resolve naturally.


## Notebook index

| Notebook | Focus | Best after |
| --- | --- | --- |
| [`notebooks/00_cli_tour.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/00_cli_tour.ipynb) | CLI entrypoints and command discovery | [Installation](../getting-started/installation.md) |
| [`notebooks/01_data_loader_end_to_end.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/01_data_loader_end_to_end.ipynb) | dataset catalogs, providers, and loading | [Manage datasets](../how-to/datasets.md) |
| [`notebooks/02_sampling_quickstart.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/02_sampling_quickstart.ipynb) | sampling plans and split artifacts | [Create and reuse sampling splits](../how-to/sampling.md) |
| [`notebooks/03_preprocess_quickstart.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/03_preprocess_quickstart.ipynb) | preprocess plans and outputs | [Run preprocessing plans](../how-to/preprocess.md) |
| [`notebooks/04_data_augmentation_quickstart.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/04_data_augmentation_quickstart.ipynb) | augmentation plans and contexts | [How to use data augmentation](../how-to/augmentation.md) |
| [`notebooks/05_views_quickstart.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/05_views_quickstart.ipynb) | view plans and generated feature views | [How to generate multi-view features](../how-to/views.md) |
| [`notebooks/06_graph_quickstart.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/06_graph_quickstart.ipynb) | graph construction and graph-derived views | [How to build graphs and views](../how-to/graph.md) |
| [`notebooks/07_supervised_quickstart.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/07_supervised_quickstart.ipynb) | supervised baselines | [API reference: supervised](../reference/api/supervised.md) |
| [`notebooks/08_inductive_quickstart.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/08_inductive_quickstart.ipynb) | inductive SSL methods | [Inductive tutorial](../tutorials/inductive-toy.md) |
| [`notebooks/09_transductive_quickstart.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/09_transductive_quickstart.ipynb) | transductive SSL methods | [Transductive tutorial](../tutorials/transductive-toy.md) |
| [`notebooks/10_hpo_bench_smoke.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/10_hpo_bench_smoke.ipynb) | HPO bench workflow | [How to run hyperparameter search](../how-to/hpo.md) |
| [`notebooks/11_bench_run_anatomy.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/11_bench_run_anatomy.ipynb) | inspect `run.json`, artifacts, and bench outputs | [Benchmarks](../reference/benchmarks.md) |
| [`notebooks/12_optional_extras_and_modalities.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/12_optional_extras_and_modalities.ipynb) | choose extras by workflow and inspect dataset requirements | [Optional extras and platform support](../getting-started/extras-and-platforms.md) |
| [`notebooks/13_official_vs_custom_splits.ipynb`](https://github.com/ModSSC/ModSSC/blob/main/notebooks/13_official_vs_custom_splits.ipynb) | compare official test splits vs user-defined overrides | [Reproducibility](../how-to/reproducibility.md) |


## Which notebooks to open first
- Start with `00_cli_tour.ipynb` if you are still discovering the surface area.
- Open `08_inductive_quickstart.ipynb` or `09_transductive_quickstart.ipynb` once you know which modeling family you need.
- Open `11_bench_run_anatomy.ipynb` when you want to understand exactly what the benchmark runner writes to disk.


## Related links
- [Example scripts](../examples/index.md)
- [Choose your path](../getting-started/choose-your-path.md)
- [Bench config cookbook](../how-to/bench-cookbook.md)
- [Troubleshooting](../how-to/troubleshooting.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/README.md"><code>README.md</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/tree/main/notebooks"><code>notebooks/</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/notebooks/00_cli_tour.ipynb"><code>notebooks/00_cli_tour.ipynb</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/notebooks/10_hpo_bench_smoke.ipynb"><code>notebooks/10_hpo_bench_smoke.ipynb</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/notebooks/11_bench_run_anatomy.ipynb"><code>notebooks/11_bench_run_anatomy.ipynb</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/notebooks/12_optional_extras_and_modalities.ipynb"><code>notebooks/12_optional_extras_and_modalities.ipynb</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/notebooks/13_official_vs_custom_splits.ipynb"><code>notebooks/13_official_vs_custom_splits.ipynb</code></a></li>
</ol>
</details>
