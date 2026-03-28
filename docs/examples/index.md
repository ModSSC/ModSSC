# Example scripts

Use this index when you want small, readable entrypoints that run directly from the repository checkout. These scripts are intentionally lighter than full benchmark configs and easier to adapt inside your own code.


## How to run them
From the repository root:

```bash
python examples/00_inductive_toy_pseudo_label.py
python examples/01_transductive_toy_label_propagation.py
python examples/02_sampling_toy_holdout.py
```

If you installed only from PyPI, clone the repository first because the `examples/` folder is a repository asset.


## Script index

| Script | Focus | Typical use |
| --- | --- | --- |
| [`examples/00_inductive_toy_pseudo_label.py`](https://github.com/ModSSC/ModSSC/blob/main/examples/00_inductive_toy_pseudo_label.py) | Inductive SSL on the built-in toy dataset | first Python API success |
| [`examples/01_transductive_toy_label_propagation.py`](https://github.com/ModSSC/ModSSC/blob/main/examples/01_transductive_toy_label_propagation.py) | Graph-based SSL with label propagation | first transductive API run |
| [`examples/01_evaluation_quickstart.py`](https://github.com/ModSSC/ModSSC/blob/main/examples/01_evaluation_quickstart.py) | Metrics from labels or score matrices | evaluation helpers |
| [`examples/02_sampling_toy_holdout.py`](https://github.com/ModSSC/ModSSC/blob/main/examples/02_sampling_toy_holdout.py) | Deterministic holdout split creation | sampling API and stats |
| [`examples/02_hpo_primitives_quickstart.py`](https://github.com/ModSSC/ModSSC/blob/main/examples/02_hpo_primitives_quickstart.py) | HPO space primitives | inspect `modssc.hpo.Space` quickly |
| [`examples/03_cli_smoke.py`](https://github.com/ModSSC/ModSSC/blob/main/examples/03_cli_smoke.py) | CLI smoke test | verify entrypoints and discover commands |
| [`examples/04_inductive_cotraining_two_views.py`](https://github.com/ModSSC/ModSSC/blob/main/examples/04_inductive_cotraining_two_views.py) | Two-view inductive SSL | multi-view data and co-training |


## Which scripts to run first
- Start with `00_inductive_toy_pseudo_label.py` if you want the shortest end-to-end Python example.
- Run `03_cli_smoke.py` if you want a quick environment check across CLI bricks.
- Run `04_inductive_cotraining_two_views.py` when you are specifically exploring `data.views`.


## Related links
- [Quickstart](../getting-started/quickstart.md)
- [Choose your path](../getting-started/choose-your-path.md)
- [Notebook tour](../notebooks/index.md)
- [Bench config cookbook](../how-to/bench-cookbook.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/README.md"><code>README.md</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/tree/main/examples"><code>examples/</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/examples/00_inductive_toy_pseudo_label.py"><code>examples/00_inductive_toy_pseudo_label.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/examples/01_transductive_toy_label_propagation.py"><code>examples/01_transductive_toy_label_propagation.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/examples/02_sampling_toy_holdout.py"><code>examples/02_sampling_toy_holdout.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/examples/03_cli_smoke.py"><code>examples/03_cli_smoke.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/examples/04_inductive_cotraining_two_views.py"><code>examples/04_inductive_cotraining_two_views.py</code></a></li>
</ol>
</details>
