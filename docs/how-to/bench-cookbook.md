# Bench config cookbook

Use this guide when you want a concrete starting config instead of building YAML from scratch. The goal is simple: start from the nearest working config, change as little as possible, and only widen the search space once the baseline runs.


## Fastest starting points

| Goal | Config | Command |
| --- | --- | --- |
| Smallest inductive walkthrough | [`bench/configs/experiments/toy_inductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml) | `python -m bench.main --config bench/configs/experiments/toy_inductive.yaml` |
| Smallest transductive walkthrough | [`bench/configs/experiments/toy_transductive.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_transductive.yaml) | `python -m bench.main --config bench/configs/experiments/toy_transductive.yaml` |
| Smallest HPO walkthrough | [`bench/configs/experiments/toy_inductive_hpo.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive_hpo.yaml) | `python -m bench.main --config bench/configs/experiments/toy_inductive_hpo.yaml` |


## Minimal configs by modality

### Inductive

| Modality | Config | Typical extras |
| --- | --- | --- |
| Audio | [`bench/configs/experiments/minimal/inductive/pseudo_label/audio/speechcommands.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/inductive/pseudo_label/audio/speechcommands.yaml) | `audio`, `inductive-torch` |
| Graph | [`bench/configs/experiments/minimal/inductive/pseudo_label/graph/cora.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/inductive/pseudo_label/graph/cora.yaml) | `graph`, `inductive-torch` |
| Tabular | [`bench/configs/experiments/minimal/inductive/pseudo_label/tabular/iris.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/inductive/pseudo_label/tabular/iris.yaml) | `openml`, `inductive-torch` |
| Text | [`bench/configs/experiments/minimal/inductive/pseudo_label/text/imdb.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/inductive/pseudo_label/text/imdb.yaml) | `hf`, `inductive-torch` |
| Vision | [`bench/configs/experiments/minimal/inductive/pseudo_label/vision/mnist.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/inductive/pseudo_label/vision/mnist.yaml) | `vision`, `inductive-torch` |

### Transductive

| Modality | Config | Typical extras |
| --- | --- | --- |
| Audio | [`bench/configs/experiments/minimal/transductive/gcn/audio/speechcommands.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/transductive/gcn/audio/speechcommands.yaml) | `audio`, `transductive-torch` |
| Graph | [`bench/configs/experiments/minimal/transductive/gcn/graph/cora.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/transductive/gcn/graph/cora.yaml) | `graph`, `transductive-torch` |
| Tabular | [`bench/configs/experiments/minimal/transductive/gcn/tabular/iris.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/transductive/gcn/tabular/iris.yaml) | `openml`, `transductive-torch` |
| Text | [`bench/configs/experiments/minimal/transductive/gcn/text/imdb.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/transductive/gcn/text/imdb.yaml) | `hf`, `transductive-torch` |
| Vision | [`bench/configs/experiments/minimal/transductive/gcn/vision/mnist.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/transductive/gcn/vision/mnist.yaml) | `vision`, `transductive-torch` |


## Which config should you copy first
- Use the toy configs when you want the fastest local validation of the full pipeline.
- Use the minimal modality configs when you want a real provider-backed dataset but still want a small surface area.
- Use the HPO toy config when you want to validate the search workflow before tuning a larger model.
- Use the minimal co-training config when you specifically need `views.plan`: [`bench/configs/experiments/minimal/inductive/co_training/text/imdb.yaml`](https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/inductive/co_training/text/imdb.yaml).


## Safe editing strategy
When you fork a working config, change fields in this order:

1. `run.name` and `run.output_dir`
2. `dataset.id` and `dataset.options`
3. `sampling.plan`
4. `preprocess.plan`
5. `graph.spec` if the method is transductive
6. `method.id` and `method.params`
7. `evaluation.metrics`

Keep one known-good baseline config untouched so you always have a rollback point.


## Common patterns
### First local smoke
Start with one of the toy configs, then rerun with `--log-level detailed` only if you need deeper diagnostics.

### First real dataset run
Start with the matching minimal config instead of copying a large benchmark suite from `bench/configs/best/`.

### Clean-room reproducibility check
Reuse the same config, set a dedicated `MODSSC_CACHE_ROOT`, and keep the copied `config.yaml` and `run.json` together under `runs/`.


## Related links
- [Benchmarks](../reference/benchmarks.md)
- [Configuration reference](../reference/configuration.md)
- [Reproducibility](reproducibility.md)
- [Troubleshooting](troubleshooting.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/minimal/README.md"><code>bench/configs/experiments/minimal/README.md</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive.yaml"><code>bench/configs/experiments/toy_inductive.yaml</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_transductive.yaml"><code>bench/configs/experiments/toy_transductive.yaml</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/configs/experiments/toy_inductive_hpo.yaml"><code>bench/configs/experiments/toy_inductive_hpo.yaml</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
</ol>
</details>
