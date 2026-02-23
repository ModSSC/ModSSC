# Minimal benchmark configs

This folder contains a reduced set of configurations to keep the repository simple:

- 1 inductive method (`pseudo_label`)
- 1 transductive method (`gcn`)
- 5 modalities (`audio`, `graph`, `tabular`, `text`, `vision`)

## Commands

### Inductive (`pseudo_label`)

```bash
python -m bench.main --config bench/configs/experiments/minimal/inductive/pseudo_label/audio/speechcommands.yaml
python -m bench.main --config bench/configs/experiments/minimal/inductive/pseudo_label/graph/cora.yaml
python -m bench.main --config bench/configs/experiments/minimal/inductive/pseudo_label/tabular/iris.yaml
python -m bench.main --config bench/configs/experiments/minimal/inductive/pseudo_label/text/imdb.yaml
python -m bench.main --config bench/configs/experiments/minimal/inductive/pseudo_label/vision/mnist.yaml
```

### Transductive (`gcn`)

```bash
python -m bench.main --config bench/configs/experiments/minimal/transductive/gcn/audio/speechcommands.yaml
python -m bench.main --config bench/configs/experiments/minimal/transductive/gcn/graph/cora.yaml
python -m bench.main --config bench/configs/experiments/minimal/transductive/gcn/tabular/iris.yaml
python -m bench.main --config bench/configs/experiments/minimal/transductive/gcn/text/imdb.yaml
python -m bench.main --config bench/configs/experiments/minimal/transductive/gcn/vision/mnist.yaml
```
