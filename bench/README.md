# ModSSC benchmark and paper-replication runner

`bench/` is the scientific execution layer shipped with both the source checkout
and the ModSSC wheel. It validates YAML experiments, orchestrates ModSSC bricks,
and records reproducible results. It remains a separate top-level package rather
than becoming `modssc.experiments`.

## What belongs here

- the end-to-end runner, schemas, and scientific stage orchestrators;
- `configs/experiments/` for examples and small runnable templates;
- `configs/best/` for the standardized benchmark matrix;
- `configs/reproductions/` for immutable paper protocol cards;
- compact fixed resources that are direct protocol inputs;
- campaign generation, execution, reconciliation, and scientific acceptance.

Campaign manifests and retry plans are scheduler-neutral. Optional deployment
tooling may render them afterwards; `bench/` neither emits batch scripts nor
reads scheduler-specific process variables.

The method implementations themselves belong under `src/modssc/`. A benchmark
or replication config selects those implementations; it must not carry a second
implementation copied from an external research repository.

## What does not belong here

- cluster launchers, deployment scripts, or resource-manager policy: keep them
  in repository or private deployment tooling outside the installed package;
- completed evidence, audit history, and audit-only source manifests: use
  `provenance/article10/`; checksums, notices, and licences needed to
  authenticate or redistribute a direct protocol input remain beside it under
  `bench/assets/`;
- site-specific queues, allocation/account names, credentials, private paths,
  caches, checkpoints, logs, and results: keep them in runtime or private site
  state; portable local resource-limit presets remain part of the runner;
- vendored copies of external research source trees: do not retain them.

`src/modssc` must never import `bench`, and scientific execution must never
import code from a provenance bundle.

## Autonomous replication contract

A checkout of ModSSC must contain all source code and protocol metadata needed
to reproduce every supported article result. No clone of an author's repository,
external JAR, hand-installed Weka runtime, or manually prepared cache is a valid
hidden prerequisite.

Datasets may remain external because of their size and licenses, but every
protocol resolves them through a ModSSC provider. The protocol records provider
options and expected identity; preparation downloads or materializes the data
through ModSSC and fails explicitly when it cannot verify them. Optional Python
backends are declared as project extras.

Fixed splits, permutations, or graphs may be committed when they are direct,
compact inputs required by the publication protocol. Their checksums belong to
the protocol contract. Reference source code used during an audit does not.

## Install

```bash
python -m pip install -e "."
# or for development checks:
python -m pip install -e ".[dev]"
```

Install the declared extras required by the selected dataset and method. Do not
install dependencies ad hoc from an upstream paper repository.

An installed wheel exposes the same autonomous command as
`modssc-reproduce`; `python -m bench.reproduce` remains available as well.

## Run

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
python -m bench.main --config bench/configs/experiments/toy_transductive.yaml
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml --num-runs 5
```

Paper replications use the same runner with a card under
`bench/configs/reproductions/`. The autonomous entry point verifies fixed
resources, prepares the declared dataset through ModSSC, and then runs the card:

```bash
python -m bench.reproduce list
python -m bench.reproduce verify
python -m bench.reproduce run METHOD/PROTOCOL
```

Do not reconstruct a paper protocol from a standardized `best/` config.

`run.benchmark_mode` is disabled by default. Enable it for fully pinned
benchmark or replication contracts:

```yaml
run:
  benchmark_mode: true
```

## Cache and outputs

Benchmark presets use explicit environment variables for outputs and caches:

```bash
export MODSSC_OUTPUT_DIR=/tmp/modssc_runs
export MODSSC_DATASET_CACHE_DIR=/tmp/modssc_cache/datasets
export MODSSC_PREPROCESS_CACHE_DIR=/tmp/modssc_cache/preprocess
export MODSSC_GRAPH_CACHE_DIR=/tmp/modssc_cache/graph
```

Keep one shared cache root when reuse is desired. Fingerprints isolate
seed-dependent preprocess and graph artifacts. Use a different root for a strict
clean-room run.

The runner writes `config.yaml`, `run.json`, and `error.txt` on failure. A
multi-seed sweep also writes `aggregate.json` beside its child run directories.
These outputs are runtime evidence and must not be committed under `bench/`.

The standardized regime inventory remains in
`bench/configs/best/regime_manifest.yaml`.
