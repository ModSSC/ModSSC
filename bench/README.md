# ModSSC benchmark runner

`bench/` is a thin, declarative execution layer. It validates one YAML card,
constructs a pipeline from registered `modssc` components, checks their declared
capabilities, executes them, and writes the results. It does not implement a
method, reproduce an article-specific code path, or manage a compute campaign.

## Boundary

Scientific behaviour belongs under `src/modssc/`:

- sampling and split semantics in `modssc.sampling`;
- preprocessing in `modssc.preprocess`;
- views, augmentation, and graph construction in their native bricks;
- inductive, transductive, and supervised methods in their registries;
- aggregation, execution identity, resume, and checkpoints in native runtime
  and evaluation modules.

`bench/` contains only:

- the YAML schema and generic runner;
- stage orchestrators that call those public bricks;
- example, standardized, and paper-reproduction cards;
- serialization of native run, execution-contract, reconciliation, and
  acceptance reports.

The runner must not branch on a method identifier, paper profile, dataset, or
cluster. Compatibility is determined from declared method and pipeline
capabilities. Adding a method-specific behaviour to `bench/` is an architecture
error: add it to the appropriate `src/modssc` brick, expose parameters in the
schema, and select it from YAML.

## Reproduction contract

A card in `configs/reproductions/` is executable through the same entry point as
any other config. Datasets are prepared separately, then authenticated by the
runner. When `dataset.integrity` is declared, the native data-loader verifier
rehashes the cached bytes before the runner compares the fingerprint, content
digest, and manifest digest, then rehashes them again before publishing the run
result. Splits, preprocessing outputs, graph topology, and checkpoints are
recomputed by native components whenever their defining algorithm is known.
Committed result bundles or paper-specific index/graph files are not runtime
dependencies.

After the environment and dataset cache have been prepared, the YAML is the
only execution input. A cache miss trains or builds every declared native
stage before running the method; a cache hit reuses the authenticated output.
Stages whose configuration and input content are identical therefore share a
single cache entry across methods, label budgets, and sampling seeds.

Every article result is assessed from the current source and card. Recovered
runs, scores, verdicts, split files, and generated graph artifacts are not
inputs to a fresh replication.

Immediately before the method starts, native execution composes the resolved
method requirements with the exact materialized inputs and bound model
contracts. A proven incompatibility always fails before `fit`. A missing proof
is reported as `unverified` and strict benchmark cards fail closed. `run.json`
stores the full canonical report under
`artifacts.method.execution_contract`, its SHA-256 beside it, and a compact
status/digest reference in `resolution`. `bench` only copies this native report;
it does not reinterpret model or modality requirements.

Cards that make a numerical paper-replication claim carry their declarative
`acceptance` block directly in the same YAML file. The block binds a protocol
and method, the exact repetition count, scientific-conformity evidence, primary
and optional secondary/diagnostic targets, required diagnostics, documented
deviations, equivalences, unknowns, and a fidelity ceiling. It is not stored in
a separate campaign registry. All 20 active reproduction cards carry an
acceptance block. The five bounded paper canaries remain under
`bench/configs/diagnostics/paper_canaries/` and cannot enter a paper result.

The schema verifies that an acceptance card runs in benchmark mode, that its
`method_id` matches `method.id`, and that `run.seeds` contains exactly the
declared repetitions. The acceptance mathematics lives in
`modssc.evaluation.acceptance`, not in `bench`. After seed reports have been
authenticated and reconciled, the runner calls that public API and serializes
its canonical result as `aggregate.json.acceptance`. The result contains
`assessment_status` (`passed`, `failed`, or `not_evaluable`),
`fidelity_status` (`paper_matched`, `paper_approx`, or `not_claimable`), the
fidelity ceiling, gate details and reasons, and `acceptance_sha256`.

## Run

Install the locked development environment, prepare the dataset, then use the
single runner:

```bash
python -m pip install -e ".[dev,full]"
modssc datasets download --dataset mnist --cache-dir "$MODSSC_DATASET_CACHE_DIR"
modssc-bench --config bench/configs/reproductions/laplace_learning/mnist-table1-1-label-per-class.yaml --seed-index 0
```

The module form is equivalent:

```bash
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml
```

Use `--seed-index N` to execute exactly one element of `run.seeds`. Use the card
without that option only when the complete sweep is intentional. Long paper
sweeps must be launched only after the short validation gate and explicit
approval.

Reports produced independently with `--seed-index` can be reconciled later from
an explicit root without any compute-platform metadata:

```bash
modssc-bench reconcile \
  --config bench/configs/reproductions/laplace_learning/mnist-table1-1-label-per-class.yaml \
  --runs-root /path/to/results
```

The command searches the root recursively for `run.json` and writes
`aggregate.json` there; `--output-dir` can select another destination. The
card's `run.seeds` remains the source of truth. Successful, failed,
not-evaluable, and missing seeds are reported as four disjoint categories.
Duplicate reports and reports for undeclared seeds are rejected as ambiguous.
Each report's requested-config hash must match the seed-specific configuration
derived from the card. The report stores the exact effective configuration used
after limits and HPO, rather than a parser-normalized reconstruction. The
reconciler recalculates `effective_config_hash` and `protocol_sha256` from that
`config` payload and `software_sha256` from `versions`; a falsified payload or
declared digest is rejected. Reports with an incomplete schema or a different
software identity are also rejected. Only successful reports contribute to
metric aggregation. If the card declares `acceptance`, the native evaluator
also checks cohort completeness, targets, diagnostics, conformity, deviations,
unknowns, and the fidelity ceiling. An incomplete or unresolved cohort remains
`not_evaluable`; it is never silently converted to failure or success. Both a
local sweep and `reconcile` return a non-zero status unless the ordinary sweep
and the declared acceptance assessment pass.

## Runtime state

Keep outputs, datasets, caches, and checkpoints outside the repository:

```bash
export MODSSC_OUTPUT_DIR=/path/to/results
export MODSSC_DATASET_CACHE_DIR=/path/to/cache/datasets
export MODSSC_PREPROCESS_CACHE_DIR=/path/to/cache/preprocess
export MODSSC_GRAPH_CACHE_DIR=/path/to/cache/graph
export MODSSC_ARTIFACT_ROOT=/path/to/read-only/external-inputs
```

The runner writes `config.yaml`, `run.json`, and `error.txt` on failure. A seed
sweep also writes `aggregate.json`. Resume behaviour is explicit in
`run.resume_policy`; checkpoints use the native execution identity rather than
an external campaign identifier. These runtime files must not be committed
under `bench/`.

When a local sweep stops early under `fail_fast`, its partial `aggregate.json`
is written before the original error is re-raised. Completed, failed,
not-evaluable, and still-missing seeds therefore remain inspectable.

External files used by a protocol can be declared generically under
`run.input_artifacts` with a relative `path`, `kind: file|tree`, and SHA-256;
`run.artifact_root` supplies the machine-local root. The root is operational,
while the declarations remain protocol identity. The runner rehashes them at
preflight and again before recording success.
