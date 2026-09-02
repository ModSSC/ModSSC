# Evaluation API

This page documents the evaluation API. For workflows, see [Evaluation how-to](../../how-to/evaluation.md).


## What it is for
The evaluation brick provides metric implementations and helpers for labels or score matrices. <sup class="cite"><a href="#source-1">[1]</a></sup>


## Examples
List metrics:

```python
from modssc.evaluation import list_metrics

print(list_metrics())
```

Evaluate accuracy and macro F1:

```python
import numpy as np
from modssc.evaluation import evaluate

y_true = np.array([0, 1, 1])
y_pred = np.array([0, 1, 0])
print(evaluate(y_true, y_pred, ["accuracy", "macro_f1"]))
```

Metrics are implemented in [`src/modssc/evaluation/metrics.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/metrics.py). <sup class="cite"><a href="#source-1">[1]</a></sup>


## Fitted-method evaluation

`evaluate_inductive_method` and `evaluate_transductive_method` are the native,
runner-independent evaluation entrypoints. Callers provide selected named
splits; the evaluation brick owns prediction payloads, metric computation,
named diagnostic outputs, backend materialization, and the separation between
transductive fit data and evaluation-only truth. <sup class="cite"><a href="#source-2">[2]</a></sup>

A fitted method may implement `predict_evaluation_proba(payload)` when its
reported predictor differs from the general `predict_proba` default. Torch
model bundles use this contract to honor `predict_with_ema`: the paper-facing
metric is then computed from the EMA model while ordinary `predict_proba`
continues to mean the terminal student. Additional outputs use the generic
`predict_evaluation_outputs(payload)` mapping; benchmark code never inspects a
method's private classifiers or dispatches on a method ID. <sup class="cite"><a href="#source-2">[2]</a></sup>

Methods with a historical reporting rule may also expose
`evaluation_metric_sets()`. The direct final-model result stays under
`metrics.<split>`; checkpoint-terminal statistics live under
`metrics.terminal.<split>`; and the statistic reported by the paper lives under
`metrics.reported.<split>`. Policy and test-selection flags travel beside the
reported values as non-numeric metadata. Generic seed aggregation therefore
aggregates all three numeric paths without mistaking the paper statistic for a
fresh terminal prediction. <sup class="cite"><a href="#source-2">[2]</a></sup>

`MethodEvaluationRuntime` records the fitted feature backend and device. Native
inductive execution exposes it both in `InductiveExecutionResult` and as the
public fitted field `evaluation_runtime_`, avoiding model-object introspection
during later evaluation. <sup class="cite"><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>


## Repeated-seed reconciliation

`reconcile_seed_reports` is the runner-independent API for joining reports from
seeds executed separately. It validates the declared seed set, rejects duplicate
or unexpected observations, partitions seeds into `success`, `failed`,
`not_evaluable`, and `missing`, and aggregates numeric metric leaves from
successful reports only. Callers may provide `expected_config_hashes` to reject
reports produced from a different seed-specific configuration and
`expected_protocol_hashes` to validate protocols whose identity legitimately
varies with the seed. Every report must carry a complete identity; software
identity must be homogeneous across the cohort. Reconciliation also recomputes
the effective-configuration and protocol digests from each report's exact
`config` payload, and the software digest from `versions`; declared digests are
not trusted. The partition and its counts are available through
`SeedReconciliation.categories()` and `SeedReconciliation.summary()`.

Portable execution identity is required by default. Reading reports written
before that field existed requires the explicit migration argument
`require_execution_identity=False`; such a cohort is not eligible for a new
native publication and its reconciliation remains non-certifiable.

```python
from modssc.evaluation import reconcile_seed_reports

result = reconcile_seed_reports(
    requested_seeds=[1, 2],
    reports=[run_for_seed_1, run_for_seed_2],
)
print(result.status, result.metrics)
```

Use `modssc-bench reconcile --config CARD --runs-root ROOT` when the reports are
stored as separate `run.json` files.


## Scientific acceptance

`AcceptanceSpec`, `parse_acceptance_spec`, and `evaluate_acceptance` form the
native, runner-independent scientific acceptance API. The evaluator receives a
declarative contract and an in-memory cohort of run payloads. It performs no
filesystem access and has no knowledge of YAML locations, schedulers, papers,
campaign names, or method-specific code paths. The caller remains responsible
for authenticating and reconciling reports before evaluation; the benchmark
runner does that before invoking this API. <sup class="cite"><a href="#source-4">[4]</a></sup>

```python
from modssc.evaluation import evaluate_acceptance, parse_acceptance_spec

spec = parse_acceptance_spec(acceptance_mapping)
report = evaluate_acceptance(spec, authenticated_run_reports)
print(report.assessment_status, report.fidelity_status)
print(report.acceptance_sha256)
```

The result deliberately separates two questions:

- `assessment_status` is `passed`, `failed`, or `not_evaluable`. An incomplete
  cohort, a non-successful repetition, a missing primary target, or unresolved
  scientific conformity cannot be evaluated. A complete evaluable cohort fails
  when its gating targets, required diagnostics, or conformity fail.
- `fidelity_status` is `paper_matched`, `paper_approx`, or `not_claimable`. It
  respects the declared fidelity ceiling, documented deviations, and critical
  unknowns instead of promoting a numerical pass automatically to a faithful
  paper match.

Primary and secondary numerical targets use the native Student 95% confidence
interval summary plus an absolute-margin gate. Diagnostic targets gate on their
margin without requiring the published mean to fall in the confidence
interval; informational targets are reported but do not decide the assessment.
Required diagnostic predicates are evaluated for every successful repetition.
The report includes all target summaries, diagnostic failures, conformity,
reasons, fidelity metadata, and `acceptance_sha256`. The digest covers the
canonical strict-JSON payload excluding the digest field itself, so a persisted
acceptance decision is self-identifying.


## API reference

::: modssc.evaluation

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/metrics.py"><code>src/modssc/evaluation/metrics.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/runtime.py"><code>src/modssc/evaluation/runtime.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/execution.py"><code>src/modssc/inductive/execution.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/acceptance.py"><code>src/modssc/evaluation/acceptance.py</code></a></li>
</ol>
</details>
