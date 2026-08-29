# Reproducibility

Use this guide when you want runs that are easy to rerun, compare, and audit. In ModSSC, reproducibility comes from a combination of fixed seeds, saved configs, stable fingerprints, and disciplined cache usage.


## What ModSSC fingerprints for you
- Datasets are identified and cached from their resolved provider, dataset ID, version, and resolved options.
- Sampling outputs are driven by the sampling plan plus the split seed.
- Preprocess cache entries depend on the dataset fingerprint, resolved preprocess plan, fit subset, and preprocess seed.
- Graph cache entries depend on the dataset fingerprint, preprocess fingerprint, graph spec, and graph seed.
- Reproduction cards can additionally pin the full cached dataset content; the
  runner rehashes it before execution and again before publishing success.
- Every run records effective-config, protocol, selected-software, and composed
  method/input/model contract hashes.
- A seed aggregate for a card with `acceptance` records the full canonical
  native acceptance report and its `acceptance_sha256`.

This means cache reuse is precise when the inputs match and intentionally broken when a meaningful upstream input changes.


## Seed strategy
- Use `run.seed` as the root seed in benchmark configs.
- When you need stage-specific control, set `sampling.seed`, `preprocess.seed`, `views.seed`, `graph.seed`, `augmentation.seed`, and `search.seed` explicitly.
- For multi-seed sweeps, keep one shared cache root and let the fingerprints separate seed-specific artifacts.

For strict comparisons, prefer one committed config per experiment family and vary only one dimension at a time.


## Strong vs best-effort determinism
- Sampling is the strongest reproducibility layer because it is fully driven by deterministic plans and explicit seeds.
- Torch-based inductive methods use a best-effort deterministic seeding helper that also enables deterministic algorithms when supported.
- Torch-based transductive GNN helpers seed torch, but backend- and device-specific kernels can still introduce small variations.
- For regression checks where exact reruns matter, prefer CPU runs or the same hardware/backend stack.


## Cache discipline
- Reuse one shared `MODSSC_CACHE_ROOT` when you want fast reruns of the same experiment family.
- Use a fresh `MODSSC_CACHE_ROOT` only when you explicitly want a clean-room comparison.
- Preprocess and graph cache identities include their relevant producer code
  and selected dependency versions. A code change invalidates the affected
  generation; manual deletion is not required for correctness.
- Treat benchmark configs and caches as trusted local artifacts only.


## Practical checklist
1. Pin the same commit, Python version, and dependency profile.
2. Save the benchmark `config.yaml` next to the produced `run.json`.
3. Keep `run.seed` fixed unless you are intentionally doing a seed sweep.
4. Save sampling artifacts when you need to reuse the exact same split outside the bench runner.
5. Prefer CPU or a fixed accelerator stack for exact comparisons.
6. Start from a known-good config from the [Bench config cookbook](bench-cookbook.md).
7. For a paper result, reconcile the complete declared seed set and inspect
   `failed`, `not_evaluable`, and `missing` separately before interpreting the
   aggregate.
8. Verify that every strict run's execution contract is `compatible`; never
   reinterpret `unverified` as evidence of compatibility.
9. Read `aggregate.json.acceptance.assessment_status` separately from
   `fidelity_status`. A numerical `passed` result can remain `paper_approx`, and
   an incomplete cohort must remain `not_evaluable`/`not_claimable`.
10. Verify `acceptance_sha256` before transferring or citing the assessment;
    the digest identifies the canonical native report, not the historical
    result bundle that motivated the card.
11. Require each new `run.json` to carry a portable `execution_identity` and
    matching digest. Legacy opt-in exists only to inspect historical reports;
    removing both identity fields must never make a new cohort certifiable.
    Confirm `aggregate.json.sweep.execution_identity_complete` before treating
    the reconciliation as native evidence.


## Acceptance and historical evidence

For a supported numerical reproduction, the `acceptance` block is stored
directly in its YAML card. This keeps the repetition count, conformity review,
published targets, diagnostic predicates, deviations, equivalences, unknowns,
and fidelity ceiling inside the protocol identity. `bench` parses that block,
authenticates and reconciles seed reports, then delegates all acceptance
mathematics to `modssc.evaluation.evaluate_acceptance`; it only serializes the
returned report. <sup class="cite"><a href="#source-10">[10]</a><a href="#source-11">[11]</a></sup>

Do not copy a verdict from an older campaign into a fresh aggregate. Historical
reports remain evidence for the source, inputs, software, and hardware identity
that produced them. Recomputed cards need a complete fresh cohort. Missing or
non-successful repetitions and unresolved conformity yield `not_evaluable`;
failed numerical/diagnostic gates yield `failed`; only an evaluable passing
cohort yields `passed`. Fidelity is then classified independently as
`paper_matched`, `paper_approx`, or `not_claimable`.


## Related links
- [Benchmarks](../reference/benchmarks.md)
- [Bench config cookbook](bench-cookbook.md)
- [Configuration reference](../reference/configuration.md)
- [Troubleshooting](troubleshooting.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/utils/io.py"><code>bench/utils/io.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/services/service.py"><code>src/modssc/sampling/services/service.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/storage.py"><code>src/modssc/sampling/storage.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/cache.py"><code>src/modssc/preprocess/cache.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/cache.py"><code>src/modssc/graph/cache.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/inductive/seed.py"><code>src/modssc/inductive/seed.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/transductive/methods/gnn/common.py"><code>src/modssc/transductive/methods/gnn/common.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/context.py"><code>bench/context.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/orchestrators/reporting.py"><code>bench/orchestrators/reporting.py</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/evaluation/acceptance.py"><code>src/modssc/evaluation/acceptance.py</code></a></li>
</ol>
</details>
