# Reproducibility

Use this guide when you want runs that are easy to rerun, compare, and audit. In ModSSC, reproducibility comes from a combination of fixed seeds, saved configs, stable fingerprints, and disciplined cache usage.


## What ModSSC fingerprints for you
- Datasets are identified and cached from their resolved provider, dataset ID, version, and resolved options.
- Sampling outputs are driven by the sampling plan plus the split seed.
- Preprocess cache entries depend on the dataset fingerprint, resolved preprocess plan, fit subset, and preprocess seed.
- Graph cache entries depend on the dataset fingerprint, preprocess fingerprint, graph spec, and graph seed.

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
- If you change code inside preprocess or graph implementations while keeping the same config, clear the impacted cache folders before comparing outputs.
- Treat benchmark configs and caches as trusted local artifacts only.


## Practical checklist
1. Pin the same commit, Python version, and dependency profile.
2. Save the benchmark `config.yaml` next to the produced `run.json`.
3. Keep `run.seed` fixed unless you are intentionally doing a seed sweep.
4. Save sampling artifacts when you need to reuse the exact same split outside the bench runner.
5. Prefer CPU or a fixed accelerator stack for exact comparisons.
6. Start from a known-good config from the [Bench config cookbook](bench-cookbook.md).


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
</ol>
</details>
