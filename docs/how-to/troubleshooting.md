# Troubleshooting

Use this guide when a run fails and you want the shortest route to a diagnosis.


## First diagnostic commands

```bash
modssc doctor --json
modssc datasets info --dataset toy
modssc preprocess steps list
modssc graph build --help
python -m bench.main --config bench/configs/experiments/toy_inductive.yaml --log-level detailed
```


## Common failure patterns

### `modssc` works but `bench/` commands or configs are missing
The benchmark runner lives in the repository and is not shipped to PyPI. Use a source checkout when you need `bench.main`, authored configs, examples, or notebooks.

### Optional dependency error during dataset load, preprocess, or method execution
Install the extra suggested by the error message. Dataset specs expose `required_extra`, preprocess steps expose step metadata, and method registries rely on the extras declared in `pyproject.toml`.

### `Unresolved environment variable(s)` when loading a YAML config
Your config contains placeholders like `${MODSSC_OUTPUT_DIR}` or `${MODSSC_CACHE_ROOT}` that are not set in the environment. Export them first or remove the placeholders from the YAML.

### A transductive run fails because no graph is available
Enable the `graph:` block for non-graph datasets, or switch to a graph dataset/provider that already supplies graph structure.

### A graph config is rejected
Graph specification validation is strict. If a backend/scheme combination is unsupported, start from a known-good config in the [Bench config cookbook](bench-cookbook.md) and change one field at a time.

### The test split does not come from the place you expected
`respect_official_test: true` keeps `dataset.test` when it exists. `allow_override_official: true` tells ModSSC to ignore the provider test split for inductive datasets and apply the user split to `dataset.train` instead.

### `method.model.factory` is rejected
This extension hook is disabled by default. You must set `run.allow_custom_factories: true` explicitly, and you should only do that for configs you control.

### A rerun seems to reuse stale artifacts
Check which cache root is active (`MODSSC_CACHE_ROOT` or per-cache overrides). Dataset, preprocess, graph, and views caches are separate layers. If code changed inside one stage, clear only the impacted cache subtree and rerun.


## Trusted-input boundaries
- Benchmark configs are trusted local inputs.
- Custom factories are for trusted configs only.
- Dataset and preprocess cache artifacts are trusted local artifacts only.

If you are debugging behavior and you do not trust a cached artifact, clear the relevant cache entry and regenerate it.


## When to escalate from CLI to code inspection
- Use the CLI and cookbook configs first when you are debugging environment or schema issues.
- Use the Python API when you need to inspect intermediate objects such as sampling stats, preprocess outputs, or graph artifacts.
- Use the [Architecture](../development/architecture.md) page when you need to know which package path is public and which one is internal.


## Related links
- [Common errors and where to go](common-errors.md)
- [Glossary](../getting-started/glossary.md)
- [Optional extras and platform support](../getting-started/extras-and-platforms.md)
- [Bench config cookbook](bench-cookbook.md)
- [Reproducibility](reproducibility.md)
- [Configuration reference](../reference/configuration.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/app.py"><code>src/modssc/cli/app.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/errors.py"><code>src/modssc/data_loader/errors.py</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/utils/io.py"><code>bench/utils/io.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/schema.py"><code>bench/schema.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py"><code>src/modssc/graph/specs.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/plan.py"><code>src/modssc/sampling/plan.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/sampling/services/service.py"><code>src/modssc/sampling/services/service.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/data_loader/storage/files.py"><code>src/modssc/data_loader/storage/files.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/preprocess/cache.py"><code>src/modssc/preprocess/cache.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/bench/README.md"><code>bench/README.md</code></a></li>
</ol>
</details>
