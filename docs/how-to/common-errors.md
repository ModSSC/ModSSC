# Common errors and where to go

Use this guide when you have an error message or a confusing symptom and want the shortest route to the right documentation page.


## Fast routing table

| Symptom or error | Most likely cause | Read this page first |
| --- | --- | --- |
| optional dependency error during dataset load or method execution | missing extra such as `graph`, `hf`, `preprocess-text`, or a torch backend | [Optional extras and platform support](../getting-started/extras-and-platforms.md) |
| `Unresolved environment variable(s)` while loading YAML | config contains `${VAR}` or `$VAR` placeholders that were not exported | [Configuration reference](../reference/configuration.md) |
| `Dataset not found` or unclear dataset ID | wrong dataset key or wrong provider assumptions | [Catalogs and registries](../reference/catalogs.md) |
| benchmark config rejected by schema validation | wrong field name, wrong block shape, or unsupported advanced field | [Configuration reference](../reference/configuration.md) |
| transductive run fails because no graph is available | graph block missing, graph build not run, or method kind does not match the data | [How to build graphs and views](graph.md) |
| `method.model.factory` rejected | trusted-only extension hook used without explicit opt-in | [Configuration reference](../reference/configuration.md) |
| split behavior is not the one you expected | confusion between official split policies and custom split policies | [How to create and reuse sampling splits](sampling.md) |
| rerun seems to reuse stale artifacts | cache reuse is happening across identical fingerprints | [Reproducibility](reproducibility.md) |
| `modssc` works but `bench/` configs or examples are missing | package installed from PyPI without repository assets | [Installation](../getting-started/installation.md) |
| command exists but available methods or datasets look incomplete | missing extras or registry filtering with `--available-only` | [Catalogs and registries](../reference/catalogs.md) |


## First commands to run
Start with the shortest environment checks:

```bash
modssc doctor
modssc --help
modssc datasets list
```

If the problem comes from a benchmark YAML:

```bash
python -m bench.main --config path/to/config.yaml --log-level detailed
```


## How to decide between this page and troubleshooting
- Use this page when you mainly need routing.
- Use [Troubleshooting](troubleshooting.md) when you already know the subsystem and need a more detailed diagnosis path.


## Terms that often cause confusion
If the error message mentions a policy or config field you do not recognize, check the [Glossary](../getting-started/glossary.md) before diving into the schema or the code.


## Related links
- [Troubleshooting](troubleshooting.md)
- [Glossary](../getting-started/glossary.md)
- [CLI reference](../reference/cli.md)
- [Configuration reference](../reference/configuration.md)
- [Catalogs and registries](../reference/catalogs.md)


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/how-to/troubleshooting.md"><code>docs/how-to/troubleshooting.md</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/reference/configuration.md"><code>docs/reference/configuration.md</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/reference/catalogs.md"><code>docs/reference/catalogs.md</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/getting-started/extras-and-platforms.md"><code>docs/getting-started/extras-and-platforms.md</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/how-to/reproducibility.md"><code>docs/how-to/reproducibility.md</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/docs/getting-started/installation.md"><code>docs/getting-started/installation.md</code></a></li>
</ol>
</details>
