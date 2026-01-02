# CLI reference

This page is the authoritative CLI reference for ModSSC commands and options. For registry lists and IDs, see [Catalogs and registries](catalogs.md).

Each command section below follows the same pattern (purpose, syntax, options, examples), so you can skim and copy quickly. Use the navigation to jump to the brick you need.


## How the CLI is installed and invoked
The CLI entry points are defined in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml) and implemented as Typer apps in [`src/modssc/cli/`](https://github.com/ModSSC/ModSSC/tree/main/src/modssc/cli). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


Primary entry point:

```bash
modssc --help
modssc --version
```

Direct entry points (also declared in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml)): <sup class="cite"><a href="#source-1">[1]</a></sup>

Use `modssc` when you want a single entry point with shared logging and `doctor`. Use the direct entry points when you prefer shorter commands or want to scope scripts to a single brick. Both map to the same Typer apps. For lists of datasets, steps, methods, and metrics, see [Catalogs and registries](catalogs.md). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a><a href="#source-3">[3]</a></sup>

```bash
modssc-datasets --help
modssc-sampling --help
modssc-preprocess --help
modssc-graph --help
modssc-inductive --help
modssc-transductive --help
modssc-augmentation --help
modssc-evaluation --help
```

## Commands and subcommands
### modssc
- Purpose: Root CLI that wires all bricks and provides `doctor` and `--version`. <sup class="cite"><a href="#source-3">[3]</a></sup>

- Syntax: `modssc [--version] [--log-level <level>] <command> [OPTIONS]`
- Options:
  - `--version`: print the package version and exit.
  - `--log-level` / `--log`: logging level (none, basic, detailed).
- Examples:

```bash
modssc doctor
modssc --log-level detailed datasets list
```

### modssc doctor
- Purpose: Report which optional CLI bricks are available and which extras are missing. <sup class="cite"><a href="#source-3">[3]</a></sup>

- Syntax: `modssc doctor [--json]`
- Options:
  - `--json`: emit machine-readable JSON.
- Examples:

```bash
modssc doctor
modssc doctor --json
```

### modssc datasets
- Purpose: List, inspect, and download datasets plus manage cache. <sup class="cite"><a href="#source-4">[4]</a></sup>

- Syntax: `modssc datasets <providers|list|info|download|cache> [OPTIONS]`
- Options (selected):
  - `list --modalities <modality>`
  - `info --dataset <id>`
  - `download --dataset <id> | --all` plus `--force`, `--cache-dir`, `--skip-cached`, `--modalities`.
- Examples:

```bash
modssc datasets list
modssc datasets info --dataset toy
```

### modssc datasets cache
- Purpose: Inspect and clean the dataset cache. <sup class="cite"><a href="#source-4">[4]</a></sup>

- Syntax: `modssc datasets cache <ls|purge|gc> [OPTIONS]`
- Options:
  - `ls --cache-dir <path>`
  - `purge <dataset_or_fp> [--fingerprint]`
  - `gc [--keep-latest/--no-keep-latest]`
- Examples:

```bash
modssc datasets cache ls
modssc datasets cache purge toy
```

### modssc sampling
- Purpose: Create and inspect deterministic SSL splits. <sup class="cite"><a href="#source-5">[5]</a></sup>

- Syntax: `modssc sampling <create|show|validate> [OPTIONS]`
- Options:
  - `create --dataset <id> --plan <file> --out <dir> [--seed <n>] [--overwrite]`
  - `show <split_dir>`
  - `validate <split_dir> --dataset <id>`
- Examples:

```bash
modssc sampling create --dataset toy --plan sampling_plan.yaml --out splits/toy
modssc sampling show splits/toy
```

### modssc preprocess
- Purpose: Run preprocessing plans and inspect registries. <sup class="cite"><a href="#source-6">[6]</a></sup>

- Syntax: `modssc preprocess <steps|models|run> [OPTIONS]`
- Options:
  - `steps list [--json]`
  - `steps info <step_id>`
  - `models list [--modality <modality>] [--json]`
  - `models info <model_id>`
  - `run --plan <file> --dataset <id> [--seed <n>] [--no-cache] [--purge-unused]`
- Examples:

```bash
modssc preprocess steps list
modssc preprocess run --plan preprocess_plan.yaml --dataset toy
```

### modssc graph
- Purpose: Build graphs and graph-derived views; inspect caches. <sup class="cite"><a href="#source-7">[7]</a></sup>

- Syntax: `modssc graph <build|views|cache> [OPTIONS]`
- Options (build):
  - `--dataset <id>`
  - `--spec <file>` (optional; full spec supports symmetrize/weights/normalize/self_loops). <sup class="cite"><a href="#source-8">[8]</a></sup>

  - `--scheme knn|epsilon|anchor`, `--metric cosine|euclidean`, `--k`, `--radius`, `--backend auto|numpy|sklearn|faiss`
  - `--chunk-size`, `--n-anchors`, `--anchors-k`, `--anchors-method`, `--candidate-limit`
  - `--faiss-exact`, `--faiss-hnsw-m`, `--faiss-ef-search`, `--faiss-ef-construction`
  - `--seed`, `--cache`, `--cache-dir`, `--edge-shard-size`, `--resume`
- Examples:

```bash
modssc graph build --dataset toy --scheme knn --metric euclidean --k 8
modssc graph views build --dataset toy --views attr diffusion
```

### modssc graph views
- Purpose: Build graph-derived views and inspect the views cache. <sup class="cite"><a href="#source-7">[7]</a></sup>

- Syntax: `modssc graph views <build|cache-ls> [OPTIONS]`
- Options (build, selected):
  - `--dataset <id>`
  - `--views <name>` (repeatable; attr, diffusion, struct)
  - `--diffusion-steps`, `--diffusion-alpha`
  - `--struct-method`, `--struct-dim`, `--walk-length`, `--num-walks-per-node`, `--window-size`, `--p`, `--q`
  - `--scheme`, `--metric`, `--k-graph`, `--radius`
- Examples:

```bash
modssc graph views build --dataset toy --views attr --views diffusion --diffusion-steps 5
modssc graph views cache-ls
```

### modssc graph cache
- Purpose: Inspect or purge graph caches. <sup class="cite"><a href="#source-7">[7]</a></sup>

- Syntax: `modssc graph cache <ls|purge>`
- Examples:

```bash
modssc graph cache ls
modssc graph cache purge
```

### modssc augmentation
- Purpose: List augmentation ops and inspect defaults. <sup class="cite"><a href="#source-9">[9]</a></sup>

- Syntax: `modssc augmentation <list|info> [OPTIONS]`
- Options:
  - `list [--modality <modality>]`
  - `info <op_id> [--as-json]`
- Examples:

```bash
modssc augmentation list --modality text
modssc augmentation info text.word_dropout --as-json
```

### modssc evaluation
- Purpose: List metrics and compute scores from `.npy` files. <sup class="cite"><a href="#source-10">[10]</a></sup>

- Syntax: `modssc evaluation <list|compute> [OPTIONS]`
- Options:
  - `list [--json]`
  - `compute --y-true <path> --y-pred <path> [--metric <name>] [--json]`
- Examples:

```bash
modssc evaluation list
modssc evaluation compute --y-true y_true.npy --y-pred y_pred.npy --metric accuracy
```

### modssc inductive
- Purpose: Inspect inductive method registry. <sup class="cite"><a href="#source-11">[11]</a></sup>

- Syntax: `modssc inductive methods <list|info> [OPTIONS]`
- Options:
  - `list [--all/--available-only]`
  - `info <method_id>`
- Examples:

```bash
modssc inductive methods list
modssc inductive methods info pseudo_label
```

### modssc transductive
- Purpose: Inspect transductive method registry. <sup class="cite"><a href="#source-12">[12]</a></sup>

- Syntax: `modssc transductive methods <list|info> [OPTIONS]`
- Options:
  - `list [--all/--available-only]`
  - `info <method_id>`
- Examples:

```bash
modssc transductive methods list
modssc transductive methods info label_propagation
```

### modssc supervised
- Purpose: List supervised baselines and their backends. <sup class="cite"><a href="#source-13">[13]</a></sup>

- Syntax: `modssc supervised <list|info> [OPTIONS]`
- Options:
  - `list [--available-only] [--json]`
  - `info <classifier_id>`
- Examples:

```bash
modssc supervised list --available-only
modssc supervised info logreg
```

## Related links
- [Catalogs and registries](catalogs.md)
- [Dataset how-to](../how-to/datasets.md)
- [Preprocess how-to](../how-to/preprocess.md)

<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/tree/main/src/modssc/cli"><code>src/modssc/cli/</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/app.py"><code>src/modssc/cli/app.py</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/datasets.py"><code>src/modssc/cli/datasets.py</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/sampling.py"><code>src/modssc/cli/sampling.py</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/preprocess.py"><code>src/modssc/cli/preprocess.py</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/graph.py"><code>src/modssc/cli/graph.py</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/graph/specs.py"><code>src/modssc/graph/specs.py</code></a></li>
  <li id="source-9"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/augmentation.py"><code>src/modssc/cli/augmentation.py</code></a></li>
  <li id="source-10"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/evaluation.py"><code>src/modssc/cli/evaluation.py</code></a></li>
  <li id="source-11"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/inductive.py"><code>src/modssc/cli/inductive.py</code></a></li>
  <li id="source-12"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/transductive.py"><code>src/modssc/cli/transductive.py</code></a></li>
  <li id="source-13"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/cli/supervised.py"><code>src/modssc/cli/supervised.py</code></a></li>
</ol>
</details>
