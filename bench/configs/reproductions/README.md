# Paper-reproduction cards

These 20 YAML files describe the ten article-replication protocols through the
generic benchmark schema. They select registered `src/modssc` components and
their parameters; they do not select a second execution engine. A
`method.profile: paper:...` value is descriptive provenance only and never
changes dispatch.

## What a card guarantees

Each card pins the dataset identity, sampling protocol, preprocessing, optional
views/augmentation/graph stages, method parameters, seeds, and reported
metrics. `dataset.download: false` keeps preparation separate from scientific
execution:

```bash
modssc datasets download --dataset DATASET_ID --cache-dir "$MODSSC_DATASET_CACHE_DIR"
modssc-bench --config bench/configs/reproductions/METHOD/CARD.yaml --seed-index 0
```

`--seed-index` addresses one entry of the card's `run.seeds`; it is the direct
interface used by local scripts and schedulers. No manifest generator, article
audit module, or provenance directory is required at runtime.

## Native protocol reconstruction

The executable cards no longer depend on packaged split, permutation, or graph
artifacts when ModSSC can reproduce their construction:

- Calder cards train the declared GraphLearning MNIST VAE2 recipe, query an
  Annoy index with the archived 2022 `10 trees / 30 candidates / search_k=-1`
  contract, retain the first ten neighbours including self, and symmetrize as
  `(W + W.T) / 2`. All ten cards share this cached graph. Poisson removes its
  diagonal inside the native paper solver, while Laplace consumes the same
  graph unchanged;
- FixMatch reconstructs Google's class-balanced stream, legacy seeded
  selection, one-item validation holdout, and inclusive unlabeled pool through
  the sampling plan;
- FlexMatch, FreeMatch, and SoftMatch reconstruct the TorchSSL seeded pools and
  inclusive unlabeled semantics through the same sampling API;
- Democratic Co-Learning draws complete deterministic native partitions. Its
  method and classifier behaviour live in `src/modssc`, not in a
  card-specific runner.

This makes the protocol reusable: another registered method may consume the
same modality and stages whenever its capability contract is satisfied.

## Claim boundary

Every result starts from a fresh execution of the current card and current
source. No recovered run, score, verdict, split file, graph array, or campaign
artifact is an input. The `fidelity_ceiling` field is only the maximum claim
allowed by source knowledge; it is never an execution result.

Source-level evidence is kept under
`docs/replications/protocols/`. Those pages identify the article, official
code when available, registered protocol, and unresolved ambiguities. They
contain no ModSSC results.

Known protocol ambiguity or backend deviation remains in the YAML and the
fresh result report. Missing seeds, failed runs, non-convergence, and
`not_evaluable` outcomes are reported rather than replaced or imputed. Result
files use strict JSON: a non-finite metric is represented by `null` together
with the explicit `not_evaluable` status.

## Long runs and Jean Zay

Do not launch a full Calder 100-seed cell, a full Match training trajectory, or
another costly reproduction before its bounded gates. First validate
configuration loading, capability compatibility, dataset identity, native
preprocessing/graph construction, checkpoint creation, and one short canary.
Production lots are promoted progressively only after their dependent gates
pass.

On Jean Zay, the scheduler wrapper remains private operational glue and invokes
the public command directly, one array index per seed:

```bash
modssc-bench --config "$CARD" --seed-index "$SLURM_ARRAY_TASK_ID"
```

Account, partition, modules, cache roots, output roots, and wall time belong to
the private job script. They must not be encoded in a reproduction card.
