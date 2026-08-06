# Article replication audit

!!! note "Historical snapshot"
    This is the initial implementation audit from 2026-07-22. Its campaign
    statuses and test totals are superseded by
    `docs/development/article10-replication-summary.md`; it is retained only as
    an audit trail.

Audit date: 2026-07-22. Code baseline: `f5fab6b1cda030e74936b1bd1a81c60ae7be5c4d`.

Update, 2026-07-24: the original Adult pilot below remains historical
diagnostic evidence. The separate Vote profile now pins ModSSC's internal
historical-classifier semantics, uses the
test-blind first 20 eligible partition lock, and has an independently
transcribed Figure 1/2 equation oracle. Its ceiling is `paper_approx`, not
`paper_matched`; the article does not publish exact learner versions or its
confidence-interval construction, and no official-code parity is claimed. See
`docs/development/scientific-conformity-oracles.md`.

## Verdict

All 51 method identifiers exposed by the article have registered, importable
ModSSC implementations: 28 inductive and 23 transductive. This is implementation
coverage, not proof of scientific reproduction. At the time of this snapshot,
the paper-fidelity registry contained no `paper_matched` method.

The article package rebuilds an analysis of a public result snapshot; it does
not retrain the methods. Its visible results reduce each method and slice to the
best test accuracy from a single run rather than an average over paired seeds.
Several visible slices also mix class filters, task sizes, or preprocessing
pipelines, so the resulting rankings cannot yet serve as reproduction evidence.

## Method inventory

- Wrappers (10): `self_training`, `pseudo_label`, `noisy_student`,
  `meta_pseudo_labels`, `setred`, `co_training`, `democratic_co_learning`,
  `tri_training`, `deep_co_training`, `trinet`.
- Consistency (5): `pi_model`, `temporal_ensembling`, `mean_teacher`, `vat`,
  `uda`.
- Hybrid inductive (10): `mixmatch`, `fixmatch`, `flexmatch`, `softmatch`,
  `free_match`, `defixmatch`, `adamatch`, `adsh`, `comatch`, `daso`.
- Classical transductive (7): `label_propagation`,
  `dynamic_label_propagation`, `label_spreading`, `laplace_learning`,
  `lazy_random_walk`, `graphhop`, `graph_mincuts`.
- PDE / variational (3): `p_laplace_learning`, `poisson_learning`,
  `poisson_mbo`.
- GNN (12): `appnp`, `chebnet`, `gat`, `gcn`, `gcnii`, `grafn`, `grand`,
  `graphsage`, `h_gcn`, `n_gcn`, `planetoid`, `sgc`.
- Additional article surface (4): `s4vm`, `tsvm`, the `supervised` control,
  and the excluded representation method `simclr_v2`.

## Findings corrected in this audit

- Restored the missing scikit-learn entropy decision-tree backend required by
  the Democratic Co-Learning configuration and made nested classifier configs
  hydrate into typed specifications.
- Corrected Democratic Co-Learning's voting rule. The implementation previously
  chose a class by summed learner weight; the paper first requires a strict
  majority by vote count and only then applies its confidence-sum gate. Exact
  majority, confidence, tie, and single-class cases now have regression tests.
- Added protocol controls needed by the Adult study: uniformly random labeled
  examples, per-seed dataset subsampling, and per-seed preprocessing. The
  reproduction card now uses an unstratified random partition, matching the
  experiment description instead of the standardized benchmark defaults.
- Fixed a preprocessing-cache key collision: `core.ensure_2d` preferred an
  upstream `features.X` value while its cache identity tracked only `raw.X`.
  Different one-hot fit scopes could therefore reuse the same cached matrix.
- Corrected 36 native graph presets. APPNP, GAT, GCN, and SGC on Cora,
  CiteSeer, and PubMed now use 1/3/5 labels per class in R1/R2/R3 instead of 10.
- Added regression tests for the classifier restoration, cache isolation, and
  graph regime budgets.
- Added worktree-aware run provenance anchored to the executed ModSSC checkout.
  Reports now distinguish a clean commit from local tracked, staged, or
  untracked changes through an opaque SHA-256 fingerprint, without storing file
  paths or diff contents.
- Persisted Democratic Co-Learning convergence and pseudo-label diagnostics in
  each run report so that its paper-specific partition eligibility can be
  checked rather than assumed.
- Added the first tracked protocol-specific reproduction card for Democratic
  Co-Learning on Adult.

## Pilot experiment

Paper target: Adult accuracy `0.784 +/- 0.021`, 20 random partitions, 60 labeled
and 1,691 unlabeled examples, using Naive Bayes, C4.5, and 3-NN learners
([Zhou and Goldman, 2004](https://doi.org/10.1109/ICTAI.2004.48)). The paper
retained only partitions on which the method labeled at least one example.

The first current-code run could not start because nested classifier mappings
were not hydrated and `decision_tree` was absent from the supervised registry.
An independent algorithm audit then found and corrected the weighted-class vote
described above. The final diagnostic protocol run also randomizes the Adult
subsample, split, labeled set, and preprocessing for each of seeds 1--20.

That final run obtained accuracy `0.7478 +/- 0.0485` and macro-F1
`0.5591 +/- 0.0349`. Accuracy is `0.0362` below the paper mean and its
dispersion is about 2.31 times larger. All 20 runs labeled at least one example
and converged before the 20-iteration cap, so none failed the paper's stated
partition condition and the cap was not binding.

The internal behavior is also materially different from the published summary.
The pilot added, on average, `835.95`, `704.10`, and `781.50` pseudo-labels for
Naive Bayes, the entropy tree, and 3-NN, versus the paper's `413`, `130`, and
`353`; it changed labels over `8.75` rounds on average versus `2.6` reported
rounds. The exact C4.5 learner is missing, and the paper does not fully specify
the confidence-interval construction needed to resolve the remaining update
differences. This result is therefore a diagnostic paper-protocol approximation,
not an exact replication or a `paper_matched` result.

An older local aggregate reported `0.7850 +/- 0.0165`, but it was produced from
an uncommitted backend and a cache identity that did not include upstream
feature provenance. It is not valid golden evidence and should not be used as a
paper-match assertion.

The final rerun recorded the baseline commit together with `git_dirty: true`
and a non-null worktree fingerprint. The numerical result is therefore
traceable to the current local corrections, but should only become a frozen
artifact after those corrections are reviewed and committed.

## End-to-end standardized smoke campaign

Six R4 presets were executed for their five configured seeds on CPU with
isolated caches and no downloads. This verifies execution, prediction, and
aggregation for every article family; it does not reproduce the source papers,
and the values below are not a cross-method comparison.

| Family | Preset | Test accuracy (mean +/- std) |
|---|---|---:|
| Wrappers | `self_training / iris` | `0.9133 +/- 0.0499` |
| Consistency | `pi_model / iris` | `0.9200 +/- 0.0499` |
| Hybrid inductive | `fixmatch / iris` | `0.9067 +/- 0.0490` |
| Classical transductive | `label_propagation / iris` | `0.8000 +/- 0.0760` |
| PDE / variational | `poisson_learning / iris` | `0.4600 +/- 0.2620` |
| GNN | `sgc / cora` | `0.7620 +/- 0.0160` |

The five presets requesting CUDA were executed from temporary copies with only
the device changed to CPU; `poisson_learning` was already a CPU preset. All
30 runs completed successfully.

## Validation

- All 5,285 executable standardized benchmark configurations parse and satisfy
  the common five-seed, validation-selection, holdout, metric, and regime-budget
  checks.
- The full test suite passes: 2,993 passed, two skipped, with 100% statement and
  branch coverage over `src/modssc`.
- Ruff and Git whitespace validation pass.

## Remaining replication work

1. Pin the exact ModSSC and dashboard revisions, dataset artifacts, dependencies,
   hardware contract, splits, and paired seed list.
2. For Democratic Co-Learning, recover or implement exact C4.5 behavior and
   validate the paper's confidence-interval and error-estimate update details;
   make the runner enforce its partition eligibility rather than checking it
   only from diagnostics.
3. Extract a complete protocol card for every method; the archive currently has
   complete experiment notes for only a minority of methods.
4. Do not revert the 214 removed `graph_mincuts` / `lazy_random_walk` presets.
   Of these, 138 turn multiclass datasets into binary tasks with class filters,
   and eight additional lazy-random-walk Iris/Toy presets are invalid. Start a
   separate binary benchmark with the 20 light, verified Breast Cancer and
   YesNo R1-R5 candidates only after strict-config migration and five-seed smoke
   tests; keep the methods `not_claimable` until their paper protocols are
   completely extracted.
5. Execute the 5,285 standardized configs over five paired seeds after datasets
   and CUDA capacity are available; the current local cache covers only six of
   21 datasets.
6. Rebuild article tables from per-seed means and uncertainty, never from the
   best test run, and reject slices whose task contract differs.
