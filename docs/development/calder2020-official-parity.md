# Calder 2020 Table 1 parity

This note defines the claim boundary for the ten MNIST Table 1 cells shared by
`laplace_learning` and `poisson_learning`.

The code, archived inputs, and completed 100-repetition cells are
algorithmically and numerically matched. Both Laplace Learning and Poisson
Learning have the final status `paper_matched`; every published mean lies in
the corresponding replication confidence interval.

## Authenticated provenance and inputs

The repository records the exact GraphLearningOld v0.0.3 provenance and keeps
only compact numerical inputs required by the protocol. It does not distribute
or execute the upstream source tree:

| Item | Pin |
|---|---|
| Repository | `https://github.com/jwcalder/GraphLearningOld` |
| Commit | `04bece45cd512cf1a3bcddb163b767ca44a746e1` |
| Upstream tree digest | `e2d16b74ac7d9ba3daab1c2d020e97b268e26bc378fba1f1077bbfd8707a3372` |
| MNIST labels | `ec01dca8550a4bf9a4c8559c5c9c1c3ed5b8dd4fb9ab2e771883b03c8635ab2e` |
| VAE kNN arrays | `5b42bb234888c83eed763958a17fdfb8a55c09a2f0071b55a61635d86dc90db5` |
| Label permutations | `4d2f9949f4ce20d2644cb4c070766421751070dc625c05a0219b1c9d60045770` |
| Laplace result CSV | `894e3b33ae18bf0e43c5413dfe72b0e150f9a40d95027bba145fd309bf429b6b` |
| Poisson result CSV | `a20e0bc231fa0a05a8b1dc341d42b387e8b7129da63df28ecbf7e5f733be4374` |

`bench/campaign/protocols/calder/official.py` holds independent hard-coded hashes and rejects
an edited manifest, extra file, symlink, wrong array layout, changed dataset
order, malformed permutation, or Table 1 statistic that no longer matches the
archive. Commits, checksums, and licences are recorded under
`provenance/article10/`; the verified numerical inputs remain under
`bench/assets/calder2020/`.

`bench/campaign/protocols/calder/oracle.py` independently authenticates the
packaged numerical parity oracle by module identity and checksums. The portable
preparation module uses explicit package, dataset-cache, and cache-root paths;
it has no scheduler, site, Git, `WORK`, or `SCRATCH` contract.

## Shared data, permutations, and graph

- The official 60,000 training and 10,000 test examples are merged in that
  order into one transductive pool. Its labels must equal the archived
  70,000-element label vector exactly.
- There are 100 trials at each of 1--5 labels per class. Permutation row
  `trial * 5 + budget - 1` is used; rows are intentionally not assumed nested.
- The first 10 stored neighbors are selected, including the query vertex.
- Directed weights are
  `exp(-4 * distance**2 / distance_to_10th_neighbor**2)`.
- The graph is symmetrized as `(W + W.T) / 2`, is not normalized, and remains
  float64. Its diagonal is retained for Laplace and removed inside Poisson.

The full 70,000-node parity test independently reconstructs the archived
SciPy CSR matrix and compares `indptr`, `indices`, and `data` exactly. Both
matrices contain 984,538 entries.

## Laplace Learning

The paper card reproduces the GraphLearningOld `laplace_learning` baseline:

- unnormalized `L = D - W`;
- harmonic system restricted to unlabeled vertices;
- Jacobi scaling `1 / sqrt(diag(L_uu) + 1e-10)`;
- archived multi-right-hand-side conjugate-gradient loop;
- tolerance `1e-5` and maximum `100000` iterations.

Two historical NumPy details are deliberate: the initial direction aliases the
residual (`p = r`) before an in-place residual update, and the initial stopping
value is the scalar `1`. Changing either changes the archived trajectory.
Every class must have a non-zero source. The exact solver is NumPy/CPU only and
reports its joint absolute residual.

## Poisson Learning

The archived Table 1 CSV was produced with `solver="graddesc"`, represented by
the ModSSC `paper_iteration` solver:

- remove the graph diagonal;
- build the centered one-hot Poisson source;
- compute inverse degrees from `W + 1e-10 I`;
- iterate `u = D^-1 b + D^-1 W^T u`;
- start the mixing chain uniformly on labeled vertices;
- compare it with the raw-degree stationary distribution;
- run at least 50 and at most 1000 iterations, stopping at maximum difference
  `<= 1 / 70000`;
- apply GraphLearningOld's default `training_balance=True` rule
  `u @ diag(1 / observed_class_fraction)`, with no external class prior.

The exact solver and decision rule are NumPy/CPU only and retain float64
scores.

## Autonomous preparation and execution

The canonical runner authenticates the committed graph and permutation inputs,
prepares MNIST through the ModSSC provider, materializes the graph cache, and
then executes the frozen card:

```bash
python -m bench.reproduce prepare laplace_learning/mnist-table1-1-label-per-class
python -m bench.reproduce run laplace_learning/mnist-table1-1-label-per-class
python -m bench.reproduce prepare poisson_learning/mnist-table1-1-label-per-class
python -m bench.reproduce run poisson_learning/mnist-table1-1-label-per-class
```

Use the corresponding cards for budgets 2--5. No GraphLearning source checkout,
external executable, manually seeded cache, or scheduler is required. Slurm is
an optional operational adapter under `tools/hpc/`; it changes resource routing
only and invokes the same scientific interface.

## Canary archive boundary

The authenticated historical Laplace CSV contains a budget-5 row at `69.00`.
It does not record permutation identifiers, so that row cannot be attributed
independently to permutation 0. Replaying the locked permutation 0 with the
authenticated source gives 48,269 correct predictions out of 69,950, or
`69.00500357%`, which formats as `69.01`. ModSSC gives the same 48,269
predictions. The official and ModSSC prediction arrays are identical, and
their full `(70000, 10)` float64 score matrices are bitwise identical. The
archive token `69.00` is compatible with 48,263 through 48,268 correct
predictions, so the same-budget comparison is separated by one node rather
than by a solver or split difference.

The sealed numerical oracle
`bench/assets/calder2020/reference_oracles/laplace-b5-permutation0-source-replay.json`
records
the source, graph, labels, permutation, environment, configuration, split,
labeled indices, predictions, scores, iterations, and residual evidence. Its
raw file SHA-256 is pinned in every generated campaign specification and in
the release code.

Its ModSSC source hash belongs to immutable execution commit
`2756f2f53c6454d726be476d7d0799fc088f1898`, not to the evolving file in the
current checkout. The repository-only validator under
`tools/replication_audit/calder/` can bind that execution to the archived
descriptor under `provenance/`. Neither path is imported by the wheel. Runtime
reproduction authenticates the packaged numerical oracle and the installed
ModSSC scientific payload instead of requiring a source checkout or an
execution-history archive.

The repository-only canary gate remains exact for every other identity. It accepts the
one-node boundary only for Laplace budget 5, permutation 0, and only when the
live result reproduces every oracle binding. Matching accuracy without
matching predictions and scores is rejected. A result already inside an
archived two-decimal interval does not use the exception.

## Acceptance boundary

For each method and label budget, all 100 archived permutation rows must
succeed from one clean commit and one verified environment. The reported
Table 1 standard deviation is the population value (`ddof=0`), matching
GraphLearningOld's `np.std`. Confidence intervals remain based on the sample
standard deviation (`ddof=1`) and Student's t critical value.

The completed acceptance assigned `paper_matched` independently to both methods
after it:

- verify 100 unique successes and no missing/duplicate seed;
- verify the graph, preprocessing, dataset, permutation, commit, and
  environment hashes in every result;
- check solver iterations/residuals and the Poisson balancing diagnostics;
- require the published mean to lie in the replication 95% interval and the
  absolute mean difference to remain within the preregistered margin.

Laplace and Poisson acceptance decisions remain independent. Success of one
cannot promote the other in a future rerun.
