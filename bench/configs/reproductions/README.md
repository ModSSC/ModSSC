# Article reproductions

These configurations are protocol-specific paper cards. They are separate from
`bench/configs/best`, which is the standardized ModSSC benchmark. A card is not
automatically an exact paper match: its `method.profile`, campaign
`fidelity_status`, split contract, seed set, target value, and known deviations
must all be reported.

All classic tabular cards are CPU jobs and perform no validation or test-based
model selection. Their strategies use ModSSC's internal NumPy classifiers,
including the historical C4.5-style, nominal Naive Bayes, and mixed-feature
3-NN backends. No Java process, Weka installation, or external JAR is part of a
supported run.

The immutable cards retain `dataset.download: false` so a scientific task never
changes its inputs mid-run. Use `python -m bench.reproduce run CARD`: the
reproduction command first downloads or materializes the declared dataset
through its ModSSC provider, authenticates protocol resources, and then invokes
the runner. A manually populated cache is not a prerequisite.

| Card | Explicit method profile | Repetitions | Fidelity |
|---|---|---:|---|
| Self-Training, Wine Table 3, 1-NN | `paper:li-zhou-2005-setred-table3-wine-self-training` | 50 | `paper_approx/failed_replication` |
| Co-Training, WebKB Course Table 2 | `paper:nigam-ghani2000-webkb-table2` | 10 | `paper_approx/failed_replication` |
| Tri-Training, WDBC Table III, J4.8 at 80% unlabeled | `paper:zhou-li-2005-wdbc-table3-j48` | 3 | `paper_approx` |
| Democratic Co-Learning, Adult Table 3 | `paper:zhou-goldman-2004-adult-table3` | 20 | `not_claimable` |
| Tri-Training, Vote Table III, J4.8 at 80% unlabeled | `paper:zhou-li-2005-vote-table3-j48` | 3 | `paper_approx` |
| Democratic Co-Learning, Vote Table 3 | `paper:zhou-goldman-2004-vote-table3` | 20 | `paper_approx` |

The Tri-Training and Democratic Co-Learning statuses in this table summarize
archived campaigns. Those campaigns predate the autonomous NumPy historical
classifier stack; the current cards must be confirmed again before their
numerical conclusions are transferred.

The completed historical replacement campaigns and their authenticated
acceptance statistics are recorded in
`docs/development/historical-replacements-v2-results.md`.

The wheel interface always starts from these canonical cards. Multi-card
deployment descriptions are optional orchestration concerns and do not alter a
card's protocol, seeds, resource hashes, or acceptance ceiling.

## Pseudo-Label: MNIST, Table 2

The tracked card reproduces the `+PL` row without DAE pretraining at 600
balanced labels: 60 labels per class, 1,000 validation examples, the official
MNIST test set, and ten random splits. It uses the paper's 5,000-unit MLP,
32/256 labeled/unlabeled batches, non-inverted hidden dropout, and the
learning-rate, momentum, and alpha schedules from Equations 12, 13, and 16.

The paper does not publish an unambiguous final stopping rule, weight
initialization, minibatch traversal rule, or visible-unit dropout setting.
The card records its explicit reconstruction choices and therefore has a
`paper_approx` ceiling. Independent equation-level evidence is recorded in
`docs/development/pseudo-label-lee2013-equation-oracle.md`.

`pseudo_label/mnist.yaml` is the single hardware-neutral scientific card. A
runtime may select CPU or accelerator execution without duplicating or changing
the protocol. Deployment-specific campaign templates and resource profiles live
outside `bench/`.

## Shared Vote cards for the four-dataset campaign

OpenML data ID 56 is pinned as the first-class `vote` catalog entry. It is
row-for-row equivalent to UCI Congressional Voting Records (UCI ID 105, DOI
`10.24432/C5C01P`): 435 rows, 16 nominal voting attributes, 267 Democrats, 168
Republicans, and missing votes. Both source articles use this same dataset.

The Tri-Training card retains one 109-example stratified test set, then requests
65 labeled and 261 unlabeled examples for each of three proportional L/U
redraws. Its Table III targets are initial error `0.076` and final error
`0.055`. It follows the authors' released `TriTrain.java` probability
aggregation (rather than Table I's printed hard vote); provenance and the
miniature parity fixture are recorded in
`docs/development/tri-training-official-parity.md`.

The Democratic Co-Learning card requests 40 labeled, 200 unlabeled, and 195
test examples for 20 random partitions. Its Table 3 target is accuracy
`0.944 +/- 0.012`. The article retains only partitions where at least one
unlabeled example is added. Screen candidate seeds without consulting test
accuracy, then SHA-pin the first 20 accepted split artifacts before production.
That test-blind selection is now frozen under
`bench/campaigns/locks/dcl-vote-zhou-goldman-2004-v1/`; production must enforce
the lock and reproduce every pinned split fingerprint and artifact digest.
Because the article does not publish its exact software versions or
confidence-interval construction, the resulting profile is `paper_approx`,
never `paper_matched`.

The public runner does not repeat that historical screening step. It
authenticates the reviewed selection descriptor and replays the 20 packaged
split artifacts byte-for-byte, without consulting test metrics or external
campaign evidence. The packaged selection file has SHA-256
`5f586b2ab21bd6c2b0e058ab9d588ec1fc04b41b7d93e5a125d0a5f2ea1b36fb`.

## Tri-Training: WDBC, Table III

The paper uses 569-row, 30-feature WDBC, keeps about 25% for test, then splits
the remaining pool into 20% labeled and 80% unlabeled data with similar class
proportions. Under each unlabeled rate it reports three independent L/U
partitions inside the retained training pool. The card therefore fixes one
142-example test set, then requests 85 labeled and 342 unlabeled examples for
seeds 1--3, with no validation partition.

The tracked cell is the J4.8 decision-tree block of Table III. Its published initial
and final errors are `0.094` and `0.075` (reported improvement `20.0%`), i.e. a
final accuracy target of `0.925`. The retained round-zero ensemble is evaluated
as `test_initial`; the final ensemble remains `test`. This is only
`paper_approx`: the first-class ModSSC dataset key `wdbc` pins OpenML data ID
1510 (and remains distinct from `breast_cancer`, which pins OpenML data ID 15)
and the backend is ModSSC's internal unpruned C4.5-style tree. The paper does not
publish its exact classifier version, test indices, RNG, or software versions,
so this internal reconstruction requires a new numerical confirmation and
cannot be claimed as exact backend parity. A fixed component seed makes the
stratified test set replayable; listed seeds randomize only the proportional
L/U selection. The fixed test set is not claimed to recover the historical
indices. The card's 100-round safety cap must be non-binding in every accepted
run; report any seed that does not converge.

## Democratic Co-Learning: Adult, Table 3

The published target is accuracy `0.784 +/- 0.021` over 20 random Adult
partitions with 60 labeled, 1,691 unlabeled, and a roughly equal-size test set.
The card uses exactly 60/1,691/1,691 with no validation partition. Every seed
randomizes the 3,442-row Adult subsample, the L/U/test partition, and fitted
preprocessing.

The archived diagnostic pilot obtained accuracy `0.7478 +/- 0.0485` and
macro-F1 `0.5591 +/- 0.0349`, but it predates the autonomous internal classifier
stack and is not acceptance evidence for the current card. The profile remains
`not_claimable` until the source-data fingerprints are pinned and a fresh
confirmation is complete. The Figure 1 confidence weights, `e'`, `q/q'`,
eligibility filter, and final Laplace-corrected vote are implemented and tested;
the runner still does not pre-filter partitions using the paper's condition
that at least one unlabeled example be added.

Before reporting a Democratic Co-Learning run, inspect
`artifacts.method.diagnostics` in every `run.json`:
`pseudo_labels_added_total` must be positive. Report any excluded seed and do
not silently replace it after seeing test accuracy.

To prepare the declared dataset through ModSSC and run one diagnostic seed:

```bash
python -m bench.reproduce run democratic_co_learning/adult --seed 1
```

## Laplace and Poisson: Calder Table 1

The ten Calder cards pin their common preprocessing and semantic graph
fingerprints directly. No VAE is trained by ModSSC. The repository contains
only the authenticated Table 1 inputs: MNIST labels, label permutations, VAE
kNN arrays, and result CSV files. The GraphLearning implementation is identified
by commit and SHA-256 for provenance but is neither distributed nor executed.

Use `python -m bench.reproduce prepare CARD` (or `run CARD`, which prepares
first). The command authenticates the bundled protocol inputs and materializes
the derived graph cache under the selected ModSSC cache root. Scheduler wrappers
may invoke the same command, but they do not add a scientific dependency.

Both routes authenticate the official bundle, verify the 70,000-row MNIST
ordering, build and reload the shared float64 graph cache, write one immutable
artifact lock, and materialize ten immutable effective cards with the verified
cache fingerprints. Production manifests use the repository-relative paths
recorded for those effective cards.
The full protocol and verification evidence are documented in
`docs/development/calder2020-official-parity.md`.
