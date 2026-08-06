# Democratic Co-Learning on Vote: primary-source audit

This note records the scientific ceiling and the immutable data evidence for
the ModSSC reproduction of the Vote result in Zhou and Goldman (2004). It
separates statements made by the 2004 Democratic Co-Learning paper from
details found only in Goldman and Zhou (2000), its earlier statistical
co-learning paper.

## Primary sources

- Yan Zhou and Sally Goldman, *Democratic Co-Learning*, ICTAI 2004,
  pp. 594-602: [DOI](https://doi.org/10.1109/ICTAI.2004.48),
  [DBLP record](https://dblp.org/rec/conf/ictai/ZhouG04).
  The auditor-local PDF, which is not distributed or required at runtime, has SHA-256
  `f14d7f8c2782476911a45d88eea73df9c72c6547c0a380b4e7620c530f30afed`.
- Sally Goldman and Yan Zhou, *Enhancing Supervised Learning with Unlabeled
  Data*, ICML 2000, pp. 327-334:
  [archived author PostScript](https://web.archive.org/web/20060909002158id_/http://siesta.cs.wustl.edu/~zy/icml2000.ps).
  The archived PostScript has SHA-256
  `33a079d2a212d9e7746a1a1eb81731b9066154b11769c06a613731e5a690b9ac`.
- UCI, *Congressional Voting Records*:
  [dataset page](https://archive.ics.uci.edu/dataset/105/congressional%2Bvoting%2Brecords),
  DOI [10.24432/C5C01P](https://doi.org/10.24432/C5C01P).
- OpenML dataset 56, `vote`:
  [dataset page](https://www.openml.org/d/56),
  [metadata API](https://www.openml.org/api/v1/json/data/56),
  [ARFF](https://www.openml.org/data/v1/download/56/vote.arff).

## Dataset identity

OpenML 56 contains the historical UCI Congressional Voting Records data, not
an altered derivative.

The following archived UCI files predate the 2004 experiment:

- [7 December 1999 snapshot](https://web.archive.org/web/19991207014211id_/http://ftp.ics.uci.edu/pub/machine-learning-databases/voting-records/house-votes-84.data)
- [8 December 2000 snapshot](https://web.archive.org/web/20001208043700id_/http://ftp.ics.uci.edu/pub/machine-learning-databases/voting-records/house-votes-84.data)

Both snapshots and the current UCI `house-votes-84.data` file are identical
byte for byte:

| Artifact | Size | Rows | SHA-256 |
|---|---:|---:|---|
| UCI 1999 snapshot | 18,171 bytes | 435 | `c87c14110a5ba91d4a1e313ec7392824458152bf071fa5f5452340488337936e` |
| UCI 2000 snapshot | 18,171 bytes | 435 | `c87c14110a5ba91d4a1e313ec7392824458152bf071fa5f5452340488337936e` |
| Current UCI data file | 18,171 bytes | 435 | `c87c14110a5ba91d4a1e313ec7392824458152bf071fa5f5452340488337936e` |
| Current UCI ZIP | - | - | `ffec9d9328f28f151d95d2f8d36fd94ed8d2b0cdd0c9bd284bd8e65031a5e6a6` |
| OpenML 56 ARFF | - | 435 | `ee647a77207729d73d02cea20646afcd274fe9de95711cbf9909c903636cd65f` |

The OpenML metadata declares MD5
`3c16059c5b92f6551f720f97d0eccc09`; the cached ARFF has exactly that MD5.

UCI stores the class in the first column. OpenML stores the same class in the
last column and names the 16 features. After moving the UCI class to the last
column, stripping ARFF quoting, and serializing every row as comma-separated
values with a final newline:

- ordered rows are exactly equal;
- the ordered canonical SHA-256 is
  `026da61cef5b33a96aab54cfae27dafbd32614e9424ed5bf67109e697557a798`;
- sorted row multisets are exactly equal;
- the sorted canonical SHA-256 is
  `8fc825184f72275c361efe698c04f8d59e395f859e7642497799414f7b6f3957`.

Both representations contain:

- 435 rows, 16 nominal `{n,y}` features, and one binary class;
- 267 `democrat` and 168 `republican` rows;
- 392 `?` values across 203 rows;
- per-feature missing counts
  `[12, 48, 11, 11, 15, 11, 14, 15, 22, 7, 21, 31, 25, 17, 28, 104]`.

Consequently, OpenML 56 is an admissible byte-audited representation of the
historical UCI data. A separate Vote dataset must not be added to ModSSC.

## Protocol stated in the 2004 paper

Section 5 and Tables 2-3 establish the following Vote protocol:

- three learners: naive Bayes, C4.5, and 3-nearest neighbor;
- 40 initially labeled examples;
- 200 unlabeled examples;
- a test set described as roughly the same size as the unlabeled pool;
- 20 random partitions into labeled, unlabeled, and test data;
- retention only of random partitions on which Democratic Co-Learning labels
  at least one example from the unlabeled pool;
- no validation split or test-based model selection is described.

With all 435 Vote rows and the published `|L|=40`, `|U|=200`, using the
remainder gives 195 test examples. The value 195 is therefore a deterministic
reconstruction from the published counts, not a number printed in the paper.

For Vote, Table 2 reports:

- 66 examples from `U` labeled for naive Bayes;
- 40 labeled for C4.5;
- 40 labeled for 3-NN;
- 2.2 co-learning rounds on average.

Table 3 reports Democratic Co-Learning accuracy as **`0.944 +/- 0.012`**.
The surrounding text explicitly says that the standard deviation is shown
only for Democratic Co-Learning. Therefore, **`0.012` is a standard
deviation, not a 95% confidence interval**. Parenthesized values for the other
methods are paired t statistics. The paper does not state whether its standard
deviation uses a population or sample denominator.

The original row indices, RNG, seeds, stratification policy, and partition
order were not published. ModSSC can reproduce the published conditioning
rule without test leakage by screening candidates solely on whether the
method adds at least one pseudo-label, then SHA-pinning the first 20 accepted
partitions. Those partitions are new replayable realizations of the published
rule, not recovery of the authors' unavailable partitions.

## Confidence intervals: 2000 versus 2004

### What Goldman and Zhou (2000) states

Section 3.1 says that, for each hypothesis and each equivalence class, the
authors use 10-fold cross-validation to count correct predictions and then
compute a 95% confidence interval for a binomial parameter. The paper further
says that cross-validation could use either the originally labeled data or all
labeled data, including pseudo-labels, and that the experiments use only the
originally labeled data `L`.

Table 2 places the overall hypothesis intervals computed from 10-fold
cross-validation on `L` before the repeat loop. It places equivalence-class
intervals inside the loop, again stating that `L` is used.

The 2000 paper does **not** specify:

- the equation used for the binomial interval;
- whether its bounds are clipped to `[0,1]`;
- fold stratification, assignment, seed, or reuse;
- whether a fold model is trained on only nine tenths of `L`, or on those
  examples plus the current pseudo-labeled examples;
- small-support behavior for an equivalence class.

It cites Larsen and Marx (1986) and calls the construction a standard
95%-confidence interval for a binomial parameter. This does not uniquely
identify an executable interval implementation.

### What Zhou and Goldman (2004) states

Figure 1 of the 2004 Democratic Co-Learning paper has the following order:

1. train learner `A_i` on its current `L_i` to obtain `H_i`;
2. use original `L` to compute `[l_i,h_i]` for `H_i`, with
   `w_i=(l_i+h_i)/2`;
3. use current `L_i` to compute another `[l_i,h_i]` for `H_i` when estimating
   the error of a proposed pseudo-label set.

The complete 2004 paper does not use the term cross-validation when defining
these intervals and does not give an interval equation. Thus the following
choices are not scientifically interchangeable:

- evaluating the current `H_i` directly on `L`, and later on `L_i`, is the
  literal execution order of Figure 1;
- performing 10-fold out-of-fold evaluation on `L` alone imports an explicit
  rule from the 2000 method, but the 2004 paper does not say it does so;
- training each fold on `(L_i minus the held-out part of L)` plus
  pseudo-labels is a statistically coherent reconstruction, but appears in
  neither paper;
- stratifying all of current `L_i` into folds, so accepted pseudo-labels can
  occur in both fold-training and fold-validation subsets, is the diagnostic
  reconstruction implemented below and is likewise specified by neither
  paper.

No one of these alternatives can be proven to be the authors' exact 2004
implementation from the published record. A clipped Wald interval is likewise
a declared reconstruction, not a recovered paper detail.

## Diagnostic cards implemented in v2

The four Table 3 controls are isolated baselines. They fit the pinned
naive-Bayes, J48, and 3-NN learners once on the original `L`; the three
single-learner cards select one of those predictions and `Combining Only`
applies Figure 2 to the same three initial learners. They do not execute a
pseudo-label round. Their diagnostics must consequently report zero rounds,
zero additions, convergence, and an empty round trace.

The confidence cards keep every other protocol component fixed:

- `training_accuracy + wald` evaluates the current `H_i` directly on original
  `L` for voting weights and on current `L_i` for proposal-error bounds. This
  is the explicit replay of the v1 convention.
- Each `kfold_oof` card computes the original-`L` voting intervals once from
  deterministic stratified 10-fold predictions on `L`. It computes each
  evolving interval from deterministic 10-fold predictions on that learner's
  current `L_i`, including any accepted pseudo-labels, and recomputes it only
  when `L_i` changes. Fold assignment uses seed `0`; each validation example
  appears exactly once and never appears in its fold's training set.
- The Wald, Wilson, and Clopper-Pearson cards differ only in the binomial
  interval. Wilson and Clopper-Pearson remain conditional diagnostics.

This `kfold_oof` behavior is a declared hybrid reconstruction. In particular,
retraining folds on the evolving `L_i` is not specified by either paper.
Trajectory agreement can make this reconstruction useful, but cannot turn it
into recovered historical code or raise it above `protocol_conformity:
pending`.

## Learner implementations are under-specified

The 2004 paper names naive Bayes, C4.5, and 3-NN but provides no software
package, release, source code, or command-line options.

- No naive Bayes smoothing, probability, or missing-value policy is given.
- No C4.5 pruning confidence, minimum leaf size, subtree-raising, binary split,
  or missing-value policy is given. Reference [17] is Quinlan's 1986
  *Induction of Decision Trees* paper, not a software/version specification.
- No 3-NN metric, normalization, distance weighting, tie handling, or
  missing-value policy is given. Reference [18] is Cover and Hart (1967), not
  an implementation specification.

Goldman and Zhou (2000) explicitly used ID3 and HOODG because both were
available in MLC++, but it provides no MLC++ version. Those two algorithms are
not the three learners used by Democratic Co-Learning in 2004, so this does
not establish that the later experiment used MLC++ or Weka.

ModSSC's internal NumPy naive Bayes, decision-tree, and 3-NN backends, together
with their pinned settings and numerical oracles, provide a reproducible,
period-appropriate reconstruction. They must not be described as the original
authors' recovered software stack.

## Private Slurm execution and public evidence

The execution-site identity, account and physical paths are private. Public
campaign names below are publication aliases, not rewritten source identifiers;
the original identifiers are authenticated by SHA-256 and retained with the
private provenance bundle.

The immutable Table 3 control execution used:

- commit `1b54e9693ada1cf7ba6334fe2279ff59218a40ee`;
- tag `replication-10m-dcl-v2-diagnostics-v2`;
- public campaign alias `dcl-vote-controls-v2-r2` (source identifier SHA-256
  `0a474eeab50efa348a3424f0e1d83ac950c611f77a0d5645d73d5d19c214617f`);
- manifest SHA-256
  `71a38b6f21af19f9597d2ed14af7e4a170a8dcf4474bf8422ee5427518ce47b3`;
- the 20 locked v1 partitions for each of the four controls, for 80 successful
  tasks out of 80.

All four cells completed without a control-integrity failure:
`protocol_failures` is empty for every protocol. The exact numerical results
from the immutable gate report are:

| Table 3 control | Published mean | Replication mean | Replication 95% CI | Absolute difference | Decision |
|---|---:|---:|---:|---:|---|
| Naive Bayes | `0.861` | `0.8946153846153846` | `[0.8846898003547028, 0.9045409688760664]` | `0.0336153846153846` | Failed |
| C4.5 | `0.942` | `0.9407692307692308` | `[0.926005464744081, 0.9555329967943806]` | `0.0012307692307691465` | Passed |
| 3-NN | `0.902` | `0.9028205128205128` | `[0.8919613911845123, 0.9136796344565132]` | `0.0008205128205127643` | Passed |
| Combining Only | `0.938` | `0.9099999999999999` | `[0.8999228481946527, 0.9200771518053471]` | `0.028000000000000025` | Failed |

The two failed controls exceed the preregistered absolute margin of `0.02`.
This is a scientific-equivalence failure, not an infrastructure or control
execution failure.

The completed v1 DCL campaign itself produced a mean test accuracy of
`0.9046153846153846`, sample standard deviation
`0.021397909116329016`, and Student 95% confidence interval
`[0.8946008548816785, 0.9146299143490907]`. Its signed difference from the
published `0.944` is `-0.03938461538461535`, outside the preregistered
two-point margin. The incompatible dynamics are also explicit:

- `changed_rounds` averages `4.55`, versus `2.2` rounds in Table 2;
- ModSSC's raw `n_iter` averages `5.55` because it also counts the terminal
  no-change pass;
- mean pseudo-label additions are `[4.45, 42.8, 5.6]`, versus the published
  `[66, 40, 40]`;
- the corresponding final learner-set sizes are
  `[44.45, 82.8, 45.6]` after adding the 40 original labels.

The last two quantities must not be conflated: Table 2 reports examples from
`U` labeled for each learner, so the additions, rather than the final sizes,
are the comparable diagnostic.

The source audit and the independent equation oracle confirm that ModSSC's
`Combining Only` implements Figure 2 exactly: it uses the midpoint of each
95%-confidence interval, excludes a classifier when its weight is at most
`0.5`, applies the published Laplace/m-estimator correction to each prediction
group, and selects the group with the largest corrected confidence. Thus the
`0.91` ensemble result is not caused by substituting an unweighted majority
vote.

The pinned Naive Bayes reconstruction is ModSSC's internal NumPy historical
backend. Vote's 16 inputs remain native nominal attributes, its nominal
probability estimates use Laplace smoothing, and missing values are skipped
rather than imputed into a nominal category. The 2004 paper does not identify
its Naive Bayes implementation,
version, smoothing, or missing-value policy. The observed Naive Bayes mismatch
therefore identifies one failed localization control, but it does not prove
that the backend is the unique cause. The authors' 20 partitions are
unavailable, while the replayed partitions were conditioned by the different
v1 DCL dynamics. The mismatch may reflect learner semantics, the realized and
conditioned partitions, or their interaction; no one of these causes is
recoverable from the published record.

The two immutable report artifacts are addressed by content:

- `modssc-artifact://replication/evidence/847a751493e47dfa99c070b5fd5e81702d1943f744a5d50a2b1538c78b8435e4`,
  SHA-256
  `847a751493e47dfa99c070b5fd5e81702d1943f744a5d50a2b1538c78b8435e4`;
- `modssc-artifact://replication/evidence/5566623051730bdbff2920a1f8e01843bc32dd494eb2b1a2f89b7a92cb31a33e`,
  SHA-256
  `5566623051730bdbff2920a1f8e01843bc32dd494eb2b1a2f89b7a92cb31a33e`.

The Table 3 diagnostic execution consumed `0.625833` V100-hours in total.

### Test-blind confidence diagnostics

The confidence comparison was subsequently executed from immutable release
`replication-10m-easy-wave1-v8`, commit
`e6e509e0840e6ee18ef55d1b5b99255798364f5a`, with environment lock
`040ef191238a49230ed3b3e035ce03ee51dc949da9aa0b662c26f305289e37c7`.
All scientific tasks ran through Slurm on V100 nodes. The frontend was used
only for manifest generation, submission, inspection and reconciliation.

The primary public alias `dcl-vote-confidence-primary-v2-v8` (source identifier
SHA-256 `da2b29125ec25f6690bb02a1b4bd3c9848c5666a15eba055910410395015ae67`)
completed 40/40 tasks: the v1-compatible
resubstitution + Wald control and stratified 10-fold out-of-fold + Wald, each
on the 20 locked partitions. Its manifest SHA-256 is
`ad8e4787c5b9ce48282c0ed105813b12775746717acb2258038b676e51e286ad`.
The conditional public alias `dcl-vote-confidence-conditional-v2-v8` (source
identifier SHA-256
`cc6adbf83c380e46f37f64e19ac3afbbc2c9c903d80b0eae891cb18a49f9ee69`)
was triggered because both primary
candidates failed and 10-fold + Wald had the smaller preregistered normalized
Table 2 distance. It completed another 40/40 tasks for 10-fold + Wilson and
10-fold + Clopper--Pearson. Its manifest SHA-256 is
`0267792444c0280b9fdeb8106b94b4782a491bb502557e1f8c1459cfa5272e39`.
Neither campaign read or reported test metrics for candidate selection.

| Confidence reconstruction | Mean `n_iter` | Mean additions `[NB, C4.5, 3-NN]` | NB receives most | Table 2 dynamics |
|---|---:|---:|---|---|
| Resubstitution + Wald | `5.55` | `[4.45, 42.80, 5.60]` | No | Failed |
| 10-fold OOF + Wald | `5.35` | `[4.05, 40.85, 4.60]` | No | Failed |
| 10-fold OOF + Wilson | `5.10` | `[3.90, 38.90, 3.65]` | No | Failed |
| 10-fold OOF + Clopper--Pearson | `4.95` | `[3.95, 35.20, 3.60]` | No | Failed |
| Article Table 2 | `2.2` | `[66, 40, 40]` | Yes | Target |

The first-round raw disagreement counts are independent of the confidence
interval because the majority class is selected by the three unweighted
predictions; confidence weights only decide candidate eligibility. Their
means `[3.45, 13.4, 2.3]` already make C4.5, rather than Naive Bayes, the main
dissenter. The inversion therefore exists before confidence filtering.
Changing the estimator or the binomial interval changes the number of accepted
examples slightly, but it does not reverse the learner roles.

These four registered confidence constructions are therefore insufficient to
explain or recover the published dynamics. This does not reject every possible
historical construction and does not identify a unique cause: the
period-specific learner semantics are not recoverable, the authors' partitions
are unavailable, and the accepted v1 partitions were conditioned under the
reconstructed DCL dynamics. Historical learner behavior, partition
realization, or their interaction remain plausible.

The immutable confidence reports are:

- primary reconciliation:
  `modssc-artifact://replication/evidence/ad1a2e4d5a6710df1f4f712c697c60f2e61e7e8b2b9e5358882e6e9015d182e0`,
  SHA-256
  `ad1a2e4d5a6710df1f4f712c697c60f2e61e7e8b2b9e5358882e6e9015d182e0`;
- primary gate report:
  `modssc-artifact://replication/evidence/42e6805a991ce4230d6158c7fdf20382884b53c86601d9a4437590faeb362990`,
  SHA-256
  `42e6805a991ce4230d6158c7fdf20382884b53c86601d9a4437590faeb362990`;
- first-round attribution:
  `modssc-artifact://replication/evidence/565bd6f5c453bdfc7cb13bd8f04b4e75c3f8db45d08f3eef816ecf8e1221069e`,
  SHA-256
  `565bd6f5c453bdfc7cb13bd8f04b4e75c3f8db45d08f3eef816ecf8e1221069e`;
- conditional reconciliation:
  `modssc-artifact://replication/evidence/1409d8daccaeb5356cbb7ad37d895c45931f4179c16db3ca57d911c0b8ebadc9`,
  SHA-256
  `1409d8daccaeb5356cbb7ad37d895c45931f4179c16db3ca57d911c0b8ebadc9`;
- conditional gate report:
  `modssc-artifact://replication/evidence/8d79278dfb3a3b40718d48ad24b81d5d036dd0b16a8a3b54ef844db8924c37dc`,
  SHA-256
  `8d79278dfb3a3b40718d48ad24b81d5d036dd0b16a8a3b54ef844db8924c37dc`.

Each reconstructed confidence candidate retains
`protocol_conformity: pending`: successful execution or failed Table 2
dynamics cannot recover which unpublished confidence procedure the authors
used. The parent Vote replication retains `protocol_conformity: failed`.

## Scientific decision

The evidence supports these classifications:

| Component | Decision |
|---|---|
| Dataset identity and row order | Matched |
| Published `40 L / 200 U / remainder test` counts | Matched by reconstruction |
| Conditioning on at least one pseudo-label | Matched when screened without test access |
| Original 20 partitions | Unrecoverable; replayable replacements only |
| Democratic Co-Learning equations in Figure 1 | Implementable |
| Figure 2 combining equation | Passed by source audit and independent oracle |
| Table 3 control integrity | Passed, with 80/80 successes and no protocol failures |
| Table 3 numerical control equivalence | Failed for Naive Bayes and Combining Only |
| Four registered confidence reconstructions | 80/80 successes; all failed Table 2 dynamics without test access |
| Four tested confidence constructions as sufficient cause | Insufficient |
| Exact binomial interval used in 2004 | Still unknown |
| Exact confidence-evaluation procedure in 2004 | Still unknown |
| Learner software, versions, and options | Unknown |
| Published Vote summary | Mean `0.944`, standard deviation `0.012` |

The final immutable v1 decision is:

- `protocol_status: paper_approx`;
- `result_status: failed_margin`;
- `equation_conformity: passed`;
- `protocol_conformity: failed`.

The result remains `paper_approx`, rather than `paper_matched`, because the
equations passed but the numerical result and the preregistered protocol
controls did not. The four executed confidence candidates all fail the
published dynamics and independently remain epistemically `pending`, because
the original procedure is not identified. Their diagnostic failures do not
replace the failed parent protocol decision. Numerical agreement from a future
unregistered setting could not recover the unpublished confidence-interval
and learner choices.

The campaign becomes `not_claimable` if it uses fewer than 20 accepted
partitions, conditions on test performance, changes the pinned data or
software between repetitions, or reports the paper's `+/- 0.012` as a
confidence interval.
