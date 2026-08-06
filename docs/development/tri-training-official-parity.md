# Tri-Training official-source parity

This note records the primary executable reference used by the Table III
reproduction. It certifies reviewed algorithmic transitions; it does not claim
that the unpublished random partitions have been recovered.

## Frozen provenance

- Paper: Zhi-Hua Zhou and Ming Li, *Tri-Training: Exploiting Unlabeled Data
  Using Three Classifiers*.
- Local PDF SHA-256:
  `a49960332f6dae3cfe547390cfcdfa850cd55482f1768b23b4e72b3a4d199cbc`.
- Audited PDF pages: 3 and 5 of the 13-page file.
- Authors' code page: `https://www.lamda.nju.edu.cn/code_TriTrain.ashx`.
- Official archive: `https://www.lamda.nju.edu.cn/files/TriTrain.rar`.
- Archive SHA-256:
  `f23cda982f521cca607e3fdc50a9f2bc4b0fe5352e7d856083c7c564af66f9e8`.
- `TriTrain.java` SHA-256:
  `0f3497f93190138e9ea93061dd72502f4134b977c4d7670b0898b94613433286`.
- `ReadMe.htm` SHA-256:
  `290711330eda5e96e6cbabb3b7523afba733777f5551a6cab358224e9019cb07`.
- Audit date: 2026-07-24.

The archive identifies Weka 3.4 as the original executable's runtime. ModSSC
does not distribute or invoke it: the reviewed learner and probability
semantics are implemented by internal NumPy backends and locked by independent
fixtures. The executable demonstration constructs J48 and explicitly calls
`setUnpruned(true)`; both Table III cards preserve that setting.

## Reviewed transitions

The released Java source and ModSSC agree on the trajectory-critical rules:

- three bootstrap samples of the original labeled set;
- initial `e'_i = 0.5` and `l'_i = 0`;
- `MeasureError` counts errors only where the other two classifiers agree;
- all agreeing unlabeled examples are candidates, without a confidence
  threshold or candidate cap;
- the first `l'_i` is
  `floor(e_i / (e'_i - e_i) + 1)`;
- a full candidate set is accepted only when
  `e_i |L_i| < e'_i l'_i`;
- the fallback sample size is the official Java expression
  `ceil(e'_i l'_i / e_i - 1)`;
- accepted classifiers are retrained on the original labeled set plus only the
  current round's selected pseudo-labels;
- the loop stops when no classifier changes.

The pinned miniature comparison is
`tests/inductive/methods/fixtures/tri_training_official_oracle.json`, exercised
by `tests/inductive/methods/test_tri_training_official_oracle.py`.

## Paper versus executable final prediction

Table I writes the final hypothesis as a hard majority vote. The authors'
released `TriTrain.java` instead sums the three class-probability
distributions, normalizes the sum, and chooses its largest component. These can
produce different labels. The reproduction profile follows the released
executable behavior (`prediction_rule: soft_average`) because the objective is
to reproduce the numerical Table III experiment. The miniature fixture
contains a vector where the executable and the paper vote deliberately differ,
so this choice cannot regress silently.

## Fidelity ceiling

The paper reports three random labeled/unlabeled partitions after a 25% test
holdout, but does not publish their indices or random stream. ModSSC records new
deterministic partitions and bootstraps; it does not claim them as the original
draws. The paper says J4.8 but does not state its pruning option; ModSSC follows
the released demonstration's explicit unpruned J48. Vote's 16 nominal
attributes and missing values are preserved and the base learner is ModSSC's
pinned NumPy reconstruction of the reviewed historical behavior.

Consequently the algorithmic gate can pass from the pinned official source,
while the numerical reproduction remains capped at `paper_approx`.
