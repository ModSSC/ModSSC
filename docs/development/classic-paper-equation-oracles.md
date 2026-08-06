# Classic paper-equation conformity oracles

Audit date: 2026-07-24.

This note records paper-equation evidence for the classic profiles. Pseudo-Label
has no recoverable official implementation and therefore uses an independent
equation oracle. Tri-Training now also has a separately pinned official-source
audit. Neither profile can justify `paper_matched` while its historical
protocol choices remain unavailable.

## Pseudo-Label (Lee, 2013)

Visual source: auditor-local copy of Lee (2013), pages 2-5. The copy is not
distributed and is not required by the executable protocol.

The `lee2013_mnist` profile implements the no-DAE `+PL` row of Table 2:

- one 5,000-unit ReLU hidden layer and sigmoid output units;
- non-inverted hidden dropout with probability 0.5;
- hard argmax pseudo-labels recomputed on every weight update (Equation 14);
- labeled and unlabeled binary cross-entropy means combined by Equation 15;
- learning-rate, momentum, and alpha schedules from Equations 12, 13, and 16;
- labeled/unlabeled batch sizes 32/256 and no confidence filter.

The independent oracle and comparison are in
`tests/inductive/methods/_pseudo_label_lee2013_oracle.py` and
`tests/inductive/methods/test_pseudo_label_lee2013_oracle.py`. Focused
production tests remain in `tests/inductive/methods/test_pseudo_label.py`.
The full audit is recorded in
`docs/development/pseudo-label-lee2013-equation-oracle.md`.

The paper does not publish weight initialization, the final stopping epoch,
the epoch/minibatch traversal rule, or whether visible-unit dropout was used in
the reported `+PL` row. The tracked card explicitly chooses PyTorch Linear
initialization, zero visible dropout, full-pool cycling, and the minimum
601-epoch horizon reaching `alpha_f`. Those are critical historical unknowns:
the card can support `paper_approx`, not `paper_matched`.

## Tri-Training (Zhou and Li, 2005)

Visual source: auditor-local copy of Zhou and Li (2005), pages 3-5. The copy is
not distributed and is not required by the executable protocol.

The authors' recovered `TriTrain.java` and the implementation agree on the
critical transitions from Table I and Equations 9-11:

- three size-|L| bootstrap samples;
- `MeasureError` counts errors only where the other two hypotheses agree on
  the original labeled set;
- candidates are all unlabeled examples on which the other two hypotheses
  agree, without a confidence threshold in the paper card;
- `e/e'`, `l/l'`, and `Subsample` use the strict inequalities and integer
  bounds in Table I;
- updates for all three hypotheses are decided from the same round and then
  applied;
- the released executable sums class-probability distributions for final
  prediction. The paper card follows that executable behavior even though
  Table I prints a hard majority vote.

The frozen official archive/source hashes, miniature transition fixture, and
the counterexample distinguishing the executable from the printed vote are
recorded in `docs/development/tri-training-official-parity.md` and
`tests/inductive/methods/test_tri_training_official_oracle.py`.

The WDBC protocol retains one approximately 25% test set and performs three
L/U redraws in the remaining pool. A fixed component seed encodes that
structure. ModSSC reproduces the reviewed historical learner semantics with an
internal NumPy backend; no external runtime is required. The exact historical
indices and RNG state were not published, so the result ceiling remains
`paper_approx`.
