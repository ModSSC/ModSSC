# Nigam--Ghani 2000 Co-Training on WebKB

## Source and registered result

The single scientific profile
`paper:nigam-ghani2000-webkb-table2` reconstructs the Co-Training row of
Table 2 in Nigam and Ghani, *Analyzing the Effectiveness and Applicability of
Co-training* (CIKM 2000, pp. 86--93; [DOI](https://doi.org/10.1145/354756.354805),
[author PDF](https://www.cs.cmu.edu/~knigam/papers/cotrain-CIKM00.pdf)). The
published WebKB Course error is **5.4%** with 12 labeled and 776 unlabeled
documents; 263 documents are held out for test. The registered absolute margin
is two percentage points over ten runs.

The supervised Table 2 rows are controls within this same profile: NB trained
on the initial 12 labels has 13.0% published error, and NB trained on all 788
true training labels has 3.3% published error. They are emitted as
`test_nb12` and `test_nb788` secondary metrics. They are not additional
scientific profiles, and neither control can affect Co-Training or select a
protocol variant.

## Frozen reconstruction

The profile uses the official two-view CMU WebKB Course archive (`fulltext`
and `inlinks`). Its cache identity is frozen as follows:

- dataset fingerprint:
  `5a1d45139e2a1ccb17abf374fb6ec17dc7d0bb3f9ff7caf08935d7731bb80683`;
- content SHA-256:
  `894e2f310924fd66239632029db7738b8e1fcd330ffb86cb201cf6937ed9a264`.

Confirmation campaign v2 uses the fresh, pre-registered seeds 21--30. For each
seed, the reconstruction draws 263 test documents, then fixes the initial
labeled set to 3 course and 9 non-course documents. Each view uses raw word
counts after deterministic HTML stripping. Its multinomial NB applies add-one
smoothing to both word likelihoods and the class prior; no feature selection is
performed.

The unlabeled pool starts at 75. At every round, both classifiers select from
the same pre-round pool. Each proposes its most confident 1 positive and 3
negative documents, subject only to the number of documents available in the
final undersized pool. The shared labeled multiset retains every proposal in
`ordered_multiset_view1_then_view2` order. A document proposed by both views
therefore occurs twice, including when the proposed labels conflict, while its
source index is removed from the unlabeled pool only once. Confidence is the
posterior probability; ranking uses the order-equivalent log posterior for
numerical stability. The final prediction multiplies the two view probabilities
by summing their log probabilities.

Training continues until all 776 unique unlabeled indices have been promoted.
The number of rounds is data-dependent because cross-view overlaps reduce the
number of unique removals without reducing the number of multiset additions.
The final labeled size is therefore `12 + pseudo_labels_added_to_shared_l`, and
the number of pseudo-label additions is `776 + overlap_count`. The pool requests
eight replenishment documents per round and takes only those still available in
the randomized reservoir.

## Ambiguities and claim ceiling

The paper does not publish the ten split identities, historical tokenizer and
vocabulary, cross-view collision policy, or the precise rule for the final
undersized pool. The reconstruction therefore preserves both same-pool proposal
streams as an ordered multiset and drains the last pool without replacement.
This collision policy is an explicit reconstruction uncertainty in the
historical confirmation report, not a recovered historical fact. These rules
are pre-registered and test-blind, and the claim is capped at `paper_approx`
even if the numerical target is recovered.

Acceptance requires the exact 12/776 and 3:9 start, same-pre-round-pool
selection, ordered multiset additions, 776 unique promotions, and an empty final
pool. It validates overlap and conflict accounting, the variable round and
training-size trajectories, replenishment, add-one smoothing, no feature
selection, and explicit evidence that neither test metrics nor the supervised
controls were used for protocol selection. It does not require 97 rounds, zero
collisions, exactly 776 multiset additions, or final labeled size 788.

## Confirmation result

The immutable confirmation v2 at commit
`24803701f178dfc7bfaa609170b2c54fa1e659e3` completed all ten registered runs.
Its mean Co-Training error is **21.90%**, with sample standard deviation
**14.09%** and Student 95% confidence interval **[11.82%, 31.98%]**. The
published 5.4% target is outside the interval, and the absolute difference of
16.50 percentage points exceeds the registered two-point margin. The numerical
result is therefore `numeric_not_matched` and the scientific result is
`paper_approx/failed_replication`.

The fully supervised 788-label control is compatible with the paper: **3.65%**
error, 95% CI **[2.72%, 4.58%]**, versus 3.3% published. The 12-label control is
not compatible: **19.16%**, 95% CI **[16.38%, 21.94%]**, versus 13.0% published.
This localizes a material discrepancy before pseudo-labeling, most plausibly in
the unavailable historical splits or under-specified historical text pipeline.
Trace analysis also shows that failed runs accumulate semantically incorrect
pseudo-labels; across the ten registered runs, Co-Training error correlates
strongly with pseudo-label accuracy (`r=-0.982`). This is a diagnostic, not a
criterion for excluding runs or selecting a variant.

No tokenizer, collision rule, seed subset, or confidence variant will be
selected from these test results. This remains the sole Co-Training paper
profile; the unresolved historical details prevent an exact numerical claim.
