# Scientific conformity oracles

## Gate policy

An article-method gate may be marked `passed` with exactly one declared
`conformity_basis`:

- `pinned_official_implementation`: a reviewed, immutable official
  implementation is identified and a miniature trajectory comparison is
  recorded.
- `independent_equation_oracle`: the primary source does not provide or cite an
  official implementation suitable for comparison. A second implementation of
  the published equations must then be written independently of the production
  method, use fixed hand-checkable vectors, and be reviewed together with
  equation/invariant tests.

The second basis is not a substitute for missing protocol details. It is allowed
only when all of the following are recorded:

1. The primary-source artifact, digest, exact figures/equations/tables, and
   extracted test vectors.
2. An explicit statement that no official-code parity is claimed.
3. Independence: the oracle computes expected values without calling the
   production helpers under test.
4. Focused comparisons for every critical published transition, plus an
   end-to-end canary using the pinned paper profile.
5. Every unspecified or unresolved choice in `known_deviations` or
   `critical_unknowns`.
6. A fidelity ceiling no higher than `paper_approx`.

The registry loader enforces the declared basis for every passed gate. A gate
pass certifies the reviewed algorithmic transitions only. It does not certify a
paper result, remove acceptance-registry deviations, or promote a screening run
into a paper repetition.

## Democratic Co-Learning oracle

Primary source: Yan Zhou and Sally Goldman, "Democratic Co-Learning", ICTAI
2004, DOI `10.1109/ICTAI.2004.48`.

- Auditor-local archive (not distributed and not required by the protocol)
- PDF SHA-256:
  `f14d7f8c2782476911a45d88eea73df9c72c6547c0a380b4e7620c530f30afed`
- Audit date: 2026-07-24
- Audited PDF pages: 4-7 of the nine-page file
- Production method metadata deliberately has `official_code=None`.

The paper does not provide or cite a source-code release. An auditor-local
inventory note records that none was found; it is not distributed or treated as
an official artifact. The DCL gate therefore uses
`independent_equation_oracle`, never an invented official-code comparison.

### Primary-source transcription

Figure 1 defines these transitions:

- initialize each learner with `L_i = L` and `e_i = 0`;
- train at least three different learners;
- choose the strict majority label by vote count;
- allow a proposal only when the majority group's sum of mean confidence
  weights is greater than every minority-group sum;
- propose the majority label only to learners that disagree with it;
- compute `q_i = |L_i| * (1 - 2 e_i / |L_i|)^2`;
- estimate proposal error from the lower confidence bounds of the classifiers
  supporting the proposed label;
- accept `L'_i` only when
  `q'_i = |L_i union L'_i| * (1 - 2 (e_i + e'_i) / |L_i union L'_i|)^2`
  is strictly greater than `q_i`;
- stop when none of the learner-specific labeled sets changes.

Figure 2 defines the final prediction:

- compute each learner weight as the mean of its 95% confidence interval on
  the original labeled set;
- ignore learners with weight at most `0.5`;
- group the remaining learners by predicted class;
- score a group with its mean confidence multiplied by the Laplace correction
  `(|G| + 0.5) / (|G| + 1)`;
- choose the group with the largest corrected score.

The independent vectors in
`tests/inductive/methods/test_democratic_co_learning_oracle.py` transcribe those
rules without calling production helpers. They cover a low-confidence strict
majority that must be rejected, an accepted proposal, the `e'`, `q`, and `q'`
calculation, and the final Laplace-corrected vote. Existing focused tests in
`tests/inductive/methods/_methods_classic.py` additionally cover NumPy/Torch
parity, ties, per-example majority groups, convergence, and pseudo-label
diagnostics.

### Vote protocol evidence

The paper states that the three base learners are Naive Bayes, C4.5, and 3-NN.
For Vote, Table 2 gives 16 attributes, 40 labeled examples, 200 unlabeled
examples, and 2.2 update rounds on average. Section 5.1 says that 20 random
`L/U/test` partitions were retained only when Democratic Co-Learning labeled at
least one example in `U`. Table 3 reports accuracy `0.944 +/- 0.012`.

The test-blind screening campaign, published under the alias
`dcl-vote-partition-screening-v6`, completed 100 of 100 tasks. Its private
source identifier is bound by SHA-256
`f9a03e169898482256fdf0c6da40b4a065db02152309bc3353ed1abe17c3a88b`;
the alias is not a rewritten `campaign_id`. Selection used only convergence,
iteration count, and positive
`pseudo_labels_added_total`; it did not inspect test labels or test metrics.
The first 20 eligible seeds were locked in:

`bench/campaigns/locks/dcl-vote-zhou-goldman-2004-v1/selected-partitions.json`

The lock file SHA-256 is
`5f586b2ab21bd6c2b0e058ab9d588ec1fc04b41b7d93e5a125d0a5f2ea1b36fb`.
Every selected row pins the split fingerprint and the SHA-256 values for
`MANIFEST.json`, `split.json`, and `arrays.npz`. Screening outputs remain
diagnostic evidence and must never enter the paper aggregate.

### Deliberate ceiling

The source does not specify the exact confidence-interval construction,
software versions, learner options, random-number generator, or original
partitions. ModSSC uses a clipped normal/Wald accuracy interval and explicit
NumPy historical-classifier backends as reconstruction choices. Their options
and numerical oracles are versioned with ModSSC; no external classifier runtime
is loaded.

These are known deviations, not hidden assumptions. Even with a passed
algorithmic gate and a valid partition lock, the Vote profile ceiling is
`paper_approx`; it cannot produce `paper_matched`. The Adult profile remains
`not_claimable` until its separate data and protocol unknowns are resolved.
