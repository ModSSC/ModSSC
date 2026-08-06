# Historical Self-Training and Co-Training: v2 results

## Frozen evidence

The two campaigns were executed locally from the clean commit
`d6b89866b7d17bd56cd66c911ebc7513e41a757a`, tagged
`replication-10m-historical-replacements-v2`. They used Python 3.12.13,
NumPy 2.3.5, and scikit-learn 1.8.0. Every run contains an authenticated
sampling replay and a distinct split fingerprint. No result from the dirty
technical canaries or the underflow-affected v1 attempt is included.

The acceptance artifacts are stored below
`modssc_cache/output/historical-replacements-v2/acceptance/`. Their provenance
seals are:

- Self-Training: `e2917e1d20ed886dafeee978b6521f06e45a1f4b4d285c1c7841518a62ebcaf6`;
- Co-Training: `0f0eea6bf4a1905108a1e312f6fc7719a9414006f57cbe1a04507e32f70e98f5`.

## Acceptance summary

| Method | Runs | Published error | Replication error, mean ± sample SD | 95% Student interval | Absolute difference | Acceptance |
|---|---:|---:|---:|---:|---:|---|
| Self-Training, Wine | 50/50 | 7.90% | 6.36% ± 3.95% | [5.24%, 7.49%] | 1.54 points | `paper_approx/failed_replication` |
| Co-Training, WebKB | 5/5 | 5.00% | 18.17% ± 5.35% | [11.54%, 24.81%] | 13.17 points | `paper_approx/failed_replication` |

Self-Training respects the pre-registered two-point absolute margin, but the
published value is outside the replication interval. The result therefore
fails the conjunction required by the acceptance rule and is not marked as a
successful numerical replication. All 50 trajectories completed 40 rounds,
added between 112 and 118 pseudo-labels (mean 115.48), and ended with between
125 and 131 labeled examples (mean 128.48).

Co-Training fails both the primary and secondary controls:

| WebKB metric | Published error | Replication mean error | 95% Student interval |
|---|---:|---:|---:|
| Combined classifier | 5.00% | 18.17% | [11.54%, 24.81%] |
| Full-text view | 6.20% | 17.87% | [11.19%, 24.56%] |
| Inlink view | 11.60% | 32.17% | [25.53%, 38.81%] |

All five Co-Training runs completed the 30 published rounds and appended 240
ordered proposals to the shared labeled multiset. They promoted 232--237
unique pages (mean 235.2), with mean same-label and conflicting overlap counts
of 3.2 and 1.6. The primary run errors ranged from 11.03% to 24.71%.

## Interpretation

Both implementations now have reproducible paper profiles and pass their
algorithmic and provenance controls, so their fidelity ceiling is
`paper_approx`. Neither campaign reproduces the published numerical result
under the frozen reconstruction. The most plausible residual causes are the
unpublished historical partitions and preprocessing choices for Self-Training,
and the unpublished WebKB parsing, tokenization, vocabulary, smoothing, and
split identities for Co-Training. No post-test parameter search was performed.
