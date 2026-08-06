# Historical Self-Training and Co-Training: v3 confirmation results

## Frozen evidence

The confirmation campaigns were executed locally from the clean commit
`1d6ac37532400f3c08b5f4b8dd262cb1917eaaf0`, tagged
`replication-10m-historical-replacements-v3`. The environment used Python
3.12.13, NumPy 2.3.5, and scikit-learn 1.8.0. Every accepted run reports
`git_dirty=false`, authenticates its dataset and replay manifest, and uses a
distinct split fingerprint.

The immutable acceptance artifacts are stored below
`modssc_cache/output/historical-replacements-v3-1d6ac375/acceptance/`. The
scientific provenance seals embedded in them are:

- Self-Training confirmation: `2ad577c089b7b66a07f1aa9f22b5af280d213dc1a362d2ea351de376d1876be6`;
- Co-Training test-blind diagnostic gate: `ccb07eb489d563f4e7dbbf57622f7276bf4a38a708db8a110ecf4de1a7e226ee`;
- Co-Training confirmation: `2931475a6991a2c05801cbcb7ec553f803961508ca9060b9a1755eaa7947aaad`.

No parameter, preprocessing choice, or implementation detail was selected
from the v3 confirmation test metrics.

## Acceptance summary

| Method | Runs | Published error | Replication error, mean +/- sample SD | 95% Student interval | Absolute difference | Result |
|---|---:|---:|---:|---:|---:|---|
| Self-Training, Wine | 50/50 | 7.90% | 8.36% +/- 4.74% | [7.02%, 9.71%] | 0.46 points | `replicated_paper_approx` |
| Co-Training, WebKB | 5/5 | 5.00% | 7.15% +/- 1.81% | [4.90%, 9.40%] | 2.15 points | `paper_approx/failed_replication` |

Self-Training satisfies both pre-registered numerical conditions: the
published value lies in the replication interval and the absolute difference
is below two percentage points. It is therefore a successful numerical
replication. Its scientific ceiling remains `paper_approx`, because the
historical split seeds, pool size, exact confidence formula, and scaling rule
were not fully specified by the source. All 50 runs completed 40 rounds and
added 105--120 pseudo-labels (mean 115.5).

Co-Training improved markedly over the preceding reconstruction, but it does
not satisfy the frozen acceptance rule. The published target lies in the 95%
interval, while the absolute difference exceeds the two-point margin by 0.15
point. The secondary controls localize the remaining discrepancy:

| WebKB metric | Published error | v3 replication mean error | 95% Student interval | Within two-point margin |
|---|---:|---:|---:|---|
| Combined classifier | 5.00% | 7.15% | [4.90%, 9.40%] | No |
| Full-text view | 6.20% | 7.00% | [4.33%, 9.66%] | Yes |
| Inlink view | 11.60% | 19.16% | [13.51%, 24.82%] | No |

The combined WebKB error fell from 18.17% in v2 to 7.15% in v3, the full-text
error from 17.87% to 7.00%, and the inlink error from 32.17% to 19.16%. The
v3 implementation is therefore substantially closer to the published method,
but the weak inlink view prevents a replication claim.

## Test-blind Co-Training gate

Before the five confirmation seeds were run, seeds 1--5 completed a diagnostic
campaign that emitted only labeled-training metrics. The fail-closed gate
verified the clean commit and environment, exact partition replay, 30 rounds,
dynamic mutual-information feature traces, finite Craven scores, and the
absence of every test metric. It then sealed the confirmation cards for seeds
6--10.

This gate prevents v3 test-guided tuning, but it is not a claim of complete
historical blindness: results from the earlier v2 reconstruction were already
known when v3 was designed.

## Self-Training split controls

The two supervised controls from the same Wine experiment were rerun on the
same fresh seeds 51--100 and the same test partitions:

| Control | Runs | Published error | Replication error, mean +/- sample SD | 95% Student interval |
|---|---:|---:|---:|---:|
| NN-L | 50/50 | 9.00% | 9.14% +/- 4.89% | [7.75%, 10.53%] |
| NN-A | 50/50 | 4.80% | 4.45% +/- 2.75% | [3.67%, 5.24%] |

Both published controls lie inside their replication intervals. This makes an
abnormal Wine split or a broken nearest-neighbor baseline unlikely explanations
for the Self-Training match.

## Interpretation and final status

- Self-Training is retained as the historical replacement for Pseudo-Label,
  with status `replicated_paper_approx`: numerically replicated, historically
  under-specified.
- Co-Training remains a scientifically useful replacement candidate for
  Democratic Co-Learning, but its v3 status is
  `paper_approx/failed_replication`. It must not be labeled replicated.
- The remaining Co-Training suspects are the historical inlink-anchor parser
  and tokenizer, the exact top-2000 feature-selection criterion, Naive Bayes
  smoothing and confidence details, and the unavailable original partitions.
- No further tuning on seeds 6--10 is permitted. A new attempt would require
  new source evidence and fresh, pre-registered confirmation seeds.
