# Democratic Co-Learning — Zhou and Goldman (2004)

## Primary source

Yan Zhou and Sally Goldman, *Democratic Co-Learning*, ICTAI 2004,
[DOI 10.1109/ICTAI.2004.48](https://doi.org/10.1109/ICTAI.2004.48).
The reviewed PDF has SHA-256
`f14d7f8c2782476911a45d88eea73df9c72c6547c0a380b4e7620c530f30afed`.
The article does not provide or cite an official source-code release.

## Registered protocol

The active cards cover the Adult and Congressional Voting Records rows of
Table 3. They use three distinct learners—Naive Bayes, C4.5 and 3-NN—and
implement the proposal, quality and final weighted-vote equations from Figures
1 and 2. Vote declares 40 labeled and 200 unlabeled examples; Adult declares 60
labeled and 1,691 unlabeled examples. Each card requests the 20 repetitions
stated by the article.

The independent native oracles transcribe Figure 1's strict-majority,
confidence and proposal-quality transitions and Figure 2's filtered,
Laplace-corrected final vote. The production helpers are checked against those
transcriptions. Figure 2 does not define a prediction when every learner is
filtered out, so the paper path fails closed in that state instead of applying
an undocumented ensemble fallback.

The source does not identify the original partitions, learner versions,
learner options, random stream, or confidence-interval construction. ModSSC
therefore records explicit native reconstruction choices and never invokes an
external classifier runtime.

## Claim boundary

The Adult card remains capped at `not_claimable` because original data and
software parity cannot be established. The Vote card is capped at
`paper_approx`: its algorithmic conformity is `passed` by the independent
equation oracle, while the missing learner versions, confidence-interval
definition, partitions and RNG remain explicit deviations or unknowns. The
fresh campaign alone determines the numerical result. This page records no
execution outcome.
