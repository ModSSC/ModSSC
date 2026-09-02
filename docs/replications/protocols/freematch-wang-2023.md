# FreeMatch — Wang et al. (2023)

## Primary sources

- Wang et al., *FreeMatch: Self-adaptive Thresholding for Semi-supervised
  Learning*, [paper](https://arxiv.org/pdf/2205.07246).
- Reference implementation:
  [Microsoft Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning/tree/1ef4cbebcc0b368158315aeb425053858cf6c845)
  at commit `1ef4cbebcc0b368158315aeb425053858cf6c845`.

## Registered protocol

The active card registers CIFAR-10 with 40 labels and three article
repetitions. It uses the common WRN-28-2/SGD/EMA stack, self-adaptive
thresholding and fairness regularization, labeled/unlabeled batches 64/448,
replacement sampling, and (2^{20}) optimizer steps. The entropy coefficient
is 0.05, following the article's Appendix Table 6 for this label budget.

Later configuration sources disagree on that coefficient and the generated
split files were not published. The article value is kept as the declared
primary choice; no alternative is selected from test results.

## Claim boundary

The fidelity ceiling is `paper_matched`; a fresh complete three-repetition
assessment is required. This page contains no run result or verdict.
