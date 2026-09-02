# FlexMatch — Zhang et al. (2021)

## Primary sources

- Zhang et al., *FlexMatch: Boosting Semi-Supervised Learning with Curriculum
  Pseudo Labeling*, [paper](https://arxiv.org/pdf/2110.08263).
- Reference implementation:
  [Microsoft Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning/tree/1ef4cbebcc0b368158315aeb425053858cf6c845)
  at commit `1ef4cbebcc0b368158315aeb425053858cf6c845`.

## Registered protocol

The active card registers CIFAR-10 with 250 labels and three article
repetitions. It uses the FixMatch WRN-28-2/SGD/EMA stack, curriculum
pseudo-labeling, labeled/unlabeled batches 64/448, replacement sampling,
(2^{20}) optimizer steps, and the source evaluation/checkpoint schedule.
The unlabeled pool includes the labeled examples as in the pinned source.

The generated index files and process-global worker random stream were not
published. Explicit loader seeds and serialized generator state provide a
portable reconstruction without claiming identical original draws.

## Claim boundary

The fidelity ceiling is `paper_matched`; a fresh complete three-repetition
assessment is required. This page contains no run result or verdict.
