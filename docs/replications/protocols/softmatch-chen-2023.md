# SoftMatch — Chen et al. (2023)

## Primary sources

- Chen et al., *SoftMatch: Addressing the Quantity-Quality Trade-off in
  Semi-supervised Learning*, [paper](https://openreview.net/pdf?id=ymt1zQXBDiF).
- Reference implementation:
  [TorchSSL/TorchSSL](https://github.com/TorchSSL/TorchSSL/tree/03193a1b7883727db1ce9c092e083091e18aedbb)
  at commit `03193a1b7883727db1ce9c092e083091e18aedbb`.

## Registered protocol

The active card registers CIFAR-10 with 250 labels and three article
repetitions. It pins the TorchSSL WRN-28-2/SGD/EMA stack, Gaussian confidence
weighting, labeled-prediction alignment, labeled/unlabeled batches 64/448,
replacement sampling, (2^{20}) optimizer steps, and the source
evaluation/checkpoint schedule.

The paper pseudocode and later implementations differ in the placement and
form of distribution alignment. The card follows the TorchSSL path identified
by the paper for the registered experiment and records the alternatives as
non-selectable source ambiguities.

## Claim boundary

The fidelity ceiling is `paper_matched`; a fresh complete three-repetition
assessment is required. This page contains no run result or verdict.
