# FixMatch — Sohn et al. (2020)

## Primary sources

- Sohn et al., *FixMatch: Simplifying Semi-Supervised Learning with Consistency
  and Confidence*, [paper](https://arxiv.org/pdf/2001.07685).
- Official implementation:
  [google-research/fixmatch](https://github.com/google-research/fixmatch/tree/d4985a158065947dba803e626ee9a6721709c570)
  at commit `d4985a158065947dba803e626ee9a6721709c570`.

## Registered protocol

The active card registers the CIFAR-10, 250-label RandAugment row: five folds,
WRN-28-2, labeled/unlabeled batches 64/448, confidence threshold 0.95,
SGD/Nesterov, EMA, cosine decay, and exactly (2^{20}) optimizer steps. It
follows the official class-balanced split construction, inclusive unlabeled
pool, BatchNorm interleave, and median-of-last-checkpoints article policy.

Generated split files and the TensorFlow parallel-input random bitstream were
not published. The card uses authenticated deterministic reconstructions with
the same declared distributions and records those equivalences explicitly.

## Claim boundary

The fidelity ceiling is `paper_matched`, but only a fresh complete five-fold
assessment may award that status. This page records source parity only and no
ModSSC execution outcome.
