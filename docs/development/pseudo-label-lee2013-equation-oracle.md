# Pseudo-Label Lee 2013 equation oracle

This note records the independent primary-source audit for the MNIST Table 2
`+PL` experiment without DAE. No official source-code release is supplied or
cited by the paper, so no official-code parity is claimed.

## Frozen provenance

- Paper: Dong-Hyun Lee, *Pseudo-Label: The Simple and Efficient
  Semi-Supervised Learning Method for Deep Neural Networks*.
- Local PDF SHA-256:
  `47a14f59e80178e554e6c205a3e2edea2c50ec7a63e0051640559358f3435bad`.
- Audited PDF pages: 2, 3, and 5 of the six-page file.
- Audit date: 2026-07-24.
- Published target: 5.03% MNIST test error with 600 labeled examples, ten
  random splits.

## Primary-source transcription

The paper specifies:

- one hidden layer with 5,000 rectified linear units and sigmoid output units;
- non-inverted hidden-unit dropout with probability 0.5;
- a 32-example labeled minibatch and a 256-example unlabeled minibatch;
- hard pseudo-labels from the maximum output probability (Equation 14);
- simultaneous labeled and unlabeled cross-entropy terms (Equation 15);
- the learning rate recurrence `epsilon(t + 1) = 0.998 epsilon(t)`, starting
  from 1.5 (Equation 12);
- momentum increasing linearly from 0.5 to 0.99 over 500 epochs (Equation 13);
- the update multiplier `(1 - p(t))` in Equation 10;
- `alpha_f = 3`, `T1 = 100`, and `T2 = 600` for the run without pretraining
  (Equation 16);
- 1,000 labeled validation examples, with the remaining training examples used
  as unlabeled data.

The independent oracle in
`tests/inductive/methods/_pseudo_label_lee2013_oracle.py` uses only NumPy and
the printed equations. Its comparison test covers the three schedules, hard
pseudo-labels, both loss terms, and a two-step momentum trajectory without
calling production helpers to compute expected values.

## Explicit reconstruction choices

The paper does not publish the weight initialization, a precise terminal epoch,
the ten original split indices, or whether the optional 20% visible-unit
dropout mentioned in Section 2.3 was used for Table 2. The ModSSC card records
the following choices:

- framework-default deterministic weight initialization;
- 601 epochs, so the no-pretraining alpha schedule reaches `T2 = 600`;
- new recorded balanced 60-per-class splits with an exact 1,000-example
  validation set;
- pixel values scaled from `[0, 255]` to `[0, 1]`;
- no visible-unit dropout and hidden-unit dropout 0.5;
- 229 updates per epoch to traverse all 58,400 unlabeled examples in
  256-example batches, while the 600 labeled examples are reshuffled and
  recycled in 32-example batches.

These are known deviations rather than inferred paper facts. Therefore the
equation gate may pass as an `independent_equation_oracle`, but the profile
ceiling remains `paper_approx`.
