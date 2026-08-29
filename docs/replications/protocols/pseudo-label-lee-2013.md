# Pseudo-Label — Lee (2013)

## Primary source

Dong-Hyun Lee, *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning
Method for Deep Neural Networks*. The reviewed six-page paper has SHA-256
`47a14f59e80178e554e6c205a3e2edea2c50ec7a63e0051640559358f3435bad`.
No official source-code release is supplied or cited by the paper.

## Registered protocol

The active card registers the MNIST Table 2 experiment without DAE pretraining
and with 600 balanced labels over ten random splits. It transcribes the
one-hidden-layer network, labeled and unlabeled minibatches, hard pseudo-label
rule, two loss terms, learning-rate and momentum schedules, and the published
alpha schedule. One thousand labeled examples are reserved for validation.

The paper does not publish its initialization, exact terminal epoch, original
split indices, traversal details, or an unambiguous statement about visible
unit dropout for this table. The card exposes deterministic reconstruction
choices for each unknown.

## Claim boundary

The fidelity ceiling is `paper_approx`. A fresh ten-repetition assessment is
required, and this page contains no ModSSC run result or verdict.
