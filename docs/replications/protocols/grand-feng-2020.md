# GRAND — Feng et al. (2020)

## Primary sources

- Feng et al., *Graph Random Neural Networks for Semi-Supervised Learning on
  Graphs*, [paper](https://arxiv.org/pdf/2005.11079).
- Official [THUDM/GRAND repository](https://github.com/THUDM/GRAND/tree/7a2fd6e7c3f20ca2c84b06ec1c5dc7f227dbfe2b)
  at commit `7a2fd6e7c3f20ca2c84b06ec1c5dc7f227dbfe2b`.

## Registered protocol

The active card registers the Cora Table 1 experiment with the public
Planetoid 140/500/1000 masks and 100 literal seeds. It pins the official hidden
size, propagation order, four stochastic augmentations, DropNode,
sharpening/consistency loss, Adam settings, patience rule, and executable
checkpoint policy.

Where the prose and pinned implementation differ—DropNode scaling, random
stream placement, initialization, and nested early stopping—the card follows
the executable source used for the table. Those choices are explicit and
covered by a fixed source-parity fixture.

## Claim boundary

The fidelity ceiling is `paper_matched`, subject to a fresh complete
100-repetition assessment on authenticated data and software. This page records
no execution outcome.
