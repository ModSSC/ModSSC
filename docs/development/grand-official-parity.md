# GRAND official-code parity

This note records the executable reference used by the Cora/Table 1
reproduction. It is evidence about algorithmic parity, not evidence that the
published accuracy has already been reproduced.

## Frozen provenance

- Paper: *Graph Random Neural Networks for Semi-Supervised Learning on Graphs*,
  NeurIPS 2020, SHA-256
  `b37c0c72d463d8ccd973d59fb85bdf23ab2a18e13bde7d5cde5916564272e7b2`.
- Supplemental material, SHA-256
  `aa4c625aa3a074e3440386bf303efe2c3cb85f5eab6b1ecbbf40e793be48a7d2`.
- Repository: `THUDM/GRAND`.
- Commit: `7a2fd6e7c3f20ca2c84b06ec1c5dc7f227dbfe2b`.
- `train_grand.py`:
  `6c6e3162937fcb382172569af7d9ddfa71c677a3a406c17ace7cd3d7a4978443`.
- `pygcn/models.py`:
  `970837c2bd448bf21bb6085ff8604d73bd8f7098260696f86fd1f1acd6082cba`.
- `pygcn/layers.py`:
  `a2715fd232e3449d76a072b88496938e2182aade355bce8d363f2056476567c4`.
- `pygcn/utils.py`:
  `a6f272c9ddcccca29a7c788b0b9c36c0b83c57b137d700f02e765d810c4a90d6`.
- `run100_cora.sh`:
  `cc94cfc7eb6194c6a6cb4a2438a89a72b518163eee10c1514542299420dc684a`.

The frozen miniature oracle is
`tests/transductive/methods/gnn/fixtures/grand_official_7a2fd6e7.json`.
Its float32 outputs were produced by an independent transcription of the
pinned functions without importing ModSSC. The parity test checks DropNode,
mixed-order propagation, sharpening, consistency loss, supervised and combined
losses, initialization, and every checkpoint/patience transition.

## Paper versus executable code

The paper and the pinned implementation differ in three trajectory-critical
places. The reproduction profile follows the executable code because Table 1
was produced by that implementation.

| Operation | Paper | Pinned executable code | ModSSC reproduction |
|---|---|---|---|
| DropNode scaling | Section 3.1 describes division by `1-delta` during training and the original features at inference. | Training masks are unscaled; inference multiplies features by `1-delta`. | Pinned-code behavior. |
| DropNode RNG | The paper specifies Bernoulli masks but not the device-specific stream. | Masks are sampled with CPU `torch.bernoulli` and then copied to CUDA, while MLP dropout uses CUDA RNG. | The same separated CPU/CUDA streams. |
| MLP initialization | Appendix A.2 says Glorot normal. | `MLPLayer.reset_parameters` calls `normal_(mean=-1/sqrt(out), std=1/sqrt(out))` for weights and biases. | The same draws in the official `(in, out)` layout, transposed into `nn.Linear`. |

The pinned checkpoint rule is also more specific than “early stopping based on
validation loss” in Appendix A.2. Patience resets when either validation loss
reaches a running minimum or validation accuracy reaches a running maximum.
The saved state changes only when validation loss improves, inside that reset
branch. ModSSC implements that nested rule and restores the saved state before
test prediction.

## Cora/Table 1 contract

`bench/configs/reproductions/grand/cora.yaml` pins the public Planetoid
140/500/1000 masks and the values from `run100_cora.sh` and Appendix Table 3:
hidden size 32, propagation order 8, four augmentations, DropNode 0.5,
temperature 0.5, consistency coefficient 1.0, input/hidden dropout 0.5, Adam
learning rate 0.01, weight decay `5e-4`, patience 200, and 100 campaign master
seeds numbered 0 through 99.

The eight raw Cora files in the ModSSC cache are byte-identical to those in
the pinned repository; their individual hashes are frozen in the parity
fixture. After preprocessing, features, labels, and the public 140/500/1000
masks are exact. The normalized adjacency has the same 13,264 nonzero entries
(including self-loops); the largest weight difference observed against the
official SciPy path is `2.98e-8`, caused by float32 normalization in ModSSC
versus float64 normalization followed by a float32 cast in the pinned code.

The GRAND paper and canary campaign cells set `model_seed_policy: literal`, so
the integers 0 through 99 reach PyTorch exactly as in `run100_cora.sh`.
Dataset and sampling component seeds remain separately derived, but they
cannot alter the frozen public Planetoid masks. The pinned paper environment
(PyTorch 1.2.0 on an RTX 2080 Ti) still differs from the configured-Slurm-site runtime, so
empirical acceptance is statistical rather than bitwise.

The deterministic miniature parity is necessary but not sufficient for a
paper-result claim. Dataset bytes and masks must remain pinned, all 100
repetitions must finish from a clean immutable build, and the configured
acceptance statistics must still pass.
