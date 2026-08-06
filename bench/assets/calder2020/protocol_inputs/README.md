# Calder 2020 protocol inputs

This directory is the complete non-dataset input bundle used by the ModSSC
Laplace and Poisson Table 1 reproductions. It contains a frozen VAE k-nearest
neighbour graph, a safe integer-only representation of the 500 published label
partitions, and the published per-run reference results.

All executable algorithms are implemented inside `src/modssc`. No source file,
Python package, pickled object array, or executable from the authors' repository
is loaded at runtime. `MANIFEST.json` authenticates every byte and retains the
upstream commit and original archive digests only as provenance.

MNIST itself is obtained through the ModSSC dataset provider. Its merged
train-then-test label ordering is authenticated by the content digest recorded
in the manifest before the protocol inputs are used.
