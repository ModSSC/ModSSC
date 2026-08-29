# Poisson Learning — Calder et al. (2020)

## Primary sources

- Calder et al., *Poisson Learning: Graph Based Semi-Supervised Learning at
  Very Low Label Rates*, [PMLR paper](https://proceedings.mlr.press/v119/calder20a.html).
- Authors' [GraphLearning repository](https://github.com/jwcalder/GraphLearning).

## Registered protocol

Five active cards cover the Poisson Learning cells of MNIST Table 1 for one
through five labels per class, with 100 repetitions per cell. Each card merges
the official MNIST splits into one transductive pool, trains the native
`graphlearning_mnist_vae2` recipe on all 70,000 images, and builds the archived
2022 GraphLearning graph protocol: an Annoy Euclidean index with 10 trees,
30 returned candidates and `search_k=-1`, followed by direct truncation to the
first 10 neighbours including self, Gaussian weights, and `(W + W^T) / 2`.
It then draws balanced labels and applies the article iteration and class-prior
correction.

No author-generated VAE, neighbour bank, graph, or label permutation is loaded.
The VAE and graph are native, content-addressed stages shared with every Laplace
card, label budget, and sampling seed. The native Poisson paper solver removes
self-loops before its degree-scaled iteration, so no method-specific graph is
constructed by the benchmark runner.

## Claim boundary

The fidelity ceiling is `paper_approx`. VAE2 is a post-publication improvement,
not the unpublished embedding used for Table 1, and its released weights and
random seeds are not runtime inputs. ModSSC retrains the documented recipe with
a fixed seed and records the Annoy version in runtime identity. The original
balanced label draws are also unavailable and are regenerated per declared
seed. The published Table 1 mean and standard deviation remain unchanged as an
honest numerical target. This page records no execution outcome.
