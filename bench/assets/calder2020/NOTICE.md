# Calder et al. replication assets

This directory contains the complete authenticated non-dataset inputs needed
by the Calder Table 1 reproductions. It deliberately contains no executable
GraphLearning source code and no object archive requiring pickle.

- The inputs originate from
  [GraphLearningOld](https://github.com/jwcalder/GraphLearningOld) at commit
  `04bece45cd512cf1a3bcddb163b767ca44a746e1`. The audited source path and its
  SHA-256 remain recorded in the parity oracle, but ModSSC neither distributes
  nor executes that implementation.
- `protocol_inputs/graph/` contains the derived VAE/k-nearest-neighbour graph.
  It is not claimed to be covered by the upstream source-code licence.
- `protocol_inputs/splits/` contains the deterministic, integer-only ModSSC
  representation of the 500 label partitions and is always read with
  `allow_pickle=False`.
- `protocol_inputs/references/` contains the published per-run result tables.
- MNIST labels are not duplicated here. ModSSC prepares MNIST through its own
  provider and authenticates their merged train-then-test byte ordering against
  the digest in `protocol_inputs/MANIFEST.json`.

ModSSC treats these files only as authenticated data inputs. Laplace and
Poisson are executed exclusively by the implementations under `src/modssc/`.
