# Native replication result registry

!!! warning "Registry state: EMPTY / HOLD"

    No new native result bundle has passed reconciliation, archive sealing, and
    publication review.

Future [paper replications](paper/index.md) reproduce article-specific
protocols. This registry does not host generic benchmark comparisons.

Each listed bundle will be an immutable directory containing a human summary,
canonical manifest, checksums, compact results, and compact observations. Raw
runs, checkpoints, caches, datasets, and scheduler logs remain in a sealed
external archive authenticated by the public manifest.

The registry lists negative, partial, and `not_evaluable` outcomes as well as
successful ones. A bundle appears here only after the checks in the
[publication policy](../publication-policy.md) pass; a canary or Slurm job
completion is never sufficient.
