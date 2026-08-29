# Replications

!!! info "Current status: Calder VAE2 execution authorised"

    The isolated Calder MNIST VAE2 reconstruction is authorised for Jean Zay.
    Its result registry remains empty until all declared seeds are reconciled
    and reviewed. This does not promote unfinished runs from other methods.

This section is the public, durable entry point for ModSSC replication work. It
separates scientific configuration, cluster operation, raw evidence, and
publication so that a result can be checked years after the allocation that
produced it has ended.

The supported execution boundary remains:

```text
YAML card -> generic bench runner -> native src/modssc components -> run.json
```

Scientific behavior belongs in `src/modssc`. A reproduction card declares the
protocol and the generic benchmark runner reads it, orchestrates native
components, and serializes their outputs. `bench` must not acquire branches for
a named method, paper, dataset, or cluster. Jean Zay scripts only map scheduler
resources and array indices to this public runner; they are private deployment
adapters, not a second campaign framework.

## Article-only scope

This section exists only to reproduce each article's own data, split, label
budget, preprocessing, training, selection, metric, and repetition contract.
The protocols are deliberately heterogeneous and cannot be treated as a common
leaderboard. Generic benchmark comparisons remain separate from this
replication registry.

## Public and private boundaries

The Git repository stores self-contained YAML protocols, source-only article
notes, compact immutable result bundles, their manifests, and explanatory
documentation. It does not store datasets, caches, checkpoints, raw per-seed
reports, scheduler logs, or Jean Zay account and filesystem details. Those raw
artifacts live in a sealed external archive identified from the public bundle
by a content manifest and SHA-256 digest.

A result is published only after transfer away from the execution checkout,
reconciliation of the complete declared seed set, redaction of private
locators, and hash verification. Failed, missing, or `not_evaluable` outcomes
remain visible; publication never turns an incomplete run into a successful
claim.

## Current programme

The first planned native campaign covers ten methods across 20 primary cards:

1. Pseudo-Label;
2. Tri-Training;
3. Democratic Co-Learning;
4. FixMatch;
5. FlexMatch;
6. FreeMatch;
7. SoftMatch;
8. Laplace Learning;
9. Poisson Learning;
10. GRAND.

The 20 cards declare 1,170 primary seed-runs: 1,056 CPU runs and 114 GPU runs.
See the [campaign design](campaigns/native-10-methods-v1.md) for its gated
execution sequence.

## Registry

- [Publication policy](publication-policy.md): required public bundle format,
  integrity, redaction, and immutability rules.
- [Publication schema v1](publication-schema-v1.md): exact per-release,
  per-card, and per-seed allow-list contract.
- [Article protocol evidence](protocols/index.md): source-only notes for the
  ten active methods, without prior results.
- [Native ten-method replication v1](campaigns/native-10-methods-v1.md): staged
  Jean Zay campaign and launch gates.
- [Calder MNIST VAE2 reconstruction v1](campaigns/calder-mnist-vae2-v1.md):
  active 1,000-run Laplace/Poisson execution and durable result layout.
- [Result registry](results/index.md): entry point for newly generated native
  evidence. It stays empty until reconciliation and publication review.

No long benchmark may be launched merely because a card parses or a canary
succeeds. Each lot is promoted progressively only when its environment, data,
resource, continuation, and identity gates pass.
