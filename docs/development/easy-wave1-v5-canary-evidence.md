# Easy-wave v5 canary evidence

This record freezes the promotion decision for the light paper-replication
wave. It does not turn a canary score into a paper result. Numerical acceptance
still requires every repetition declared by the production protocol.

## Release and execution boundary

- Release: `replication-10m-easy-wave1-v5`,
  `8cbee2e53a029c39d53b0f1557b68dbbd9653e77`.
- configured Slurm site build manifest:
  `evidence://modssc/build-manifests/8cbee2e53a02.json`.
- Every preflight, method run, reconciliation, daily report, and paper
  evaluation ran under Slurm. The login node only generated manifests,
  submitted jobs, and inspected small reports.
- The exact machine-readable values used below are frozen in
  `provenance/article10/evidence/easy-wave1-v5-canaries.json`.

The first three arrays were refused before `run-task` because the submission
environment omitted `MODSSC_ENVIRONMENT_LOCK_SHA256`. No method was imported
and no scientific result was produced. Slurm reconciliation emitted signed
retry campaigns. Fresh architecture-specific preflights then passed, and only
those retry wrappers were submitted. This failure remains in the audit record;
it was not overwritten.

## Tri-Training

The full Vote historical-backend canary completed in 24 seconds and reconciled
1/1 success with no retry or reprofile request. The V100 preflight verified the
pinned learner contract; the autonomous card now resolves the corresponding
NumPy backend from ModSSC and requires no external executable.

The run retained the initial ensemble, converged in two rounds, used the
released-code `soft_average` rule, and selected 59 pseudo-labels. The score was
91.7431% on this one locked partition. The canary therefore validates the
execution path and reviewed algorithm, not the three-partition paper mean.

Algorithmic conformity moves to `passed` using the recorded source hashes, the
independent fixture replay, and this full-profile canary; no Java or Weka code
is distributed or executed. The fidelity
ceiling remains `paper_approx`: the historical test indices and RNG state were
not published, and the released probability vote and unpruned J48 demonstration
are recorded deviations from the paper prose.

## Pseudo-Label

The full MNIST/A100 canary completed in 282 seconds and reconciled 1/1 success
with no retry or reprofile request. It completed 601 epochs, 229 updates per
epoch and 137,629 parameter updates; alpha reached 3.0, no confidence threshold
was applied, and all 58,400 unlabeled examples received final pseudo-labels.

The score was 93.05% on this one locked split. Ten runs project to about 0.783
A100-hours, below both production guards. Algorithmic conformity moves to
`passed` using the independent Lee-equation oracle plus this full-profile
trajectory. The fidelity ceiling remains `paper_approx` because the ten
historical split indices, initialization, optional visible-unit dropout and
some traversal details are not recoverable from the paper.

## GRAND

The three literal official seeds completed in 85, 91 and 112 seconds. Their
accuracies were 85.2%, 85.8% and 85.4%, averaging exactly the published 85.4%.
All ten registered diagnostics were present, including official DropNode
scaling and CPU RNG, mixed-order propagation, initialization and checkpoint
policy. The p95 duration was 109.9 seconds; `1.25 × p95`, rounded up to the next
minute, fixes production walltime at three minutes with concurrency `%10`.
