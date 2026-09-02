# Native ten-method replication v1

!!! danger "Campaign state: HOLD"

    Zero new native paper runs are accepted yet. Production remains gated by
    the Jean Zay environment, data, canary, continuation, and resource checks
    below.

## Scope

This campaign is the first clean execution of the current native replication
cards. It uses the public `modssc-bench --config CARD --seed-index N` interface
for every task. Private Jean Zay adapters select resources and map a Slurm array
index to `N`; they do not parse article protocols or implement method behavior.

The 20 primary cards cover ten methods and 1,170 runs:

| Family | Methods | Cards | Runs | Intended resource class |
|---|---|---:|---:|---|
| Classic inductive | Pseudo-Label, Tri-Training, Democratic Co-Learning | 5 | 56 | CPU |
| Transductive Calder Table 1 | Laplace Learning, Poisson Learning | 10 | 1,000 | CPU |
| Deep Match | FixMatch, FlexMatch, FreeMatch, SoftMatch | 4 | 14 | GPU |
| Graph neural | GRAND | 1 | 100 | GPU |
| **Primary total** | **10 methods** | **20** | **1,170** | **1,056 CPU; 114 GPU** |

The classic count comprises one Pseudo-Label card, two Tri-Training cards, and
two Democratic Co-Learning cards. This campaign has no additional screening
cards or diagnostic article variants.

## Non-negotiable boundaries

- YAML declares scientific parameters and inputs.
- `bench` performs generic validation, orchestration, reconciliation, and
  serialization only.
- Sampling, preprocessing, graph construction, models, methods, checkpoints,
  and acceptance behavior live in `src/modssc`.
- Scheduler files, module names, accounts, partitions, quotas, paths, array
  throttles, and retry policy stay outside the repository.
- A scheduler success code is insufficient: each task must produce an
  authenticated native report with its expected protocol, seed, software, and
  execution identity.
- No recovered replication artefact is a runtime dependency. Calder recomputes its native
  preprocessing and graph artefacts from authenticated input data.
- Production lots start progressively only after their bounded canaries and
  resource gates pass.

## Gated execution plan

Every gate produces a small review record. A failed gate stops dependent waves;
there is no silent fallback, resource substitution, seed reduction, or
promotion from canary to production without a recorded gate decision.

### G0 — Freeze source and campaign inventory

Record a clean Git commit, the exact 20 card paths, their declared seed sets,
and their protocol hashes. Build the
immutable source snapshot that all jobs will use. Reject a dirty checkout or a
card-count/run-count mismatch.

**Exit evidence:** source digest, card inventory digest, `20 / 1,170` primary
totals, and confirmation that only the five bounded paper canaries remain.

### G1 — Authenticate the Linux environment

Create the environment on a Jean Zay login node from an immutable Linux lock or
wheelhouse. Record Python, ModSSC distribution, native libraries, optional
backends, and accelerator stack. Import every required native component in a
short allocation; do not infer compatibility from the macOS environment.

**Exit evidence:** environment manifest and digest, import report, and clean
source verification from the compute node.

### G2 — Authenticate data and storage

Stage each licensed dataset and validate the card-declared dataset fingerprint
before training. Verify compute-node access and write/rename semantics for data,
preprocess, graph, checkpoint, output, and log roots. Confirm that execution
does not require internet access. Keep raw locations private.

**Exit evidence:** sanitized dataset-fingerprint matrix, storage probe, and
external location for the future raw-evidence manifest.

### G3 — Validate all cards without training

Load every reproduction YAML, resolve capabilities, bind the declared
input/model contracts where possible, check zero-based seed indices, and verify
that output identities are unique. This gate remains static and short.

**Exit evidence:** complete card-validation report with every rejected or
unverified composition listed explicitly.

### G4 — Run bounded CPU and Calder canaries

Run one bounded seed for each classic execution shape. For Calder, run a cold
canary that recomputes preprocessing and graph artefacts, then a warm canary
that reuses only authenticated native caches. Verify that both paths produce
the same scientific identities and that no recovered legacy artefact was read.

**Exit evidence:** per-seed reports, cold/warm cache manifests, timing and peak
resource observations, and negative-first canary review. These are diagnostics,
not paper repetitions.

### G5 — Run bounded GPU and continuation canaries

Run one GRAND canary on the selected V100-compatible resource class and one
representative Match canary on the selected H100-compatible class. Exercise
checkpoint save, planned interruption, resume, final report publication, and
identity rejection after a scientific change before any long Match job is
eligible.

**Exit evidence:** accelerator compatibility report, checkpoint/resume proof,
measured time and memory, and complete canary reports. If native Match
continuation is not proven end to end, Match remains on hold independently of
the other methods.

### G6 — Calibrate resources and authorize bounded production lots

Use G4/G5 measurements to set conservative wall time, memory, CPU, GPU, array
throttle, retry limit, storage, and normalized allocation budgets. Include the
cost of cold preprocessing, retries, and reconciliation. Review current site
quotas and scheduler policy from
the authenticated Jean Zay session.

**Exit evidence:** bounded resource envelope, quota check, launch matrix, and
an explicit recorded gate decision. The pure native publication builder and verifier
must also be implemented and exercised on retained G4/G5 evidence before
production promotion. A failed or incomplete prerequisite keeps only its
dependent production lots on `HOLD`.

### G7 — Execute CPU production lots

Launch small classic CPU lots first, reconcile them, then launch Calder with
throttled arrays and authenticated shared caches.

**Exit evidence:** immutable raw attempts, scheduler accounting, complete seed
accounting, and native aggregate/acceptance reports for each closed lot.

### G8 — Execute GPU production lots

Launch GRAND separately from Match so that V100 and H100 resource envelopes,
failure domains, and accounting remain visible. Match begins only after its
continuation gate passes. Retries retain the failed attempt and must reproduce
the expected execution identity.

**Exit evidence:** 100/100 declared GRAND seeds and 14/14 declared Match seeds,
or an explicit list of every failed/missing seed and a non-certifiable status.

### G9 — Reconcile, seal, review, and publish

Close each lot before cross-card reporting. Verify hashes, reconcile the exact
declared seeds, run the native acceptance evaluator, seal and transfer the raw
archive, and independently verify it. Generate the compact public bundle only
from the transferred evidence and review it under the
[publication policy](../publication-policy.md).

**Exit evidence:** immutable raw archive plus digest, compact paper bundle plus
`SHA256SUMS`, negative-first scientific summary, and an explicit publication
review. A partial campaign may be published only as partial or
`not_evaluable`, never as a completed replication.

## Lot isolation and retries

CPU classic, Calder, GRAND, and Match use separate lot
identifiers, output roots, logs, resource limits, and reconciliations. Within a
lot, every seed has a deterministic execution identity. A retry writes a new
attempt directory and records why the preceding attempt failed; it never
overwrites evidence.

A dependency failure pauses only dependent lots. For example, a Match
continuation failure must not force already validated CPU work onto H100, and a
Calder cache failure must not change its preprocessing contract to make the job
fit. Any scientific change creates a new protocol identity and returns the
affected lot to G0.

## Completion definition

The campaign is complete only when all 1,170 primary seeds are accounted for,
every aggregate and acceptance report is authenticated, failures are stated,
the raw archive has been verified after transfer, and the compact paper bundle
has passed publication review. Scheduler completion or a set of plausible
means is not sufficient.
