# Match paper wave 1: scientific audit

This note records the conformity decision for the four paper-only Match
profiles. It does not authorize the standardized benchmark.

## Pinned provenance

- Google Research FixMatch at
  `d4985a158065947dba803e626ee9a6721709c570`;
- TorchSSL at `03193a1b7883727db1ce9c092e083091e18aedbb`;
- Microsoft USB at `1ef4cbebcc0b368158315aeb425053858cf6c845`.

The consulted commits, upstream paths, SHA-256 digests, and licences are
recorded under `provenance/article10/`. No Google FixMatch, TorchSSL, or USB
source file is distributed or executed by the replication. ModSSC's own Match
implementation is validated against compact numerical and pixel-level fixtures
whose provenance is recorded by those manifests.

## Conformity decision

The shared trainer reproduces the registered batch contract (64 labeled and
448 unlabeled samples), SGD/Nesterov schedule, EMA, Wide ResNet topology,
BatchNorm interleave for Google FixMatch, online weak/strong augmentation,
checkpoint state, and exactly \(2^{20}\) optimizer steps. Multi-iteration
oracles cover FixMatch masking, FlexMatch CPL, FreeMatch SAT/SAF, and SoftMatch
Gaussian weighting and alignment. An interrupted run must reproduce the same
sampler, RNG, model, optimizer, scheduler, EMA, adaptive state, and evaluation
history as an uninterrupted run. FlexMatch, FreeMatch, and SoftMatch evaluate
after TorchSSL iteration zero and then every 5,000 iterations for the complete
run, as preregistered by their campaign cards; the late 1,000-iteration
cadence present in the historical training loop is not enabled.

All four methods are classified as `pinned_official_implementation` parity.
For FreeMatch, the training stack and equations are pinned to TorchSSL while
the primary entropy coefficient remains the paper's explicit `lambda_e=0.05`
rather than a value from a later configuration card. The SoftMatch paper
explicitly states that its Section 4.1 results were produced with TorchSSL, so
the 4.82-percent profile follows that pinned trajectory: Gaussian statistics
on raw weak predictions, labeled-prediction EMA alignment, aligned confidence
weights and raw hard pseudo-labels.

All registered repetitions, numerical margins, confidence intervals,
diagnostics, and scientific gates subsequently passed. FixMatch, FlexMatch,
FreeMatch, and SoftMatch therefore have the final status `paper_matched`; the
signed results remain recorded in the article-replication summary.

## Registered limitations

- Official generated CIFAR-10 index files were not published. The ordered
  splits are authenticated deterministic outputs of the pinned official
  generators.
- TorchSSL index sampling uses its pinned CPU
  `RandomSampler(replacement=True)`/`torch.randint` primitive exactly, with a
  serialized `torch.Generator`. ModSSC allocates explicit independent loader
  seeds rather than consuming TorchSSL's implicit process-global seeds, whose
  values depend on unrelated model and worker RNG consumption.
- Google FixMatch uses the pinned TensorFlow
  `repeat().shuffle(8192)` fill/replace transition and repeat order. The TF1
  graph and parallel-input RNG bitstream was not published and is not portable,
  so the uniform slot generator is local, authenticated and serialized.
- Augmentation primitives and distributions match the pinned sources, but
  authenticated stateless weak/strong streams replace the historical
  multiprocessing draw order. These equivalences make interrupted runs exactly
  replayable without claiming bit-for-bit historical random draws.
- TorchSSL changes its evaluation interval from 5,000 to 1,000 iterations
  after 80 percent of training. The preregistered Match cards instead freeze
  the interval at 5,000 throughout, so historical checkpoint selection follows
  the declared campaign protocol rather than that late source-code change.
- TorchSSL leaves duplicate-index CPL writes undefined. ModSSC uses a
  deterministic sequential equivalent in which the last accepted occurrence
  in batch order wins, allowing exact checkpoint continuation.
- The FreeMatch paper gives `lambda_e=0.05` for CIFAR-10 with 40 labels, whereas
  the later TorchSSL and USB cards use different values. The paper value is the
  preregistered primary; alternatives are test-blind diagnostics only.
- SoftMatch Algorithm 1 describes uniform alignment, while the TorchSSL code
  identified by the paper uses labeled-prediction alignment and USB later
  applies uniform alignment before updating Gaussian statistics. The
  preregistered primary follows the TorchSSL path that produced the target;
  the other paths are documented secondary controls, not selectable variants.

## Historical execution evidence

Hardware bridges and forced-continuation jobs were used while auditing this
replication. They are historical evidence, not part of the public reproduction
API and not prerequisites for running a packaged card. The first 4,096-step
bridge found architecture-dependent divergence despite identical initial
states. A second bridge pinned matrix-multiplication and convolution precision
to IEEE and confirmed exact same-architecture continuation, but it still did
not meet the preregistered cross-architecture tolerance.

| Bridge | A100 campaign | H100 campaign | Resume campaign | Terminal / historical / acceptance deltas | Status |
|---|---|---|---|---|---|
| v1 | `article10-match-fix-bridge-a100-v1` | `article10-match-fix-bridge-h100-v1` | `article10-match-fix-resume-v1` | `3.64 / 2.33 / 0.01091` | `failed`, informative only |
| v2 | `article10-match-fix-bridge-a100-v2` | `article10-match-fix-bridge-h100-v2` | `article10-match-fix-resume-v2` | `1.31 / 0.94 / 0.00152` | `failed`, informative only |

Both rows retain the original absolute tolerances `0.5 / 0.5 / 0.02`.

Production was subsequently kept on one architecture, and exact checkpoint
continuation was verified over model, optimizer, scheduler, EMA, samplers,
adaptive method state, evaluation history, and all Python/NumPy/Torch RNG
states. This operational choice did not reinterpret either failed bridge as a
pass and did not alter a scientific threshold.

The immutable aggregate and its acceptance status are retained in
`provenance/article10/evidence/article10-replication-summary.json`. Public
`bench` cards now express only the scientific protocol and portable checkpoint
contract. They neither consult those historical hardware policies nor require
the private execution history; repository-side scheduler adapters may consume
the same scheduler-neutral cards independently.
