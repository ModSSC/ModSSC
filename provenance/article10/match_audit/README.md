# Match reference provenance

This directory records the exact upstream revisions used to audit the four
paper Match profiles. It contains no upstream implementation source. ModSSC's
own Match implementation is the only code executed by the replications.

The pinned repositories are:

- Google Research FixMatch,
  `d4985a158065947dba803e626ee9a6721709c570`;
- TorchSSL,
  `03193a1b7883727db1ce9c092e083091e18aedbb`;
- Microsoft USB (`Semi-supervised-learning`),
  `1ef4cbebcc0b368158315aeb425053858cf6c845`.

For every repository, `MANIFEST.json` records the upstream path and SHA-256 of
each file consulted during the completed audit. These records authenticate the
review without making the upstream repositories runtime dependencies. In particular,
Google `libml/data.py` freezes the `repeat().shuffle(8192)` pipeline and
TorchSSL `datasets/data_utils.py` freezes
`RandomSampler(replacement=True, num_samples=batch_size * num_iters)`. The
hashes preserve the exact reviewed revisions. No upstream file is imported,
executed, or required by the benchmark runner.

`PIXEL_FIXTURES.json` records the deterministic input, operation parameters,
output encoding, source revision, and expected digest for every pixel oracle used by
`tests/data_augmentation/test_cifar_reference.py`.

The three license files are byte-for-byte copies from the repository roots at
the pinned commits:

- `LICENSE.google-fixmatch.apache-2.0.txt`;
- `LICENSE.torchssl.mit.txt`;
- `LICENSE.usb.mit.txt`.

Pixel fixtures authenticate the tested outputs, not an exact upstream RNG
trajectory. ModSSC matches TorchSSL's replacement-index primitive exactly with
a serialized CPU `torch.Generator`, but allocates explicit independent loader
seeds instead of consuming an implicit process-global seed. Its Google path
matches the TensorFlow shuffle-buffer transition and repeat order, while a
local serialized slot generator replaces the unpublished, non-portable TF1
graph/parallel-input bitstream. Augmentations likewise preserve the pinned
primitive distributions with authenticated independent replayable streams;
they do not claim the historical multiprocessing draw order. Pillow-level
results also remain covered by the immutable execution environment manifest
used for a scientific run.
