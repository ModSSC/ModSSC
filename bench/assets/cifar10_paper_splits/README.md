# CIFAR-10 paper partition artifacts

These immutable archives reconstruct the ordered pools produced by the pinned
official split generators:

- Google FixMatch commit `d4985a158065947dba803e626ee9a6721709c570`,
  `scripts/create_split.py` and `scripts/create_unlabeled.py`, seeds 1–5,
  250 labels, and the original one-example validation prefix;
- TorchSSL commit `03193a1b7883727db1ce9c092e083091e18aedbb`,
  `datasets/data_utils.py`, seeds 0–2, 40 or 250 labels, with
  `include_lb_to_ulb=True`; Microsoft USB commit
  `1ef4cbebcc0b368158315aeb425053858cf6c845` is recorded as a secondary
  implementation control.

The upstream repositories do not publish the generated CIFAR index files.
These are deterministic reconstructions from the canonical CIFAR-10 Python
label order, not upstream-distributed split files. Every reproduction card pins
the complete artifact SHA-256. Each archive also embeds the canonical label
order hash, source archive hash, upstream commit and ordered arrays for train,
validation, test, labeled and unlabeled pools.

The checked-in archives are sufficient for every supported run. The public
preparation path downloads or materializes CIFAR-10 through the ModSSC provider
and authenticates these arrays against that dataset:

```text
python -m bench.reproduce prepare fixmatch/cifar10-250
```

The reconstruction utility under `bench.campaign.protocols.match` is retained
only for provenance audits; it is not a replication prerequisite and must not
replace provider-based dataset preparation.

The writer fixes ZIP member timestamps and ordering, so identical NumPy arrays
produce identical bytes. The generated numeric indices are distributed under
the ModSSC license; ModSSC's independently implemented reconstruction rules are
documented with the upstream notices in `LICENSES.md`.
