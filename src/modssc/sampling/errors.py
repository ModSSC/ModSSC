from __future__ import annotations


class SamplingError(RuntimeError):
    """Base error for sampling."""


class MissingDatasetFingerprintError(SamplingError):
    def __init__(self) -> None:
        super().__init__(
            "dataset_fingerprint is required to compute a split fingerprint and to cache splits. "
            "Pass dataset_fingerprint explicitly or provide it in dataset.meta['dataset_fingerprint']."
        )


class SamplingValidationError(SamplingError):
    """Raised when a sampled split violates invariants."""
