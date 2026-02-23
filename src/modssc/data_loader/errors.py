from __future__ import annotations

from dataclasses import dataclass

from modssc.dependency_errors import MissingExtraErrorMessageMixin


class DataLoaderError(RuntimeError):
    """Base error for modssc.data_loader."""


class UnknownDatasetError(DataLoaderError):
    def __init__(self, key: str) -> None:
        super().__init__(f"Unknown dataset key: {key!r}")


class InvalidDatasetURIError(DataLoaderError):
    def __init__(self, uri: str) -> None:
        super().__init__(f"Invalid dataset URI: {uri!r}. Expected format '<provider>:<reference>'.")


class ProviderNotFoundError(DataLoaderError):
    def __init__(self, provider: str) -> None:
        super().__init__(f"Unknown provider: {provider!r}")


@dataclass(frozen=True)
class OptionalDependencyError(MissingExtraErrorMessageMixin, DataLoaderError):
    """Raised when an optional dependency (extra) required by a provider is missing."""

    extra: str
    purpose: str | None = None


class DatasetNotCachedError(DataLoaderError):
    def __init__(self, dataset_id: str) -> None:
        super().__init__(
            f"Dataset {dataset_id!r} is not available in the processed cache. "
            "Set download=True or run: modssc datasets download --dataset <id>"
        )


class ManifestError(DataLoaderError):
    pass
