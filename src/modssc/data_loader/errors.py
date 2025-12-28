from __future__ import annotations

from dataclasses import dataclass


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
class OptionalDependencyError(DataLoaderError):
    """Raised when an optional dependency (extra) required by a provider is missing."""

    extra: str
    purpose: str | None = None

    def __str__(self) -> str:
        msg = f"Missing optional dependency extra: {self.extra!r}."
        if self.purpose:
            msg += f" Required for: {self.purpose}."
        msg += f' Install with: pip install "modssc[{self.extra}]"'
        return msg


class DatasetNotCachedError(DataLoaderError):
    def __init__(self, dataset_id: str) -> None:
        super().__init__(
            f"Dataset {dataset_id!r} is not available in the processed cache. "
            "Set download=True or run: modssc datasets download --dataset <id>"
        )


class ManifestError(DataLoaderError):
    pass
