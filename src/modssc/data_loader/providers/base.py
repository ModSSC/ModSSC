from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modssc.data_loader.types import DatasetIdentity, LoadedDataset
from modssc.data_loader.uri import ParsedURI


class BaseProvider(ABC):
    """Provider interface.

    Providers must:
    - resolve a ParsedURI + options into a DatasetIdentity
    - load a canonical dataset into memory (official splits only if provided)
    - use raw_dir as their download/cache location whenever the upstream API supports it
    """

    name: str
    required_extra: str | None

    @abstractmethod
    def resolve(self, parsed: ParsedURI, *, options: Mapping[str, Any]) -> DatasetIdentity:
        raise NotImplementedError

    @abstractmethod
    def load_canonical(self, identity: DatasetIdentity, *, raw_dir: Path) -> LoadedDataset:
        raise NotImplementedError

    def list(self, *, query: str | None = None) -> list[str] | None:
        """Optional dynamic listing. Providers may return None if unsupported."""
        return None
