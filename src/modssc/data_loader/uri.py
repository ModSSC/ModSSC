from __future__ import annotations

from dataclasses import dataclass

from modssc.data_loader.errors import InvalidDatasetURIError


@dataclass(frozen=True)
class ParsedURI:
    provider: str
    reference: str

    @property
    def uri(self) -> str:
        return f"{self.provider}:{self.reference}"


def is_uri(dataset_id: str) -> bool:
    if ":" not in dataset_id:
        return False
    provider, ref = dataset_id.split(":", 1)
    return bool(provider.strip()) and bool(ref.strip())


def parse_uri(uri: str) -> ParsedURI:
    if ":" not in uri:
        raise InvalidDatasetURIError(uri)
    provider, ref = uri.split(":", 1)
    provider = provider.strip()
    ref = ref.strip()
    if not provider or not ref:
        raise InvalidDatasetURIError(uri)
    return ParsedURI(provider=provider, reference=ref)
