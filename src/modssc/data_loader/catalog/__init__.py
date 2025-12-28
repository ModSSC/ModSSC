from __future__ import annotations

from modssc.data_loader.catalog.audio import AUDIO_CATALOG
from modssc.data_loader.catalog.graph import GRAPH_CATALOG
from modssc.data_loader.catalog.tabular import TABULAR_CATALOG
from modssc.data_loader.catalog.text import TEXT_CATALOG
from modssc.data_loader.catalog.toy import TOY_CATALOG
from modssc.data_loader.catalog.vision import VISION_CATALOG
from modssc.data_loader.types import DatasetSpec


def _merge(*parts: dict[str, DatasetSpec]) -> dict[str, DatasetSpec]:
    merged: dict[str, DatasetSpec] = {}
    for part in parts:
        overlap = set(merged).intersection(part)
        if overlap:
            raise ValueError(f"Duplicate dataset keys in catalog: {sorted(overlap)}")
        merged.update(part)
    return merged


DATASET_CATALOG: dict[str, DatasetSpec] = _merge(
    TOY_CATALOG,
    TABULAR_CATALOG,
    TEXT_CATALOG,
    VISION_CATALOG,
    AUDIO_CATALOG,
    GRAPH_CATALOG,
)
