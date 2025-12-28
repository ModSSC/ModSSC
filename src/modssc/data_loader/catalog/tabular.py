from __future__ import annotations

from modssc.data_loader.types import DatasetSpec

# Notes on reproducibility:
# Prefer OpenML data_id over name to avoid ambiguity when multiple "active" versions exist.

TABULAR_CATALOG: dict[str, DatasetSpec] = {
    "iris": DatasetSpec(
        key="iris",
        provider="openml",
        uri="openml:61",
        modality="tabular",
        task="classification",
        description="Iris (OpenML data_id=61). No official split.",
        required_extra="openml",
        source_kwargs={"data_id": 61},
    ),
    "adult": DatasetSpec(
        key="adult",
        provider="openml",
        uri="openml:1590",
        modality="tabular",
        task="classification",
        description="Adult (OpenML data_id=1590). No official split.",
        required_extra="openml",
        source_kwargs={"data_id": 1590},
    ),
    "breast_cancer": DatasetSpec(
        key="breast_cancer",
        provider="openml",
        uri="openml:15",
        modality="tabular",
        task="classification",
        description="Breast Cancer Wisconsin (OpenML data_id=15). Binary, numeric features.",
        required_extra="openml",
        source_kwargs={"data_id": 15},
    ),
}
