from __future__ import annotations

from modssc.data_loader.types import DatasetSpec

TOY_CATALOG: dict[str, DatasetSpec] = {
    "toy": DatasetSpec(
        key="toy",
        provider="toy",
        uri="toy:default",
        modality="tabular",
        task="classification",
        description="Deterministic synthetic dataset used for tests and examples.",
        required_extra=None,
        source_kwargs={
            "seed": 0,
            "n_samples": 64,
            "n_features": 4,
            "n_classes": 3,
            "official_test": True,
        },
    ),
}
