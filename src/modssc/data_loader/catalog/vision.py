from __future__ import annotations

from modssc.data_loader.types import DatasetSpec

VISION_CATALOG: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        key="mnist",
        provider="torchvision",
        uri="torchvision:MNIST",
        modality="vision",
        task="classification",
        description="MNIST (torchvision).",
        required_extra="vision",
        source_kwargs={},
    ),
    "cifar10": DatasetSpec(
        key="cifar10",
        provider="torchvision",
        uri="torchvision:CIFAR10",
        modality="vision",
        task="classification",
        description="CIFAR-10 (torchvision).",
        required_extra="vision",
        source_kwargs={},
    ),
    "cifar100": DatasetSpec(
        key="cifar100",
        provider="torchvision",
        uri="torchvision:CIFAR100",
        modality="vision",
        task="classification",
        description="CIFAR-100 (torchvision).",
        required_extra="vision",
        source_kwargs={},
    ),
    "svhn": DatasetSpec(
        key="svhn",
        provider="torchvision",
        uri="torchvision:SVHN",
        modality="vision",
        task="classification",
        description="SVHN (torchvision). Uses train/test splits; extra split not loaded by default.",
        required_extra="vision",
        source_kwargs={},
    ),
    "stl10": DatasetSpec(
        key="stl10",
        provider="torchvision",
        uri="torchvision:STL10",
        modality="vision",
        task="classification",
        description="STL-10 (torchvision). Uses train/test splits; unlabeled split not loaded by default.",
        required_extra="vision",
        source_kwargs={},
    ),
}
