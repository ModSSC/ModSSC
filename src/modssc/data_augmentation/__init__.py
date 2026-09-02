"""ModSSC data augmentation brick.

This brick provides **training-time** (stochastic) transformations for multiple modalities
(vision, text, tabular, audio, graph). It is designed to be:

- **Deterministic** when requested (seed + epoch + sample_id => same output)
- **Backend-aware** (NumPy by default; supports torch tensors without requiring torch at import)
- **Composable** through a small plan/pipeline system
- **Extensible** via a registry (contributors can add new operations without touching core code)

Notes
-----
This is intentionally separate from :mod:`modssc.preprocess`, which is meant for offline and/or
cacheable feature engineering (including embeddings with pretrained models). Augmentations are
applied on-the-fly during training loops (future brick/orchestrator).
"""

from .api import (
    AugmentationPipeline,
    AugmentationStrategy,
    available_ops,
    build_pipeline,
    get_op,
    make_context_rng,
)
from .cifar_reference import (
    CIFAR_REFERENCE_AUGMENTER_ID,
    CIFAR_REFERENCE_CONTRACT_SCHEMA_VERSION,
    CifarAugmentationDraws,
    CifarReferenceAugmentation,
    cifar_reference_runtime_identity,
    resolve_cifar_augmentation_profile,
)
from .plan import AugmentationPlan, StepConfig, parse_augmentation_plan
from .registry import available_online_augmenters, get_online_augmenter, register_op
from .runtime import (
    OnlineAugmentation,
    UnlabeledAugmentationResult,
    build_online_augmentation,
    materialize_views,
    prepare_unlabeled_augmentation,
    validate_augmentation_regime,
)
from .types import AugmentationContext, GraphSample, Modality

__all__ = [
    # Types / plan
    "AugmentationContext",
    "AugmentationPlan",
    "StepConfig",
    "parse_augmentation_plan",
    "AugmentationStrategy",
    "AugmentationPipeline",
    "GraphSample",
    "Modality",
    # Registry / API
    "available_ops",
    "available_online_augmenters",
    "build_pipeline",
    "get_op",
    "get_online_augmenter",
    "register_op",
    "make_context_rng",
    "OnlineAugmentation",
    "UnlabeledAugmentationResult",
    "build_online_augmentation",
    "materialize_views",
    "prepare_unlabeled_augmentation",
    "validate_augmentation_regime",
    "CIFAR_REFERENCE_AUGMENTER_ID",
    "CIFAR_REFERENCE_CONTRACT_SCHEMA_VERSION",
    "CifarAugmentationDraws",
    "CifarReferenceAugmentation",
    "cifar_reference_runtime_identity",
    "resolve_cifar_augmentation_profile",
]
