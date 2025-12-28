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
from .plan import AugmentationPlan, StepConfig
from .registry import register_op
from .types import AugmentationContext, GraphSample, Modality

__all__ = [
    # Types / plan
    "AugmentationContext",
    "AugmentationPlan",
    "StepConfig",
    "AugmentationStrategy",
    "AugmentationPipeline",
    "GraphSample",
    "Modality",
    # Registry / API
    "available_ops",
    "build_pipeline",
    "get_op",
    "register_op",
    "make_context_rng",
]
