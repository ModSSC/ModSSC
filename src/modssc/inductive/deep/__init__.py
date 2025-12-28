"""Torch-only deep model bundle for inductive SSL methods."""

from .bundles import build_torch_bundle_from_classifier
from .types import TorchModelBundle
from .validation import validate_torch_model_bundle

__all__ = [
    "TorchModelBundle",
    "build_torch_bundle_from_classifier",
    "validate_torch_model_bundle",
]
