"""Inductive semi-supervised learning.

This package exposes inductive SSL datasets, registries, validation helpers,
and method integrations used by the inductive brick.
"""

from .adapters import NumpyDataset, TorchDataset, to_numpy_dataset, to_torch_dataset
from .base import InductiveMethod, MethodInfo
from .deep import TorchModelBundle, validate_torch_model_bundle
from .errors import InductiveNotImplementedError, InductiveValidationError, OptionalDependencyError
from .registry import available_methods, get_method_class, get_method_info, register_method
from .seed import make_numpy_rng, seed_everything
from .types import DeviceSpec, InductiveDataset
from .validation import validate_inductive_dataset

__all__ = [
    "DeviceSpec",
    "InductiveDataset",
    "InductiveMethod",
    "MethodInfo",
    "NumpyDataset",
    "OptionalDependencyError",
    "TorchDataset",
    "TorchModelBundle",
    "InductiveNotImplementedError",
    "InductiveValidationError",
    "available_methods",
    "get_method_class",
    "get_method_info",
    "register_method",
    "seed_everything",
    "make_numpy_rng",
    "to_numpy_dataset",
    "to_torch_dataset",
    "validate_torch_model_bundle",
    "validate_inductive_dataset",
]
