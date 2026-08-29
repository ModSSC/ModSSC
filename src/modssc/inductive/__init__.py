"""Inductive semi-supervised learning.

This package exposes inductive SSL datasets, registries, validation helpers,
and method integrations used by the inductive brick.
"""

from modssc.capabilities import (
    IncompatiblePipelineError,
    MethodCapabilities,
    PipelineCapabilities,
    check_pipeline_compatibility,
    validate_pipeline_compatibility,
)
from modssc.runtime.method_spec import build_method_spec

from .adapters import NumpyDataset, TorchDataset, to_numpy_dataset, to_torch_dataset
from .base import InductiveMethod, MethodInfo
from .deep import TorchModelBundle, validate_torch_model_bundle
from .errors import InductiveNotImplementedError, InductiveValidationError, OptionalDependencyError
from .execution import (
    InductiveExecutionConfig,
    InductiveExecutionError,
    InductiveExecutionInput,
    InductiveExecutionResult,
    execute_inductive_method,
    prepare_inductive_dataset,
    requires_torch_inputs,
)
from .model_binding import (
    ModelBindingError,
    ModelBindingSpec,
    ModelBuildConfig,
    bind_model_to_spec,
)
from .registry import available_methods, get_method_class, get_method_info, register_method
from .seed import make_numpy_rng, seed_everything
from .types import DeviceSpec, InductiveDataset
from .validation import validate_inductive_dataset

__all__ = [
    "DeviceSpec",
    "InductiveDataset",
    "InductiveExecutionConfig",
    "InductiveExecutionError",
    "InductiveExecutionInput",
    "InductiveExecutionResult",
    "InductiveMethod",
    "IncompatiblePipelineError",
    "MethodInfo",
    "ModelBindingError",
    "ModelBindingSpec",
    "ModelBuildConfig",
    "MethodCapabilities",
    "NumpyDataset",
    "OptionalDependencyError",
    "PipelineCapabilities",
    "TorchDataset",
    "TorchModelBundle",
    "InductiveNotImplementedError",
    "InductiveValidationError",
    "available_methods",
    "bind_model_to_spec",
    "build_method_spec",
    "check_pipeline_compatibility",
    "execute_inductive_method",
    "get_method_class",
    "get_method_info",
    "prepare_inductive_dataset",
    "register_method",
    "requires_torch_inputs",
    "seed_everything",
    "make_numpy_rng",
    "to_numpy_dataset",
    "to_torch_dataset",
    "validate_torch_model_bundle",
    "validate_inductive_dataset",
    "validate_pipeline_compatibility",
]
