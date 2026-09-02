"""Transductive semi supervised learning.

This package provides the math and integration layer:
- backend abstraction (numpy, torch)
- graph operators (normalization, laplacian, spmm)
- generic solvers (fixed point, conjugate gradient)
- PyG adapter (optional)
- strict input validation
- classic label diffusion methods
- PDE-inspired methods
- GNN-based transductive methods
"""

from modssc.capabilities import (
    IncompatiblePipelineError,
    MethodCapabilities,
    PipelineCapabilities,
    check_pipeline_compatibility,
    validate_pipeline_compatibility,
)

from .data import (
    NodeEvaluationData,
    PreparedNodeData,
    build_node_dataset,
    graph_from_dataset,
    masks_from_indices,
    masks_from_sampling,
    prepare_node_data,
)
from .errors import OptionalDependencyError, TransductiveDataError, TransductiveValidationError
from .execution import (
    TransductiveExecutionConfig,
    TransductiveExecutionError,
    TransductiveExecutionInput,
    TransductiveExecutionResult,
    execute_transductive_method,
)
from .registry import available_methods, get_method_class, get_method_info, register_method
from .types import DeviceSpec
from .validation import validate_node_dataset

__all__ = [
    "DeviceSpec",
    "IncompatiblePipelineError",
    "MethodCapabilities",
    "NodeEvaluationData",
    "OptionalDependencyError",
    "PipelineCapabilities",
    "PreparedNodeData",
    "TransductiveValidationError",
    "TransductiveDataError",
    "TransductiveExecutionConfig",
    "TransductiveExecutionError",
    "TransductiveExecutionInput",
    "TransductiveExecutionResult",
    "build_node_dataset",
    "graph_from_dataset",
    "masks_from_indices",
    "masks_from_sampling",
    "prepare_node_data",
    "validate_node_dataset",
    "available_methods",
    "check_pipeline_compatibility",
    "execute_transductive_method",
    "get_method_class",
    "get_method_info",
    "register_method",
    "validate_pipeline_compatibility",
]
