"""Transductive semi supervised learning.

This package provides the math and integration layer:
- backend abstraction (numpy, torch)
- graph operators (normalization, laplacian, spmm)
- generic solvers (fixed point, conjugate gradient)
- PyG adapter (optional)
- strict input validation

Algorithms (Label Propagation, Poisson Learning, GNNs, etc.) are added in later waves.
"""

from .errors import OptionalDependencyError, TransductiveValidationError
from .registry import available_methods, get_method_class, get_method_info, register_method
from .types import DeviceSpec
from .validation import validate_node_dataset

__all__ = [
    "DeviceSpec",
    "OptionalDependencyError",
    "TransductiveValidationError",
    "validate_node_dataset",
    "available_methods",
    "get_method_class",
    "get_method_info",
    "register_method",
]
