"""Adapters for inductive datasets."""

from .numpy import NumpyDataset, to_numpy_dataset
from .torch import TorchDataset, to_torch_dataset

__all__ = [
    "NumpyDataset",
    "TorchDataset",
    "to_numpy_dataset",
    "to_torch_dataset",
]
