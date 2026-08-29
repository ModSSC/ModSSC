"""Backend-preserving row selection for canonical ModSSC data containers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .errors import DatasetSelectionError

_EDGE_ALIGNED_KEYS = frozenset(
    {
        "edge_attr",
        "edge_attrs",
        "edge_feature",
        "edge_features",
        "edge_label",
        "edge_labels",
        "edge_time",
        "edge_type",
        "edge_weight",
    }
)


def _is_torch_tensor(value: Any) -> bool:
    try:
        import importlib

        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return False
    return isinstance(value, torch.Tensor)


def _as_indices(indices: np.ndarray | list[int]) -> np.ndarray:
    normalized = np.asarray(indices, dtype=np.int64).reshape(-1)
    if normalized.size and int(normalized.min()) < 0:
        raise DatasetSelectionError(
            "indices must be >= 0",
            code="E_DATA_SELECTION_NEGATIVE_INDEX",
        )
    return normalized


def _leading_size(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if len(shape) > 0:
                return int(shape[0])
        except (TypeError, ValueError):
            return None
    if isinstance(value, list):
        return len(value)
    return None


def _infer_population_size(container: Mapping[str, Any]) -> int | None:
    features = container.get("x")
    size = _leading_size(features)
    if size is not None:
        return size
    value = container.get("num_nodes")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_indices(indices: np.ndarray, *, size: int, context: str) -> None:
    if indices.size and int(indices.max()) >= int(size):
        raise DatasetSelectionError(
            f"{context}: index {int(indices.max())} is outside a population of size {int(size)}",
            code="E_DATA_SELECTION_INDEX_BOUNDS",
        )


def _normalize_edge_index(edge_index: Any) -> tuple[Any, int]:
    shape = getattr(edge_index, "shape", None)
    if shape is None:
        edge_index = np.asarray(edge_index)
        shape = edge_index.shape
    try:
        valid = len(shape) == 2 and int(shape[0]) == 2
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise DatasetSelectionError(
            "edge_index must have shape (2, E)",
            code="E_DATA_SELECTION_GRAPH_SHAPE",
        )
    return edge_index, int(shape[1])


def _induce_edges(
    edge_index: Any,
    indices: np.ndarray,
    *,
    num_nodes: int | None,
) -> tuple[Any, Any, int]:
    """Induce and relabel a graph without requiring PyTorch Geometric.

    Returns the selected edge index, the backend-native edge mask, and the
    original edge count so edge-aligned fields can be selected consistently.
    """

    normalized_edges, n_edges = _normalize_edge_index(edge_index)
    if np.unique(indices).size != indices.size:
        raise DatasetSelectionError(
            "graph node selection does not permit duplicate indices",
            code="E_DATA_SELECTION_GRAPH_DUPLICATE_NODE",
        )

    def resolve_size(*, inferred: int, minimum: int) -> int:
        if minimum < 0:
            raise DatasetSelectionError(
                "edge_index contains negative node ids",
                code="E_DATA_SELECTION_GRAPH_BOUNDS",
            )
        if num_nodes is not None and (int(num_nodes) < 0 or inferred > int(num_nodes)):
            raise DatasetSelectionError(
                "edge_index contains node ids outside [0, num_nodes)",
                code="E_DATA_SELECTION_GRAPH_BOUNDS",
            )
        size = int(num_nodes) if num_nodes is not None else inferred
        if indices.size and size == 0:
            raise DatasetSelectionError(
                "graph population size is unknown for a non-empty selection",
                code="E_DATA_SELECTION_GRAPH_BOUNDS",
            )
        return size

    if _is_torch_tensor(normalized_edges):
        import importlib

        torch = importlib.import_module("torch")
        edges = normalized_edges
        if edges.dtype == torch.bool or edges.dtype.is_floating_point:
            raise DatasetSelectionError(
                "edge_index must use an integer dtype",
                code="E_DATA_SELECTION_GRAPH_DTYPE",
            )
        inferred = int(edges.max().item()) + 1 if edges.numel() else 0
        minimum = int(edges.min().item()) if edges.numel() else 0
        size = resolve_size(inferred=inferred, minimum=minimum)
        _validate_indices(indices, size=size, context="graph node selection")
        mapping = torch.full((size,), -1, dtype=torch.long, device=edges.device)
        subset = torch.as_tensor(indices, dtype=torch.long, device=edges.device)
        mapping[subset] = torch.arange(subset.numel(), dtype=torch.long, device=edges.device)
        edge_mask = (mapping[edges[0].long()] >= 0) & (mapping[edges[1].long()] >= 0)
        selected = torch.stack(
            (mapping[edges[0].long()[edge_mask]], mapping[edges[1].long()[edge_mask]]),
            dim=0,
        ).to(dtype=edges.dtype)
        return selected, edge_mask, n_edges

    edges = np.asarray(normalized_edges)
    if not np.issubdtype(edges.dtype, np.integer):
        raise DatasetSelectionError(
            "edge_index must use an integer dtype",
            code="E_DATA_SELECTION_GRAPH_DTYPE",
        )
    inferred = int(edges.max()) + 1 if edges.size else 0
    minimum = int(edges.min()) if edges.size else 0
    size = resolve_size(inferred=inferred, minimum=minimum)
    _validate_indices(indices, size=size, context="graph node selection")
    mapping = np.full((size,), -1, dtype=np.int64)
    mapping[indices] = np.arange(indices.size, dtype=np.int64)
    edge_mask = (mapping[edges[0]] >= 0) & (mapping[edges[1]] >= 0)
    selected = np.stack((mapping[edges[0, edge_mask]], mapping[edges[1, edge_mask]]), axis=0)
    return selected.astype(edges.dtype, copy=False), edge_mask, n_edges


def _slice_tensor(value: Any, indices: Any) -> Any:
    import importlib

    torch = importlib.import_module("torch")
    torch_indices = torch.as_tensor(indices, device=value.device)
    if torch_indices.dtype != torch.bool:
        torch_indices = torch_indices.to(dtype=torch.long)
    return value[torch_indices]


def _slice_value(value: Any, indices: Any) -> Any:
    if _is_torch_tensor(value):
        return _slice_tensor(value, indices)
    if isinstance(value, np.ndarray):
        if _is_torch_tensor(indices):
            indices = indices.detach().cpu().numpy()
        return value[np.asarray(indices)]
    if isinstance(value, list):
        if _is_torch_tensor(indices):
            indices = indices.detach().cpu().numpy()
        normalized = np.asarray(indices)
        if normalized.dtype == bool:
            normalized = np.flatnonzero(normalized)
        return [value[int(index)] for index in normalized.tolist()]
    return value[indices]


def _is_edge_aligned(key: str, value: Any, *, n_edges: int) -> bool:
    if key not in _EDGE_ALIGNED_KEYS and not (
        key.startswith("edge_") and not key.endswith("_index")
    ):
        return False
    size = _leading_size(value)
    if size != n_edges:
        raise DatasetSelectionError(
            f"{key} must have leading dimension E={n_edges}, got {size}",
            code="E_DATA_SELECTION_EDGE_ALIGNMENT",
        )
    return True


def select_rows(
    values: Any,
    indices: np.ndarray | list[int],
    *,
    context: str = "data",
) -> Any:
    """Select rows without changing the NumPy/Torch/container backend.

    For mappings, only fields whose leading dimension is exactly the canonical
    population size are row-aligned. Graph mappings additionally induce and
    relabel edges, and apply the same edge mask to edge-aligned attributes.
    Other metadata is preserved verbatim.
    """

    normalized = _as_indices(indices)
    if values is None:
        return None

    if isinstance(values, Mapping):
        selected: dict[str, Any] = {}
        population_size = _infer_population_size(values)
        edge_mask: Any | None = None
        n_edges: int | None = None
        if population_size is not None:
            _validate_indices(normalized, size=population_size, context=context)
        if "edge_index" in values:
            selected["edge_index"], edge_mask, n_edges = _induce_edges(
                values["edge_index"], normalized, num_nodes=population_size
            )
        for key, value in values.items():
            if key == "edge_index":
                continue
            if key == "num_nodes" and edge_mask is not None:
                selected[key] = int(normalized.size)
                continue
            if key == "n_edges" and edge_mask is not None:
                count = edge_mask.sum().item() if _is_torch_tensor(edge_mask) else edge_mask.sum()
                selected[key] = int(count)
                continue
            if (
                edge_mask is not None
                and n_edges is not None
                and _is_edge_aligned(str(key), value, n_edges=n_edges)
            ):
                selected[key] = _slice_value(value, edge_mask)
            elif population_size is not None and _leading_size(value) == population_size:
                selected[key] = _slice_value(value, normalized)
            else:
                selected[key] = value
        return selected

    size = _leading_size(values)
    if size is not None:
        _validate_indices(normalized, size=size, context=context)
    try:
        return _slice_value(values, normalized)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise DatasetSelectionError(
            f"{context}: unsupported row-selection contract for {type(values).__name__}",
            code="E_DATA_SELECTION_UNSUPPORTED",
        ) from exc


__all__ = ["select_rows"]
