from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from modssc.data_augmentation.utils import is_torch_tensor

from ..errors import BenchRuntimeError


def _as_idx(idx: np.ndarray | list[int]) -> np.ndarray:
    idx_arr = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx_arr.size and int(idx_arr.min()) < 0:
        raise BenchRuntimeError("E_BENCH_SLICE_NEGATIVE_INDEX", "indices must be >= 0")
    return idx_arr


def _infer_num_nodes(X: Mapping[str, Any]) -> int | None:
    x = X.get("x")
    shape = getattr(x, "shape", None)
    if shape is not None:
        try:
            if len(shape) > 0:
                return int(shape[0])
        except (TypeError, ValueError):
            pass

    num_nodes = X.get("num_nodes")
    if num_nodes is None:
        return None
    try:
        return int(num_nodes)
    except (TypeError, ValueError):
        return None


def _slice_edge_index(edge_index: Any, idx: np.ndarray, *, num_nodes: int | None = None) -> Any:
    try:
        import importlib

        torch = importlib.import_module("torch")
        pyg_utils = importlib.import_module("torch_geometric.utils")
    except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - optional deps
        raise BenchRuntimeError(
            "E_BENCH_GRAPH_SLICE_DEP_MISSING",
            "graph slicing requires torch and torch_geometric",
        ) from exc

    if is_torch_tensor(edge_index):
        ei = edge_index
        device = ei.device
    else:
        ei = torch.as_tensor(np.asarray(edge_index), dtype=torch.long)
        device = ei.device

    subset = torch.as_tensor(idx, dtype=torch.long, device=device)
    try:
        sub_ei, _ = pyg_utils.subgraph(
            subset,
            ei,
            relabel_nodes=True,
            num_nodes=None if num_nodes is None else int(num_nodes),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise BenchRuntimeError(
            "E_BENCH_GRAPH_SLICE_ERROR",
            f"failed to slice edge_index with relabel_nodes=True: {exc}",
        ) from exc

    if is_torch_tensor(edge_index):
        return sub_ei
    return sub_ei.detach().cpu().numpy()


def _slice_tensor(x: Any, idx: np.ndarray) -> Any:
    import importlib

    torch = importlib.import_module("torch")
    idx_t = torch.as_tensor(idx, device=x.device, dtype=torch.long)
    return x[idx_t]


def _slice_sequence(x: list[Any], idx: np.ndarray) -> list[Any]:
    return [x[int(i)] for i in idx.tolist()]


def _should_slice_by_dim0(x: Any, idx: np.ndarray) -> bool:
    if idx.size == 0:
        return True
    shape = getattr(x, "shape", None)
    if shape is None:
        return False
    try:
        if len(shape) == 0:
            return False
        return int(shape[0]) > int(idx.max())
    except (TypeError, ValueError):
        return False


def select_rows(
    X: Any,
    idx: np.ndarray | list[int],
    *,
    context: str,
) -> Any:
    idx_arr = _as_idx(idx)
    if X is None:
        return None

    if isinstance(X, Mapping):
        out: dict[str, Any] = {}
        num_nodes = _infer_num_nodes(X)
        if "edge_index" in X:
            out["edge_index"] = _slice_edge_index(X["edge_index"], idx_arr, num_nodes=num_nodes)
        for k, v in X.items():
            if k == "edge_index":
                continue
            if is_torch_tensor(v):
                if _should_slice_by_dim0(v, idx_arr):
                    out[k] = _slice_tensor(v, idx_arr)
                else:
                    out[k] = v
                continue
            if isinstance(v, np.ndarray):
                if _should_slice_by_dim0(v, idx_arr):
                    out[k] = v[idx_arr]
                else:
                    out[k] = v
                continue
            if isinstance(v, list):
                if _should_slice_by_dim0(np.asarray(v, dtype=object), idx_arr):
                    out[k] = _slice_sequence(v, idx_arr)
                else:
                    out[k] = v
                continue
            out[k] = v
        return out

    if is_torch_tensor(X):
        return _slice_tensor(X, idx_arr)
    if isinstance(X, np.ndarray):
        return X[idx_arr]
    if isinstance(X, list):
        return _slice_sequence(X, idx_arr)

    try:
        return X[idx_arr]
    except Exception as exc:
        raise BenchRuntimeError(
            "E_BENCH_SLICE_ERROR",
            f"{context}: unsupported slicing contract for object type={type(X).__name__}",
        ) from exc
