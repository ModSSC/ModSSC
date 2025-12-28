from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry import register_op
from ..types import AugmentationContext, GraphSample, Modality
from ..utils import ensure_edge_index_2xE, is_torch_tensor, to_numpy
from .base import AugmentationOp


def _extract_graph(obj: Any) -> tuple[Any, Any, Any | None, str]:
    """Return (x, edge_index, edge_weight, kind) where kind in {'graphsample','dict','attr'}."""
    if isinstance(obj, GraphSample):
        return obj.x, obj.edge_index, obj.edge_weight, "graphsample"
    if isinstance(obj, dict):
        return obj.get("x"), obj.get("edge_index"), obj.get("edge_weight"), "dict"
    if hasattr(obj, "x") and hasattr(obj, "edge_index"):
        ew = getattr(obj, "edge_weight", None)
        return obj.x, obj.edge_index, ew, "attr"
    raise TypeError(
        "Graph ops expect GraphSample, dict(x, edge_index), or object with .x and .edge_index"
    )


def _rebuild_graph(obj: Any, kind: str, *, x: Any, edge_index: Any, edge_weight: Any | None) -> Any:
    if kind == "graphsample":
        assert isinstance(obj, GraphSample)
        return GraphSample(x=x, edge_index=edge_index, edge_weight=edge_weight, meta=dict(obj.meta))
    if kind == "dict":
        assert isinstance(obj, dict)
        out = dict(obj)
        out["x"] = x
        out["edge_index"] = edge_index
        if edge_weight is not None:
            out["edge_weight"] = edge_weight
        else:
            out.pop("edge_weight", None)
        return out
    # attr
    out = copy.copy(obj)
    out.x = x
    out.edge_index = edge_index
    if edge_weight is not None:
        out.edge_weight = edge_weight
    elif hasattr(out, "edge_weight"):
        delattr(out, "edge_weight")
    return out


@register_op("graph.edge_dropout")
@dataclass
class EdgeDropout(AugmentationOp):
    """Randomly drop edges."""

    op_id: str = "graph.edge_dropout"
    modality: Modality = "graph"
    p: float = 0.2

    def apply(self, g: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        p = float(self.p)
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if p == 0.0:
            return g

        x, edge_index, edge_weight, kind = _extract_graph(g)
        ei = ensure_edge_index_2xE(edge_index)
        E = int(ei.shape[1])

        keep = rng.random(size=(E,)) >= p
        keep_idx = np.flatnonzero(keep).astype(np.int64)

        use_torch = (
            is_torch_tensor(x) or is_torch_tensor(edge_index) or is_torch_tensor(edge_weight)
        )
        if use_torch:
            import importlib

            torch = importlib.import_module("torch")

            device = None
            for obj in (edge_index, edge_weight, x):
                if is_torch_tensor(obj):
                    device = obj.device
                    break

            ei_t = (
                ei if is_torch_tensor(ei) else torch.as_tensor(ei, device=device, dtype=torch.long)
            )
            idx_t = torch.as_tensor(keep_idx, device=device, dtype=torch.long)

            ei2 = ei_t.index_select(1, idx_t)
            ew2 = None
            if edge_weight is not None:
                ew_t = (
                    edge_weight
                    if is_torch_tensor(edge_weight)
                    else torch.as_tensor(edge_weight, device=device)
                )
                ew_t = ew_t.reshape(-1)
                ew2 = ew_t.index_select(0, idx_t)

            return _rebuild_graph(g, kind, x=x, edge_index=ei2, edge_weight=ew2)

        ei2 = ei[:, keep_idx]
        ew2 = None
        if edge_weight is not None:
            ew_np = to_numpy(edge_weight).astype(np.float32, copy=False)
            ew2 = ew_np[keep_idx]

        return _rebuild_graph(g, kind, x=x, edge_index=ei2, edge_weight=ew2)


@register_op("graph.feature_mask")
@dataclass
class FeatureMask(AugmentationOp):
    """Randomly mask node features (set to zero)."""

    op_id: str = "graph.feature_mask"
    modality: Modality = "graph"
    p: float = 0.1

    def apply(self, g: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        p = float(self.p)
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if p == 0.0:
            return g

        x, edge_index, edge_weight, kind = _extract_graph(g)
        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")
            mask = torch.from_numpy((rng.random(size=tuple(x.shape)) >= p).astype(np.float32))
            mask = mask.to(device=x.device, dtype=x.dtype)
            x2 = x * mask
        else:
            x_np = np.asarray(x)
            mask = (rng.random(size=x_np.shape) >= p).astype(x_np.dtype, copy=False)
            x2 = x_np * mask

        return _rebuild_graph(g, kind, x=x2, edge_index=edge_index, edge_weight=edge_weight)
