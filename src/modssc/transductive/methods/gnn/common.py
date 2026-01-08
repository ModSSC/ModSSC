from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from modssc.device import resolve_device_name
from modssc.transductive.optional import optional_import
from modssc.transductive.validation import validate_node_dataset

# Optional dependency (keeps core import lightweight).
#
# NOTE: This module is only imported when a torch-based method is instantiated
# (through the method registry), so importing torch here is acceptable.
torch = optional_import("torch", extra="transductive-torch")

logger = logging.getLogger(__name__)

NormMode = Literal["rw", "sym"]


def normalize_device_name(device: str | None) -> str:
    return resolve_device_name(device, torch=torch) or "cpu"


def set_torch_seed(seed: int) -> None:
    """Best-effort deterministic seeding for torch."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    # Torch tensor
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    return X


def _labels_to_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        if y.shape[1] == 0:
            raise ValueError("y has zero columns")
        return y.argmax(axis=1).astype(np.int64)
    return y.reshape(-1).astype(np.int64)


def _as_edge_index(x: Any) -> np.ndarray:
    ei = _as_numpy(x).astype(np.int64, copy=False)
    if ei.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape {ei.shape}")
    if ei.shape[0] == 2:
        return ei
    if ei.shape[1] == 2:
        return ei.T
    raise ValueError(f"edge_index must have shape (2, E) or (E, 2), got {ei.shape}")


def _as_mask(x: Any, n: int, *, name: str) -> np.ndarray:
    m = _as_numpy(x).astype(bool, copy=False).reshape(-1)
    if m.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {m.shape}")
    return m


@dataclass
class PreparedData:
    X: Any  # torch.Tensor
    y: Any  # torch.LongTensor
    edge_index: Any  # torch.LongTensor (2, E)
    edge_weight: Any  # torch.FloatTensor (E,)
    train_mask: Any  # torch.BoolTensor (N,)
    val_mask: Any | None
    n_nodes: int
    n_classes: int
    device: Any


def coalesce_edges(edge_index: Any, edge_weight: Any, *, n_nodes: int) -> tuple[Any, Any]:
    """Coalesce duplicate edges by summing weights.

    The internal adjacency convention in ModSSC is PyG-like: edge_index is
    (src, dst) and corresponds to adjacency A[dst, src]. We therefore build a
    sparse tensor with indices (dst, src).
    """
    src, dst = edge_index[0], edge_index[1]
    idx = torch.stack([dst, src], dim=0)
    A = torch.sparse_coo_tensor(idx, edge_weight, size=(n_nodes, n_nodes)).coalesce()
    idx2 = A.indices()
    w2 = A.values()
    dst2, src2 = idx2[0], idx2[1]
    return torch.stack([src2, dst2], dim=0), w2


def add_self_loops_coalesce(
    edge_index: Any,
    edge_weight: Any,
    *,
    n_nodes: int,
    fill_value: float = 1.0,
) -> tuple[Any, Any]:
    loop_idx = torch.arange(n_nodes, device=edge_index.device, dtype=edge_index.dtype)
    loops = torch.stack([loop_idx, loop_idx], dim=0)
    edge_index2 = torch.cat([edge_index, loops], dim=1)
    edge_weight2 = torch.cat(
        [
            edge_weight,
            torch.full(
                (n_nodes,), float(fill_value), device=edge_weight.device, dtype=edge_weight.dtype
            ),
        ],
        dim=0,
    )
    return coalesce_edges(edge_index2, edge_weight2, n_nodes=n_nodes)


def normalize_edge_weight(
    *,
    edge_index: Any,
    edge_weight: Any,
    n_nodes: int,
    mode: NormMode,
    eps: float = 1e-12,
) -> Any:
    """Normalize edge weights.

    - rw: row-stochastic with respect to destination node (A rows correspond to dst)
    - sym: symmetric normalization (D^{-1/2} A D^{-1/2}) using a single degree vector
    """
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros((n_nodes,), device=edge_weight.device, dtype=edge_weight.dtype)
    deg.scatter_add_(0, dst, edge_weight)
    deg = deg.clamp_min(eps)

    if mode == "rw":
        return edge_weight / deg[dst]

    if mode == "sym":
        return edge_weight * (deg[src].rsqrt() * deg[dst].rsqrt())

    raise ValueError(f"Unknown normalization mode: {mode}")


def spmm(edge_index: Any, edge_weight: Any, X: Any, *, n_nodes: int) -> Any:
    """Sparse matrix multiplication (A @ X) for adjacency A[dst, src]."""
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros((n_nodes, X.shape[1]), device=X.device, dtype=X.dtype)
    out.index_add_(0, dst, X[src] * edge_weight.unsqueeze(1))
    return out


def prepare_data(
    data: Any,
    *,
    device: str | Any = "cpu",
    add_self_loops: bool = True,
    norm_mode: NormMode = "sym",
    dtype: Any | None = None,
) -> PreparedData:
    """Validate and convert a NodeDatasetLike into torch tensors."""
    validate_node_dataset(data)

    X_np = _ensure_2d(_as_numpy(data.X)).astype(np.float32, copy=False)
    y_np = _labels_to_int(_as_numpy(data.y))

    n_nodes = int(X_np.shape[0])

    masks = getattr(data, "masks", {}) or {}
    if "train_mask" not in masks:
        raise ValueError("data.masks must contain 'train_mask'")

    train_mask_np = _as_mask(masks["train_mask"], n_nodes, name="train_mask")
    val_mask_np = None
    if "val_mask" in masks and masks["val_mask"] is not None:
        try:
            val_mask_np = _as_mask(masks["val_mask"], n_nodes, name="val_mask")
        except Exception:
            val_mask_np = None

    g = data.graph
    edge_index_np = _as_edge_index(g.edge_index)
    edge_weight_raw = getattr(g, "edge_weight", None)
    if edge_weight_raw is None:
        edge_weight_np = np.ones((edge_index_np.shape[1],), dtype=np.float32)
    else:
        edge_weight_np = _as_numpy(edge_weight_raw).astype(np.float32, copy=False).reshape(-1)
        if edge_weight_np.shape[0] != edge_index_np.shape[1]:
            raise ValueError(
                f"edge_weight length mismatch: got {edge_weight_np.shape[0]} for E={edge_index_np.shape[1]}"
            )

    if isinstance(device, str) or device is None:
        dev = torch.device(normalize_device_name(device))
    else:
        dev = device
    X = torch.as_tensor(X_np, device=dev, dtype=dtype or torch.float32)
    y = torch.as_tensor(y_np, device=dev, dtype=torch.long)
    edge_index = torch.as_tensor(edge_index_np, device=dev, dtype=torch.long)
    edge_weight = torch.as_tensor(edge_weight_np, device=dev, dtype=torch.float32)

    train_mask = torch.as_tensor(train_mask_np, device=dev, dtype=torch.bool)
    val_mask = (
        torch.as_tensor(val_mask_np, device=dev, dtype=torch.bool)
        if val_mask_np is not None
        else None
    )

    if add_self_loops:
        edge_index, edge_weight = add_self_loops_coalesce(
            edge_index, edge_weight, n_nodes=n_nodes, fill_value=1.0
        )

    edge_weight = normalize_edge_weight(
        edge_index=edge_index, edge_weight=edge_weight, n_nodes=n_nodes, mode=norm_mode
    )

    n_classes = int(y.max().item()) + 1 if y.numel() > 0 else 0

    return PreparedData(
        X=X,
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        train_mask=train_mask,
        val_mask=val_mask,
        n_nodes=n_nodes,
        n_classes=n_classes,
        device=dev,
    )


def _prep_cache_key(
    data: Any,
    *,
    device: str | Any,
    add_self_loops: bool,
    norm_mode: NormMode,
    dtype: Any | None,
) -> tuple[Any, ...]:
    graph = getattr(data, "graph", None)
    edge_index = getattr(graph, "edge_index", None) if graph is not None else None
    edge_weight = getattr(graph, "edge_weight", None) if graph is not None else None
    device_key = (
        normalize_device_name(device) if isinstance(device, str) or device is None else str(device)
    )
    return (
        id(data),
        id(getattr(data, "X", None)),
        id(getattr(data, "y", None)),
        id(graph),
        id(edge_index),
        id(edge_weight),
        id(getattr(data, "masks", None)),
        device_key,
        bool(add_self_loops),
        str(norm_mode),
        None if dtype is None else str(dtype),
    )


def prepare_data_cached(
    data: Any,
    *,
    device: str | Any = "cpu",
    add_self_loops: bool = True,
    norm_mode: NormMode = "sym",
    dtype: Any | None = None,
    cache: dict[str, Any],
) -> PreparedData:
    """Prepare data with a simple identity-based cache for repeated predict calls."""
    key = _prep_cache_key(
        data,
        device=device,
        add_self_loops=add_self_loops,
        norm_mode=norm_mode,
        dtype=dtype,
    )
    cached_key = cache.get("key")
    cached_prep = cache.get("prep")
    if cached_key == key and cached_prep is not None:
        return cached_prep
    prep = prepare_data(
        data,
        device=device,
        add_self_loops=add_self_loops,
        norm_mode=norm_mode,
        dtype=dtype,
    )
    cache["key"] = key
    cache["prep"] = prep
    return prep


def accuracy_from_logits(logits: Any, y: Any, mask: Any) -> float:
    if mask is None or mask.numel() == 0:
        return float("nan")
    if not bool(mask.any()):
        return float("nan")
    pred = logits.argmax(dim=1)
    return float((pred[mask] == y[mask]).float().mean().item())


@dataclass
class TrainResult:
    n_epochs: int
    best_epoch: int | None
    best_val_loss: float | None


def train_fullbatch(
    *,
    model: Any,
    forward_fn: Callable[[], Any],
    y: Any,
    train_mask: Any,
    val_mask: Any | None,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int = 0,
) -> TrainResult:
    """Generic full-batch training loop for node classification."""
    set_torch_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state: dict[str, Any] | None = None
    best_val_loss: float | None = None
    best_epoch: int | None = None
    bad_epochs = 0

    for epoch in range(int(max_epochs)):
        model.train()
        logits = forward_fn()
        loss = torch.nn.functional.cross_entropy(logits[train_mask], y[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if val_mask is not None and bool(val_mask.any()):
            model.eval()
            with torch.no_grad():
                logits_val = forward_fn()
                val_loss = torch.nn.functional.cross_entropy(
                    logits_val[val_mask], y[val_mask]
                ).item()

            if best_val_loss is None or val_loss < best_val_loss - 1e-9:
                best_val_loss = float(val_loss)
                best_epoch = int(epoch)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
                logger.debug(
                    "train_fullbatch epoch=%s val_loss=%.4f best updated",
                    epoch,
                    val_loss,
                )
            else:
                bad_epochs += 1
                logger.debug(
                    "train_fullbatch epoch=%s val_loss=%.4f bad_epochs=%s/%s",
                    epoch,
                    val_loss,
                    bad_epochs,
                    patience,
                )
                if bad_epochs >= int(patience):
                    logger.debug(
                        "train_fullbatch early_stop epoch=%s best_epoch=%s best_val_loss=%.4f",
                        epoch,
                        best_epoch,
                        best_val_loss if best_val_loss is not None else float("nan"),
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.debug(
        "train_fullbatch done n_epochs=%s best_epoch=%s best_val_loss=%s",
        epoch + 1,
        best_epoch,
        best_val_loss,
    )
    return TrainResult(n_epochs=epoch + 1, best_epoch=best_epoch, best_val_loss=best_val_loss)
