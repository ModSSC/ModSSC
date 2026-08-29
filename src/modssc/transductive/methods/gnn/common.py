from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from modssc.runtime.device import resolve_device_name
from modssc.transductive.optional import optional_import
from modssc.transductive.validation import validate_node_dataset
from modssc.utils.numpy import to_numpy as _as_numpy

# Optional dependency (keeps core import lightweight).
#
# NOTE: This module is only imported when a torch-based method is instantiated
# (through the method registry), so importing torch here is acceptable.
torch = optional_import("torch", extra="transductive-torch")

logger = logging.getLogger(__name__)

NormMode = Literal["rw", "sym"]
SelectionMetric = Literal["val_loss", "val_acc_then_loss", "val_acc_then_loss_reset_any"]
WeightDecayScope = Literal["all", "first_layer", "non_bias", "none"]


def normalize_device_name(device: str | None) -> str:
    return resolve_device_name(device, torch=torch) or "cpu"


def set_torch_seed(seed: int) -> None:
    """Best-effort deterministic seeding for torch."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
    (src, dst) and corresponds to adjacency A[dst, src]. Duplicate edges are
    aggregated deterministically by their (dst, src) key.
    """
    if edge_index.numel() and (
        bool((edge_index < 0).any().item()) or bool((edge_index >= int(n_nodes)).any().item())
    ):
        raise ValueError("edge_index contains node id outside [0, n_nodes)")
    src, dst = edge_index[0], edge_index[1]
    idx = torch.stack([dst, src], dim=0)
    idx2, inverse = torch.unique(idx, dim=1, sorted=True, return_inverse=True)
    w2 = torch.zeros((int(idx2.shape[1]),), device=edge_weight.device, dtype=edge_weight.dtype)
    w2.scatter_add_(0, inverse, edge_weight)
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


class TwoLayerMLP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        *,
        dropout: float,
        input_dropout: float | None = None,
        hidden_dropout: float | None = None,
        batch_norm: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.input_dropout = float(dropout if input_dropout is None else input_dropout)
        self.hidden_dropout = float(dropout if hidden_dropout is None else hidden_dropout)
        self.batch_norm = bool(batch_norm)
        self.bn_input = torch.nn.BatchNorm1d(in_channels) if self.batch_norm else None
        self.bn_hidden = torch.nn.BatchNorm1d(hidden_dim) if self.batch_norm else None
        self.lin1 = torch.nn.Linear(in_channels, hidden_dim, bias=bool(bias))
        self.lin2 = torch.nn.Linear(hidden_dim, out_channels, bias=bool(bias))

    def forward(self, x: Any) -> Any:
        if self.bn_input is not None:
            x = self.bn_input(x)
        x = torch.nn.functional.dropout(x, p=self.input_dropout, training=self.training)
        x = torch.relu(self.lin1(x))
        if self.bn_hidden is not None:
            x = self.bn_hidden(x)
        x = torch.nn.functional.dropout(x, p=self.hidden_dropout, training=self.training)
        x = self.lin2(x)
        return x


def two_layer_gnn_forward(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
    x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
    x = torch.relu(self.conv1(x, edge_index, edge_weight, n_nodes=n_nodes))
    x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
    x = self.conv2(x, edge_index, edge_weight, n_nodes=n_nodes)
    return x


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
    best_val_acc: float | None


def _optimizer_param_groups(model: Any, *, weight_decay: float, scope: WeightDecayScope) -> Any:
    wd = float(weight_decay)
    if scope == "all":
        return model.parameters()
    if scope == "none" or wd == 0.0:
        return [{"params": list(model.parameters()), "weight_decay": 0.0}]
    if scope == "non_bias":
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            is_bias = name.rsplit(".", 1)[-1] == "bias"
            if int(param.ndim) >= 2 and not is_bias:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
    if scope != "first_layer":
        raise ValueError(f"Unknown weight_decay_scope: {scope}")

    first_weight_id: int | None = None
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if weight is not None and getattr(weight, "requires_grad", False) and int(weight.ndim) == 2:
            first_weight_id = id(weight)
            break
    if first_weight_id is None:
        return [{"params": list(model.parameters()), "weight_decay": 0.0}]

    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if id(param) == first_weight_id:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _is_better_checkpoint(
    *,
    selection_metric: SelectionMetric,
    val_loss: float,
    val_acc: float,
    best_val_loss: float | None,
    best_val_acc: float | None,
) -> bool:
    if selection_metric == "val_loss":
        return best_val_loss is None or val_loss < best_val_loss - 1e-9
    if selection_metric in {"val_acc_then_loss", "val_acc_then_loss_reset_any"}:
        if best_val_acc is None or val_acc > best_val_acc + 1e-9:
            return True
        if abs(val_acc - best_val_acc) <= 1e-9:
            return best_val_loss is None or val_loss < best_val_loss - 1e-9
        return False
    raise ValueError(f"Unknown selection_metric: {selection_metric}")


def _should_reset_patience(
    *,
    selection_metric: SelectionMetric,
    checkpoint_updated: bool,
    val_loss: float,
    val_acc: float,
    best_stop_val_loss: float | None,
    best_stop_val_acc: float | None,
) -> bool:
    if selection_metric != "val_acc_then_loss_reset_any":
        return checkpoint_updated
    return (
        best_stop_val_acc is None
        or val_acc > best_stop_val_acc + 1e-9
        or best_stop_val_loss is None
        or val_loss < best_stop_val_loss - 1e-9
    )


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
    weight_decay_scope: WeightDecayScope = "all",
    selection_metric: SelectionMetric = "val_loss",
) -> TrainResult:
    """Generic full-batch training loop for node classification."""
    set_torch_seed(seed)

    optimizer = torch.optim.Adam(
        _optimizer_param_groups(model, weight_decay=float(weight_decay), scope=weight_decay_scope),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    best_state: dict[str, Any] | None = None
    best_val_loss: float | None = None
    best_val_acc: float | None = None
    best_stop_val_loss: float | None = None
    best_stop_val_acc: float | None = None
    best_epoch: int | None = None
    bad_epochs = 0
    n_epochs = 0

    for epoch in range(int(max_epochs)):
        n_epochs = epoch + 1
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
                val_acc = accuracy_from_logits(logits_val, y, val_mask)

            checkpoint_updated = _is_better_checkpoint(
                selection_metric=selection_metric,
                val_loss=float(val_loss),
                val_acc=float(val_acc),
                best_val_loss=best_val_loss,
                best_val_acc=best_val_acc,
            )
            reset_patience = _should_reset_patience(
                selection_metric=selection_metric,
                checkpoint_updated=checkpoint_updated,
                val_loss=float(val_loss),
                val_acc=float(val_acc),
                best_stop_val_loss=best_stop_val_loss,
                best_stop_val_acc=best_stop_val_acc,
            )
            if best_stop_val_loss is None or float(val_loss) < best_stop_val_loss - 1e-9:
                best_stop_val_loss = float(val_loss)
            if best_stop_val_acc is None or float(val_acc) > best_stop_val_acc + 1e-9:
                best_stop_val_acc = float(val_acc)

            if checkpoint_updated:
                best_val_loss = float(val_loss)
                best_val_acc = float(val_acc)
                best_epoch = int(epoch)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            if reset_patience:
                bad_epochs = 0
                logger.debug(
                    "train_fullbatch epoch=%s val_loss=%.4f val_acc=%.4f checkpoint_updated=%s",
                    epoch,
                    val_loss,
                    val_acc,
                    checkpoint_updated,
                )
            else:
                bad_epochs += 1
                logger.debug(
                    "train_fullbatch epoch=%s val_loss=%.4f val_acc=%.4f bad_epochs=%s/%s",
                    epoch,
                    val_loss,
                    val_acc,
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
        "train_fullbatch done n_epochs=%s best_epoch=%s best_val_loss=%s best_val_acc=%s",
        n_epochs,
        best_epoch,
        best_val_loss,
        best_val_acc,
    )
    return TrainResult(
        n_epochs=n_epochs,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
    )
