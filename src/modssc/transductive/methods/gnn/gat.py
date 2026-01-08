from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.optional import optional_import

from .common import normalize_device_name, prepare_data_cached, torch, train_fullbatch

logger = logging.getLogger(__name__)


def _edge_softmax(logits: Any, dst: Any, *, n_nodes: int, eps: float = 1e-16) -> Any:
    """Softmax over incoming edges for each destination node.

    Parameters
    ----------
    logits:
        Tensor of shape (E, H) where H=heads.
    dst:
        Tensor of shape (E,) with destination node indices.
    n_nodes:
        Number of nodes.

    Returns
    -------
    Tensor of shape (E, H) with softmax-normalized values per dst node.
    """
    heads = int(logits.shape[1])
    idx = dst.unsqueeze(1).expand(-1, heads)

    # max over incoming edges (dst)
    max_per_dst = torch.full(
        (n_nodes, heads), -float("inf"), device=logits.device, dtype=logits.dtype
    )
    max_per_dst.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)

    exp_logits = torch.exp(logits - max_per_dst[dst])
    denom = torch.zeros((n_nodes, heads), device=logits.device, dtype=logits.dtype)
    denom.scatter_add_(0, idx, exp_logits)
    return exp_logits / (denom[dst] + eps)


class _GATConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.out_channels = int(out_channels)
        self.concat = bool(concat)
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)

        self.lin = torch.nn.Linear(in_channels, self.heads * self.out_channels, bias=False)
        self.att_src = torch.nn.Parameter(torch.empty(self.heads, self.out_channels))
        self.att_dst = torch.nn.Parameter(torch.empty(self.heads, self.out_channels))

        if bias:
            out_dim = self.heads * self.out_channels if self.concat else self.out_channels
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Any, edge_index: Any, *, n_nodes: int) -> Any:
        # edge_index: (2, E) (src,dst)
        src, dst = edge_index[0], edge_index[1]
        h = self.lin(x).view(n_nodes, self.heads, self.out_channels)  # (N, H, C)

        # attention scores per node/head
        e_src = (h * self.att_src).sum(dim=-1)  # (N, H)
        e_dst = (h * self.att_dst).sum(dim=-1)  # (N, H)

        logits = e_src[src] + e_dst[dst]  # (E, H)
        logits = torch.nn.functional.leaky_relu(logits, negative_slope=self.negative_slope)

        alpha = _edge_softmax(logits, dst, n_nodes=n_nodes)  # (E, H)
        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)

        # messages: (E, H, C)
        msg = h[src] * alpha.unsqueeze(-1)
        msg_flat = msg.reshape(msg.shape[0], self.heads * self.out_channels)  # (E, H*C)

        out_flat = torch.zeros(
            (n_nodes, self.heads * self.out_channels), device=x.device, dtype=x.dtype
        )
        out_flat.index_add_(0, dst, msg_flat)

        if self.concat:
            out = out_flat
        else:
            out = out_flat.view(n_nodes, self.heads, self.out_channels).mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out


class _GATNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        head_dim: int,
        heads: int,
        out_channels: int,
        dropout: float,
        negative_slope: float,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = _GATConv(
            in_channels,
            head_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            negative_slope=negative_slope,
            bias=True,
        )
        self.conv2 = _GATConv(
            head_dim * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            negative_slope=negative_slope,
            bias=True,
        )

    def forward(self, x: Any, edge_index: Any, *, n_nodes: int) -> Any:
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, n_nodes=n_nodes)
        x = torch.nn.functional.elu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, n_nodes=n_nodes)
        return x


@dataclass(frozen=True)
class GATSpec:
    """Hyperparameters for a full-batch GAT baseline.

    This is a small, torch-only implementation (no torch-geometric dependency).
    """

    head_dim: int = 8
    heads: int = 8
    dropout: float = 0.6
    negative_slope: float = 0.2
    lr: float = 0.005
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 100
    add_self_loops: bool = True


class GATMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="gat",
        name="GAT",
        year=2018,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Graph Attention Networks",
        paper_pdf="https://arxiv.org/abs/1710.10903",
        official_code="https://github.com/PetarV-/GAT",
    )

    def __init__(self, spec: GATSpec | None = None) -> None:
        self.spec = spec or GATSpec()
        self._model: Any | None = None
        self._device: Any | None = None
        self._prep_cache: dict[str, Any] = {}

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> GATMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        # Make the dependency explicit in the exception message.
        optional_import("torch", extra="transductive-torch")

        self._device = normalize_device_name(device)
        prep = prepare_data_cached(
            data,
            device=self._device,
            add_self_loops=self.spec.add_self_loops,
            norm_mode="rw",  # edge_weight unused by GAT, but keep deterministic preprocessing
            cache=self._prep_cache,
        )
        val_count = int(prep.val_mask.sum()) if prep.val_mask is not None else None
        logger.info(
            "GAT sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )

        self._model = _GATNet(
            prep.X.shape[1],
            head_dim=self.spec.head_dim,
            heads=self.spec.heads,
            out_channels=prep.n_classes,
            dropout=self.spec.dropout,
            negative_slope=self.spec.negative_slope,
        ).to(self._device)

        train_fullbatch(
            model=self._model,
            forward_fn=lambda: self._model(prep.X, prep.edge_index, n_nodes=prep.n_nodes),
            y=prep.y,
            train_mask=prep.train_mask,
            val_mask=prep.val_mask,
            lr=self.spec.lr,
            weight_decay=self.spec.weight_decay,
            max_epochs=self.spec.max_epochs,
            patience=self.spec.patience,
            seed=seed,
        )
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GATMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=self.spec.add_self_loops,
            norm_mode="rw",
            cache=self._prep_cache,
        )

        self._model.eval()
        with torch.no_grad():
            logits = self._model(prep.X, prep.edge_index, n_nodes=prep.n_nodes)
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
