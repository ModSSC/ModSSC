from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.optional import optional_import

from .common import (
    _as_edge_index,
    _as_numpy,
    normalize_device_name,
    prepare_data_cached,
    torch,
    train_fullbatch,
)

logger = logging.getLogger(__name__)


def _build_dense_adjacency(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    *,
    n_nodes: int,
    symmetrize: bool,
) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    w = edge_weight.astype(np.float32, copy=False)

    for s, d, weight in zip(src.tolist(), dst.tolist(), w.tolist(), strict=True):
        if s == d:
            continue
        adj[s, d] += weight

    if symmetrize:
        adj = np.maximum(adj, adj.T)

    np.fill_diagonal(adj, 0.0)
    return adj


def _normalize_adjacency(adj: Any, *, add_self_loops: bool, eps: float = 1e-12) -> Any:
    if add_self_loops:
        eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        adj = adj + eye
    deg = adj.sum(dim=1).clamp_min(eps)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def _coarsen_graph(
    adj: np.ndarray, node_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = int(adj.shape[0])
    marked = np.zeros(n_nodes, dtype=bool)
    neighbors: list[tuple[int, ...]] = []

    for i in range(n_nodes):
        neigh = np.flatnonzero(adj[i] != 0)
        neigh = neigh[neigh != i]
        neighbors.append(tuple(neigh.tolist()))

    groups: list[list[int]] = []

    # Structural equivalence grouping (SEG).
    by_neigh: dict[tuple[int, ...], list[int]] = {}
    for i, neigh in enumerate(neighbors):
        by_neigh.setdefault(neigh, []).append(i)
    for nodes in by_neigh.values():
        if len(nodes) < 2:
            continue
        nodes = sorted(nodes)
        for idx in range(0, len(nodes) - 1, 2):
            a = nodes[idx]
            b = nodes[idx + 1]
            if marked[a] or marked[b]:
                continue
            marked[a] = True
            marked[b] = True
            groups.append([a, b])

    # Structural similarity grouping (SSG).
    degrees = np.array([len(n) for n in neighbors], dtype=np.int64)
    order = np.argsort(degrees, kind="stable")
    for v in order.tolist():
        if marked[v]:
            continue
        best = -1
        best_score = None
        for u in neighbors[v]:
            if marked[u]:
                continue
            score = adj[v, u] / np.sqrt(float(node_weights[v]) * float(node_weights[u]))
            if best_score is None or score > best_score or (score == best_score and u < best):
                best_score = score
                best = u
        if best == -1:
            marked[v] = True
            groups.append([v])
        else:
            marked[v] = True
            marked[best] = True
            groups.append([v, best])

    if not bool(marked.all()):
        for v in range(n_nodes):
            if not marked[v]:
                marked[v] = True
                groups.append([v])

    n_next = len(groups)
    M = np.zeros((n_nodes, n_next), dtype=np.float32)
    for idx, group in enumerate(groups):
        M[group, idx] = 1.0

    weights_next = np.array([node_weights[group].sum() for group in groups], dtype=np.int64)
    adj_next = M.T @ adj @ M

    return M, adj_next, weights_next


class _MultiChannelGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.channels = int(channels)
        self.channel_logits = torch.nn.Parameter(torch.zeros(self.channels))
        self.lins = torch.nn.ModuleList(
            [torch.nn.Linear(in_channels, out_channels, bias=False) for _ in range(self.channels)]
        )

    def forward(
        self,
        x: Any,
        adj_norm: Any,
        *,
        node_weights: Any,
        weight_embed: Any | None,
    ) -> Any:
        if weight_embed is not None:
            w_emb = weight_embed(node_weights)
            x = torch.cat([x, w_emb], dim=1)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = adj_norm @ x
        outs = [torch.relu(lin(x)) for lin in self.lins]
        weights = torch.softmax(self.channel_logits, dim=0)
        return sum(w * out for w, out in zip(weights, outs, strict=True))


class _HGCNNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        *,
        num_layers: int,
        channels: int,
        weight_embed_dim: int,
        max_weight: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.num_coarsen = (self.num_layers - 1) // 2
        self.dropout = float(dropout)

        if weight_embed_dim > 0:
            self.weight_embed = torch.nn.Embedding(int(max_weight) + 1, int(weight_embed_dim))
        else:
            self.weight_embed = None

        layers = []
        for layer_idx in range(2 * self.num_coarsen):
            base_in = in_channels if layer_idx == 0 else self.hidden_dim
            in_dim = base_in + int(weight_embed_dim)
            layers.append(
                _MultiChannelGCN(
                    in_dim,
                    self.hidden_dim,
                    channels=channels,
                    dropout=self.dropout,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.out_lin = torch.nn.Linear(self.hidden_dim, self.out_channels, bias=False)

    def forward(
        self,
        x: Any,
        *,
        adj_norm_levels: list[Any],
        group_mats: list[Any],
        node_weights_levels: list[Any],
    ) -> Any:
        H = x
        g_skip: list[Any] = []

        for i in range(self.num_coarsen):
            G = self.layers[i](
                H,
                adj_norm_levels[i],
                node_weights=node_weights_levels[i],
                weight_embed=self.weight_embed,
            )
            g_skip.append(G)
            H = group_mats[i].T @ G

        for offset, i in enumerate(range(self.num_coarsen - 1, -1, -1)):
            layer = self.layers[self.num_coarsen + offset]
            G = layer(
                H,
                adj_norm_levels[i + 1],
                node_weights=node_weights_levels[i + 1],
                weight_embed=self.weight_embed,
            )
            H = group_mats[i] @ G + g_skip[i]

        H = torch.nn.functional.dropout(H, p=self.dropout, training=self.training)
        H = adj_norm_levels[0] @ H
        H = self.out_lin(H)
        return torch.relu(H)


@dataclass(frozen=True)
class HGCNSpec:
    """Hyperparameters for H-GCN."""

    hidden_dim: int = 64
    weight_embed_dim: int = 8
    num_layers: int = 9
    channels: int = 4
    dropout: float = 0.15
    lr: float = 0.03
    weight_decay: float = 7e-4
    max_epochs: int = 250
    patience: int = 50
    add_self_loops: bool = True
    symmetrize: bool = True


class HGCNMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="h_gcn",
        name="H-GCN",
        year=2019,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Hierarchical Graph Convolutional Networks for Semi-Supervised Node Classification",
        paper_pdf="https://arxiv.org/abs/1902.06667",
        official_code="https://github.com/CRIPAC-DIG/H-GCN",
    )

    def __init__(self, spec: HGCNSpec | None = None) -> None:
        self.spec = spec or HGCNSpec()
        self._model: Any | None = None
        self._device: Any | None = None
        self._prep_cache: dict[str, Any] = {}
        self._adj_norm_levels: list[Any] = []
        self._group_mats: list[Any] = []
        self._node_weights_levels: list[Any] = []
        self._n_nodes: int | None = None
        self._n_classes: int | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> HGCNMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        if self.spec.num_layers < 3 or self.spec.num_layers % 2 == 0:
            raise ValueError("num_layers must be odd and >= 3 for H-GCN")
        if self.spec.channels < 1:
            raise ValueError("channels must be >= 1")
        if self.spec.weight_embed_dim < 1:
            raise ValueError("weight_embed_dim must be >= 1")

        self._device = normalize_device_name(device)
        prep = prepare_data_cached(
            data,
            device=self._device,
            add_self_loops=False,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        self._n_nodes = prep.n_nodes
        self._n_classes = prep.n_classes
        val_count = int(prep.val_mask.sum()) if prep.val_mask is not None else None
        logger.info(
            "H-GCN sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )

        edge_index_np = _as_edge_index(_as_numpy(data.graph.edge_index))
        edge_weight_raw = getattr(data.graph, "edge_weight", None)
        if edge_weight_raw is None:
            edge_weight_np = np.ones((edge_index_np.shape[1],), dtype=np.float32)
        else:
            edge_weight_np = _as_numpy(edge_weight_raw).astype(np.float32, copy=False).reshape(-1)
            if edge_weight_np.shape[0] != edge_index_np.shape[1]:
                raise ValueError(
                    "edge_weight length mismatch: "
                    f"got {edge_weight_np.shape[0]} for E={edge_index_np.shape[1]}"
                )

        adj = _build_dense_adjacency(
            edge_index_np,
            edge_weight_np,
            n_nodes=prep.n_nodes,
            symmetrize=self.spec.symmetrize,
        )
        node_weights = np.ones((prep.n_nodes,), dtype=np.int64)

        num_coarsen = (self.spec.num_layers - 1) // 2
        adj_levels = [adj]
        weight_levels = [node_weights]
        group_mats = []

        for _ in range(num_coarsen):
            M, adj_next, weights_next = _coarsen_graph(adj_levels[-1], weight_levels[-1])
            group_mats.append(M)
            adj_levels.append(adj_next)
            weight_levels.append(weights_next)

        max_weight = int(max(int(w.max()) for w in weight_levels))
        if max_weight < 1:
            max_weight = 1

        device_t = torch.device(self._device)
        self._adj_norm_levels = [
            _normalize_adjacency(
                torch.as_tensor(level, device=device_t, dtype=torch.float32),
                add_self_loops=self.spec.add_self_loops,
            )
            for level in adj_levels
        ]
        self._group_mats = [
            torch.as_tensor(M, device=device_t, dtype=torch.float32) for M in group_mats
        ]
        self._node_weights_levels = [
            torch.as_tensor(w, device=device_t, dtype=torch.long) for w in weight_levels
        ]

        self._model = _HGCNNet(
            prep.X.shape[1],
            self.spec.hidden_dim,
            prep.n_classes,
            num_layers=self.spec.num_layers,
            channels=self.spec.channels,
            weight_embed_dim=self.spec.weight_embed_dim,
            max_weight=max_weight,
            dropout=self.spec.dropout,
        ).to(device_t)

        train_fullbatch(
            model=self._model,
            forward_fn=lambda: self._model(
                prep.X,
                adj_norm_levels=self._adj_norm_levels,
                group_mats=self._group_mats,
                node_weights_levels=self._node_weights_levels,
            ),
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
            raise RuntimeError("HGCNMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=False,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        if self._n_nodes is not None and prep.n_nodes != self._n_nodes:
            raise ValueError(f"H-GCN was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}")

        self._model.eval()
        with torch.no_grad():
            logits = self._model(
                prep.X,
                adj_norm_levels=self._adj_norm_levels,
                group_mats=self._group_mats,
                node_weights_levels=self._node_weights_levels,
            )
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
