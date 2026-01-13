from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.device import resolve_device_name
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.optional import require
from modssc.preprocess.steps.base import get_X
from modssc.preprocess.store import ArtifactStore


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim != 2:
        raise PreprocessValidationError(f"graph.dgi requires 2D features, got shape {X.shape}")
    return X


def _as_edge_index(edge_index: Any, *, n_nodes: int) -> np.ndarray:
    edge_index_np = np.asarray(edge_index)
    if edge_index_np.ndim != 2:
        raise PreprocessValidationError("graph.edge_index must be a 2D array")
    if edge_index_np.shape[0] != 2 and edge_index_np.shape[1] == 2:
        edge_index_np = edge_index_np.T
    if edge_index_np.shape[0] != 2:
        raise PreprocessValidationError("graph.edge_index must have shape (2, E)")
    if edge_index_np.size:
        flat = edge_index_np.ravel()
        min_idx = int(min(flat))
        max_idx = int(max(flat))
        if min_idx < 0 or max_idx >= n_nodes:
            raise PreprocessValidationError("graph.edge_index has out of range node indices")
    return edge_index_np.astype(np.int64, copy=False)


def _as_edge_weight(edge_weight: Any, *, n_edges: int) -> np.ndarray:
    edge_weight_np = np.asarray(edge_weight, dtype=np.float32).reshape(-1)
    if edge_weight_np.shape[0] != n_edges:
        raise PreprocessValidationError(
            f"graph.edge_weight length mismatch: got {edge_weight_np.shape[0]} for E={n_edges}"
        )
    return edge_weight_np


def _train_dgi(
    X: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    *,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    unsup_epochs: int,
    unsup_lr: float,
    add_self_loops: bool,
    device: str,
    seed: int,
) -> np.ndarray:
    torch = require(module="torch", extra="transductive-torch", purpose="graph.dgi")

    def _coalesce_edges(edge_index_t, edge_weight_t, *, n_nodes: int):
        src, dst = edge_index_t[0], edge_index_t[1]
        idx = torch.stack([dst, src], dim=0)
        adj = torch.sparse_coo_tensor(idx, edge_weight_t, size=(n_nodes, n_nodes)).coalesce()
        idx2 = adj.indices()
        w2 = adj.values()
        dst2, src2 = idx2[0], idx2[1]
        return torch.stack([src2, dst2], dim=0), w2

    def _add_self_loops(edge_index_t, edge_weight_t, *, n_nodes: int):
        loop_idx = torch.arange(n_nodes, device=edge_index_t.device, dtype=edge_index_t.dtype)
        loops = torch.stack([loop_idx, loop_idx], dim=0)
        edge_index_t = torch.cat([edge_index_t, loops], dim=1)
        edge_weight_t = torch.cat(
            [
                edge_weight_t,
                torch.full(
                    (n_nodes,),
                    1.0,
                    device=edge_weight_t.device,
                    dtype=edge_weight_t.dtype,
                ),
            ],
            dim=0,
        )
        return _coalesce_edges(edge_index_t, edge_weight_t, n_nodes=n_nodes)

    def _normalize_edge_weight(edge_index_t, edge_weight_t, *, n_nodes: int, eps: float = 1e-12):
        src, dst = edge_index_t[0], edge_index_t[1]
        deg = torch.zeros((n_nodes,), device=edge_weight_t.device, dtype=edge_weight_t.dtype)
        deg.scatter_add_(0, dst, edge_weight_t)
        deg = deg.clamp_min(eps)
        return edge_weight_t * (deg[src].rsqrt() * deg[dst].rsqrt())

    def _spmm(edge_index_t, edge_weight_t, X_t, *, n_nodes: int):
        src, dst = edge_index_t[0], edge_index_t[1]
        out = torch.zeros((n_nodes, X_t.shape[1]), device=X_t.device, dtype=X_t.dtype)
        out.index_add_(0, dst, X_t[src] * edge_weight_t.unsqueeze(1))
        return out

    class _GCNConv(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)

        def forward(self, x: Any, edge_index_t: Any, edge_weight_t: Any, *, n_nodes: int) -> Any:
            x = self.lin(x)
            return _spmm(edge_index_t, edge_weight_t, x, n_nodes=n_nodes)

    class _GCNEncoder(torch.nn.Module):
        def __init__(self, in_channels: int, hidden: int, out_dim: int, *, dropout: float) -> None:
            super().__init__()
            self.dropout = float(dropout)
            self.conv1 = _GCNConv(in_channels, hidden)
            self.conv2 = _GCNConv(hidden, out_dim)

        def forward(self, x: Any, edge_index_t: Any, edge_weight_t: Any, *, n_nodes: int) -> Any:
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = torch.relu(self.conv1(x, edge_index_t, edge_weight_t, n_nodes=n_nodes))
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index_t, edge_weight_t, n_nodes=n_nodes)
            return x

    class _Discriminator(torch.nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = torch.nn.Linear(dim, dim, bias=False)

        def forward(self, h: Any, s: Any) -> Any:
            ws = self.weight(s)
            return (h * ws).sum(dim=-1)

    dev_name = resolve_device_name(device, torch=torch) or "cpu"
    dev = torch.device(dev_name)
    X_t = torch.as_tensor(X, device=dev, dtype=torch.float32)
    edge_index_t = torch.as_tensor(edge_index, device=dev, dtype=torch.long)
    edge_weight_t = torch.as_tensor(edge_weight, device=dev, dtype=torch.float32)

    n_nodes = int(X_t.shape[0])
    if add_self_loops:
        edge_index_t, edge_weight_t = _add_self_loops(edge_index_t, edge_weight_t, n_nodes=n_nodes)
    edge_weight_t = _normalize_edge_weight(edge_index_t, edge_weight_t, n_nodes=n_nodes)

    encoder = _GCNEncoder(
        X_t.shape[1],
        hidden=int(hidden_dim),
        out_dim=int(embedding_dim),
        dropout=float(dropout),
    ).to(dev)
    disc = _Discriminator(int(embedding_dim)).to(dev)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(disc.parameters()), lr=float(unsup_lr)
    )
    bce = torch.nn.BCEWithLogitsLoss()

    torch.manual_seed(int(seed))

    for _epoch in range(int(unsup_epochs)):
        encoder.train()
        disc.train()

        perm = torch.randperm(n_nodes, device=X_t.device)
        x_corrupt = X_t[perm]

        h_pos = encoder(X_t, edge_index_t, edge_weight_t, n_nodes=n_nodes)
        h_neg = encoder(x_corrupt, edge_index_t, edge_weight_t, n_nodes=n_nodes)

        s = torch.sigmoid(h_pos.mean(dim=0))

        logits_pos = disc(h_pos, s)
        logits_neg = disc(h_neg, s)

        lbl_pos = torch.ones_like(logits_pos)
        lbl_neg = torch.zeros_like(logits_neg)

        loss = bce(logits_pos, lbl_pos) + bce(logits_neg, lbl_neg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    with torch.no_grad():
        emb = encoder(X_t, edge_index_t, edge_weight_t, n_nodes=n_nodes).detach()
    return emb.cpu().numpy().astype(np.float32, copy=False)


@dataclass
class GraphDGIStep:
    """Learn DGI embeddings from graph edges and store as features.X."""

    embedding_dim: int = 512
    hidden_dim: int = 512
    dropout: float = 0.0
    unsup_epochs: int = 100
    unsup_lr: float = 0.001
    add_self_loops: bool = True
    device: str = "cpu"

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        edge_index = store.get("graph.edge_index")
        if edge_index is None:
            raise PreprocessValidationError("graph.dgi requires graph.edge_index")

        try:
            X = get_X(store)
        except KeyError as exc:
            raise PreprocessValidationError("graph.dgi requires raw.X or features.X") from exc
        try:
            X_np = np.asarray(X, dtype=np.float32)
        except Exception as exc:
            raise PreprocessValidationError("graph.dgi requires numeric features") from exc
        X_np = _ensure_2d(X_np)

        n_nodes = int(X_np.shape[0])
        edge_index_np = _as_edge_index(edge_index, n_nodes=n_nodes)

        edge_weight = store.get("graph.edge_weight")
        if edge_weight is None:
            edge_weight_np = np.ones((edge_index_np.shape[1],), dtype=np.float32)
        else:
            edge_weight_np = _as_edge_weight(edge_weight, n_edges=edge_index_np.shape[1])

        if int(self.embedding_dim) <= 0:
            raise PreprocessValidationError("embedding_dim must be > 0")
        if int(self.hidden_dim) <= 0:
            raise PreprocessValidationError("hidden_dim must be > 0")
        if int(self.unsup_epochs) <= 0:
            raise PreprocessValidationError("unsup_epochs must be > 0")
        if float(self.unsup_lr) <= 0:
            raise PreprocessValidationError("unsup_lr must be > 0")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise PreprocessValidationError("dropout must be in [0, 1)")

        seed = int(rng.integers(0, 1 << 31))

        emb = _train_dgi(
            X_np,
            edge_index_np,
            edge_weight_np,
            embedding_dim=int(self.embedding_dim),
            hidden_dim=int(self.hidden_dim),
            dropout=float(self.dropout),
            unsup_epochs=int(self.unsup_epochs),
            unsup_lr=float(self.unsup_lr),
            add_self_loops=bool(self.add_self_loops),
            device=str(self.device),
            seed=seed,
        )
        return {"features.X": emb}
