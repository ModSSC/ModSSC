from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

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
    add_self_loops: bool,
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

    if add_self_loops:
        adj = adj + np.eye(n_nodes, dtype=np.float32)

    return adj


def _normalize_adjacency(adj: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    deg = adj.sum(axis=1).astype(np.float32, copy=False)
    deg = np.maximum(deg, eps)
    deg_inv_sqrt = np.power(deg, -0.5, dtype=np.float32)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


class _DenseGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        dims = [int(in_channels)]
        for _ in range(max(0, int(num_layers) - 2)):
            dims.append(int(hidden_dim))
        if int(num_layers) > 1:
            dims.append(int(hidden_dim))
        dims.append(int(out_channels))

        self.lins = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(len(dims) - 1)]
        )

    def forward(self, adj_norm: Any, x: Any) -> Any:
        z = x
        for idx, lin in enumerate(self.lins):
            z = torch.nn.functional.dropout(z, p=self.dropout, training=self.training)
            z = adj_norm @ z
            z = lin(z)
            if idx < len(self.lins) - 1:
                z = torch.relu(z)
        return z


class _NGCNNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int,
        gcn_out_dim: int,
        num_layers: int,
        K: int,
        r: int,
        classifier: Literal["fc", "attention"],
        dropout: float,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.r = int(r)
        self.classifier = classifier
        self.gcn_out_dim = int(gcn_out_dim)
        self.n_classes = int(n_classes)

        self.gcns = torch.nn.ModuleList(
            [
                _DenseGCN(
                    in_channels,
                    hidden_dim=hidden_dim,
                    out_channels=gcn_out_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for _ in range(self.K * self.r)
            ]
        )

        if self.classifier == "fc":
            self.fc = torch.nn.Linear(self.K * self.r * self.gcn_out_dim, self.n_classes)
            self.att_logits = None
            self.out_lin = None
        else:
            self.fc = None
            self.att_logits = torch.nn.Parameter(torch.zeros(self.K * self.r))
            self.out_lin = (
                torch.nn.Linear(self.gcn_out_dim, self.n_classes)
                if self.gcn_out_dim != self.n_classes
                else None
            )

    def forward(self, adj_powers: list[Any], x: Any) -> Any:
        outputs = []
        idx = 0
        for k in range(self.K):
            adj = adj_powers[k]
            for _ in range(self.r):
                outputs.append(self.gcns[idx](adj, x))
                idx += 1

        if self.classifier == "fc":
            stacked = torch.cat(outputs, dim=1)
            return self.fc(stacked)

        weights = torch.softmax(self.att_logits, dim=0)
        out = sum(w * o for w, o in zip(weights, outputs, strict=True))
        if self.out_lin is not None:
            out = self.out_lin(out)
        return out


@dataclass(frozen=True)
class NGCNSpec:
    """Hyperparameters for N-GCN."""

    hidden_dim: int = 16
    gcn_layers: int = 2
    K: int = 6
    r: int = 4
    classifier: Literal["fc", "attention"] = "fc"
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 1e-5
    max_epochs: int = 600
    patience: int = 50
    add_self_loops: bool = True
    symmetrize: bool = True


class NGCNMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="n_gcn",
        name="N-GCN",
        year=2020,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="N-GCN: Multi-scale Graph Convolution for Semi-supervised Node Classification",
        paper_pdf="https://arxiv.org/abs/1802.08888",
        official_code=None,
    )

    def __init__(self, spec: NGCNSpec | None = None) -> None:
        self.spec = spec or NGCNSpec()
        self._model: Any | None = None
        self._device: Any | None = None
        self._prep_cache: dict[str, Any] = {}
        self._adj_powers: list[Any] = []
        self._n_nodes: int | None = None
        self._n_classes: int | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> NGCNMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        if self.spec.K < 1:
            raise ValueError("K must be >= 1")
        if self.spec.r < 1:
            raise ValueError("r must be >= 1")
        if self.spec.gcn_layers < 1:
            raise ValueError("gcn_layers must be >= 1")
        if self.spec.classifier not in {"fc", "attention"}:
            raise ValueError("classifier must be 'fc' or 'attention'")

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
            "N-GCN sizes: n_nodes=%s n_classes=%s train=%s val=%s",
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
            add_self_loops=self.spec.add_self_loops,
        )
        adj_norm = _normalize_adjacency(adj)

        device_t = torch.device(self._device)
        adj_norm_t = torch.as_tensor(adj_norm, device=device_t, dtype=torch.float32)
        P = torch.eye(prep.n_nodes, device=device_t, dtype=torch.float32)
        self._adj_powers = []
        for _ in range(self.spec.K):
            self._adj_powers.append(P)
            P = adj_norm_t @ P

        gcn_out_dim = prep.n_classes
        self._model = _NGCNNet(
            prep.X.shape[1],
            hidden_dim=self.spec.hidden_dim,
            gcn_out_dim=gcn_out_dim,
            num_layers=self.spec.gcn_layers,
            K=self.spec.K,
            r=self.spec.r,
            classifier=self.spec.classifier,
            dropout=self.spec.dropout,
            n_classes=prep.n_classes,
        ).to(device_t)

        train_fullbatch(
            model=self._model,
            forward_fn=lambda: self._model(self._adj_powers, prep.X),
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
            raise RuntimeError("NGCNMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=False,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        if self._n_nodes is not None and prep.n_nodes != self._n_nodes:
            raise ValueError(f"N-GCN was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}")

        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._adj_powers, prep.X)
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
