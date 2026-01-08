from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.optional import optional_import

from .common import normalize_device_name, prepare_data_cached, spmm, torch, train_fullbatch

logger = logging.getLogger(__name__)


class _MLP(torch.nn.Module):
    def __init__(
        self, in_channels: int, hidden_dim: int, out_channels: int, *, dropout: float
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.lin1 = torch.nn.Linear(in_channels, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, out_channels)

    def forward(self, x: Any) -> Any:
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.lin1(x))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class _APPNPNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int,
        out_channels: int,
        k: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mlp = _MLP(in_channels, hidden_dim, out_channels, dropout=dropout)
        self.k = int(k)
        self.alpha = float(alpha)

    def forward(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        h0 = self.mlp(x)
        h = h0
        for _ in range(self.k):
            h = (1.0 - self.alpha) * spmm(
                edge_index, edge_weight, h, n_nodes=n_nodes
            ) + self.alpha * h0
        return h


@dataclass(frozen=True)
class APPNPSpec:
    """Hyperparameters for APPNP."""

    hidden_dim: int = 64
    dropout: float = 0.5
    k: int = 10
    alpha: float = 0.1
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 50
    add_self_loops: bool = True


class APPNPMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="appnp",
        name="APPNP",
        year=2019,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Predict then Propagate: Graph Neural Networks meet Personalized PageRank",
        paper_pdf="https://arxiv.org/abs/1810.05997",
        official_code="https://github.com/klicperajo/ppnp",
    )

    def __init__(self, spec: APPNPSpec | None = None) -> None:
        self.spec = spec or APPNPSpec()
        self._model: Any | None = None
        self._device: Any | None = None
        self._prep_cache: dict[str, Any] = {}

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> APPNPMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        self._device = normalize_device_name(device)
        prep = prepare_data_cached(
            data,
            device=self._device,
            add_self_loops=self.spec.add_self_loops,
            norm_mode="rw",  # APPNP uses random-walk / PPR propagation
            cache=self._prep_cache,
        )
        val_count = int(prep.val_mask.sum()) if prep.val_mask is not None else None
        logger.info(
            "APPNP sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )

        self._model = _APPNPNet(
            prep.X.shape[1],
            hidden_dim=self.spec.hidden_dim,
            out_channels=prep.n_classes,
            k=self.spec.k,
            alpha=self.spec.alpha,
            dropout=self.spec.dropout,
        ).to(self._device)

        train_fullbatch(
            model=self._model,
            forward_fn=lambda: self._model(
                prep.X, prep.edge_index, prep.edge_weight, n_nodes=prep.n_nodes
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
            raise RuntimeError("APPNPMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=self.spec.add_self_loops,
            norm_mode="rw",
            cache=self._prep_cache,
        )

        self._model.eval()
        with torch.no_grad():
            logits = self._model(prep.X, prep.edge_index, prep.edge_weight, n_nodes=prep.n_nodes)
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
