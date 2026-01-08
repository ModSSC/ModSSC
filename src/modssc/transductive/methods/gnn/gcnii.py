from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.optional import optional_import

from .common import normalize_device_name, prepare_data_cached, spmm, torch, train_fullbatch

logger = logging.getLogger(__name__)


class _GCNIINet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        alpha: float,
        lam: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.dropout = float(dropout)

        self.lin_in = torch.nn.Linear(in_channels, self.hidden_dim)
        self.lins = torch.nn.ModuleList(
            [torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.lin_out = torch.nn.Linear(self.hidden_dim, self.out_channels)

    def forward(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.lin_in(x))
        x0 = x

        for i, lin in enumerate(self.lins, start=1):
            x = (1.0 - self.alpha) * spmm(
                edge_index, edge_weight, x, n_nodes=n_nodes
            ) + self.alpha * x0

            beta = math.log(self.lam / float(i) + 1.0)
            x = (1.0 - beta) * x + beta * lin(x)

            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.lin_out(x)
        return x


@dataclass(frozen=True)
class GCNIISpec:
    """Hyperparameters for GCNII."""

    hidden_dim: int = 64
    num_layers: int = 16
    alpha: float = 0.1
    lam: float = 0.5
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 50
    add_self_loops: bool = True


class GCNIIMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="gcnii",
        name="GCNII",
        year=2020,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Simple and Deep Graph Convolutional Networks",
        paper_pdf="https://arxiv.org/abs/2007.02133",
        official_code="https://github.com/chennnM/GCNII",
    )

    def __init__(self, spec: GCNIISpec | None = None) -> None:
        self.spec = spec or GCNIISpec()
        self._model: Any | None = None
        self._device: Any | None = None
        self._prep_cache: dict[str, Any] = {}

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> GCNIIMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        self._device = normalize_device_name(device)
        prep = prepare_data_cached(
            data,
            device=self._device,
            add_self_loops=self.spec.add_self_loops,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        val_count = int(prep.val_mask.sum()) if prep.val_mask is not None else None
        logger.info(
            "GCNII sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )

        self._model = _GCNIINet(
            prep.X.shape[1],
            hidden_dim=self.spec.hidden_dim,
            out_channels=prep.n_classes,
            num_layers=self.spec.num_layers,
            alpha=self.spec.alpha,
            lam=self.spec.lam,
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
            raise RuntimeError("GCNIIMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=self.spec.add_self_loops,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        self._model.eval()
        with torch.no_grad():
            logits = self._model(prep.X, prep.edge_index, prep.edge_weight, n_nodes=prep.n_nodes)
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
