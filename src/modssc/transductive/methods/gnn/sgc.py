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


def _propagate_features(x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int, k: int) -> Any:
    out = x
    for _ in range(int(k)):
        out = spmm(edge_index, edge_weight, out, n_nodes=n_nodes)
    return out


@dataclass(frozen=True)
class SGCSpec:
    """Hyperparameters for the SGC baseline."""

    k: int = 2
    lr: float = 0.1
    weight_decay: float = 0.0
    max_epochs: int = 200
    patience: int = 50
    add_self_loops: bool = True


class SGCMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="sgc",
        name="SGC",
        year=2019,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Simplifying Graph Convolutional Networks",
        paper_pdf="https://arxiv.org/abs/1902.07153",
        official_code="https://github.com/Tiiiger/SGC",
    )

    def __init__(self, spec: SGCSpec | None = None) -> None:
        self.spec = spec or SGCSpec()
        self._device: Any | None = None
        self._model: Any | None = None
        self._X_prop: Any | None = None
        self._n_nodes: int | None = None
        self._prep_cache: dict[str, Any] = {}

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> SGCMethod:
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
            "SGC sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )

        self._n_nodes = prep.n_nodes
        self._X_prop = _propagate_features(
            prep.X,
            prep.edge_index,
            prep.edge_weight,
            n_nodes=prep.n_nodes,
            k=self.spec.k,
        ).detach()

        model = torch.nn.Linear(self._X_prop.shape[1], prep.n_classes).to(self._device)
        self._model = model

        train_fullbatch(
            model=model,
            forward_fn=lambda: model(self._X_prop),
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
        if self._model is None or self._X_prop is None or self._n_nodes is None:
            raise RuntimeError("SGCMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=self.spec.add_self_loops,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        if prep.n_nodes != self._n_nodes:
            raise ValueError(f"SGC was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}")

        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._X_prop)
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
