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


class _ChebConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, k: int, bias: bool = True) -> None:
        super().__init__()
        self.k = int(k)
        self.lins = torch.nn.ModuleList(
            [torch.nn.Linear(in_channels, out_channels, bias=False) for _ in range(self.k + 1)]
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        # Approximate lambda_max=2 -> scaled Laplacian operator becomes -D^{-1/2} A D^{-1/2}
        def lap_op(z: Any) -> Any:
            return -spmm(edge_index, edge_weight, z, n_nodes=n_nodes)

        out = self.lins[0](x)
        if self.k >= 1:
            t0 = x
            t1 = lap_op(x)
            out = out + self.lins[1](t1)
        else:
            t0 = x
            t1 = None

        for kk in range(2, self.k + 1):
            assert t1 is not None
            t2 = 2.0 * lap_op(t1) - t0
            out = out + self.lins[kk](t2)
            t0, t1 = t1, t2

        if self.bias is not None:
            out = out + self.bias
        return out


class _ChebNet(torch.nn.Module):
    def __init__(
        self, in_channels: int, *, hidden_dim: int, out_channels: int, k: int, dropout: float
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = _ChebConv(in_channels, hidden_dim, k=k, bias=True)
        self.conv2 = _ChebConv(hidden_dim, out_channels, k=k, bias=True)

    def forward(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.conv1(x, edge_index, edge_weight, n_nodes=n_nodes))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight, n_nodes=n_nodes)
        return x


@dataclass(frozen=True)
class ChebNetSpec:
    """Hyperparameters for a ChebNet-style graph CNN."""

    hidden_dim: int = 32
    k: int = 2
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 50
    add_self_loops: bool = True


class ChebNetMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="chebnet",
        name="ChebNet",
        year=2016,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering",
        paper_pdf="https://arxiv.org/abs/1606.09375",
        official_code="https://github.com/mdeff/cnn_graph",
    )

    def __init__(self, spec: ChebNetSpec | None = None) -> None:
        self.spec = spec or ChebNetSpec()
        self._model: Any | None = None
        self._device: Any | None = None
        self._prep_cache: dict[str, Any] = {}

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> ChebNetMethod:
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
            "ChebNet sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )

        self._model = _ChebNet(
            prep.X.shape[1],
            hidden_dim=self.spec.hidden_dim,
            out_channels=prep.n_classes,
            k=self.spec.k,
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
            raise RuntimeError("ChebNetMethod is not fitted yet. Call fit() first.")

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
