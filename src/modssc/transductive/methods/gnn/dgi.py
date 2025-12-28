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


class _GCNConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        x = self.lin(x)
        return spmm(edge_index, edge_weight, x, n_nodes=n_nodes)


class _GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = _GCNConv(in_channels, hidden_dim)
        self.conv2 = _GCNConv(hidden_dim, out_dim)

    def forward(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.conv1(x, edge_index, edge_weight, n_nodes=n_nodes))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight, n_nodes=n_nodes)
        return x


class _Discriminator(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, h: Any, s: Any) -> Any:
        # h: (N, D), s: (D,)
        ws = self.weight(s)  # (D,)
        return (h * ws).sum(dim=-1)  # (N,)


@dataclass(frozen=True)
class DGISpec:
    """Hyperparameters for DGI + linear classifier."""

    embedding_dim: int = 512
    hidden_dim: int = 512
    dropout: float = 0.0
    unsup_epochs: int = 100
    unsup_lr: float = 0.001
    classifier_lr: float = 0.1
    classifier_weight_decay: float = 0.0
    classifier_max_epochs: int = 200
    classifier_patience: int = 50
    add_self_loops: bool = True


class DGIMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="dgi",
        name="DGI",
        year=2019,
        family="self_supervised",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Deep Graph Infomax",
        paper_pdf="https://arxiv.org/abs/1809.10341",
        official_code="https://github.com/PetarV-/DGI",
    )

    def __init__(self, spec: DGISpec | None = None) -> None:
        self.spec = spec or DGISpec()
        self._device: Any | None = None
        self._encoder: Any | None = None
        self._disc: Any | None = None
        self._emb: Any | None = None
        self._clf: Any | None = None
        self._n_nodes: int | None = None
        self._n_classes: int | None = None
        self._prep_cache: dict[str, Any] = {}

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> DGIMethod:
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
        n_nodes = prep.n_nodes
        self._n_nodes = n_nodes
        self._n_classes = prep.n_classes
        logger.info(
            "DGI sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            int(prep.val_mask.sum()),
        )

        encoder = _GCNEncoder(
            prep.X.shape[1],
            hidden_dim=self.spec.hidden_dim,
            out_dim=self.spec.embedding_dim,
            dropout=self.spec.dropout,
        ).to(torch.device(self._device))
        disc = _Discriminator(self.spec.embedding_dim).to(torch.device(self._device))

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(disc.parameters()), lr=self.spec.unsup_lr
        )
        bce = torch.nn.BCEWithLogitsLoss()

        # determinism
        torch.manual_seed(int(seed))

        for _epoch in range(self.spec.unsup_epochs):
            encoder.train()
            disc.train()

            perm = torch.randperm(n_nodes, device=prep.X.device)
            x_corrupt = prep.X[perm]

            h_pos = encoder(prep.X, prep.edge_index, prep.edge_weight, n_nodes=n_nodes)
            h_neg = encoder(x_corrupt, prep.edge_index, prep.edge_weight, n_nodes=n_nodes)

            s = torch.sigmoid(h_pos.mean(dim=0))  # (D,)

            logits_pos = disc(h_pos, s)
            logits_neg = disc(h_neg, s)

            lbl_pos = torch.ones_like(logits_pos)
            lbl_neg = torch.zeros_like(logits_neg)

            loss = bce(logits_pos, lbl_pos) + bce(logits_neg, lbl_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # embeddings
        encoder.eval()
        with torch.no_grad():
            self._emb = encoder(prep.X, prep.edge_index, prep.edge_weight, n_nodes=n_nodes).detach()

        self._encoder = encoder
        self._disc = disc

        # classifier
        clf = torch.nn.Linear(self.spec.embedding_dim, prep.n_classes).to(
            torch.device(self._device)
        )
        self._clf = clf
        X_emb = self._emb

        train_fullbatch(
            model=clf,
            forward_fn=lambda: clf(X_emb),
            y=prep.y,
            train_mask=prep.train_mask,
            val_mask=prep.val_mask,
            lr=self.spec.classifier_lr,
            weight_decay=self.spec.classifier_weight_decay,
            max_epochs=self.spec.classifier_max_epochs,
            patience=self.spec.classifier_patience,
            seed=seed,
        )
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if self._emb is None or self._clf is None or self._n_nodes is None:
            raise RuntimeError("DGIMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=self.spec.add_self_loops,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        if prep.n_nodes != self._n_nodes:
            raise ValueError(f"DGI was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}")

        self._clf.eval()
        with torch.no_grad():
            logits = self._clf(self._emb)
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
