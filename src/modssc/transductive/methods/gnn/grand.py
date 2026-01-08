from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.optional import optional_import

from .common import normalize_device_name, prepare_data_cached, spmm, torch

logger = logging.getLogger(__name__)


def _propagate(x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int, steps: int) -> Any:
    out = x
    for _ in range(int(steps)):
        out = spmm(edge_index, edge_weight, out, n_nodes=n_nodes)
    return out


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


@dataclass(frozen=True)
class GRANDSpec:
    """Hyperparameters for a lightweight GRAND-style baseline."""

    hidden_dim: int = 64
    mlp_dropout: float = 0.5
    prop_steps: int = 8
    dropnode: float = 0.5
    num_samples: int = 4
    lambda_consistency: float = 1.0
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 50
    add_self_loops: bool = True


class GRANDMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="grand",
        name="GRAND",
        year=2021,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Graph Random Neural Networks for Semi-Supervised Learning on Graphs",
        paper_pdf="https://arxiv.org/abs/2005.11079",
        official_code="https://github.com/THUDM/GRAND",
    )

    def __init__(self, spec: GRANDSpec | None = None) -> None:
        self.spec = spec or GRANDSpec()
        self._device: Any | None = None
        self._model: Any | None = None
        self._edge_index: Any | None = None
        self._edge_weight: Any | None = None
        self._n_nodes: int | None = None
        self._prep_cache: dict[str, Any] = {}

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> GRANDMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        self._device = normalize_device_name(device)
        prep = prepare_data_cached(
            data,
            device=self._device,
            add_self_loops=self.spec.add_self_loops,
            norm_mode="rw",  # random-walk propagation
            cache=self._prep_cache,
        )
        val_count = int(prep.val_mask.sum()) if prep.val_mask is not None else None
        logger.info(
            "GRAND sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )
        self._n_nodes = prep.n_nodes
        self._edge_index = prep.edge_index
        self._edge_weight = prep.edge_weight

        model = _MLP(
            prep.X.shape[1], self.spec.hidden_dim, prep.n_classes, dropout=self.spec.mlp_dropout
        ).to(torch.device(self._device))
        self._model = model

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.spec.lr, weight_decay=self.spec.weight_decay
        )

        best_state: dict[str, Any] | None = None
        best_val = float("inf")
        bad_epochs = 0

        torch.manual_seed(int(seed))

        for _epoch in range(self.spec.max_epochs):
            model.train()

            probs = []
            sup_loss = 0.0
            for _s in range(self.spec.num_samples):
                x_aug = torch.nn.functional.dropout(prep.X, p=self.spec.dropnode, training=True)
                x_prop = _propagate(
                    x_aug,
                    prep.edge_index,
                    prep.edge_weight,
                    n_nodes=prep.n_nodes,
                    steps=self.spec.prop_steps,
                )
                logits = model(x_prop)

                sup_loss = sup_loss + torch.nn.functional.cross_entropy(
                    logits[prep.train_mask], prep.y[prep.train_mask]
                )
                probs.append(torch.softmax(logits, dim=1))

            sup_loss = sup_loss / float(self.spec.num_samples)

            p_stack = torch.stack(probs, dim=0)  # (S, N, C)
            p_bar = p_stack.mean(dim=0).clamp_min(1e-12)

            # KL(p_i || p_bar), averaged over i and nodes
            kl = (
                (
                    p_stack.clamp_min(1e-12)
                    * (torch.log(p_stack.clamp_min(1e-12)) - torch.log(p_bar))
                )
                .sum(dim=2)
                .mean()
            )

            loss = sup_loss + self.spec.lambda_consistency * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # early stopping on val loss (deterministic forward)
            if prep.val_mask is not None and prep.val_mask.any():
                model.eval()
                with torch.no_grad():
                    x_prop = _propagate(
                        prep.X,
                        prep.edge_index,
                        prep.edge_weight,
                        n_nodes=prep.n_nodes,
                        steps=self.spec.prop_steps,
                    )
                    logits = model(x_prop)
                    val_loss = torch.nn.functional.cross_entropy(
                        logits[prep.val_mask], prep.y[prep.val_mask]
                    ).item()

                if val_loss < best_val - 1e-9:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                    logger.debug("GRAND epoch=%s val_loss=%.4f best updated", _epoch, val_loss)
                else:
                    bad_epochs += 1
                    logger.debug(
                        "GRAND epoch=%s val_loss=%.4f bad_epochs=%s/%s",
                        _epoch,
                        val_loss,
                        bad_epochs,
                        self.spec.patience,
                    )
                    if bad_epochs >= self.spec.patience:
                        logger.debug(
                            "GRAND early_stop epoch=%s best_val=%.4f",
                            _epoch,
                            best_val,
                        )
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if (
            self._model is None
            or self._edge_index is None
            or self._edge_weight is None
            or self._n_nodes is None
        ):
            raise RuntimeError("GRANDMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=self.spec.add_self_loops,
            norm_mode="rw",
            cache=self._prep_cache,
        )
        if prep.n_nodes != self._n_nodes:
            raise ValueError(f"GRAND was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}")

        self._model.eval()
        with torch.no_grad():
            x_prop = _propagate(
                prep.X,
                prep.edge_index,
                prep.edge_weight,
                n_nodes=prep.n_nodes,
                steps=self.spec.prop_steps,
            )
            logits = self._model(x_prop)
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
