from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.optional import optional_import

from .common import (
    _as_edge_index,
    _as_mask,
    _as_numpy,
    add_self_loops_coalesce,
    normalize_device_name,
    normalize_edge_weight,
    prepare_data_cached,
    set_torch_seed,
    spmm,
    torch,
)

logger = logging.getLogger(__name__)


def _drop_features(x: Any, *, p: float, rng: Any) -> Any:
    if p <= 0.0:
        return x
    mask = rng.random((x.shape[1],)) < float(p)
    if not np.any(mask):
        return x
    out = x.clone()
    out[:, torch.as_tensor(mask, device=x.device, dtype=torch.bool)] = 0.0
    return out


def _drop_edges(edge_index: Any, edge_weight: Any, *, p: float, rng: Any) -> tuple[Any, Any]:
    if p <= 0.0:
        return edge_index, edge_weight
    E = int(edge_index.shape[1])
    keep = rng.random((E,)) >= float(p)
    if not np.any(keep):
        return edge_index, edge_weight
    mask = torch.as_tensor(keep, device=edge_index.device, dtype=torch.bool)
    return edge_index[:, mask], edge_weight[mask]


def _build_augmented_view(
    *,
    X: Any,
    edge_index: Any,
    edge_weight: Any,
    drop_edge_p: float,
    drop_feat_p: float,
    n_nodes: int,
    rng: Any,
) -> tuple[Any, Any, Any]:
    X_aug = _drop_features(X, p=drop_feat_p, rng=rng)
    ei, ew = _drop_edges(edge_index, edge_weight, p=drop_edge_p, rng=rng)
    ei, ew = add_self_loops_coalesce(ei, ew, n_nodes=n_nodes, fill_value=1.0)
    ew = normalize_edge_weight(edge_index=ei, edge_weight=ew, n_nodes=n_nodes, mode="sym")
    return X_aug, ei, ew


def _sample_support(
    *,
    y: np.ndarray,
    train_mask: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    label_dict: list[np.ndarray] = []
    for c in range(n_classes):
        idx = np.flatnonzero(train_mask & (y == c))
        if idx.size == 0:
            raise ValueError("GraFN requires at least one labeled node per class.")
        label_dict.append(idx)

    batch_size = min(int(idx.size) for idx in label_dict)
    samples = []
    for idx in label_dict:
        replace = idx.size < batch_size
        samples.append(rng.choice(idx, size=batch_size, replace=replace))

    support_index = np.concatenate(samples, axis=0).astype(np.int64, copy=False)

    label_matrix = np.zeros((batch_size * n_classes, n_classes), dtype=np.float32)
    for c in range(n_classes):
        label_matrix[c * batch_size : (c + 1) * batch_size, c] = 1.0

    return support_index, label_matrix, batch_size


class _GCNConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.act = torch.nn.PReLU()

    def forward(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        x = self.lin(x)
        x = spmm(edge_index, edge_weight, x, n_nodes=n_nodes)
        x = self.bn(x)
        x = self.act(x)
        return x


class _GCNEncoder(torch.nn.Module):
    def __init__(self, layer_sizes: Sequence[int]) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")
        self.layers = torch.nn.ModuleList(
            [_GCNConv(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )

    def forward(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight, n_nodes=n_nodes)
        return x


class _GraFNNet(torch.nn.Module):
    def __init__(self, encoder: _GCNEncoder, *, n_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(encoder.layers[-1].lin.out_features, n_classes, bias=True)

    def encode(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        return self.encoder(x, edge_index, edge_weight, n_nodes=n_nodes)

    def classify(self, x: Any, edge_index: Any, edge_weight: Any, *, n_nodes: int) -> Any:
        z = self.encode(x, edge_index, edge_weight, n_nodes=n_nodes)
        return self.classifier(z)


@dataclass(frozen=True)
class GraFNSpec:
    """Hyperparameters for GraFN."""

    hidden_dims: tuple[int, ...] = (128, 128)
    lr: float = 0.01
    weight_decay: float = 1e-5
    max_epochs: int = 1000
    patience: int = 200
    tau: float = 0.1
    thres: float = 0.9
    lam_consistency: float = 0.5
    lam_nodewise: float = 0.5
    drop_feat_strong: float = 0.5
    drop_edge_strong: float = 0.5
    drop_feat_weak: float = 0.1
    drop_edge_weak: float = 0.1


class GraFNMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="grafn",
        name="GraFN",
        year=2022,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title=(
            "GraFN: Semi-Supervised Node Classification on Graph with Few Labels "
            "via Non-Parametric Distribution Assignment"
        ),
        paper_pdf="https://arxiv.org/abs/2204.01303",
        official_code="https://github.com/Junseok0207/GraFN",
    )

    def __init__(self, spec: GraFNSpec | None = None) -> None:
        self.spec = spec or GraFNSpec()
        self._model: Any | None = None
        self._device: Any | None = None
        self._prep_cache: dict[str, Any] = {}
        self._base_edge_index: Any | None = None
        self._base_edge_weight: Any | None = None
        self._n_nodes: int | None = None
        self._n_classes: int | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> GraFNMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        if self.spec.tau <= 0:
            raise ValueError("tau must be > 0")
        if not (0.0 <= self.spec.thres <= 1.0):
            raise ValueError("thres must be in [0, 1]")

        self._device = normalize_device_name(device)
        set_torch_seed(seed)

        prep = prepare_data_cached(
            data,
            device=self._device,
            add_self_loops=True,
            norm_mode="sym",
            cache=self._prep_cache,
        )

        train_mask_np = _as_mask(prep.train_mask, prep.n_nodes, name="train_mask")
        if not bool(train_mask_np.any()):
            raise ValueError("train_mask is empty")

        self._n_nodes = prep.n_nodes
        self._n_classes = prep.n_classes

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

        device_t = torch.device(self._device)
        X = prep.X
        y = prep.y
        train_mask = torch.as_tensor(train_mask_np, device=device_t, dtype=torch.bool)
        val_mask = prep.val_mask

        edge_index_raw = torch.as_tensor(edge_index_np, device=device_t, dtype=torch.long)
        edge_weight_raw_t = torch.as_tensor(edge_weight_np, device=device_t, dtype=torch.float32)

        self._base_edge_index = prep.edge_index
        self._base_edge_weight = prep.edge_weight

        hidden_dims = (X.shape[1],) + tuple(int(d) for d in self.spec.hidden_dims)
        encoder = _GCNEncoder(hidden_dims)
        self._model = _GraFNNet(encoder, n_classes=prep.n_classes).to(device_t)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=float(self.spec.lr),
            weight_decay=float(self.spec.weight_decay),
        )

        rng = np.random.default_rng(seed)
        best_state: dict[str, Any] | None = None
        best_val = None
        bad_epochs = 0

        y_np = _as_numpy(prep.y).astype(np.int64, copy=False).reshape(-1)

        for _epoch in range(int(self.spec.max_epochs)):
            self._model.train()

            support_index_np, label_matrix_np, _ = _sample_support(
                y=y_np, train_mask=train_mask_np, n_classes=prep.n_classes, rng=rng
            )
            support_index = torch.as_tensor(support_index_np, device=device_t, dtype=torch.long)
            label_matrix = torch.as_tensor(label_matrix_np, device=device_t, dtype=torch.float32)

            rng_state = np.random.default_rng(rng.integers(0, 2**32 - 1))
            X_anchor, ei_anchor, ew_anchor = _build_augmented_view(
                X=X,
                edge_index=edge_index_raw,
                edge_weight=edge_weight_raw_t,
                drop_edge_p=self.spec.drop_edge_strong,
                drop_feat_p=self.spec.drop_feat_strong,
                n_nodes=prep.n_nodes,
                rng=rng_state,
            )
            rng_state = np.random.default_rng(rng.integers(0, 2**32 - 1))
            X_pos, ei_pos, ew_pos = _build_augmented_view(
                X=X,
                edge_index=edge_index_raw,
                edge_weight=edge_weight_raw_t,
                drop_edge_p=self.spec.drop_edge_weak,
                drop_feat_p=self.spec.drop_feat_weak,
                n_nodes=prep.n_nodes,
                rng=rng_state,
            )

            anchor_rep = self._model.encode(X_anchor, ei_anchor, ew_anchor, n_nodes=prep.n_nodes)
            pos_rep = self._model.encode(X_pos, ei_pos, ew_pos, n_nodes=prep.n_nodes)

            anchor_support = anchor_rep[support_index]
            pos_support = pos_rep[support_index]

            probs = _snn(anchor_rep, anchor_support, label_matrix, tau=self.spec.tau)
            with torch.no_grad():
                targets = _snn(pos_rep, pos_support, label_matrix, tau=self.spec.tau)
                values = targets.max(dim=1).values
                include = torch.logical_or(values > float(self.spec.thres), train_mask)
                indices = torch.arange(targets.shape[0], device=device_t)[include]
                targets[targets < 1e-4] *= 0.0
                gt_labels = y[train_mask].unsqueeze(-1)
                gt_matrix = torch.zeros(
                    (int(train_mask.sum()), prep.n_classes), device=device_t, dtype=torch.float32
                )
                gt_matrix.scatter_(1, gt_labels, 1.0)
                targets[train_mask] = gt_matrix
                targets = targets[indices]

            probs = probs[indices]
            consistency_loss = torch.mean(torch.sum(torch.log(probs ** (-targets)), dim=1))

            logits_anchor = self._model.classify(
                X_anchor, ei_anchor, ew_anchor, n_nodes=prep.n_nodes
            )
            logits_pos = self._model.classify(X_pos, ei_pos, ew_pos, n_nodes=prep.n_nodes)
            sup_loss = torch.nn.functional.cross_entropy(logits_anchor[train_mask], y[train_mask])
            sup_loss = sup_loss + torch.nn.functional.cross_entropy(
                logits_pos[train_mask], y[train_mask]
            )
            sup_loss = sup_loss * 0.5

            unsup_loss = (
                2.0
                - 2.0 * torch.nn.functional.cosine_similarity(anchor_rep, pos_rep, dim=-1).mean()
            )

            loss = (
                sup_loss
                + float(self.spec.lam_consistency) * consistency_loss
                + float(self.spec.lam_nodewise) * unsup_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if val_mask is not None and bool(val_mask.any()):
                self._model.eval()
                with torch.no_grad():
                    logits = self._model.classify(
                        X, self._base_edge_index, self._base_edge_weight, n_nodes=prep.n_nodes
                    )
                    pred = logits.argmax(dim=1)
                    val_acc = float((pred[val_mask] == y[val_mask]).float().mean().item())
                if best_val is None or val_acc > best_val:
                    best_val = val_acc
                    best_state = {
                        k: v.detach().clone() for k, v in self._model.state_dict().items()
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(self.spec.patience):
                        break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if self._model is None or self._base_edge_index is None or self._base_edge_weight is None:
            raise RuntimeError("GraFNMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=True,
            norm_mode="sym",
            cache=self._prep_cache,
        )
        if self._n_nodes is not None and prep.n_nodes != self._n_nodes:
            raise ValueError(f"GraFN was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}")

        self._model.eval()
        with torch.no_grad():
            logits = self._model.classify(
                prep.X, self._base_edge_index, self._base_edge_weight, n_nodes=prep.n_nodes
            )
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()


def _snn(query: Any, supports: Any, labels: Any, *, tau: float) -> Any:
    query = torch.nn.functional.normalize(query, dim=1)
    supports = torch.nn.functional.normalize(supports, dim=1)
    scores = query @ supports.T / float(tau)
    weights = torch.softmax(scores, dim=1)
    return weights @ labels
