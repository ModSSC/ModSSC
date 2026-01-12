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
    _as_mask,
    _as_numpy,
    normalize_device_name,
    prepare_data_cached,
    set_torch_seed,
    spmm,
    torch,
)

logger = logging.getLogger(__name__)


def _build_adj_list(edge_index: np.ndarray, *, n_nodes: int, symmetrize: bool) -> list[list[int]]:
    adj = [set() for _ in range(n_nodes)]
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    for s, d in zip(src.tolist(), dst.tolist(), strict=True):
        if s == d:
            continue
        if 0 <= s < n_nodes and 0 <= d < n_nodes:
            adj[s].add(d)
            if symmetrize:
                adj[d].add(s)
    return [sorted(neigh) for neigh in adj]


def _build_hop_edges(adj: list[list[int]], *, n_nodes: int, max_hops: int) -> list[np.ndarray]:
    edges_per_hop: list[list[tuple[int, int]]] = [[] for _ in range(max_hops)]
    for i in range(n_nodes):
        visited = {i}
        frontier = {i}
        for hop in range(max_hops):
            next_set: set[int] = set()
            for node in frontier:
                next_set.update(adj[node])
            next_set -= visited
            visited |= next_set
            for u in next_set:
                edges_per_hop[hop].append((u, i))
            frontier = next_set
    hop_edges = []
    for edges in edges_per_hop:
        if edges:
            hop_edges.append(np.asarray(edges, dtype=np.int64).T)
        else:
            hop_edges.append(np.empty((2, 0), dtype=np.int64))
    return hop_edges


def _row_normalize_weights(edge_index: np.ndarray, *, n_nodes: int) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.empty((0,), dtype=np.float32)
    dst = edge_index[1]
    deg = np.zeros((n_nodes,), dtype=np.float32)
    np.add.at(deg, dst, 1.0)
    w = np.zeros((edge_index.shape[1],), dtype=np.float32)
    mask = deg[dst] > 0
    w[mask] = 1.0 / deg[dst[mask]]
    return w


def _concat_features(feats: list[Any]) -> Any:
    return torch.cat(feats, dim=1)


class _LogisticRegression(torch.nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, num_classes)

    def forward(self, x: Any) -> Any:
        return torch.nn.functional.log_softmax(self.linear(x), dim=1)

    def predict_soft_labels(self, x: Any) -> Any:
        return torch.softmax(self.linear(x), dim=1)

    def predict_temp_soft_labels(self, x: Any, *, temperature: float) -> Any:
        return torch.softmax(self.linear(x) / float(temperature), dim=1)


def _train_lr(
    *,
    step: int,
    X: Any,
    labels: Any,
    train_mask: Any,
    val_mask: Any | None,
    y_val: Any | None,
    prev_model: _LogisticRegression | None,
    spec: GraphHopSpec,
    device: Any,
) -> _LogisticRegression:
    num_feat = int(X.shape[1])
    num_classes = int(labels.shape[1])

    if step <= 1 or prev_model is None:
        model = _LogisticRegression(num_feat, num_classes).to(device)
    else:
        model = prev_model

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(spec.lr), weight_decay=float(spec.weight_decay)
    )

    pseudo_mask = ~train_mask
    num_train = int(train_mask.sum())
    num_pseudo = int(pseudo_mask.sum())

    best_state = None
    prev_loss_val = float("inf")
    bad_epochs = 0

    for _epoch in range(int(spec.max_epochs)):
        model.train()
        optimizer.zero_grad()
        log_prob = model(X)

        if step == 0:
            loss = -(labels[train_mask] * log_prob[train_mask]).sum() / max(num_train, 1)
        else:
            term_train = (labels[train_mask] * log_prob[train_mask]).sum() / max(num_train, 1)
            term_pseudo = (labels[pseudo_mask] * log_prob[pseudo_mask]).sum() / max(
                num_pseudo * num_classes, 1
            )
            term_entropy = (torch.exp(log_prob[pseudo_mask]) * log_prob[pseudo_mask]).sum() / max(
                num_pseudo * num_classes, 1
            )
            loss = -(term_train + float(spec.alpha) * term_pseudo + float(spec.beta) * term_entropy)

        loss.backward()
        optimizer.step()

        if val_mask is None or y_val is None:
            continue

        model.eval()
        with torch.no_grad():
            log_prob_val = model(X[val_mask])
            loss_val = -(y_val * log_prob_val).sum()

        if loss_val - prev_loss_val > 0 or prev_loss_val - loss_val < float(spec.min_delta):
            bad_epochs += 1
        else:
            bad_epochs = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if bad_epochs >= int(spec.patience):
            break
        prev_loss_val = float(loss_val.item())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@dataclass(frozen=True)
class GraphHopSpec:
    """Hyperparameters for GraphHop."""

    hops: int = 2
    max_iter: int = 50
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 1000
    patience: int = 10
    min_delta: float = 1e-2
    temperature: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    weight_temp: tuple[float, ...] | None = None
    weight_pred: tuple[float, ...] | None = None
    symmetrize: bool = True


class GraphHopMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="graphhop",
        name="GraphHop",
        year=2023,
        family="gnn",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="GraphHop: An Enhanced Label Propagation Method for Node Classification",
        paper_pdf="https://arxiv.org/abs/2301.12368",
        official_code="https://github.com/TianXieUSC/GraphHop",
    )

    def __init__(self, spec: GraphHopSpec | None = None) -> None:
        self.spec = spec or GraphHopSpec()
        self._device: Any | None = None
        self._prep_cache: dict[str, Any] = {}
        self._proba: np.ndarray | None = None
        self._n_nodes: int | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> GraphHopMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        if self.spec.hops < 1:
            raise ValueError("hops must be >= 1")
        if self.spec.temperature <= 0:
            raise ValueError("temperature must be > 0")

        self._device = normalize_device_name(device)
        set_torch_seed(seed)

        prep = prepare_data_cached(
            data,
            device=self._device,
            add_self_loops=False,
            norm_mode="rw",
            cache=self._prep_cache,
        )

        train_mask_np = _as_mask(prep.train_mask, prep.n_nodes, name="train_mask")
        if not bool(train_mask_np.any()):
            raise ValueError("train_mask is empty")

        val_mask_np = None
        if prep.val_mask is not None:
            val_mask_np = _as_mask(prep.val_mask, prep.n_nodes, name="val_mask")
            if not bool(val_mask_np.any()):
                val_mask_np = None

        y_np = _as_numpy(prep.y).astype(np.int64, copy=False).reshape(-1)
        if y_np.size == 0:
            raise ValueError("y must contain at least one label")
        n_classes = int(y_np.max()) + 1

        label_counts = np.bincount(y_np[train_mask_np], minlength=n_classes)
        if np.any(label_counts == 0):
            raise ValueError("GraphHop requires at least one labeled node per class.")

        edge_index_np = _as_edge_index(_as_numpy(data.graph.edge_index))
        adj_list = _build_adj_list(
            edge_index_np, n_nodes=prep.n_nodes, symmetrize=self.spec.symmetrize
        )
        hop_edges_np = _build_hop_edges(adj_list, n_nodes=prep.n_nodes, max_hops=self.spec.hops)
        hop_weights_np = [_row_normalize_weights(ei, n_nodes=prep.n_nodes) for ei in hop_edges_np]

        device_t = torch.device(self._device)
        X = prep.X
        y = prep.y
        train_mask = torch.as_tensor(train_mask_np, device=device_t, dtype=torch.bool)
        val_mask = (
            torch.as_tensor(val_mask_np, device=device_t, dtype=torch.bool)
            if val_mask_np is not None
            else None
        )
        y_val = None
        if val_mask is not None:
            y_val_np = np.zeros((int(val_mask.sum()), n_classes), dtype=np.float32)
            labels_val = y_np[val_mask_np]
            y_val_np[np.arange(labels_val.size), labels_val] = 1.0
            y_val = torch.as_tensor(y_val_np, device=device_t, dtype=torch.float32)

        hop_edges = [torch.as_tensor(ei, device=device_t, dtype=torch.long) for ei in hop_edges_np]
        hop_weights = [
            torch.as_tensor(w, device=device_t, dtype=torch.float32) for w in hop_weights_np
        ]

        # Precompute attribute aggregations X_m
        X_hops: list[Any] = [X]
        for m in range(self.spec.hops):
            if hop_edges[m].numel() == 0:
                X_hops.append(torch.zeros_like(X))
            else:
                X_hops.append(spmm(hop_edges[m], hop_weights[m], X, n_nodes=prep.n_nodes))

        X_concat = [_concat_features(X_hops[: m + 2]) for m in range(self.spec.hops)]

        # Initialization (step 0): train LR on labeled nodes only.
        one_hot = torch.zeros((prep.n_nodes, n_classes), device=device_t, dtype=torch.float32)
        one_hot[torch.arange(prep.n_nodes, device=device_t), y] = 1.0

        prev_models: list[_LogisticRegression | None] = [None for _ in range(self.spec.hops)]
        model_init = _train_lr(
            step=0,
            X=X_concat[-1],
            labels=one_hot,
            train_mask=train_mask,
            val_mask=val_mask,
            y_val=y_val,
            prev_model=None,
            spec=self.spec,
            device=device_t,
        )

        H = model_init.predict_soft_labels(X_concat[-1]).detach()
        pseudo_labels = model_init.predict_temp_soft_labels(
            X_concat[-1], temperature=self.spec.temperature
        ).detach()
        pseudo_labels[train_mask] = one_hot[train_mask]

        # Iteration
        weight_temp = (
            np.asarray(self.spec.weight_temp, dtype=np.float32)
            if self.spec.weight_temp is not None
            else np.full((self.spec.hops,), 1.0 / float(self.spec.hops), dtype=np.float32)
        )
        weight_pred = (
            np.asarray(self.spec.weight_pred, dtype=np.float32)
            if self.spec.weight_pred is not None
            else np.full((self.spec.hops,), 1.0 / float(self.spec.hops), dtype=np.float32)
        )
        if weight_temp.shape != (self.spec.hops,):
            raise ValueError("weight_temp must have length equal to hops")
        if weight_pred.shape != (self.spec.hops,):
            raise ValueError("weight_pred must have length equal to hops")

        for step in range(1, int(self.spec.max_iter) + 1):
            H_hops: list[Any] = [H]
            for m in range(self.spec.hops):
                if hop_edges[m].numel() == 0:
                    H_hops.append(torch.zeros_like(H))
                else:
                    H_hops.append(spmm(hop_edges[m], hop_weights[m], H, n_nodes=prep.n_nodes))

            H_concat = [_concat_features(H_hops[: m + 2]) for m in range(self.spec.hops)]

            preds_temp = []
            preds_soft = []
            for m in range(self.spec.hops):
                model = _train_lr(
                    step=step,
                    X=H_concat[m],
                    labels=pseudo_labels,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    y_val=y_val,
                    prev_model=prev_models[m],
                    spec=self.spec,
                    device=device_t,
                )
                prev_models[m] = model
                preds_temp.append(
                    model.predict_temp_soft_labels(
                        H_concat[m], temperature=self.spec.temperature
                    ).detach()
                )
                preds_soft.append(model.predict_soft_labels(H_concat[m]).detach())

            pseudo_labels = sum(
                float(weight_temp[m]) * preds_temp[m] for m in range(self.spec.hops)
            )
            pseudo_labels[train_mask] = one_hot[train_mask]
            H = sum(float(weight_pred[m]) * preds_soft[m] for m in range(self.spec.hops))

        self._proba = H.detach().cpu().numpy()
        self._n_nodes = prep.n_nodes
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if self._proba is None or self._n_nodes is None:
            raise RuntimeError("GraphHopMethod is not fitted yet. Call fit() first.")
        n_nodes = int(np.asarray(data.y).shape[0])
        if n_nodes != self._n_nodes:
            raise ValueError(f"GraphHop was fitted on n={self._n_nodes} nodes, got n={n_nodes}")
        return np.asarray(self._proba, dtype=np.float32)
