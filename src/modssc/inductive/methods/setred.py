from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import NormalDist
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    concat_data,
    get_torch_device,
    get_torch_len,
    slice_data,
)
from modssc.inductive.methods.utils import (
    BaseClassifierSpec,
    build_classifier,
    detect_backend,
    ensure_1d_labels,
    ensure_1d_labels_torch,
    ensure_classifier_backend,
    ensure_cpu_device,
    ensure_numpy_data,
    ensure_torch_data,
    predict_scores,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


def _get_torch_x(obj: Any) -> Any:
    if isinstance(obj, dict) and "x" in obj:
        return obj["x"]
    return obj


def _pairwise_distances_numpy(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise InductiveValidationError("X must be 2D for distance computation.")
    norms = np.sum(X * X, axis=1, keepdims=True)
    dist2 = norms + norms.T - 2.0 * (X @ X.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.sqrt(dist2).astype(np.float32, copy=False)


def _build_rng_graph_numpy(X: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    dist = _pairwise_distances_numpy(X)
    n = int(dist.shape[0])
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i, j]
            if np.any((dist[i] < d) & (dist[j] < d)):
                continue
            neighbors[i].append(j)
            neighbors[j].append(i)
    return [np.asarray(sorted(nbrs), dtype=np.int64) for nbrs in neighbors], dist


def _build_knn_graph_numpy(X: np.ndarray, *, k: int) -> tuple[list[np.ndarray], np.ndarray]:
    dist = _pairwise_distances_numpy(X)
    n = int(dist.shape[0])
    k = max(1, min(int(k), max(0, n - 1)))
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        order = np.argsort(dist[i])
        for j in order[1 : k + 1]:
            neighbors[i].add(int(j))
            neighbors[int(j)].add(int(i))
    return [np.asarray(sorted(nbrs), dtype=np.int64) for nbrs in neighbors], dist


def _build_graph_numpy(
    X: np.ndarray,
    *,
    graph_type: str,
    knn_k: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    if graph_type == "rng":
        return _build_rng_graph_numpy(X)
    if graph_type == "knn":
        return _build_knn_graph_numpy(X, k=knn_k)
    raise InductiveValidationError(f"Unknown graph_type: {graph_type!r}.")


def _edge_weights_numpy(
    dist: np.ndarray,
    *,
    mode: str,
    eps: float,
) -> np.ndarray:
    if mode == "uniform":
        return np.ones_like(dist, dtype=np.float32)
    if mode == "distance":
        return dist.astype(np.float32, copy=False)
    if mode == "inverse_distance":
        return (1.0 / (dist + float(eps))).astype(np.float32, copy=False)
    raise InductiveValidationError(f"Unknown edge_weight mode: {mode!r}.")


def _allocate_per_class(total: int, counts: np.ndarray) -> np.ndarray:
    total = int(total)
    if total <= 0:
        return np.zeros_like(counts, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.float64)
    denom = float(counts.sum())
    if denom <= 0.0:
        return np.zeros_like(counts, dtype=np.int64)
    raw = counts / denom * float(total)
    base = np.floor(raw).astype(np.int64)
    remainder = int(total - int(base.sum()))
    if remainder > 0:
        frac = raw - base
        order = np.argsort(frac)[::-1]
        for idx in order:
            if remainder == 0:
                break
            base[int(idx)] += 1
            remainder -= 1
    return base


def _select_candidates_by_class(
    scores: np.ndarray,
    pred: np.ndarray,
    class_labels: np.ndarray,
    class_counts: np.ndarray,
    *,
    max_new: int,
    threshold: float | None,
) -> np.ndarray:
    conf = scores.max(axis=1)
    total_new = min(int(max_new), int(scores.shape[0]))
    if total_new <= 0:
        return np.empty((0,), dtype=np.int64)
    per_class = _allocate_per_class(total_new, class_counts)
    selected: list[int] = []
    for label, k in zip(class_labels, per_class, strict=True):
        if int(k) <= 0:
            continue
        mask = pred == label
        if threshold is not None:
            mask = mask & (conf >= float(threshold))
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        order = np.argsort(conf[idx])[::-1]
        take = idx[order[: int(k)]]
        selected.extend(take.tolist())
    if not selected:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.asarray(selected, dtype=np.int64))


def _filter_setred_numpy(
    X_all: np.ndarray,
    y_all: np.ndarray,
    idx_l0: np.ndarray,
    class_probs: dict[int, float],
    *,
    theta: float,
    graph_type: str,
    knn_k: int,
    edge_weight: str,
    eps: float,
) -> np.ndarray:
    neighbors, dist = _build_graph_numpy(X_all, graph_type=graph_type, knn_k=knn_k)
    z = float(NormalDist().inv_cdf(float(theta)))
    keep = np.zeros((int(idx_l0.shape[0]),), dtype=bool)
    for pos, idx in enumerate(idx_l0.tolist()):
        nbrs = neighbors[int(idx)]
        if nbrs.size == 0:
            keep[int(pos)] = True
            continue
        weights = _edge_weights_numpy(dist[int(idx), nbrs], mode=edge_weight, eps=eps)
        label_i = int(y_all[int(idx)])
        diff = y_all[nbrs] != label_i
        o_i = float(weights[diff].sum())
        sum_w = float(weights.sum())
        sum_w2 = float(np.square(weights).sum())
        p = float(class_probs.get(label_i, 0.0))
        mu = (1.0 - p) * sum_w
        sigma2 = p * (1.0 - p) * sum_w2
        threshold = mu if sigma2 <= 0.0 else mu + z * float(np.sqrt(sigma2))
        keep[int(pos)] = o_i <= threshold
    return keep


@dataclass(frozen=True)
class SetredSpec(BaseClassifierSpec):
    max_iter: int = 10
    pool_size: int | None = None
    max_new_labels: int | None = None
    min_new_labels: int = 1
    confidence_threshold: float | None = None
    theta: float = 0.05
    graph_type: Literal["rng", "knn"] = "rng"
    knn_k: int = 10
    edge_weight: Literal["uniform", "distance", "inverse_distance"] = "uniform"
    eps: float = 1e-12


class SetredMethod(InductiveMethod):
    """Setred: self-training with editing via local cut edge statistics (CPU/GPU)."""

    info = MethodInfo(
        method_id="setred",
        name="Setred",
        year=2005,
        family="classic",
        supports_gpu=True,
        paper_title="SETRED: Self-Training with Editing",
        paper_pdf=None,
        official_code="",
    )

    def __init__(self, spec: SetredSpec | None = None) -> None:
        self.spec = spec or SetredSpec()
        self._clf: Any | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> SetredMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)

        if not (0.0 < float(self.spec.theta) < 1.0):
            raise InductiveValidationError("theta must be in (0, 1).")
        if int(self.spec.min_new_labels) < 0:
            raise InductiveValidationError("min_new_labels must be >= 0.")
        if int(self.spec.knn_k) <= 0:
            raise InductiveValidationError("knn_k must be >= 1.")

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")

            if ds.X_u is None or np.asarray(ds.X_u).size == 0:
                clf = build_classifier(self.spec, seed=seed)
                clf.fit(ds.X_l, y_l)
                self._clf = clf
                self._backend = backend
                logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
                return self

            X_l = np.asarray(ds.X_l)
            X_u = np.asarray(ds.X_u)
            y_l = np.asarray(y_l)
            logger.info(
                "Setred sizes: n_labeled=%s n_unlabeled=%s",
                int(X_l.shape[0]),
                int(X_u.shape[0]),
            )

            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            rng = np.random.default_rng(int(seed))
            clf = build_classifier(self.spec, seed=seed)
            clf.fit(X_l, y_l)

            X_u_curr = X_u
            iter_count = 0
            while iter_count < int(self.spec.max_iter):
                n_u = int(X_u_curr.shape[0])
                if n_u == 0:
                    break

                pool_size = int(self.spec.pool_size) if self.spec.pool_size is not None else n_u
                pool_size = min(pool_size, n_u)
                if pool_size <= 0:
                    break

                if pool_size < n_u:
                    pool_idx = rng.choice(n_u, size=pool_size, replace=False)
                else:
                    pool_idx = np.arange(n_u, dtype=np.int64)
                X_pool = X_u_curr[pool_idx]

                scores = predict_scores(clf, X_pool, backend=backend)
                pred = np.asarray(clf.predict(X_pool))

                classes, counts = np.unique(y_l, return_counts=True)
                total_new = pool_size
                if self.spec.max_new_labels is not None:
                    total_new = min(total_new, int(self.spec.max_new_labels))

                sel_idx = _select_candidates_by_class(
                    scores,
                    pred,
                    classes,
                    counts,
                    max_new=total_new,
                    threshold=self.spec.confidence_threshold,
                )
                if sel_idx.size == 0:
                    logger.debug("Setred iter=%s no candidates selected; stopping.", iter_count)
                    break

                X_l0 = X_pool[sel_idx]
                y_l0 = pred[sel_idx]

                X_all = np.concatenate([X_l, X_l0], axis=0)
                y_all = np.concatenate([y_l, y_l0], axis=0)
                total = float(counts.sum())
                class_probs = {
                    int(cls): float(cnt) / total for cls, cnt in zip(classes, counts, strict=True)
                }
                idx_l0 = np.arange(int(X_l.shape[0]), int(X_all.shape[0]), dtype=np.int64)
                keep_mask = _filter_setred_numpy(
                    X_all,
                    y_all,
                    idx_l0,
                    class_probs,
                    theta=float(self.spec.theta),
                    graph_type=str(self.spec.graph_type),
                    knn_k=int(self.spec.knn_k),
                    edge_weight=str(self.spec.edge_weight),
                    eps=float(self.spec.eps),
                )
                kept = int(keep_mask.sum())
                logger.debug(
                    "Setred iter=%s pool=%s selected=%s kept=%s",
                    iter_count,
                    int(pool_size),
                    int(sel_idx.size),
                    kept,
                )
                if kept < int(self.spec.min_new_labels):
                    break

                X_keep = X_l0[keep_mask]
                y_keep = y_l0[keep_mask]

                X_l = np.concatenate([X_l, X_keep], axis=0)
                y_l = np.concatenate([y_l, y_keep], axis=0)

                accepted_pool_idx = sel_idx[keep_mask]
                accepted_u_idx = pool_idx[accepted_pool_idx]
                keep_u = np.ones((int(X_u_curr.shape[0]),), dtype=bool)
                keep_u[accepted_u_idx] = False
                X_u_curr = X_u_curr[keep_u]

                clf.fit(X_l, y_l)
                iter_count += 1

            clf.fit(X_l, y_l)
            self._clf = clf
            self._backend = backend
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u is None or int(get_torch_len(ds.X_u)) == 0:
            clf = build_classifier(self.spec, seed=seed)
            clf.fit(ds.X_l, y_l)
            self._clf = clf
            self._backend = backend
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        X_l = ds.X_l
        X_u = ds.X_u
        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        logger.info(
            "Setred sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u)),
        )

        rng = np.random.default_rng(int(seed))
        clf = build_classifier(self.spec, seed=seed)
        clf.fit(X_l, y_l)

        X_u_curr = X_u
        iter_count = 0
        while iter_count < int(self.spec.max_iter):
            n_u = int(get_torch_len(X_u_curr))
            if n_u == 0:
                break

            pool_size = int(self.spec.pool_size) if self.spec.pool_size is not None else n_u
            pool_size = min(pool_size, n_u)
            if pool_size <= 0:
                break

            if pool_size < n_u:
                pool_idx = rng.choice(n_u, size=pool_size, replace=False)
            else:
                pool_idx = np.arange(n_u, dtype=np.int64)
            pool_idx_t = torch.tensor(pool_idx, dtype=torch.long, device=get_torch_device(X_u_curr))
            X_pool = slice_data(X_u_curr, pool_idx_t)

            scores = predict_scores(clf, X_pool, backend=backend)
            pred = clf.predict(X_pool)
            scores_np = scores.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            y_l_np = y_l.detach().cpu().numpy()

            classes, counts = np.unique(y_l_np, return_counts=True)
            total_new = pool_size
            if self.spec.max_new_labels is not None:
                total_new = min(total_new, int(self.spec.max_new_labels))

            sel_idx = _select_candidates_by_class(
                scores_np,
                pred_np,
                classes,
                counts,
                max_new=total_new,
                threshold=self.spec.confidence_threshold,
            )
            if sel_idx.size == 0:
                logger.debug("Setred iter=%s no candidates selected; stopping.", iter_count)
                break

            sel_idx_t = torch.tensor(sel_idx, dtype=torch.long, device=get_torch_device(X_u_curr))
            X_l0 = slice_data(X_pool, sel_idx_t)
            y_l0 = pred[sel_idx_t]

            X_l_feat = _get_torch_x(X_l)
            X_l0_feat = _get_torch_x(X_l0)
            X_all_np = np.concatenate(
                [X_l_feat.detach().cpu().numpy(), X_l0_feat.detach().cpu().numpy()],
                axis=0,
            )
            y_all_np = np.concatenate(
                [y_l_np, y_l0.detach().cpu().numpy()],
                axis=0,
            )
            total = float(counts.sum())
            class_probs = {
                int(cls): float(cnt) / total for cls, cnt in zip(classes, counts, strict=True)
            }
            idx_l0 = np.arange(int(get_torch_len(X_l)), int(X_all_np.shape[0]), dtype=np.int64)
            keep_mask_np = _filter_setred_numpy(
                X_all_np,
                y_all_np,
                idx_l0,
                class_probs,
                theta=float(self.spec.theta),
                graph_type=str(self.spec.graph_type),
                knn_k=int(self.spec.knn_k),
                edge_weight=str(self.spec.edge_weight),
                eps=float(self.spec.eps),
            )
            kept = int(keep_mask_np.sum())
            logger.debug(
                "Setred iter=%s pool=%s selected=%s kept=%s",
                iter_count,
                int(pool_size),
                int(sel_idx.size),
                kept,
            )
            if kept < int(self.spec.min_new_labels):
                break

            keep_mask_t = torch.tensor(
                keep_mask_np, dtype=torch.bool, device=get_torch_device(X_u_curr)
            )
            X_keep = slice_data(X_l0, keep_mask_t)
            y_keep = y_l0[keep_mask_t]

            X_l = concat_data([X_l, X_keep])
            y_l = torch.cat([y_l, y_keep], dim=0)

            accepted_pool_idx = sel_idx[keep_mask_np]
            accepted_u_idx = pool_idx[accepted_pool_idx]
            keep_u = torch.ones((n_u,), dtype=torch.bool, device=get_torch_device(X_u_curr))
            if accepted_u_idx.size > 0:
                keep_u[
                    torch.tensor(
                        accepted_u_idx, dtype=torch.long, device=get_torch_device(X_u_curr)
                    )
                ] = False
            X_u_curr = slice_data(X_u_curr, keep_u)

            clf.fit(X_l, y_l)
            iter_count += 1

        clf.fit(X_l, y_l)
        self._clf = clf
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("SetredMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")
        scores = predict_scores(self._clf, X, backend=backend)
        if backend == "numpy":
            row_sum = scores.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)
        torch = optional_import("torch", extra="inductive-torch")
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def predict(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("SetredMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict input backend mismatch.")
        return self._clf.predict(X)
