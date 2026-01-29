from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

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

_GROUP_KEYS_U = (
    "group_u",
    "groups_u",
    "group_ids_u",
    "u_group_ids",
    "u_groups",
    "discourse_u",
    "discourse_ids_u",
    "u_discourse_ids",
    "group_ids",
    "groups",
    "discourse_ids",
    "discourse",
)
_GROUP_KEYS_L = (
    "group_l",
    "groups_l",
    "group_ids_l",
    "l_group_ids",
    "l_groups",
    "discourse_l",
    "discourse_ids_l",
    "l_discourse_ids",
)


@dataclass(frozen=True)
class SelfTrainingSpec(BaseClassifierSpec):
    max_iter: int = 10
    confidence_threshold: float | None = 0.95
    max_new_labels: int | None = None
    min_new_labels: int = 1
    use_group_propagation: bool | None = None
    group_key: str | None = None
    group_min_count: int = 2
    group_min_fraction: float = 1.0
    group_confidence_threshold: float | None = None


def _normalize_group_ids_numpy(value: Any, *, n_expected: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise InductiveValidationError(f"{name} must be 1D group ids.")
    if arr.shape[0] != n_expected:
        raise InductiveValidationError(
            f"{name} must have {n_expected} entries, got {arr.shape[0]}."
        )
    return arr


def _normalize_group_ids_torch(value: Any, *, n_expected: int, name: str):
    torch = optional_import("torch", extra="inductive-torch")
    if not isinstance(value, torch.Tensor):
        raise InductiveValidationError(f"{name} must be a torch.Tensor.")
    if value.ndim != 1:
        raise InductiveValidationError(f"{name} must be 1D group ids.")
    if int(value.shape[0]) != n_expected:
        raise InductiveValidationError(
            f"{name} must have {n_expected} entries, got {int(value.shape[0])}."
        )
    if value.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise InductiveValidationError(f"{name} must have an integer dtype.")
    return value


def _resolve_group_ids(
    meta: Mapping[str, Any] | None,
    *,
    group_key: str | None,
    n_expected: int,
    backend: str,
    name: str,
    key_candidates: tuple[str, ...],
) -> Any | None:
    if meta is None:
        return None
    if not isinstance(meta, Mapping):
        raise InductiveValidationError("meta must be a mapping when provided.")
    if group_key is not None:
        if group_key not in meta:
            raise InductiveValidationError(f"meta is missing key {group_key!r}.")
        value = meta[group_key]
        if backend == "numpy":
            return _normalize_group_ids_numpy(value, n_expected=n_expected, name=name)
        return _normalize_group_ids_torch(value, n_expected=n_expected, name=name)
    for key in key_candidates:
        if key not in meta:
            continue
        value = meta[key]
        try:
            if backend == "numpy":
                return _normalize_group_ids_numpy(value, n_expected=n_expected, name=name)
            return _normalize_group_ids_torch(value, n_expected=n_expected, name=name)
        except InductiveValidationError:
            continue
    return None


def _select_candidates_numpy(
    scores: np.ndarray,
    pred: np.ndarray,
    *,
    threshold: float | None,
    max_new: int | None,
    use_group: bool,
    group_u: np.ndarray | None,
    group_l: np.ndarray | None,
    y_l: np.ndarray | None,
    group_min_count: int,
    group_min_fraction: float,
    group_conf_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    conf = scores.max(axis=1)
    if threshold is None:
        direct_mask = np.ones(conf.shape[0], dtype=bool)
    else:
        direct_mask = conf >= float(threshold)
    direct_idx = np.where(direct_mask)[0]

    candidates: dict[int, tuple[Any, float]] = {
        int(i): (pred[int(i)], float(conf[int(i)])) for i in direct_idx
    }
    group_added = 0

    if use_group and group_u is not None:
        if group_conf_threshold is None:
            group_conf_threshold = threshold
        if group_conf_threshold is None:
            group_conf_threshold = -np.inf

        for gid in np.unique(group_u):
            group_idx = np.where(group_u == gid)[0]
            if group_idx.size == 0:
                continue

            votes: list[Any] = []
            vote_conf: list[float] = []

            if group_l is not None and y_l is not None:
                group_l_idx = np.where(group_l == gid)[0]
                if group_l_idx.size:
                    votes.extend(y_l[group_l_idx].tolist())
                    vote_conf.extend([1.0] * int(group_l_idx.size))

            group_direct_idx = group_idx[direct_mask[group_idx]]
            if group_direct_idx.size:
                votes.extend(pred[group_direct_idx].tolist())
                vote_conf.extend(conf[group_direct_idx].tolist())

            if len(votes) < int(group_min_count):
                continue

            labels, counts = np.unique(np.asarray(votes, dtype=object), return_counts=True)
            major_pos = int(counts.argmax())
            major_label = labels[major_pos]
            fraction = float(counts[major_pos]) / float(counts.sum())
            if fraction < float(group_min_fraction):
                continue

            major_conf = float(
                np.mean([c for v, c in zip(votes, vote_conf, strict=False) if v == major_label])
            )
            if major_conf < float(group_conf_threshold):
                continue

            for idx in group_idx.tolist():
                if int(idx) in candidates:
                    continue
                candidates[int(idx)] = (major_label, major_conf)
                group_added += 1

    items = list(candidates.items())
    if not items:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=object), int(direct_idx.size), 0

    items.sort(key=lambda item: item[1][1], reverse=True)
    if max_new is not None and len(items) > int(max_new):
        items = items[: int(max_new)]

    idx = np.asarray([i for i, _ in items], dtype=np.int64)
    labels = np.asarray([label for _, (label, _) in items], dtype=object)
    return idx, labels, int(direct_idx.size), group_added


class SelfTrainingMethod(InductiveMethod):
    """Self-training with optional one-sense-per-group propagation (CPU/GPU)."""

    info = MethodInfo(
        method_id="self_training",
        name="Self Training",
        year=1995,
        family="classic",
        supports_gpu=True,
        paper_title="Unsupervised Word Sense Disambiguation Rivaling Supervised Methods",
        paper_pdf="",
        official_code="",
    )

    def __init__(self, spec: SelfTrainingSpec | None = None) -> None:
        self.spec = spec or SelfTrainingSpec()
        self._clf: Any | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> SelfTrainingMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)

        if self.spec.group_min_count < 1:
            raise InductiveValidationError("group_min_count must be >= 1.")
        if not (0.0 <= float(self.spec.group_min_fraction) <= 1.0):
            raise InductiveValidationError("group_min_fraction must be in [0, 1].")

        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)

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
            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            group_u = _resolve_group_ids(
                ds.meta,
                group_key=self.spec.group_key,
                n_expected=int(X_u.shape[0]),
                backend=backend,
                name="meta[group_u]",
                key_candidates=_GROUP_KEYS_U,
            )
            group_l = _resolve_group_ids(
                ds.meta,
                group_key=None,
                n_expected=int(X_l.shape[0]),
                backend=backend,
                name="meta[group_l]",
                key_candidates=_GROUP_KEYS_L,
            )

            use_group = self.spec.use_group_propagation
            if use_group is None:
                use_group = group_u is not None
            if use_group and group_u is None:
                raise InductiveValidationError(
                    "SelfTraining requires meta group ids for group propagation."
                )

            clf = build_classifier(self.spec, seed=seed)
            X_u_curr = X_u
            group_u_curr = group_u
            iter_count = 0

            while iter_count < int(self.spec.max_iter):
                clf.fit(X_l, y_l)
                if X_u_curr.shape[0] == 0:
                    break

                scores = predict_scores(clf, X_u_curr, backend=backend)
                pred = clf.predict(X_u_curr)
                idx, labels, direct_count, group_added = _select_candidates_numpy(
                    scores,
                    pred,
                    threshold=self.spec.confidence_threshold,
                    max_new=self.spec.max_new_labels,
                    use_group=bool(use_group),
                    group_u=group_u_curr,
                    group_l=group_l,
                    y_l=y_l,
                    group_min_count=self.spec.group_min_count,
                    group_min_fraction=self.spec.group_min_fraction,
                    group_conf_threshold=self.spec.group_confidence_threshold,
                )
                labels = labels.astype(y_l.dtype, copy=False)

                logger.debug(
                    "Self-training iter=%s direct=%s group_added=%s total_new=%s remaining=%s",
                    iter_count,
                    int(direct_count),
                    int(group_added),
                    int(idx.size),
                    int(X_u_curr.shape[0]),
                )
                if int(idx.size) < int(self.spec.min_new_labels):
                    break

                X_l = np.concatenate([X_l, X_u_curr[idx]], axis=0)
                y_l = np.concatenate([y_l, labels], axis=0)

                keep = np.ones((X_u_curr.shape[0],), dtype=bool)
                keep[idx] = False
                X_u_curr = X_u_curr[keep]
                if group_u_curr is not None:
                    group_u_curr = group_u_curr[keep]

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

        group_u_t = _resolve_group_ids(
            ds.meta,
            group_key=self.spec.group_key,
            n_expected=int(get_torch_len(X_u)),
            backend=backend,
            name="meta[group_u]",
            key_candidates=_GROUP_KEYS_U,
        )
        group_l_t = _resolve_group_ids(
            ds.meta,
            group_key=None,
            n_expected=int(get_torch_len(X_l)),
            backend=backend,
            name="meta[group_l]",
            key_candidates=_GROUP_KEYS_L,
        )

        use_group = self.spec.use_group_propagation
        if use_group is None:
            use_group = group_u_t is not None
        if use_group and group_u_t is None:
            raise InductiveValidationError(
                "SelfTraining requires meta group ids for group propagation."
            )

        group_u_curr = group_u_t.detach().cpu().numpy() if group_u_t is not None else None
        group_l = group_l_t.detach().cpu().numpy() if group_l_t is not None else None

        clf = build_classifier(self.spec, seed=seed)
        X_u_curr = X_u
        iter_count = 0

        while iter_count < int(self.spec.max_iter):
            clf.fit(X_l, y_l)
            if int(get_torch_len(X_u_curr)) == 0:
                break

            scores = predict_scores(clf, X_u_curr, backend=backend)
            pred = clf.predict(X_u_curr)

            scores_np = scores.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            y_l_np = y_l.detach().cpu().numpy()

            idx_np, labels_np, direct_count, group_added = _select_candidates_numpy(
                scores_np,
                pred_np,
                threshold=self.spec.confidence_threshold,
                max_new=self.spec.max_new_labels,
                use_group=bool(use_group),
                group_u=group_u_curr,
                group_l=group_l,
                y_l=y_l_np,
                group_min_count=self.spec.group_min_count,
                group_min_fraction=self.spec.group_min_fraction,
                group_conf_threshold=self.spec.group_confidence_threshold,
            )

            logger.debug(
                "Self-training iter=%s direct=%s group_added=%s total_new=%s remaining=%s",
                iter_count,
                int(direct_count),
                int(group_added),
                int(idx_np.size),
                int(get_torch_len(X_u_curr)),
            )
            if int(idx_np.size) < int(self.spec.min_new_labels):
                break

            idx = torch.tensor(idx_np, dtype=torch.long, device=get_torch_device(X_u_curr))
            labels_np = labels_np.astype(y_l_np.dtype, copy=False)
            labels = torch.tensor(labels_np, dtype=y_l.dtype, device=get_torch_device(X_u_curr))

            X_l = concat_data([X_l, slice_data(X_u_curr, idx)])
            y_l = torch.cat([y_l, labels], dim=0)

            mask = torch.ones(
                (int(get_torch_len(X_u_curr)),),
                dtype=torch.bool,
                device=get_torch_device(X_u_curr),
            )
            mask[idx] = False
            X_u_curr = slice_data(X_u_curr, mask)
            if group_u_curr is not None:
                group_u_curr = group_u_curr[mask.detach().cpu().numpy()]

            iter_count += 1

        clf.fit(X_l, y_l)
        self._clf = clf
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("SelfTrainingMethod is not fitted yet. Call fit() first.")
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
            raise RuntimeError("SelfTrainingMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict input backend mismatch.")
        return self._clf.predict(X)
