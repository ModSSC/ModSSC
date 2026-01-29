from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from statistics import NormalDist
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
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


def _z_value(confidence_level: float) -> float:
    if not (0.0 < float(confidence_level) < 1.0):
        raise InductiveValidationError("confidence_level must be in (0, 1).")
    return NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0)


def _accuracy_confidence_interval(
    correct: int, total: int, *, confidence_level: float
) -> tuple[float, float]:
    if total <= 0:
        raise InductiveValidationError("Cannot compute confidence interval with total=0.")
    p_hat = float(correct) / float(total)
    z = _z_value(confidence_level)
    se = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / float(total))
    lo = max(0.0, p_hat - z * se)
    hi = min(1.0, p_hat + z * se)
    return lo, hi


def _confidence_interval_numpy(
    y_true: np.ndarray, y_pred: np.ndarray, *, confidence_level: float
) -> tuple[float, float]:
    correct = int(np.sum(y_true == y_pred))
    return _accuracy_confidence_interval(
        correct, int(y_true.shape[0]), confidence_level=confidence_level
    )


def _confidence_interval_torch(y_true: Any, y_pred: Any, *, confidence_level: float):
    correct = int((y_true == y_pred).sum().item())
    return _accuracy_confidence_interval(
        correct, int(y_true.numel()), confidence_level=confidence_level
    )


def _resolve_classifier_specs(spec: DemocraticCoLearningSpec) -> list[BaseClassifierSpec]:
    if spec.classifier_specs is not None:
        specs = list(spec.classifier_specs)
        if len(specs) < 3:
            raise InductiveValidationError("DemocraticCoLearning requires at least 3 learners.")
        return specs
    if int(spec.n_learners) < 3:
        raise InductiveValidationError("n_learners must be >= 3.")
    return [
        BaseClassifierSpec(
            classifier_id=spec.classifier_id,
            classifier_backend=spec.classifier_backend,
            classifier_params=spec.classifier_params,
        )
        for _ in range(int(spec.n_learners))
    ]


def _resolve_classes_numpy(clfs: list[Any], y_l: np.ndarray) -> np.ndarray:
    classes = None
    for clf in clfs:
        c = getattr(clf, "classes_", None)
        if c is None:
            continue
        c = np.asarray(c)
        if classes is None:
            classes = c
        elif not np.array_equal(classes, c):
            raise InductiveValidationError(
                "DemocraticCoLearning classifiers disagree on class labels."
            )
    if classes is None:
        classes = np.unique(y_l)
    return np.asarray(classes)


def _resolve_classes_torch(clfs: list[Any], y_l: Any):
    torch = optional_import("torch", extra="inductive-torch")
    classes_t = torch.unique(y_l, sorted=True)
    classes_np = None
    for clf in clfs:
        c_t = getattr(clf, "classes_t_", None)
        if c_t is not None and not torch.equal(c_t.to(classes_t.device), classes_t):
            raise InductiveValidationError(
                "DemocraticCoLearning classifiers disagree on class labels."
            )
        c_np = getattr(clf, "classes_", None)
        if c_np is not None:
            c_np = np.asarray(c_np)
            if classes_np is None:
                classes_np = c_np
            elif not np.array_equal(classes_np, c_np):
                raise InductiveValidationError(
                    "DemocraticCoLearning classifiers disagree on class labels."
                )
    if classes_np is not None and not np.array_equal(classes_np, classes_t.detach().cpu().numpy()):
        raise InductiveValidationError("DemocraticCoLearning classifiers disagree on class labels.")
    return classes_t


def _encode_predictions_numpy(preds: list[np.ndarray], classes: np.ndarray) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(classes.tolist())}
    idx_all = []
    for pred in preds:
        pred = np.asarray(pred).reshape(-1)
        idx = np.vectorize(mapping.get, otypes=[int])(pred)
        idx_all.append(idx)
    return np.stack(idx_all, axis=0)


def _encode_predictions_torch(preds: list[Any], classes_t: Any):
    torch = optional_import("torch", extra="inductive-torch")
    idx_all = []
    for pred in preds:
        if pred.ndim != 1:
            pred = pred.reshape(-1)
        if pred.dtype != classes_t.dtype:
            pred = pred.to(classes_t.dtype)
        idx_all.append(torch.searchsorted(classes_t, pred))
    return torch.stack(idx_all, dim=0)


def _weighted_majority_numpy(
    preds_idx: np.ndarray, weights: np.ndarray, *, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    n_learners, n_samples = preds_idx.shape
    scores = np.zeros((n_samples, n_classes), dtype=np.float64)
    row_idx = np.arange(n_samples)
    for i in range(n_learners):
        scores[row_idx, preds_idx[i]] += float(weights[i])
    majority_idx = scores.argmax(axis=1)
    if n_classes <= 1:
        return majority_idx, np.ones((n_samples,), dtype=bool)
    max_vals = scores.max(axis=1)
    second_vals = np.partition(scores, -2, axis=1)[:, -2]
    return majority_idx, max_vals > second_vals


def _weighted_majority_torch(preds_idx: Any, weights: Any, *, n_classes: int):
    torch = optional_import("torch", extra="inductive-torch")
    n_learners, n_samples = preds_idx.shape
    one_hot = torch.nn.functional.one_hot(preds_idx, num_classes=int(n_classes)).to(
        dtype=weights.dtype
    )
    scores = (one_hot * weights.view(n_learners, 1, 1)).sum(dim=0)
    majority_idx = scores.argmax(dim=1)
    if int(n_classes) <= 1:
        majority_ok = torch.ones((n_samples,), dtype=torch.bool, device=preds_idx.device)
        return majority_idx, majority_ok
    top2 = torch.topk(scores, k=2, dim=1).values
    majority_ok = top2[:, 0] > top2[:, 1]
    return majority_idx, majority_ok


def _combine_scores_numpy(
    preds_idx: np.ndarray, weights: np.ndarray, *, n_classes: int, min_confidence: float
) -> np.ndarray:
    n_learners, n_samples = preds_idx.shape
    eligible = weights > float(min_confidence)
    if not np.any(eligible):
        eligible = np.ones_like(eligible, dtype=bool)
    scores = np.zeros((n_samples, n_classes), dtype=np.float64)
    counts = np.zeros((n_samples, n_classes), dtype=np.float64)
    row_idx = np.arange(n_samples)
    for i in range(n_learners):
        if not eligible[i]:
            continue
        scores[row_idx, preds_idx[i]] += float(weights[i])
        counts[row_idx, preds_idx[i]] += 1.0
    avg = scores / np.maximum(counts, 1.0)
    corr = (counts + 0.5) / (counts + 1.0)
    out = avg * corr
    out[counts == 0] = 0.0
    return out


def _combine_scores_torch(preds_idx: Any, weights: Any, *, n_classes: int, min_confidence: float):
    torch = optional_import("torch", extra="inductive-torch")
    n_learners, n_samples = preds_idx.shape
    eligible = weights > float(min_confidence)
    if not bool(eligible.any()):
        eligible = torch.ones_like(eligible, dtype=torch.bool)
    one_hot = torch.nn.functional.one_hot(preds_idx, num_classes=int(n_classes)).to(
        dtype=weights.dtype
    )
    weights_eff = weights * eligible.to(weights.dtype)
    scores = (one_hot * weights_eff.view(n_learners, 1, 1)).sum(dim=0)
    counts = (one_hot * eligible.to(weights.dtype).view(n_learners, 1, 1)).sum(dim=0)
    avg = scores / torch.where(counts == 0, torch.ones_like(counts), counts)
    corr = (counts + 0.5) / (counts + 1.0)
    out = avg * corr
    out = torch.where(counts == 0, torch.zeros_like(out), out)
    return out


@dataclass(frozen=True)
class DemocraticCoLearningSpec(BaseClassifierSpec):
    max_iter: int = 20
    confidence_level: float = 0.95
    min_confidence: float = 0.5
    n_learners: int = 3
    classifier_specs: tuple[BaseClassifierSpec, ...] | None = None


class DemocraticCoLearningMethod(InductiveMethod):
    """Democratic co-learning with multiple learners (CPU/GPU)."""

    info = MethodInfo(
        method_id="democratic_co_learning",
        name="Democratic Co-Learning",
        year=2004,
        family="classic",
        supports_gpu=True,
        paper_title="Democratic Co-Learning",
        paper_pdf="",
        official_code="",
    )

    def __init__(self, spec: DemocraticCoLearningSpec | None = None) -> None:
        self.spec = spec or DemocraticCoLearningSpec()
        self._clfs: list[Any] = []
        self._backend: str | None = None
        self._weights: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._classes_t: Any | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> DemocraticCoLearningMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        logger.debug("backend=%s", backend)
        specs = _resolve_classifier_specs(self.spec)
        for spec in specs:
            ensure_classifier_backend(spec, backend=backend)

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")
            X_l = np.asarray(ds.X_l)
            X_u = np.asarray(ds.X_u) if ds.X_u is not None else None

            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            clfs = [build_classifier(spec, seed=seed + i) for i, spec in enumerate(specs)]
            if X_u is None or X_u.size == 0:
                for clf in clfs:
                    clf.fit(X_l, y_l)
                self._finalize_numpy(clfs, X_l, y_l)
                logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
                return self

            n_u = int(X_u.shape[0])
            X_l_i = [X_l for _ in clfs]
            y_l_i = [y_l for _ in clfs]
            e_i = [0.0 for _ in range(len(clfs))]
            added_mask = [np.zeros((n_u,), dtype=bool) for _ in range(len(clfs))]

            iter_count = 0
            while iter_count < int(self.spec.max_iter):
                for i, clf in enumerate(clfs):
                    clf.fit(X_l_i[i], y_l_i[i])

                weights = self._weights_from_labeled_numpy(clfs, X_l, y_l)
                classes = _resolve_classes_numpy(clfs, y_l)
                preds = [clf.predict(X_u) for clf in clfs]
                preds_idx = _encode_predictions_numpy(preds, classes)
                majority_idx, majority_ok = _weighted_majority_numpy(
                    preds_idx, weights, n_classes=int(classes.size)
                )
                majority_labels = classes[majority_idx]

                idx_per = []
                for i in range(len(clfs)):
                    mask = majority_ok & (preds_idx[i] != majority_idx) & (~added_mask[i])
                    idx_per.append(np.where(mask)[0])

                lower_bounds = []
                for i, clf in enumerate(clfs):
                    pred_l = np.asarray(clf.predict(X_l_i[i]))
                    lo, _hi = _confidence_interval_numpy(
                        y_l_i[i], pred_l, confidence_level=float(self.spec.confidence_level)
                    )
                    lower_bounds.append(lo)
                avg_lower = float(np.mean(lower_bounds)) if lower_bounds else 0.0
                avg_lower = min(max(avg_lower, 0.0), 1.0)

                changed = False
                for i, idx in enumerate(idx_per):
                    if idx.size == 0:
                        continue
                    n_i = int(y_l_i[i].shape[0])
                    q_i = float(n_i) * (1.0 - 2.0 * (e_i[i] / float(n_i))) ** 2
                    e_prime = (1.0 - avg_lower) * float(idx.size)
                    n_new = n_i + int(idx.size)
                    q_prime = float(n_new) * (1.0 - 2.0 * ((e_i[i] + e_prime) / float(n_new))) ** 2
                    if q_prime > q_i:
                        X_l_i[i] = np.concatenate([X_l_i[i], X_u[idx]], axis=0)
                        y_l_i[i] = np.concatenate([y_l_i[i], majority_labels[idx]], axis=0)
                        added_mask[i][idx] = True
                        e_i[i] += e_prime
                        changed = True

                logger.debug("Democratic co-learning iter=%s changed=%s", iter_count, changed)
                if not changed:
                    break
                iter_count += 1

            for i, clf in enumerate(clfs):
                clf.fit(X_l_i[i], y_l_i[i])

            self._finalize_numpy(clfs, X_l, y_l)
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")

        X_l = ds.X_l
        X_u = ds.X_u
        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")

        clfs = [build_classifier(spec, seed=seed + i) for i, spec in enumerate(specs)]
        if X_u is None or int(get_torch_len(X_u)) == 0:
            for clf in clfs:
                clf.fit(X_l, y_l)
            self._finalize_torch(clfs, X_l, y_l)
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        n_u = int(get_torch_len(X_u))
        X_l_i = [X_l for _ in clfs]
        y_l_i = [y_l for _ in clfs]
        e_i = [0.0 for _ in range(len(clfs))]
        added_mask = [
            torch.zeros((n_u,), dtype=torch.bool, device=get_torch_device(X_l)) for _ in clfs
        ]

        iter_count = 0
        while iter_count < int(self.spec.max_iter):
            for i, clf in enumerate(clfs):
                clf.fit(X_l_i[i], y_l_i[i])

            weights = self._weights_from_labeled_torch(clfs, X_l, y_l)
            classes_t = _resolve_classes_torch(clfs, y_l)
            preds = [clf.predict(X_u) for clf in clfs]
            preds_idx = _encode_predictions_torch(preds, classes_t)
            majority_idx, majority_ok = _weighted_majority_torch(
                preds_idx,
                torch.tensor(weights, device=get_torch_device(X_l), dtype=torch.float32),
                n_classes=int(classes_t.numel()),
            )
            majority_labels = classes_t[majority_idx]

            idx_per = []
            for i in range(len(clfs)):
                mask = majority_ok & (preds_idx[i] != majority_idx) & (~added_mask[i])
                idx_per.append(mask.nonzero(as_tuple=False).reshape(-1))

            lower_bounds = []
            for i, clf in enumerate(clfs):
                pred_l = clf.predict(X_l_i[i])
                lo, _hi = _confidence_interval_torch(
                    y_l_i[i], pred_l, confidence_level=float(self.spec.confidence_level)
                )
                lower_bounds.append(lo)
            avg_lower = float(np.mean(lower_bounds)) if lower_bounds else 0.0
            avg_lower = min(max(avg_lower, 0.0), 1.0)

            changed = False
            for i, idx in enumerate(idx_per):
                if int(idx.numel()) == 0:
                    continue
                n_i = int(y_l_i[i].shape[0])
                q_i = float(n_i) * (1.0 - 2.0 * (e_i[i] / float(n_i))) ** 2
                e_prime = (1.0 - avg_lower) * float(int(idx.numel()))
                n_new = n_i + int(idx.numel())
                q_prime = float(n_new) * (1.0 - 2.0 * ((e_i[i] + e_prime) / float(n_new))) ** 2
                if q_prime > q_i:
                    X_l_i[i] = concat_data([X_l_i[i], slice_data(X_u, idx)])
                    y_l_i[i] = torch.cat([y_l_i[i], majority_labels[idx]], dim=0)
                    added_mask[i][idx] = True
                    e_i[i] += e_prime
                    changed = True

            logger.debug("Democratic co-learning iter=%s changed=%s", iter_count, changed)
            if not changed:
                break
            iter_count += 1

        for i, clf in enumerate(clfs):
            clf.fit(X_l_i[i], y_l_i[i])

        self._finalize_torch(clfs, X_l, y_l)
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def _weights_from_labeled_numpy(self, clfs: list[Any], X_l: Any, y_l: Any) -> np.ndarray:
        weights = []
        for clf in clfs:
            pred = np.asarray(clf.predict(X_l))
            lo, hi = _confidence_interval_numpy(
                np.asarray(y_l), pred, confidence_level=float(self.spec.confidence_level)
            )
            weights.append((lo + hi) / 2.0)
        return np.asarray(weights, dtype=np.float64)

    def _weights_from_labeled_torch(self, clfs: list[Any], X_l: Any, y_l: Any) -> np.ndarray:
        weights = []
        for clf in clfs:
            pred = clf.predict(X_l)
            lo, hi = _confidence_interval_torch(
                y_l, pred, confidence_level=float(self.spec.confidence_level)
            )
            weights.append((lo + hi) / 2.0)
        return np.asarray(weights, dtype=np.float64)

    def _finalize_numpy(self, clfs: list[Any], X_l: np.ndarray, y_l: np.ndarray) -> None:
        self._weights = self._weights_from_labeled_numpy(clfs, X_l, y_l)
        self._classes = _resolve_classes_numpy(clfs, y_l)
        self._classes_t = None
        self._clfs = clfs
        self._backend = "numpy"

    def _finalize_torch(self, clfs: list[Any], X_l: Any, y_l: Any) -> None:
        self._weights = self._weights_from_labeled_torch(clfs, X_l, y_l)
        self._classes_t = _resolve_classes_torch(clfs, y_l)
        self._classes = self._classes_t.detach().cpu().numpy()
        self._clfs = clfs
        self._backend = "torch"

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self._clfs:
            raise RuntimeError("DemocraticCoLearningMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")
        if self._weights is None:
            raise RuntimeError("DemocraticCoLearningMethod missing weights; fit() was not called.")

        if backend == "numpy":
            weights = np.asarray(self._weights, dtype=np.float64)
            if self._classes is None:
                raise RuntimeError(
                    "DemocraticCoLearningMethod missing classes; fit() was not called."
                )
            preds = [clf.predict(X) for clf in self._clfs]
            preds_idx = _encode_predictions_numpy(preds, self._classes)
            scores = _combine_scores_numpy(
                preds_idx,
                weights,
                n_classes=int(self._classes.size),
                min_confidence=float(self.spec.min_confidence),
            )
            row_sum = scores.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)

        torch = optional_import("torch", extra="inductive-torch")
        weights_t = torch.tensor(self._weights, device=X.device, dtype=torch.float32)
        if self._classes_t is None:
            raise RuntimeError("DemocraticCoLearningMethod missing classes; fit() was not called.")
        preds = [clf.predict(X) for clf in self._clfs]
        preds_idx = _encode_predictions_torch(preds, self._classes_t)
        scores = _combine_scores_torch(
            preds_idx,
            weights_t,
            n_classes=int(self._classes_t.numel()),
            min_confidence=float(self.spec.min_confidence),
        )
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        backend = self._backend or detect_backend(X)
        if backend == "numpy":
            idx = proba.argmax(axis=1)
            if self._classes is None:
                return idx
            return np.asarray(self._classes)[idx]
        idx = proba.argmax(dim=1)
        if self._classes_t is None:
            return idx
        return self._classes_t[idx]
