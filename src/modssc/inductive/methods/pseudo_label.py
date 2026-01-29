from __future__ import annotations

import logging
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
    select_confident,
    select_confident_torch,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PseudoLabelSpec(BaseClassifierSpec):
    max_iter: int = 10
    confidence_threshold: float = 0.95
    max_new_labels: int | None = None
    min_new_labels: int = 1


class PseudoLabelMethod(InductiveMethod):
    """Classic pseudo-labeling with iterative refinement (CPU/GPU)."""

    info = MethodInfo(
        method_id="pseudo_label",
        name="Pseudo Label",
        year=2019,
        family="classic",
        supports_gpu=True,
        paper_title="Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks",
        paper_pdf="https://arxiv.org/abs/1905.12265",
        official_code="",
    )

    def __init__(self, spec: PseudoLabelSpec | None = None) -> None:
        self.spec = spec or PseudoLabelSpec()
        self._clf: Any | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> PseudoLabelMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
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
            logger.info(
                "Pseudo-label sizes: n_labeled=%s n_unlabeled=%s",
                int(X_l.shape[0]),
                int(X_u.shape[0]),
            )

            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            clf = build_classifier(self.spec, seed=seed)

            X_u_curr = X_u
            iter_count = 0
            while iter_count < int(self.spec.max_iter):
                clf.fit(X_l, y_l)

                if X_u_curr.shape[0] == 0:
                    break

                scores = predict_scores(clf, X_u_curr, backend=backend)
                idx = select_confident(
                    scores,
                    threshold=float(self.spec.confidence_threshold),
                    max_new=self.spec.max_new_labels,
                )
                logger.debug(
                    "Pseudo-label iter=%s accepted=%s remaining=%s",
                    iter_count,
                    int(idx.size),
                    int(X_u_curr.shape[0]),
                )
                if idx.size < int(self.spec.min_new_labels):
                    break

                y_u = np.asarray(clf.predict(X_u_curr[idx]))
                X_l = np.concatenate([X_l, X_u_curr[idx]], axis=0)
                y_l = np.concatenate([y_l, y_u], axis=0)

                keep = np.ones((X_u_curr.shape[0],), dtype=bool)
                keep[idx] = False
                X_u_curr = X_u_curr[keep]

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
            "Pseudo-label sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u)),
        )

        clf = build_classifier(self.spec, seed=seed)

        X_u_curr = X_u
        iter_count = 0
        while iter_count < int(self.spec.max_iter):
            clf.fit(X_l, y_l)

            if int(get_torch_len(X_u_curr)) == 0:
                break

            scores = predict_scores(clf, X_u_curr, backend=backend)
            idx = select_confident_torch(
                scores,
                threshold=float(self.spec.confidence_threshold),
                max_new=self.spec.max_new_labels,
            )
            logger.debug(
                "Pseudo-label iter=%s accepted=%s remaining=%s",
                iter_count,
                int(idx.numel()),
                int(get_torch_len(X_u_curr)),
            )
            if int(idx.numel()) < int(self.spec.min_new_labels):
                break

            x_u_sel = slice_data(X_u_curr, idx)
            y_u = clf.predict(x_u_sel)
            X_l = concat_data([X_l, x_u_sel])
            y_l = torch.cat([y_l, y_u], dim=0)

            mask = torch.ones(
                (int(get_torch_len(X_u_curr)),),
                dtype=torch.bool,
                device=get_torch_device(X_u_curr),
            )
            mask[idx] = False
            X_u_curr = slice_data(X_u_curr, mask)

            iter_count += 1

        clf.fit(X_l, y_l)
        self._clf = clf
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("PseudoLabelMethod is not fitted yet. Call fit() first.")
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
            raise RuntimeError("PseudoLabelMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict input backend mismatch.")
        return self._clf.predict(X)
