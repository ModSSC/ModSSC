from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import get_torch_len
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


@dataclass(frozen=True)
class SupervisedSpec(BaseClassifierSpec):
    pass


class SupervisedMethod(InductiveMethod):
    """Pure supervised inductive baseline using only labeled samples."""

    info = MethodInfo(
        method_id="supervised",
        name="Supervised",
        family="baseline",
        supports_gpu=True,
    )

    def __init__(self, spec: SupervisedSpec | None = None) -> None:
        self.spec = spec or SupervisedSpec()
        self._clf: Any | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> SupervisedMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            X_l = np.asarray(ds.X_l)
            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")
            y_l = ensure_1d_labels(ds.y_l, name="y_l")
            n_unlabeled = 0 if ds.X_u is None else int(np.asarray(ds.X_u).shape[0])
            logger.info(
                "Supervised sizes: n_labeled=%s n_unlabeled=%s",
                int(X_l.shape[0]),
                n_unlabeled,
            )
            clf = build_classifier(self.spec, seed=seed)
            clf.fit(X_l, np.asarray(y_l))
            self._clf = clf
            self._backend = backend
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        if int(get_torch_len(ds.X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        n_unlabeled = 0 if ds.X_u is None else int(get_torch_len(ds.X_u))
        logger.info(
            "Supervised sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(ds.X_l)),
            n_unlabeled,
        )
        clf = build_classifier(self.spec, seed=seed)
        clf.fit(ds.X_l, y_l)
        self._clf = clf
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("SupervisedMethod is not fitted yet. Call fit() first.")
        backend = detect_backend(X)
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
            raise RuntimeError("SupervisedMethod is not fitted yet. Call fit() first.")
        backend = detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict input backend mismatch.")
        return self._clf.predict(X)
