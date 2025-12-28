from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class SklearnRidgeClassifier(BaseSupervisedClassifier):
    classifier_id = "ridge"
    backend = "sklearn"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        solver: Literal[
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
            "lbfgs",
        ] = "auto",
        max_iter: int | None = None,
        tol: float = 1e-3,
        class_weight: str | dict[str, float] | None = None,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.alpha = float(alpha)
        self.solver = str(solver)
        self.max_iter = None if max_iter is None else int(max_iter)
        self.tol = float(tol)
        self.class_weight = class_weight
        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return False

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params alpha=%s solver=%s max_iter=%s tol=%s class_weight=%s seed=%s n_jobs=%s",
            self.alpha,
            self.solver,
            self.max_iter,
            self.tol,
            self.class_weight,
            self.seed,
            self.n_jobs,
        )
        sklearn_linear = optional_import(
            "sklearn.linear_model", extra="sklearn", feature="supervised:ridge"
        )
        RidgeClassifier = sklearn_linear.RidgeClassifier

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        model = RidgeClassifier(
            alpha=float(self.alpha),
            solver=str(self.solver),
            max_iter=self.max_iter,
            tol=float(self.tol),
            class_weight=self.class_weight,
            random_state=None if self.seed is None else int(self.seed),
        )
        model.fit(X2, y_enc)
        self._model = model

        self._fit_result = FitResult(
            n_samples=int(X2.shape[0]),
            n_features=int(X2.shape[1]),
            n_classes=int(self.n_classes_),
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def predict_scores(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        X2 = ensure_2d(X)
        scores = self._model.decision_function(X2)
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return scores

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        X2 = ensure_2d(X)
        pred_enc = self._model.predict(X2)
        return self._decode(np.asarray(pred_enc, dtype=np.int64))
