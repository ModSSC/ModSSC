from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class SklearnLogRegClassifier(BaseSupervisedClassifier):
    classifier_id = "logreg"
    backend = "sklearn"

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: Literal["lbfgs", "liblinear", "saga", "sag", "newton-cg"] = "lbfgs",
        penalty: str = "l2",
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.solver = str(solver)
        self.penalty = str(penalty)
        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params C=%s max_iter=%s solver=%s penalty=%s seed=%s n_jobs=%s",
            self.C,
            self.max_iter,
            self.solver,
            self.penalty,
            self.seed,
            self.n_jobs,
        )
        sklearn_linear = optional_import(
            "sklearn.linear_model", extra="sklearn", feature="supervised:logreg"
        )
        LogisticRegression = sklearn_linear.LogisticRegression

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        # Only pass penalty if it's not the default 'l2' to avoid sklearn deprecation warning
        kwargs = {
            "C": float(self.C),
            "max_iter": int(self.max_iter),
            "solver": str(self.solver),
            "n_jobs": self.n_jobs,
            "random_state": None if self.seed is None else int(self.seed),
        }
        if str(self.penalty) != "l2":
            kwargs["penalty"] = str(self.penalty)

        model = LogisticRegression(**kwargs)
        model.fit(X2, y_enc)
        self._model = model

        self._fit_result = FitResult(
            n_samples=int(X2.shape[0]),
            n_features=int(X2.shape[1]),
            n_classes=int(self.n_classes_),
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        X2 = ensure_2d(X)
        proba = self._model.predict_proba(X2)
        return np.asarray(proba, dtype=np.float32)

    def predict_scores(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        X2 = ensure_2d(X)
        if hasattr(self._model, "decision_function"):
            scores = self._model.decision_function(X2)
            scores = np.asarray(scores, dtype=np.float32)
            if scores.ndim == 1:
                # binary -> (n, 2)
                scores = np.stack([-scores, scores], axis=1)
            return scores
        return self.predict_proba(X2)

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        X2 = ensure_2d(X)
        pred_enc = self._model.predict(X2)
        return self._decode(np.asarray(pred_enc, dtype=np.int64))
