from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class SklearnSVRBFClassifier(BaseSupervisedClassifier):
    classifier_id = "svm_rbf"
    backend = "sklearn"

    def __init__(
        self,
        *,
        C: float = 1.0,
        gamma: float | str = "scale",
        sigma: float | None = None,
        probability: bool = False,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.C = float(C)
        self.gamma = gamma
        self.sigma = None if sigma is None else float(sigma)
        self.probability = bool(probability)
        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return bool(self.probability)

    def _resolve_gamma(self) -> float | str:
        if self.sigma is None:
            return self.gamma
        # gamma = 1 / (2*sigma^2)
        return 1.0 / (2.0 * float(self.sigma) * float(self.sigma))

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params C=%s gamma=%s sigma=%s probability=%s seed=%s n_jobs=%s",
            self.C,
            self.gamma,
            self.sigma,
            self.probability,
            self.seed,
            self.n_jobs,
        )
        sklearn_svm = optional_import("sklearn.svm", extra="sklearn", feature="supervised:svm_rbf")
        SVC = sklearn_svm.SVC

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        model = SVC(
            kernel="rbf",
            C=float(self.C),
            gamma=self._resolve_gamma(),
            probability=bool(self.probability),
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
        if hasattr(self._model, "decision_function"):
            scores = np.asarray(self._model.decision_function(X2), dtype=np.float32)
            if scores.ndim == 1:
                scores = np.stack([-scores, scores], axis=1)
            return scores
        # fallback
        return self.predict_proba(X2) if self.supports_proba else super().predict_scores(X2)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        if not self.probability:
            return super().predict_proba(X)
        X2 = ensure_2d(X)
        proba = self._model.predict_proba(X2)
        return np.asarray(proba, dtype=np.float32)

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        X2 = ensure_2d(X)
        pred_enc = self._model.predict(X2)
        return self._decode(np.asarray(pred_enc, dtype=np.int64))
