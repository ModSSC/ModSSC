from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class SklearnGradientBoostingClassifier(BaseSupervisedClassifier):
    classifier_id = "gradient_boosting"
    backend = "sklearn"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        max_features: str | int | float | None = None,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.max_features = max_features
        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params n_estimators=%s learning_rate=%s max_depth=%s subsample=%s max_features=%s "
            "seed=%s n_jobs=%s",
            self.n_estimators,
            self.learning_rate,
            self.max_depth,
            self.subsample,
            self.max_features,
            self.seed,
            self.n_jobs,
        )
        sklearn_ensemble = optional_import(
            "sklearn.ensemble", extra="sklearn", feature="supervised:gradient_boosting"
        )
        GradientBoostingClassifier = sklearn_ensemble.GradientBoostingClassifier

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        model = GradientBoostingClassifier(
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            max_depth=int(self.max_depth),
            subsample=float(self.subsample),
            max_features=self.max_features,
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

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        X2 = ensure_2d(X)
        proba = self._model.predict_proba(X2)
        return np.asarray(proba, dtype=np.float32)

    def predict_scores(self, X: Any) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        X2 = ensure_2d(X)
        pred_enc = self._model.predict(X2)
        return self._decode(np.asarray(pred_enc, dtype=np.int64))
