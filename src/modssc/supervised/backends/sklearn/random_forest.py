from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class SklearnRandomForestClassifier(BaseSupervisedClassifier):
    classifier_id = "random_forest"
    backend = "sklearn"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        max_features: str | int | float | None = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        class_weight: str | dict[str, float] | None = None,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.bootstrap = bool(bootstrap)
        self.class_weight = class_weight
        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params n_estimators=%s max_depth=%s max_features=%s min_samples_split=%s "
            "min_samples_leaf=%s bootstrap=%s class_weight=%s seed=%s n_jobs=%s",
            self.n_estimators,
            self.max_depth,
            self.max_features,
            self.min_samples_split,
            self.min_samples_leaf,
            self.bootstrap,
            self.class_weight,
            self.seed,
            self.n_jobs,
        )
        sklearn_ensemble = optional_import(
            "sklearn.ensemble", extra="sklearn", feature="supervised:random_forest"
        )
        RandomForestClassifier = sklearn_ensemble.RandomForestClassifier

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        model = RandomForestClassifier(
            n_estimators=int(self.n_estimators),
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=int(self.min_samples_split),
            min_samples_leaf=int(self.min_samples_leaf),
            bootstrap=bool(self.bootstrap),
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
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
