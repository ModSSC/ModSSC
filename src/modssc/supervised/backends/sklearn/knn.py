from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Literal

from modssc.supervised.backends.sklearn.common import SklearnProbaClassifier
from modssc.supervised.base import FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class SklearnKNNClassifier(SklearnProbaClassifier):
    classifier_id = "knn"
    backend = "sklearn"

    def __init__(
        self,
        *,
        k: int = 5,
        metric: Literal["euclidean", "cosine", "minkowski"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
        seed: int | None = 0,
        n_jobs: int | None = None,
        algorithm: str = "auto",
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.k = int(k)
        self.metric = str(metric)
        self.weights = str(weights)
        self.algorithm = str(algorithm)
        self._model: Any | None = None

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params k=%s metric=%s weights=%s algorithm=%s seed=%s n_jobs=%s",
            self.k,
            self.metric,
            self.weights,
            self.algorithm,
            self.seed,
            self.n_jobs,
        )
        sklearn_neighbors = optional_import(
            "sklearn.neighbors", extra="sklearn", feature="supervised:knn"
        )
        KNeighborsClassifier = sklearn_neighbors.KNeighborsClassifier

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        model = KNeighborsClassifier(
            n_neighbors=int(self.k),
            metric=str(self.metric),
            weights=str(self.weights),
            n_jobs=self.n_jobs,
            algorithm=str(self.algorithm),
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
