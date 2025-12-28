from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class SklearnLinearSVMClassifier(BaseSupervisedClassifier):
    classifier_id = "linear_svm"
    backend = "sklearn"

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 2000,
        loss: Literal["hinge", "squared_hinge"] = "squared_hinge",
        dual: bool = True,
        class_weight: str | dict[str, float] | None = None,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.loss = str(loss)
        self.dual = bool(dual)
        self.class_weight = class_weight
        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return False

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params C=%s max_iter=%s loss=%s dual=%s class_weight=%s seed=%s n_jobs=%s",
            self.C,
            self.max_iter,
            self.loss,
            self.dual,
            self.class_weight,
            self.seed,
            self.n_jobs,
        )
        sklearn_svm = optional_import(
            "sklearn.svm", extra="sklearn", feature="supervised:linear_svm"
        )
        LinearSVC = sklearn_svm.LinearSVC

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        model = LinearSVC(
            C=float(self.C),
            max_iter=int(self.max_iter),
            loss=str(self.loss),
            dual=bool(self.dual),
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
