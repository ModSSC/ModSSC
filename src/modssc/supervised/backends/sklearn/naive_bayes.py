from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import numpy as np

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class SklearnGaussianNBClassifier(BaseSupervisedClassifier):
    classifier_id = "gaussian_nb"
    backend = "sklearn"

    def __init__(self, *, var_smoothing: float = 1e-9, seed: int | None = 0):
        super().__init__(seed=seed, n_jobs=None)
        self.var_smoothing = float(var_smoothing)
        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug("params var_smoothing=%s seed=%s", self.var_smoothing, self.seed)
        sklearn_nb = optional_import(
            "sklearn.naive_bayes", extra="sklearn", feature="supervised:gaussian_nb"
        )
        GaussianNB = sklearn_nb.GaussianNB

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        model = GaussianNB(var_smoothing=float(self.var_smoothing))
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


class SklearnMultinomialNBClassifier(BaseSupervisedClassifier):
    classifier_id = "multinomial_nb"
    backend = "sklearn"

    def __init__(self, *, alpha: float = 1.0, fit_prior: bool = True, seed: int | None = 0):
        super().__init__(seed=seed, n_jobs=None)
        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug("params alpha=%s fit_prior=%s seed=%s", self.alpha, self.fit_prior, self.seed)
        sklearn_nb = optional_import(
            "sklearn.naive_bayes", extra="sklearn", feature="supervised:multinomial_nb"
        )
        MultinomialNB = sklearn_nb.MultinomialNB

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        model = MultinomialNB(alpha=float(self.alpha), fit_prior=bool(self.fit_prior))
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


class SklearnBernoulliNBClassifier(BaseSupervisedClassifier):
    classifier_id = "bernoulli_nb"
    backend = "sklearn"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        binarize: float | None = 0.0,
        fit_prior: bool = True,
        seed: int | None = 0,
    ):
        super().__init__(seed=seed, n_jobs=None)
        self.alpha = float(alpha)
        self.binarize = binarize if binarize is None else float(binarize)
        self.fit_prior = bool(fit_prior)
        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params alpha=%s binarize=%s fit_prior=%s seed=%s",
            self.alpha,
            self.binarize,
            self.fit_prior,
            self.seed,
        )
        sklearn_nb = optional_import(
            "sklearn.naive_bayes", extra="sklearn", feature="supervised:bernoulli_nb"
        )
        BernoulliNB = sklearn_nb.BernoulliNB

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)

        model = BernoulliNB(
            alpha=float(self.alpha),
            binarize=self.binarize,
            fit_prior=bool(self.fit_prior),
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
