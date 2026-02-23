from __future__ import annotations

from typing import Any

import numpy as np

from modssc.supervised.base import (
    BaseSupervisedClassifier,
    PredictScoresFromProbaMixin,
    SupportsProbaMixin,
)
from modssc.supervised.utils import ensure_2d


def require_fitted_model(model: Any) -> Any:
    if model is None:
        raise RuntimeError("Model is not fitted")
    return model


def predict_proba_float32(model: Any, X: Any) -> np.ndarray:
    fitted = require_fitted_model(model)
    X2 = ensure_2d(X)
    proba = fitted.predict_proba(X2)
    return np.asarray(proba, dtype=np.float32)


def decision_scores_float32(model: Any, X: Any) -> np.ndarray:
    fitted = require_fitted_model(model)
    X2 = ensure_2d(X)
    scores = fitted.decision_function(X2)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return scores


def predict_decoded_labels(
    classifier: BaseSupervisedClassifier,
    *,
    model: Any,
    X: Any,
) -> np.ndarray:
    fitted = require_fitted_model(model)
    X2 = ensure_2d(X)
    pred_enc = fitted.predict(X2)
    return classifier._decode(np.asarray(pred_enc, dtype=np.int64))


class _SklearnPredictDecodedMixin:
    def predict(self, X: Any) -> np.ndarray:
        model = getattr(self, "_model", None)
        return predict_decoded_labels(self, model=model, X=X)


class SklearnProbaClassifier(
    SupportsProbaMixin,
    PredictScoresFromProbaMixin,
    _SklearnPredictDecodedMixin,
    BaseSupervisedClassifier,
):
    def predict_proba(self, X: Any) -> np.ndarray:
        model = getattr(self, "_model", None)
        return predict_proba_float32(model, X)


class SklearnDecisionFunctionClassifier(_SklearnPredictDecodedMixin, BaseSupervisedClassifier):
    def predict_scores(self, X: Any) -> np.ndarray:
        model = getattr(self, "_model", None)
        return decision_scores_float32(model, X)
