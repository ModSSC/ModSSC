from __future__ import annotations

from typing import Any

import numpy as np

from modssc.supervised.base import (
    BaseSupervisedClassifier,
    PredictScoresFromProbaMixin,
    SupportsProbaMixin,
)
from modssc.supervised.errors import SupervisedValidationError


def make_activation(name: str, torch):
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise SupervisedValidationError(f"Unknown activation: {name!r}")


def predict_from_scores(*, classes_t: Any, scores: Any):
    if classes_t is None:
        raise RuntimeError("Model is not fitted")
    idx = scores.argmax(dim=1)
    return classes_t[idx]


class _RequiresScoresMixin:
    def _scores(self, X: Any):
        raise NotImplementedError


class TorchArgmaxPredictMixin(_RequiresScoresMixin):
    _classes_t: Any | None

    def predict(self, X: Any):
        return predict_from_scores(classes_t=self._classes_t, scores=self._scores(X))


class TorchSupportsProbaMixin(SupportsProbaMixin):
    pass


class TorchScoresProbaMixin(
    _RequiresScoresMixin,
    PredictScoresFromProbaMixin,
    TorchSupportsProbaMixin,
):
    def predict_proba(self, X: Any):
        return self._scores(X)


class TorchNumpyProbaPredictMixin:
    def predict_proba(self, X: Any):
        raise NotImplementedError("Subclasses must implement predict_proba().")

    def predict(self, X: Any):
        probs = self.predict_proba(X)
        if isinstance(probs, np.ndarray):
            return np.argmax(probs, axis=1)
        if hasattr(probs, "argmax"):
            try:
                idx = probs.argmax(dim=1)
            except Exception:
                idx = probs.argmax(axis=1)
            if hasattr(idx, "cpu"):
                idx = idx.cpu()
            if hasattr(idx, "numpy"):
                return idx.numpy()
            return np.asarray(idx)
        return np.asarray(probs).argmax(axis=1)


class TorchScoresClassifierBase(
    TorchArgmaxPredictMixin,
    TorchScoresProbaMixin,
    BaseSupervisedClassifier,
):
    """Explicit torch classifier base for score/proba/predict behavior."""

    @property
    def supports_proba(self) -> bool:
        return True

    def predict(self, X: Any):
        return TorchArgmaxPredictMixin.predict(self, X)

    def predict_proba(self, X: Any):
        return TorchScoresProbaMixin.predict_proba(self, X)

    def predict_scores(self, X: Any):
        return PredictScoresFromProbaMixin.predict_scores(self, X)


class TorchNumpyProbaClassifierBase(
    TorchNumpyProbaPredictMixin,
    TorchSupportsProbaMixin,
    BaseSupervisedClassifier,
):
    """Explicit torch classifier base using numpy/tensor predict_proba outputs."""

    @property
    def supports_proba(self) -> bool:
        return True

    def predict(self, X: Any):
        return TorchNumpyProbaPredictMixin.predict(self, X)


class TorchSupportsProbaClassifierBase(BaseSupervisedClassifier):
    """Explicit torch classifier base for implementations that support probabilities."""

    @property
    def supports_proba(self) -> bool:
        return True


def make_softmax_scores_method(*, torch_getter):
    def _scores(self, X: Any):
        torch = torch_getter()
        if self._model is None or self._classes_t is None:
            raise RuntimeError("Model is not fitted")
        X4 = self._prepare_X(X, torch, allow_infer=False)
        if X4.device != self._classes_t.device:
            raise SupervisedValidationError("X must be on the same device as the model.")
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X4.to(dtype=torch.float32))
            return torch.softmax(logits, dim=1)

    return _scores
