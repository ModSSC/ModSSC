from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.supervised.errors import NotSupportedError
from modssc.supervised.utils import encode_labels, onehot


@dataclass
class FitResult:
    n_samples: int
    n_features: int
    n_classes: int


@dataclass(frozen=True)
class ClassifierCapabilities:
    """Explicit semantic capabilities exposed by a classifier backend."""

    predict: bool
    scores: bool
    probabilities: bool
    classes: bool

    @property
    def supports_proba(self) -> bool:
        """Backward-compatible spelling for probability support."""

        return self.probabilities


def classifier_capabilities(classifier: Any) -> ClassifierCapabilities:
    """Return an explicit capability contract for native and external classifiers."""

    declared = getattr(classifier, "capabilities", None)
    if isinstance(declared, ClassifierCapabilities):
        return declared

    supports_proba = getattr(classifier, "supports_proba", None)
    if isinstance(supports_proba, (bool, np.bool_)):
        probabilities = bool(supports_proba)
    else:
        probabilities = callable(getattr(classifier, "predict_proba", None))
    return ClassifierCapabilities(
        predict=callable(getattr(classifier, "predict", None)),
        scores=callable(getattr(classifier, "predict_scores", None)) or probabilities,
        probabilities=probabilities,
        classes=hasattr(classifier, "classes_") or hasattr(classifier, "classes_t_"),
    )


class SupportsProbaMixin:
    @property
    def supports_proba(self) -> bool:
        return True


class PredictScoresFromProbaMixin:
    def predict_scores(self, X: Any) -> np.ndarray:
        return self.predict_proba(X)


class BaseSupervisedClassifier:
    """Backend-agnostic classifier interface.

    All implementations must:
    - accept arbitrary label types in fit (int, str, etc.)
    - expose classes_ (original labels, sorted unique)
    - return predictions in original label space
    """

    classifier_id: str = "unknown"
    backend: str = "unknown"

    def __init__(self, *, seed: int | None = 0, n_jobs: int | None = None):
        self.seed = seed
        self.n_jobs = n_jobs
        self.classes_: np.ndarray | None = None
        self._fit_result: FitResult | None = None

    def fit(self, X: Any, y: Any) -> FitResult:
        raise NotImplementedError

    def predict(self, X: Any) -> np.ndarray:
        raise NotImplementedError

    def predict_scores(self, X: Any) -> np.ndarray:
        """Return class scores, shape (n_samples, n_classes).

        Default implementation:
        - if predict_proba is implemented, returns probabilities
        - otherwise returns one-hot predictions
        """
        if self.supports_proba:
            return self.predict_proba(X)
        pred = self.predict(X)
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted (classes_ is None)")
        # map predictions back to indices by search
        idx = np.searchsorted(self.classes_, pred)
        return onehot(idx.astype(np.int64), n_classes=int(self.classes_.size))

    def predict_proba(self, X: Any) -> np.ndarray:
        raise NotSupportedError(
            f"{self.classifier_id} backend={self.backend} does not support predict_proba()"
        )

    @property
    def supports_proba(self) -> bool:
        return False

    @property
    def capabilities(self) -> ClassifierCapabilities:
        return ClassifierCapabilities(
            predict=True,
            scores=True,
            probabilities=bool(self.supports_proba),
            classes=True,
        )

    @property
    def n_classes_(self) -> int:
        if self.classes_ is None:
            return 0
        return int(self.classes_.size)

    def _set_classes_from_y(self, y: Any) -> np.ndarray:
        y_enc, classes = encode_labels(y)
        self.classes_ = classes
        return y_enc

    def _decode(self, y_enc: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted (classes_ is None)")
        return self.classes_[np.asarray(y_enc, dtype=np.int64)]

    @property
    def classes_t_(self) -> Any | None:
        return getattr(self, "_classes_t", None)
