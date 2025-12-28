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
