from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.supervised.backends.numpy.tabular import TabularEncoder, TabularFeature
from modssc.supervised.base import (
    BaseSupervisedClassifier,
    FitResult,
)
from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class NumpyKNNClassifier(BaseSupervisedClassifier):
    """Pure numpy kNN classifier.

    Notes
    -----
    This is intended as a lightweight baseline and a fallback when scikit-learn
    is not available. It is O(n_train * n_query) and can be slow for large datasets.
    """

    classifier_id = "knn"
    backend = "numpy"

    @property
    def supports_proba(self) -> bool:
        return True

    def __init__(
        self,
        *,
        k: int = 5,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
        batch_size: int = 1024,
        eps: float = 1e-12,
        feature_schema: Sequence[Mapping[str, Any] | str] | None = None,
        missing_values: Sequence[Any] = ("?",),
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.k = int(k)
        self.metric = str(metric)
        self.weights = str(weights)
        self.batch_size = int(batch_size)
        self.eps = float(eps)
        self.feature_schema = feature_schema
        self.missing_values = tuple(missing_values)

        self._X_train: np.ndarray | None = None
        self._y_train_enc: np.ndarray | None = None
        self._tabular_encoder: TabularEncoder | None = None
        self.feature_schema_: tuple[TabularFeature, ...] | None = None
        self._numeric_min: np.ndarray | None = None
        self._numeric_range: np.ndarray | None = None

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params k=%s metric=%s weights=%s batch_size=%s eps=%s seed=%s n_jobs=%s",
            self.k,
            self.metric,
            self.weights,
            self.batch_size,
            self.eps,
            self.seed,
            self.n_jobs,
        )
        if self.feature_schema is None:
            X2 = ensure_2d(X).astype(np.float32, copy=False)
            self._tabular_encoder = None
            self.feature_schema_ = None
        else:
            encoder = TabularEncoder(
                feature_schema=self.feature_schema,
                missing_values=self.missing_values,
                classifier_name="NumPy kNN",
            )
            X2 = encoder.fit_transform(X)
            self._tabular_encoder = encoder
            self.feature_schema_ = encoder.features_
        if X2.size == 0:
            raise SupervisedValidationError("X must be non-empty")
        y_enc = self._set_classes_from_y(y)
        y_enc = np.asarray(y_enc, dtype=np.int64).reshape(-1)

        if X2.shape[0] != y_enc.size:
            raise SupervisedValidationError(
                f"X and y have incompatible sizes: {X2.shape[0]} vs {y_enc.size}"
            )
        if self.k <= 0:
            raise SupervisedValidationError("k must be >= 1")
        if self.metric not in {"euclidean", "cosine"}:
            raise SupervisedValidationError(f"Unknown metric: {self.metric!r}")
        if self._tabular_encoder is not None and self.metric != "euclidean":
            raise SupervisedValidationError(
                "feature_schema is supported only with metric='euclidean'."
            )
        if self.weights not in {"uniform", "distance"}:
            raise SupervisedValidationError(f"Unknown weights: {self.weights!r}")

        self._X_train = X2
        self._y_train_enc = y_enc
        self._numeric_min = None
        self._numeric_range = None
        if self.feature_schema_ is not None:
            numeric_min = np.zeros((X2.shape[1],), dtype=np.float64)
            numeric_range = np.ones((X2.shape[1],), dtype=np.float64)
            for column, feature in enumerate(self.feature_schema_):
                if feature.kind != "numeric":
                    continue
                known = X2[np.isfinite(X2[:, column]), column]
                if known.size:
                    numeric_min[column] = float(np.min(known))
                    span = float(np.max(known) - numeric_min[column])
                    numeric_range[column] = span if span > 0.0 else 1.0
            self._numeric_min = numeric_min
            self._numeric_range = numeric_range

        self._fit_result = FitResult(
            n_samples=int(X2.shape[0]),
            n_features=int(X2.shape[1]),
            n_classes=int(self.n_classes_),
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def _pairwise_scores(self, Q: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise RuntimeError("Model is not fitted (X_train is None)")
        X = self._X_train

        if self.feature_schema_ is not None:
            if self._numeric_min is None or self._numeric_range is None:
                raise RuntimeError("Model is not fitted (numeric ranges are missing)")
            distances = np.zeros((Q.shape[0], X.shape[0]), dtype=np.float64)
            for column, feature in enumerate(self.feature_schema_):
                query = Q[:, column][:, None]
                train = X[:, column][None, :]
                query_missing = ~np.isfinite(query)
                train_missing = ~np.isfinite(train)
                either_missing = query_missing | train_missing
                if feature.kind == "nominal":
                    difference = (query != train).astype(np.float64)
                    difference[either_missing] = 1.0
                else:
                    query_normalized = (query - self._numeric_min[column]) / self._numeric_range[
                        column
                    ]
                    train_normalized = (train - self._numeric_min[column]) / self._numeric_range[
                        column
                    ]
                    difference = np.abs(query_normalized - train_normalized)
                    one_missing = query_missing ^ train_missing
                    known_side = np.where(query_missing, train_normalized, query_normalized)
                    missing_difference = np.maximum(known_side, 1.0 - known_side)
                    difference[one_missing] = missing_difference[one_missing]
                    difference[query_missing & train_missing] = 1.0
                distances += difference * difference
            return -distances

        if self.metric == "euclidean":
            # negative squared distance so that larger is better
            q2 = (Q * Q).sum(axis=1, keepdims=True)
            x2 = (X * X).sum(axis=1, keepdims=True).T
            dist2 = q2 + x2 - 2.0 * (Q @ X.T)
            return -dist2
        # cosine similarity
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + self.eps)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + self.eps)
        return Qn @ Xn.T

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._X_train is None or self._y_train_enc is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted")
        if self._tabular_encoder is None:
            Q = ensure_2d(X).astype(np.float32, copy=False)
        else:
            Q = self._tabular_encoder.transform(X)

        n_query = int(Q.shape[0])
        n_train = int(self._X_train.shape[0])
        n_classes = int(self.classes_.size)

        k = min(int(self.k), n_train)
        out = np.zeros((n_query, n_classes), dtype=np.float32)

        for start in range(0, n_query, int(self.batch_size)):
            stop = min(n_query, start + int(self.batch_size))
            Qb = Q[start:stop]

            scores = self._pairwise_scores(Qb)  # larger is better
            # top-k by score
            idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            top_scores = np.take_along_axis(scores, idx, axis=1)

            neigh = self._y_train_enc[idx]  # shape (b, k)

            if self.weights == "uniform":
                w = np.ones_like(top_scores, dtype=np.float32)
            else:
                # convert score to a distance-like quantity
                if self.metric == "euclidean":
                    dist = np.maximum(-top_scores, 0.0)
                else:
                    dist = np.maximum(1.0 - top_scores, 0.0)
                w = 1.0 / (dist.astype(np.float32) + float(self.eps))

            rows = np.repeat(np.arange(stop - start), k)
            cols = neigh.reshape(-1)
            np.add.at(out[start:stop], (rows, cols), w.reshape(-1))

        row_sum = out.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        out = out / row_sum
        return out

    def predict_scores(self, X: Any) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        pred_enc = proba.argmax(axis=1)
        return self._decode(pred_enc)
