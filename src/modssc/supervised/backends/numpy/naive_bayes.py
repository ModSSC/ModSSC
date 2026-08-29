from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np

from modssc.supervised.backends.numpy.tabular import TabularEncoder, TabularFeature
from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.errors import SupervisedValidationError

logger = logging.getLogger(__name__)


class NumpyNaiveBayesClassifier(BaseSupervisedClassifier):
    """Naive Bayes for historical mixed nominal/numeric datasets.

    Nominal likelihoods and class priors use Laplace smoothing. Numeric
    likelihoods use per-class Gaussian estimates. Missing feature values are
    ignored for that feature, matching the usual historical Naive Bayes rule.
    """

    classifier_id = "gaussian_nb"
    backend = "numpy"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_prior: bool = True,
        var_smoothing: float = 1e-9,
        feature_schema: Sequence[Mapping[str, Any] | str] | None = None,
        missing_values: Sequence[Any] = ("?",),
        seed: int | None = 0,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self.var_smoothing = float(var_smoothing)
        self.feature_schema = feature_schema
        self.missing_values = tuple(missing_values)
        self._encoder: TabularEncoder | None = None
        self.feature_schema_: tuple[TabularFeature, ...] | None = None
        self.class_log_prior_: np.ndarray | None = None
        self._nominal_log_prob: dict[int, np.ndarray] = {}
        self._numeric_mean: dict[int, np.ndarray] = {}
        self._numeric_var: dict[int, np.ndarray] = {}

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        if self.alpha <= 0.0:
            raise SupervisedValidationError("alpha must be > 0.")
        if self.var_smoothing <= 0.0:
            raise SupervisedValidationError("var_smoothing must be > 0.")
        encoder = TabularEncoder(
            feature_schema=self.feature_schema,
            missing_values=self.missing_values,
            classifier_name="NumPy historical Naive Bayes",
        )
        X_encoded = encoder.fit_transform(X)
        y_encoded = np.asarray(self._set_classes_from_y(y), dtype=np.int64).reshape(-1)
        if int(X_encoded.shape[0]) != int(y_encoded.size):
            raise SupervisedValidationError(
                f"X and y have incompatible sizes: {X_encoded.shape[0]} vs {y_encoded.size}"
            )
        self._encoder = encoder
        self.feature_schema_ = encoder.features_
        n_classes = int(self.n_classes_)
        class_counts = np.bincount(y_encoded, minlength=n_classes).astype(np.float64)
        if self.fit_prior:
            prior = (class_counts + self.alpha) / (float(y_encoded.size) + self.alpha * n_classes)
        else:
            prior = np.full((n_classes,), 1.0 / n_classes, dtype=np.float64)
        self.class_log_prior_ = np.log(prior)
        self._nominal_log_prob = {}
        self._numeric_mean = {}
        self._numeric_var = {}

        if self.feature_schema_ is None:  # pragma: no cover - encoder invariant
            raise RuntimeError("Tabular encoder did not expose its fitted schema")
        for column, feature in enumerate(self.feature_schema_):
            values = X_encoded[:, column]
            known = np.isfinite(values)
            if feature.kind == "nominal":
                n_values = len(feature.values)
                counts = np.zeros((n_classes, n_values), dtype=np.float64)
                if np.any(known):
                    np.add.at(
                        counts,
                        (y_encoded[known], values[known].astype(np.int64, copy=False)),
                        1.0,
                    )
                denominator = counts.sum(axis=1, keepdims=True) + self.alpha * n_values
                self._nominal_log_prob[column] = np.log((counts + self.alpha) / denominator)
                continue

            global_known = values[known]
            global_mean = float(np.mean(global_known)) if global_known.size else 0.0
            global_var = float(np.var(global_known)) if global_known.size else 1.0
            variance_floor = max(self.var_smoothing * max(global_var, 1.0), np.finfo(float).eps)
            means = np.empty((n_classes,), dtype=np.float64)
            variances = np.empty((n_classes,), dtype=np.float64)
            for class_index in range(n_classes):
                selected = values[known & (y_encoded == class_index)]
                means[class_index] = float(np.mean(selected)) if selected.size else global_mean
                variance = float(np.var(selected)) if selected.size else global_var
                variances[class_index] = max(variance, variance_floor)
            self._numeric_mean[column] = means
            self._numeric_var[column] = variances

        self._fit_result = FitResult(
            n_samples=int(X_encoded.shape[0]),
            n_features=int(X_encoded.shape[1]),
            n_classes=n_classes,
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def _joint_log_likelihood(self, X: Any) -> np.ndarray:
        if (
            self._encoder is None
            or self.feature_schema_ is None
            or self.class_log_prior_ is None
            or self.classes_ is None
        ):
            raise RuntimeError("Model is not fitted")
        encoded = self._encoder.transform(X)
        joint = np.broadcast_to(
            self.class_log_prior_, (int(encoded.shape[0]), self.n_classes_)
        ).copy()
        for column, feature in enumerate(self.feature_schema_):
            values = encoded[:, column]
            known_indices = np.flatnonzero(np.isfinite(values))
            if known_indices.size == 0:
                continue
            if feature.kind == "nominal":
                probabilities = self._nominal_log_prob[column]
                codes = values[known_indices].astype(np.int64, copy=False)
                joint[known_indices] += probabilities[:, codes].T
                continue
            means = self._numeric_mean[column]
            variances = self._numeric_var[column]
            differences = values[known_indices, None] - means[None, :]
            joint[known_indices] += -0.5 * (
                np.log(2.0 * np.pi * variances)[None, :]
                + differences * differences / variances[None, :]
            )
        return joint

    def predict_proba(self, X: Any) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        shifted = joint - joint.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict_scores(self, X: Any) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self._decode(self.predict_proba(X).argmax(axis=1))
