from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.supervised.backends.numpy.tabular import TabularEncoder, TabularFeature
from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.errors import SupervisedValidationError

logger = logging.getLogger(__name__)


@dataclass
class _TreeNode:
    probabilities: np.ndarray
    feature: int | None = None
    threshold: float | None = None
    branches: tuple[float, ...] = ()
    branch_probabilities: tuple[float, ...] = ()
    children: tuple[_TreeNode, ...] = ()


@dataclass(frozen=True)
class _Split:
    feature: int
    threshold: float | None
    branches: tuple[float, ...]
    gain: float
    gain_ratio: float
    known_branch_weights: tuple[float, ...]


def _entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probabilities = counts[counts > 0.0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


class NumpyC45Classifier(BaseSupervisedClassifier):
    """Deterministic unpruned C4.5-like decision tree.

    Splits use gain ratio, nominal features use multi-way branches, and numeric
    features use deterministic midpoint thresholds. During fitting, missing
    values are distributed fractionally using observed branch frequencies;
    prediction combines the same child distributions for missing values.
    """

    classifier_id = "decision_tree"
    backend = "numpy"

    def __init__(
        self,
        *,
        min_num_obj: int = 2,
        max_depth: int | None = None,
        min_gain: float = 1e-12,
        probability_smoothing: float = 0.0,
        feature_schema: Sequence[Mapping[str, Any] | str] | None = None,
        missing_values: Sequence[Any] = ("?",),
        unpruned: bool = True,
        binary_splits: bool = False,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.min_num_obj = int(min_num_obj)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_gain = float(min_gain)
        self.probability_smoothing = float(probability_smoothing)
        self.feature_schema = feature_schema
        self.missing_values = tuple(missing_values)
        self.unpruned = bool(unpruned)
        self.binary_splits = bool(binary_splits)
        self._encoder: TabularEncoder | None = None
        self.feature_schema_: tuple[TabularFeature, ...] | None = None
        self.tree_: _TreeNode | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def _validate_params(self) -> None:
        if self.min_num_obj < 1:
            raise SupervisedValidationError("min_num_obj must be >= 1.")
        if self.max_depth is not None and self.max_depth < 0:
            raise SupervisedValidationError("max_depth must be >= 0 or None.")
        if self.min_gain < 0.0:
            raise SupervisedValidationError("min_gain must be >= 0.")
        if self.probability_smoothing < 0.0:
            raise SupervisedValidationError("probability_smoothing must be >= 0.")
        if not self.unpruned:
            raise SupervisedValidationError(
                "The NumPy C4.5 backend implements the unpruned historical tree only; "
                "set unpruned=True."
            )
        if self.binary_splits:
            raise SupervisedValidationError(
                "The NumPy C4.5 backend uses canonical multi-way nominal splits; "
                "set binary_splits=False."
            )

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        self._validate_params()
        encoder = TabularEncoder(
            feature_schema=self.feature_schema,
            missing_values=self.missing_values,
            classifier_name="NumPy C4.5",
        )
        X_encoded = encoder.fit_transform(X)
        y_encoded = np.asarray(self._set_classes_from_y(y), dtype=np.int64).reshape(-1)
        if int(X_encoded.shape[0]) != int(y_encoded.size):
            raise SupervisedValidationError(
                f"X and y have incompatible sizes: {X_encoded.shape[0]} vs {y_encoded.size}"
            )
        self._encoder = encoder
        self.feature_schema_ = encoder.features_
        self._X = X_encoded
        self._y = y_encoded
        rows = np.arange(y_encoded.size, dtype=np.int64)
        weights = np.ones((y_encoded.size,), dtype=np.float64)
        nominal_available = frozenset(
            index
            for index, feature in enumerate(self.feature_schema_ or ())
            if feature.kind == "nominal"
        )
        self.tree_ = self._build_tree(rows, weights, depth=0, nominal_available=nominal_available)
        self._fit_result = FitResult(
            n_samples=int(X_encoded.shape[0]),
            n_features=int(X_encoded.shape[1]),
            n_classes=int(self.n_classes_),
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def _class_counts(self, rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if self._y is None:  # pragma: no cover - fit invariant
            raise RuntimeError("Model is not fitted")
        return np.bincount(self._y[rows], weights=weights, minlength=self.n_classes_).astype(
            np.float64,
            copy=False,
        )

    def _leaf_probabilities(self, counts: np.ndarray) -> np.ndarray:
        smoothed = counts + self.probability_smoothing
        total = float(np.sum(smoothed))
        if total <= 0.0:  # pragma: no cover - tree never constructs an empty node
            return np.full((self.n_classes_,), 1.0 / self.n_classes_, dtype=np.float64)
        return smoothed / total

    def _candidate_split(
        self,
        rows: np.ndarray,
        weights: np.ndarray,
        *,
        feature_index: int,
    ) -> list[_Split]:
        if self._X is None or self._y is None or self.feature_schema_ is None:
            raise RuntimeError("Model is not fitted")
        values = self._X[rows, feature_index]
        known = np.isfinite(values)
        known_rows = rows[known]
        known_weights = weights[known]
        known_total = float(np.sum(known_weights))
        total = float(np.sum(weights))
        if known_total < 2.0 * self.min_num_obj or total <= 0.0:
            return []
        parent_counts = self._class_counts(known_rows, known_weights)
        parent_entropy = _entropy(parent_counts)
        if parent_entropy <= 0.0:
            return []
        feature = self.feature_schema_[feature_index]
        branch_sets: list[tuple[float | None, tuple[float, ...]]] = []
        if feature.kind == "nominal":
            branches = tuple(
                float(code)
                for code in range(len(feature.values))
                if np.any(values[known] == float(code))
            )
            if len(branches) >= 2:
                branch_sets.append((None, branches))
        else:
            known_values = values[known]
            order = np.argsort(known_values, kind="stable")
            sorted_values = known_values[order]
            sorted_labels = self._y[known_rows[order]]
            thresholds = []
            for position in range(1, int(sorted_values.size)):
                left_value = float(sorted_values[position - 1])
                right_value = float(sorted_values[position])
                if (
                    left_value >= right_value
                    or sorted_labels[position - 1] == sorted_labels[position]
                ):
                    continue
                thresholds.append(left_value + (right_value - left_value) / 2.0)
            for threshold in thresholds:
                branch_sets.append((threshold, (0.0, 1.0)))

        candidates: list[_Split] = []
        missing_weight = total - known_total
        for threshold, branches in branch_sets:
            if threshold is None:
                masks = [known & (values == branch) for branch in branches]
            else:
                masks = [known & (values <= threshold), known & (values > threshold)]
            branch_weights = tuple(float(np.sum(weights[mask])) for mask in masks)
            if any(weight < self.min_num_obj for weight in branch_weights):
                continue
            child_entropy = 0.0
            for mask, branch_weight in zip(masks, branch_weights, strict=True):
                child_entropy += (branch_weight / known_total) * _entropy(
                    self._class_counts(rows[mask], weights[mask])
                )
            gain = (known_total / total) * (parent_entropy - child_entropy)
            proportions = [weight / total for weight in branch_weights]
            if missing_weight > 0.0:
                proportions.append(missing_weight / total)
            split_info = _entropy(np.asarray(proportions, dtype=np.float64))
            gain_ratio = gain / split_info if split_info > 0.0 else 0.0
            if gain > self.min_gain and gain_ratio > 0.0:
                candidates.append(
                    _Split(
                        feature=feature_index,
                        threshold=threshold,
                        branches=branches,
                        gain=float(gain),
                        gain_ratio=float(gain_ratio),
                        known_branch_weights=branch_weights,
                    )
                )
        return candidates

    def _best_split(
        self,
        rows: np.ndarray,
        weights: np.ndarray,
        *,
        nominal_available: frozenset[int],
    ) -> _Split | None:
        if self.feature_schema_ is None:  # pragma: no cover - fit invariant
            raise RuntimeError("Model is not fitted")
        candidates: list[_Split] = []
        for feature_index, feature in enumerate(self.feature_schema_):
            if feature.kind == "nominal" and feature_index not in nominal_available:
                continue
            candidates.extend(self._candidate_split(rows, weights, feature_index=feature_index))
        if not candidates:
            return None
        average_gain = float(np.mean([candidate.gain for candidate in candidates]))
        eligible = [candidate for candidate in candidates if candidate.gain + 1e-15 >= average_gain]
        return max(
            eligible,
            key=lambda candidate: (
                candidate.gain_ratio,
                candidate.gain,
                -candidate.feature,
                -(candidate.threshold if candidate.threshold is not None else -np.inf),
            ),
        )

    def _build_tree(
        self,
        rows: np.ndarray,
        weights: np.ndarray,
        *,
        depth: int,
        nominal_available: frozenset[int],
    ) -> _TreeNode:
        if self._X is None or self._y is None or self.feature_schema_ is None:
            raise RuntimeError("Model is not fitted")
        counts = self._class_counts(rows, weights)
        node = _TreeNode(probabilities=self._leaf_probabilities(counts))
        if (
            np.count_nonzero(counts > 0.0) <= 1
            or float(np.sum(weights)) < 2.0 * self.min_num_obj
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node
        split = self._best_split(rows, weights, nominal_available=nominal_available)
        if split is None:
            return node
        values = self._X[rows, split.feature]
        known = np.isfinite(values)
        missing = ~known
        known_total = float(sum(split.known_branch_weights))
        branch_probabilities = tuple(weight / known_total for weight in split.known_branch_weights)
        children: list[_TreeNode] = []
        for branch_index, (branch, branch_probability) in enumerate(
            zip(split.branches, branch_probabilities, strict=True)
        ):
            if split.threshold is None:
                selected = known & (values == branch)
            elif branch_index == 0:
                selected = known & (values <= split.threshold)
            else:
                selected = known & (values > split.threshold)
            child_rows = np.concatenate((rows[selected], rows[missing]))
            child_weights = np.concatenate(
                (weights[selected], weights[missing] * branch_probability)
            )
            next_nominal = nominal_available
            if split.threshold is None:
                next_nominal = nominal_available - {split.feature}
            children.append(
                self._build_tree(
                    child_rows,
                    child_weights,
                    depth=depth + 1,
                    nominal_available=next_nominal,
                )
            )
        node.feature = split.feature
        node.threshold = split.threshold
        node.branches = split.branches
        node.branch_probabilities = branch_probabilities
        node.children = tuple(children)
        return node

    def _predict_node(self, row: np.ndarray, node: _TreeNode) -> np.ndarray:
        if node.feature is None or not node.children:
            return node.probabilities
        value = row[node.feature]
        child_index: int | None = None
        if np.isfinite(value):
            if node.threshold is not None:
                child_index = 0 if value <= node.threshold else 1
            else:
                for index, branch in enumerate(node.branches):
                    if value == branch:
                        child_index = index
                        break
        if child_index is not None:
            return self._predict_node(row, node.children[child_index])
        combined = np.zeros((self.n_classes_,), dtype=np.float64)
        for probability, child in zip(node.branch_probabilities, node.children, strict=True):
            combined += probability * self._predict_node(row, child)
        return combined

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._encoder is None or self.tree_ is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted")
        encoded = self._encoder.transform(X)
        return np.vstack([self._predict_node(row, self.tree_) for row in encoded])

    def predict_scores(self, X: Any) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self._decode(self.predict_proba(X).argmax(axis=1))
