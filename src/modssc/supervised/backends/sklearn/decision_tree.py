from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Literal

from modssc.supervised.backends.sklearn.common import SklearnProbaClassifier
from modssc.supervised.base import FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import ensure_2d

logger = logging.getLogger(__name__)


class SklearnDecisionTreeClassifier(SklearnProbaClassifier):
    classifier_id = "decision_tree"
    backend = "sklearn"

    def __init__(
        self,
        *,
        criterion: Literal["gini", "entropy", "log_loss"] = "entropy",
        splitter: Literal["best", "random"] = "best",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = None,
        class_weight: str | dict[str, float] | None = None,
        ccp_alpha: float = 0.0,
        seed: int | None = 0,
    ):
        super().__init__(seed=seed, n_jobs=None)
        self.criterion = str(criterion)
        self.splitter = str(splitter)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.class_weight = class_weight
        self.ccp_alpha = float(ccp_alpha)
        self._model: Any | None = None

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params criterion=%s splitter=%s max_depth=%s max_features=%s "
            "min_samples_split=%s min_samples_leaf=%s class_weight=%s ccp_alpha=%s seed=%s",
            self.criterion,
            self.splitter,
            self.max_depth,
            self.max_features,
            self.min_samples_split,
            self.min_samples_leaf,
            self.class_weight,
            self.ccp_alpha,
            self.seed,
        )
        sklearn_tree = optional_import(
            "sklearn.tree", extra="sklearn", feature="supervised:decision_tree"
        )
        DecisionTreeClassifier = sklearn_tree.DecisionTreeClassifier

        X2 = ensure_2d(X)
        y_enc = self._set_classes_from_y(y)
        model = DecisionTreeClassifier(
            criterion=str(self.criterion),
            splitter=str(self.splitter),
            max_depth=self.max_depth,
            min_samples_split=int(self.min_samples_split),
            min_samples_leaf=int(self.min_samples_leaf),
            max_features=self.max_features,
            class_weight=self.class_weight,
            ccp_alpha=float(self.ccp_alpha),
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
