from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class DummyClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._n_classes = 0

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        self._n_classes = int(np.unique(y).size) if y.size else 0
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        n = int(X.shape[0])
        if self._n_classes <= 0:
            return np.zeros((n, 0), dtype=np.float32)
        return np.full((n, self._n_classes), 1.0 / float(self._n_classes), dtype=np.float32)

    def predict(self, X):
        X = np.asarray(X)
        return np.zeros((int(X.shape[0]),), dtype=np.int64)


class DummyLinearModel(DummyClassifier):
    def decision_function(self, X):
        X = np.asarray(X)
        return np.zeros((int(X.shape[0]),), dtype=np.float32)


class DummyLinearModel2D(DummyClassifier):
    def decision_function(self, X):
        X = np.asarray(X)
        return np.zeros((int(X.shape[0]), 3), dtype=np.float32)


def _check_proba_classifier(clf_cls, module, monkeypatch):
    dummy_module = SimpleNamespace(
        ExtraTreesClassifier=DummyClassifier,
        GradientBoostingClassifier=DummyClassifier,
        RandomForestClassifier=DummyClassifier,
        GaussianNB=DummyClassifier,
        MultinomialNB=DummyClassifier,
        BernoulliNB=DummyClassifier,
    )
    monkeypatch.setattr(module, "optional_import", lambda *a, **k: dummy_module)

    X = np.random.default_rng(0).normal(size=(6, 3)).astype(np.float32)
    y = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)

    clf = clf_cls()
    assert clf.supports_proba

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_proba(X)

    clf.fit(X, y)
    proba = clf.predict_proba(X[:2])
    scores = clf.predict_scores(X[:2])
    pred = clf.predict(X[:2])

    assert proba.shape == (2, 2)
    assert scores.shape == (2, 2)
    assert pred.shape == (2,)


def test_extra_trees_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.extra_trees as mod
    from modssc.supervised.backends.sklearn.extra_trees import SklearnExtraTreesClassifier

    _check_proba_classifier(SklearnExtraTreesClassifier, mod, monkeypatch)


def test_gradient_boosting_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.gradient_boosting as mod
    from modssc.supervised.backends.sklearn.gradient_boosting import (
        SklearnGradientBoostingClassifier,
    )

    _check_proba_classifier(SklearnGradientBoostingClassifier, mod, monkeypatch)


def test_random_forest_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.random_forest as mod
    from modssc.supervised.backends.sklearn.random_forest import SklearnRandomForestClassifier

    _check_proba_classifier(SklearnRandomForestClassifier, mod, monkeypatch)


def test_naive_bayes_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.naive_bayes as mod
    from modssc.supervised.backends.sklearn.naive_bayes import (
        SklearnBernoulliNBClassifier,
        SklearnGaussianNBClassifier,
        SklearnMultinomialNBClassifier,
    )

    for cls in (
        SklearnGaussianNBClassifier,
        SklearnMultinomialNBClassifier,
        SklearnBernoulliNBClassifier,
    ):
        _check_proba_classifier(cls, mod, monkeypatch)


def test_linear_svm_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.linear_svm as mod
    from modssc.supervised.backends.sklearn.linear_svm import SklearnLinearSVMClassifier

    dummy_module = SimpleNamespace(LinearSVC=DummyLinearModel)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_module)

    X = np.random.default_rng(0).normal(size=(4, 2)).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    clf = SklearnLinearSVMClassifier()
    assert not clf.supports_proba

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_scores(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(X)

    clf.fit(X, y)
    scores = clf.predict_scores(X[:2])
    pred = clf.predict(X[:2])

    assert scores.shape == (2, 2)
    assert pred.shape == (2,)


def test_linear_svm_wrapper_multiclass_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.linear_svm as mod
    from modssc.supervised.backends.sklearn.linear_svm import SklearnLinearSVMClassifier

    dummy_module = SimpleNamespace(LinearSVC=DummyLinearModel2D)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_module)

    X = np.random.default_rng(1).normal(size=(5, 2)).astype(np.float32)
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)

    clf = SklearnLinearSVMClassifier()
    clf.fit(X, y)

    scores = clf.predict_scores(X[:2])
    assert scores.shape == (2, 3)


def test_ridge_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.ridge as mod
    from modssc.supervised.backends.sklearn.ridge import SklearnRidgeClassifier

    dummy_module = SimpleNamespace(RidgeClassifier=DummyLinearModel)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_module)

    X = np.random.default_rng(0).normal(size=(4, 2)).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    clf = SklearnRidgeClassifier()
    assert not clf.supports_proba

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_scores(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(X)

    clf.fit(X, y)
    scores = clf.predict_scores(X[:2])
    pred = clf.predict(X[:2])

    assert scores.shape == (2, 2)
    assert pred.shape == (2,)


def test_ridge_wrapper_multiclass_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.ridge as mod
    from modssc.supervised.backends.sklearn.ridge import SklearnRidgeClassifier

    dummy_module = SimpleNamespace(RidgeClassifier=DummyLinearModel2D)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_module)

    X = np.random.default_rng(1).normal(size=(5, 2)).astype(np.float32)
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)

    clf = SklearnRidgeClassifier()
    clf.fit(X, y)

    scores = clf.predict_scores(X[:2])
    assert scores.shape == (2, 3)
