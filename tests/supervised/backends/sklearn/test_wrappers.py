from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from modssc.supervised.api import create_classifier
from modssc.supervised.backends.sklearn.logreg import SklearnLogRegClassifier
from modssc.supervised.errors import NotSupportedError


class DummyKNeighborsClassifier:
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


class DummyLogisticRegression:
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

    def decision_function(self, X):
        X = np.asarray(X)
        n = int(X.shape[0])
        if self._n_classes <= 0:
            return np.zeros((n, 0), dtype=np.float32)
        if self._n_classes == 2:
            return np.zeros((n,), dtype=np.float32)
        return np.zeros((n, self._n_classes), dtype=np.float32)

    def predict(self, X):
        X = np.asarray(X)
        return np.zeros((int(X.shape[0]),), dtype=np.int64)


class DummySVC:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._n_classes = 0
        self.probability = bool(kwargs.get("probability", False))

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        self._n_classes = int(np.unique(y).size) if y.size else 0
        return self

    def decision_function(self, X):
        X = np.asarray(X)
        n = int(X.shape[0])
        if self._n_classes <= 0:
            return np.zeros((n, 0), dtype=np.float32)
        if self._n_classes == 2:
            return np.zeros((n,), dtype=np.float32)
        return np.zeros((n, self._n_classes), dtype=np.float32)

    def predict(self, X):
        X = np.asarray(X)
        return np.zeros((int(X.shape[0]),), dtype=np.int64)

    def predict_proba(self, X):
        if not self.probability:
            raise RuntimeError("probability disabled")
        X = np.asarray(X)
        n = int(X.shape[0])
        if self._n_classes <= 0:
            return np.zeros((n, 0), dtype=np.float32)
        return np.full((n, self._n_classes), 1.0 / float(self._n_classes), dtype=np.float32)


def test_sklearn_knn_wrapper_with_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.knn as mod

    dummy_neighbors = SimpleNamespace(KNeighborsClassifier=DummyKNeighborsClassifier)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_neighbors)

    X = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)

    clf = create_classifier("knn", backend="sklearn", params={"k": 3})
    assert clf.supports_proba

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_proba(X)

    clf.fit(X, y)
    proba = clf.predict_proba(X[:2])
    assert proba.shape == (2, 2)
    pred = clf.predict(X[:2])
    assert pred.shape == (2,)

    scores = clf.predict_scores(X[:2])
    assert scores.shape == (2, 2)

    clf_algo = create_classifier(
        "knn", backend="sklearn", params={"k": 3, "algorithm": "ball_tree"}
    )
    assert clf_algo.algorithm == "ball_tree"
    clf_algo.fit(X, y)

    assert clf_algo._model.kwargs["algorithm"] == "ball_tree"


def test_sklearn_logreg_wrapper_with_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.logreg as mod

    dummy_linear = SimpleNamespace(LogisticRegression=DummyLogisticRegression)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_linear)

    X = np.random.default_rng(0).normal(size=(12, 4)).astype(np.float32)
    y = np.array([0, 1, 2] * 4, dtype=np.int64)

    clf = create_classifier("logreg", backend="sklearn")
    assert clf.supports_proba

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_proba(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_scores(X)

    clf.fit(X, y)
    scores = clf.predict_scores(X[:3])
    assert scores.shape == (3, 3)
    proba = clf.predict_proba(X[:3])
    assert proba.shape == (3, 3)
    pred = clf.predict(X[:3])
    assert pred.shape == (3,)


def test_logreg_predict_scores_falls_back_to_proba() -> None:
    clf = SklearnLogRegClassifier()

    class DummyNoDecision:
        def predict_proba(self, X):
            X = np.asarray(X)
            return np.full((int(X.shape[0]), 2), 0.5, dtype=np.float32)

        def predict(self, X):
            X = np.asarray(X)
            return np.zeros((int(X.shape[0]),), dtype=np.int64)

    clf._model = DummyNoDecision()

    scores = clf.predict_scores(np.zeros((3, 2)))
    assert scores.shape == (3, 2)


def test_sklearn_svm_rbf_wrapper_with_dummy_probability_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import modssc.supervised.backends.sklearn.svm_rbf as mod

    dummy_svm = SimpleNamespace(SVC=DummySVC)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_svm)

    X = np.random.default_rng(0).normal(size=(12, 2)).astype(np.float32)
    y = np.array([0, 1, 2] * 4, dtype=np.int64)

    clf = create_classifier("svm_rbf", backend="sklearn", params={"sigma": 1.5})
    clf.fit(X, y)
    scores = clf.predict_scores(X[:4])
    assert scores.shape == (4, 3)
    with pytest.raises(NotSupportedError):
        _ = clf.predict_proba(X[:4])


def test_sklearn_svm_rbf_wrapper_with_dummy_probability_on(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.svm_rbf as mod

    dummy_svm = SimpleNamespace(SVC=DummySVC)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_svm)

    X = np.random.default_rng(0).normal(size=(10, 2)).astype(np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)

    clf = create_classifier(
        "svm_rbf", backend="sklearn", params={"sigma": 1.0, "probability": True}
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X[:4])
    assert proba.shape == (4, 2)
    assert clf.supports_proba


def test_sklearn_logreg_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.logreg as mod

    dummy_linear = SimpleNamespace(LogisticRegression=DummyLogisticRegression)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_linear)

    clf = create_classifier("logreg", backend="sklearn")
    X = np.random.randn(5, 2)

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_proba(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_scores(X)


def test_sklearn_logreg_binary_and_params(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.logreg as mod

    dummy_linear = SimpleNamespace(LogisticRegression=DummyLogisticRegression)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_linear)

    X = np.random.randn(10, 2).astype(np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)

    clf = create_classifier("logreg", backend="sklearn", params={"penalty": "l1"})
    assert clf.penalty == "l1"

    clf.fit(X, y)
    assert clf._model.kwargs["penalty"] == "l1"

    scores = clf.predict_scores(X[:3])
    assert scores.shape == (3, 2)


def test_sklearn_svm_rbf_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.svm_rbf as mod

    dummy_svm = SimpleNamespace(SVC=DummySVC)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_svm)

    clf = create_classifier("svm_rbf", backend="sklearn")
    X = np.random.randn(5, 2)

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_proba(X)
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_scores(X)


def test_sklearn_svm_rbf_binary_and_params(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.svm_rbf as mod

    dummy_svm = SimpleNamespace(SVC=DummySVC)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_svm)

    X = np.random.randn(10, 2).astype(np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)

    clf = create_classifier("svm_rbf", backend="sklearn", params={"probability": True})
    assert clf.probability

    clf.fit(X, y)
    assert clf._model.kwargs["probability"]

    scores = clf.predict_scores(X[:3])
    assert scores.shape == (3, 2)


def test_sklearn_svm_rbf_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.svm_rbf as mod

    class DummySVCNoDecision:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._n_classes = 0
            self.probability = bool(kwargs.get("probability", False))

        def fit(self, X, y):
            y = np.asarray(y).reshape(-1)
            self._n_classes = int(np.unique(y).size) if y.size else 0
            return self

        def predict(self, X):
            X = np.asarray(X)
            return np.zeros((int(X.shape[0]),), dtype=np.int64)

        def predict_proba(self, X):
            if not self.probability:
                raise RuntimeError("probability disabled")
            X = np.asarray(X)
            n = int(X.shape[0])
            if self._n_classes <= 0:
                return np.zeros((n, 0), dtype=np.float32)
            return np.full((n, self._n_classes), 1.0 / float(self._n_classes), dtype=np.float32)

    dummy_svm = SimpleNamespace(SVC=DummySVCNoDecision)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_svm)

    X = np.random.randn(10, 2).astype(np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)

    clf = create_classifier("svm_rbf", backend="sklearn", params={"probability": True})
    clf.fit(X, y)

    scores = clf.predict_scores(X[:3])
    assert scores.shape == (3, 2)

    pred = clf.predict(X[:3])
    assert pred.shape == (3,)


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

    _check_proba_classifier(mod.SklearnExtraTreesClassifier, mod, monkeypatch)


def test_gradient_boosting_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.gradient_boosting as mod

    _check_proba_classifier(mod.SklearnGradientBoostingClassifier, mod, monkeypatch)


def test_random_forest_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.random_forest as mod

    _check_proba_classifier(mod.SklearnRandomForestClassifier, mod, monkeypatch)


def test_naive_bayes_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.naive_bayes as mod

    for cls in (
        mod.SklearnGaussianNBClassifier,
        mod.SklearnMultinomialNBClassifier,
        mod.SklearnBernoulliNBClassifier,
    ):
        _check_proba_classifier(cls, mod, monkeypatch)


def test_linear_svm_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.linear_svm as mod

    dummy_module = SimpleNamespace(LinearSVC=DummyLinearModel)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_module)

    X = np.random.default_rng(0).normal(size=(4, 2)).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    clf = mod.SklearnLinearSVMClassifier()
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

    dummy_module = SimpleNamespace(LinearSVC=DummyLinearModel2D)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_module)

    X = np.random.default_rng(1).normal(size=(5, 2)).astype(np.float32)
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)

    clf = mod.SklearnLinearSVMClassifier()
    clf.fit(X, y)

    scores = clf.predict_scores(X[:2])
    assert scores.shape == (2, 3)


def test_ridge_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import modssc.supervised.backends.sklearn.ridge as mod

    dummy_module = SimpleNamespace(RidgeClassifier=DummyLinearModel)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_module)

    X = np.random.default_rng(0).normal(size=(4, 2)).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    clf = mod.SklearnRidgeClassifier()
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

    dummy_module = SimpleNamespace(RidgeClassifier=DummyLinearModel2D)
    monkeypatch.setattr(mod, "optional_import", lambda *a, **k: dummy_module)

    X = np.random.default_rng(1).normal(size=(5, 2)).astype(np.float32)
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)

    clf = mod.SklearnRidgeClassifier()
    clf.fit(X, y)

    scores = clf.predict_scores(X[:2])
    assert scores.shape == (2, 3)
