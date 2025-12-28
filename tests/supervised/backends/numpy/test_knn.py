from __future__ import annotations

import numpy as np
import pytest

from modssc.supervised.api import create_classifier
from modssc.supervised.errors import SupervisedValidationError


def _make_easy_binary() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=-2.0, scale=0.2, size=(20, 2))
    X1 = rng.normal(loc=+2.0, scale=0.2, size=(20, 2))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    return X, y


def test_numpy_knn_fit_predict_proba_shapes() -> None:
    X, y = _make_easy_binary()
    clf = create_classifier("knn", backend="numpy", params={"k": 3, "metric": "euclidean"})
    clf.fit(X, y)
    proba = clf.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    scores = clf.predict_scores(X[:5])
    assert scores.shape == (5, 2)

    pred = clf.predict(X[:5])
    assert pred.shape == (5,)


def test_numpy_knn_supports_proba_property() -> None:
    clf = create_classifier("knn", backend="numpy")
    assert clf.supports_proba is True


def test_numpy_knn_reasonable_accuracy() -> None:
    X, y = _make_easy_binary()
    clf = create_classifier("knn", backend="numpy", params={"k": 1, "metric": "euclidean"})
    clf.fit(X, y)
    pred = clf.predict(X)
    acc = float((pred == y).mean())
    assert acc >= 0.95


def test_numpy_knn_metric_cosine_works() -> None:
    X, y = _make_easy_binary()
    clf = create_classifier("knn", backend="numpy", params={"k": 3, "metric": "cosine"})
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape


def test_numpy_knn_validates_inputs() -> None:
    X, y = _make_easy_binary()
    clf = create_classifier("knn", backend="numpy", params={"k": 0})
    try:
        clf.fit(X, y)
    except SupervisedValidationError:
        return
    raise AssertionError("Expected SupervisedValidationError")


def test_numpy_knn_validation_errors() -> None:
    X, y = _make_easy_binary()
    clf = create_classifier("knn", backend="numpy", params={"k": 3})

    with pytest.raises(SupervisedValidationError, match="X must be non-empty"):
        clf.fit(np.zeros((0, 2)), np.array([]))

    with pytest.raises(SupervisedValidationError, match="incompatible sizes"):
        clf.fit(X, y[:10])

    clf_bad_metric = create_classifier("knn", backend="numpy", params={"k": 3, "metric": "bad"})
    with pytest.raises(SupervisedValidationError, match="Unknown metric"):
        clf_bad_metric.fit(X, y)

    clf_bad_weights = create_classifier("knn", backend="numpy", params={"k": 3, "weights": "bad"})
    with pytest.raises(SupervisedValidationError, match="Unknown weights"):
        clf_bad_weights.fit(X, y)


def test_numpy_knn_not_fitted_errors() -> None:
    clf = create_classifier("knn", backend="numpy", params={"k": 3})
    X = np.random.randn(5, 2)

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict(X)

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_proba(X)

    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf._pairwise_scores(X)


def test_numpy_knn_distance_weights() -> None:
    X, y = _make_easy_binary()

    clf = create_classifier(
        "knn", backend="numpy", params={"k": 3, "weights": "distance", "metric": "euclidean"}
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape

    clf = create_classifier(
        "knn", backend="numpy", params={"k": 3, "weights": "distance", "metric": "cosine"}
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape


def test_numpy_knn_list_input() -> None:
    X = [[0.0, 1.0], [1.0, 0.0]]
    y = [0, 1]
    clf = create_classifier("knn", backend="numpy", params={"k": 1})
    clf.fit(X, y)
    assert clf._X_train.dtype == np.float32
