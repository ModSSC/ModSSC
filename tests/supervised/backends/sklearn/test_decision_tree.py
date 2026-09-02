from __future__ import annotations

import importlib

import numpy as np
import pytest

from modssc.supervised import create_classifier


def test_module_importable() -> None:
    importlib.import_module("modssc.supervised.backends.sklearn.decision_tree")


def test_real_sklearn_decision_tree_fit_predict_and_proba() -> None:
    pytest.importorskip("sklearn")

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array(["low", "low", "mid", "mid", "high", "high"])
    classifier = create_classifier(
        "decision_tree",
        backend="sklearn",
        params={"criterion": "entropy", "seed": 17},
    )

    fit_result = classifier.fit(X, y)
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)

    assert fit_result.n_samples == X.shape[0]
    assert fit_result.n_features == X.shape[1]
    assert fit_result.n_classes == 3
    np.testing.assert_array_equal(predictions, y)
    assert probabilities.shape == (X.shape[0], 3)
    assert probabilities.dtype == np.float32
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
