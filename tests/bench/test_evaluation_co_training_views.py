from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bench.errors import BenchRuntimeError
from bench.orchestrators import evaluation
from modssc.evaluation import InductiveEvaluationSplit
from modssc.inductive.base import MethodInfo
from modssc.inductive.types import InductiveDataset


class _DatasetPredictionMethod:
    info = MethodInfo(
        method_id="two_view_method",
        name="Two-view method",
        requires_views=True,
        prediction_input="dataset",
        evaluation_reference_splits=("train_labeled", "train"),
    )

    def __init__(self) -> None:
        self.combined_payload: Any | None = None
        self.named_payload: Any | None = None

    def predict_proba(self, data):
        self.combined_payload = data
        return np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]], dtype=np.float32)

    def predict_evaluation_outputs(self, data):
        self.named_payload = data
        return {
            "page": np.array(
                [[0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
                dtype=np.float32,
            ),
            "links": np.array(
                [[0.1, 0.9], [0.9, 0.1], [0.2, 0.8]],
                dtype=np.float32,
            ),
        }


def _evaluate(
    monkeypatch: pytest.MonkeyPatch,
    method: _DatasetPredictionMethod,
    *,
    views: object | None = None,
):
    X = np.zeros((3, 1), dtype=np.float32)
    y = np.array([0, 1, 1], dtype=np.int64)
    monkeypatch.setattr(
        evaluation,
        "make_inductive_split_provider",
        lambda **_kwargs: (
            lambda _split: InductiveEvaluationSplit(
                X=X,
                y_true=y,
                views={"page": {"X": X}, "links": {"X": X}},
            )
        ),
    )
    return evaluation.evaluate_inductive(
        method=method,
        pre=SimpleNamespace(),
        sampling=SimpleNamespace(),
        report_splits=["test"],
        metrics=["accuracy"],
        views=SimpleNamespace() if views is None else views,
        strict=False,
    )


def test_dataset_prediction_contract_reports_combined_and_named_outputs(monkeypatch) -> None:
    method = _DatasetPredictionMethod()

    results = _evaluate(monkeypatch, method)

    assert results == {
        "test": {"accuracy": 1.0},
        "test_page": {"accuracy": 1.0},
        "test_links": {"accuracy": 1.0 / 3.0},
    }
    assert isinstance(method.combined_payload, InductiveDataset)
    assert method.named_payload is method.combined_payload
    np.testing.assert_array_equal(method.combined_payload.X_l, np.zeros((3, 1)))
    assert method.combined_payload.y_l is None
    assert set(method.combined_payload.views or {}) == {"page", "links"}
    references = method.combined_payload.meta["evaluation_reference_splits"]
    assert set(references) == {"train_labeled", "train"}
    assert all(isinstance(reference, InductiveDataset) for reference in references.values())
    assert all(set(reference.views or {}) == {"page", "links"} for reference in references.values())


def test_dataset_prediction_contract_requires_declared_views(monkeypatch) -> None:
    method = _DatasetPredictionMethod()
    X = np.zeros((3, 1), dtype=np.float32)
    y = np.array([0, 1, 1], dtype=np.int64)
    monkeypatch.setattr(
        evaluation,
        "make_inductive_split_provider",
        lambda **_kwargs: lambda _split: InductiveEvaluationSplit(X=X, y_true=y),
    )

    with pytest.raises(BenchRuntimeError, match="requires views"):
        evaluation.evaluate_inductive(
            method=method,
            pre=SimpleNamespace(),
            sampling=SimpleNamespace(),
            report_splits=["test"],
            metrics=["accuracy"],
            views=None,
            strict=False,
        )


def test_named_prediction_contract_requires_a_mapping(monkeypatch) -> None:
    method = _DatasetPredictionMethod()
    method.predict_evaluation_outputs = lambda _data: []  # type: ignore[method-assign]

    with pytest.raises(BenchRuntimeError, match="must return a mapping"):
        _evaluate(monkeypatch, method)
