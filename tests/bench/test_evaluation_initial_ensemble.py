from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bench.orchestrators import evaluation
from modssc.evaluation import InductiveEvaluationSplit


class _InitialEnsembleMethod:
    def __init__(self) -> None:
        self.diagnostics_: dict[str, object] = {}

    def predict_proba(self, X):
        return np.eye(2, dtype=np.float32)[np.asarray(X, dtype=np.int64).reshape(-1)]

    def predict_evaluation_outputs(self, X):
        labels = 1 - np.asarray(X, dtype=np.int64).reshape(-1)
        return {"initial": np.eye(2, dtype=np.float32)[labels]}

    def record_evaluation_metrics(self, *, split, output, metrics):
        assert output == "initial"
        self.diagnostics_.setdefault("initial_evaluation", {})[split] = dict(metrics)


def test_inductive_evaluation_reports_retained_round_zero_ensemble(monkeypatch) -> None:
    method = _InitialEnsembleMethod()
    X = np.array([0, 1], dtype=np.int64)
    y = np.array([0, 1], dtype=np.int64)
    monkeypatch.setattr(
        evaluation,
        "make_inductive_split_provider",
        lambda **_kwargs: lambda _split: InductiveEvaluationSplit(X=X, y_true=y),
    )

    result = evaluation.evaluate_inductive(
        method=method,
        pre=SimpleNamespace(),
        sampling=SimpleNamespace(),
        report_splits=["test"],
        metrics=["accuracy"],
        views=None,
        strict=False,
    )

    assert result == {
        "test": {"accuracy": 1.0},
        "test_initial": {"accuracy": 0.0},
    }
    assert method.diagnostics_["initial_evaluation"] == {"test": {"accuracy": 0.0}}
