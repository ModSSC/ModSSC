from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bench.orchestrators import evaluation


class _InitialEnsembleMethod:
    def __init__(self) -> None:
        self._initial_clfs = [object()]
        self.diagnostics_: dict[str, object] = {}

    def predict_proba(self, X):
        return np.eye(2, dtype=np.float32)[np.asarray(X, dtype=np.int64).reshape(-1)]

    def predict_proba_initial(self, X):
        labels = 1 - np.asarray(X, dtype=np.int64).reshape(-1)
        return np.eye(2, dtype=np.float32)[labels]


def test_inductive_evaluation_reports_retained_round_zero_ensemble(monkeypatch) -> None:
    method = _InitialEnsembleMethod()
    X = np.array([0, 1], dtype=np.int64)
    y = np.array([0, 1], dtype=np.int64)
    monkeypatch.setattr(evaluation, "_split_data", lambda *_args, **_kwargs: (X, y))

    result = evaluation.evaluate_inductive(
        method=method,
        pre=SimpleNamespace(),
        sampling=SimpleNamespace(),
        report_splits=["test"],
        metrics=["accuracy"],
        method_id="tri_training",
        views=None,
        strict=False,
    )

    assert result == {
        "test": {"accuracy": 1.0},
        "test_initial": {"accuracy": 0.0},
    }
    assert method.diagnostics_["initial_evaluation"] == {"test": {"accuracy": 0.0}}
