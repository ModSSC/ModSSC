from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bench.errors import BenchRuntimeError
from bench.orchestrators import evaluation


class _CoTrainingMethod:
    def __init__(self, *, protocol: str, view_keys=("page", "links")) -> None:
        self.spec = SimpleNamespace(protocol=protocol)
        self._view_keys = view_keys
        self.view_calls: list[str] = []
        self.diagnostics_: dict | None = {}

    def predict_proba(self, _data):
        return np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]], dtype=np.float32)

    def predict_view_proba(self, _data, view_key: str):
        self.view_calls.append(view_key)
        if view_key == "page":
            return np.array([[0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32)
        return np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8]], dtype=np.float32)


def _evaluate(monkeypatch: pytest.MonkeyPatch, method: _CoTrainingMethod):
    X = np.zeros((3, 1), dtype=np.float32)
    y = np.array([0, 1, 1], dtype=np.int64)
    monkeypatch.setattr(evaluation, "_split_data", lambda *_args, **_kwargs: (X, y))
    monkeypatch.setattr(
        evaluation,
        "_views_for_split",
        lambda *_args, **_kwargs: {
            "page": {"X": X},
            "links": {"X": X},
        },
    )
    return evaluation.evaluate_inductive(
        method=method,
        pre=SimpleNamespace(),
        sampling=SimpleNamespace(),
        report_splits=["test"],
        metrics=["accuracy"],
        method_id="co_training",
        views=SimpleNamespace(),
        strict=False,
    )


def test_blum_mitchell_evaluation_reports_combined_and_each_view(monkeypatch) -> None:
    method = _CoTrainingMethod(protocol="fixed_pool_binary")

    results = _evaluate(monkeypatch, method)

    assert results == {
        "test": {"accuracy": 1.0},
        "test_page": {"accuracy": 1.0},
        "test_links": {"accuracy": 1.0 / 3.0},
    }
    assert method.view_calls == ["page", "links"]


def test_legacy_co_training_evaluation_preserves_combined_only(monkeypatch) -> None:
    method = _CoTrainingMethod(protocol="legacy")

    results = _evaluate(monkeypatch, method)

    assert results == {"test": {"accuracy": 1.0}}
    assert method.view_calls == []


def test_nigam_ghani_evaluation_reports_views_and_supervised_controls(monkeypatch) -> None:
    method = _CoTrainingMethod(protocol="shared_pool_exhaustive_multiset")
    monkeypatch.setattr(
        evaluation,
        "_nigam_ghani_supervised_controls",
        lambda **_kwargs: {
            "test_nb12": {"accuracy": 2.0 / 3.0},
            "test_nb788": {"accuracy": 1.0},
        },
    )

    results = _evaluate(monkeypatch, method)

    assert results == {
        "test": {"accuracy": 1.0},
        "test_page": {"accuracy": 1.0},
        "test_links": {"accuracy": 1.0 / 3.0},
        "test_nb12": {"accuracy": 2.0 / 3.0},
        "test_nb788": {"accuracy": 1.0},
    }
    assert method.view_calls == ["page", "links"]
    assert method.diagnostics_ == {
        "supervised_controls": {
            "nb12_training_size": 12,
            "nb788_training_size": 788,
            "feature_space": "concatenated_namespaced_views",
            "class_prior_smoothing": "add_one",
            "test_metrics_used_for_protocol_selection": False,
        }
    }


def test_nigam_ghani_supervised_control_oracle(monkeypatch) -> None:
    y_by_split = {
        "train_labeled": np.array([0, 1], dtype=np.int64),
        "train": np.array([0, 0, 1, 1], dtype=np.int64),
    }
    matrices = {
        "train_labeled": {
            "page": {"X": np.array([[4.0], [0.0]])},
            "links": {"X": np.array([[0.0], [4.0]])},
        },
        "train": {
            "page": {"X": np.array([[4.0], [3.0], [0.0], [0.0]])},
            "links": {"X": np.array([[0.0], [0.0], [4.0], [3.0]])},
        },
    }
    monkeypatch.setattr(
        evaluation,
        "_split_data",
        lambda _pre, _sampling, *, split: (
            np.zeros((len(y_by_split[split]), 1)),
            y_by_split[split],
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "_views_for_split",
        lambda _views, *, split, **_kwargs: matrices[split],
    )
    test_views = {
        "page": {"X": np.array([[5.0], [0.0]])},
        "links": {"X": np.array([[0.0], [5.0]])},
    }

    controls = evaluation._nigam_ghani_supervised_controls(
        pre=object(),
        sampling=object(),
        views=object(),
        test_views=test_views,
        y_test=np.array([0, 1], dtype=np.int64),
        view_keys=("page", "links"),
        metrics=["accuracy"],
        strict=False,
    )

    assert controls == {
        "test_nb12": {"accuracy": 1.0},
        "test_nb788": {"accuracy": 1.0},
    }


def test_nigam_ghani_controls_do_not_require_mutable_method_diagnostics(monkeypatch) -> None:
    method = _CoTrainingMethod(protocol="shared_pool_exhaustive_multiset")
    method.diagnostics_ = None
    monkeypatch.setattr(
        evaluation,
        "_nigam_ghani_supervised_controls",
        lambda **_kwargs: {
            "test_nb12": {"accuracy": 1.0},
            "test_nb788": {"accuracy": 1.0},
        },
    )

    results = _evaluate(monkeypatch, method)

    assert results["test_nb12"] == {"accuracy": 1.0}
    assert method.diagnostics_ is None


@pytest.mark.parametrize("view_keys", [("page",), ["page", "links"]])
def test_blum_mitchell_evaluation_requires_two_fitted_view_keys(monkeypatch, view_keys) -> None:
    method = _CoTrainingMethod(protocol="fixed_pool_binary", view_keys=view_keys)

    with pytest.raises(BenchRuntimeError, match="exactly two view keys"):
        _evaluate(monkeypatch, method)
