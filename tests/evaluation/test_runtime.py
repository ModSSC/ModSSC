from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from modssc.evaluation import (
    EvaluationError,
    InductiveEvaluationSplit,
    MethodEvaluationRuntime,
    evaluate_inductive_method,
    evaluate_transductive_method,
    make_inductive_split_provider,
)
from modssc.inductive.base import MethodInfo
from modssc.inductive.types import InductiveDataset


class _NamedMethod:
    info = MethodInfo(method_id="named", name="Named")

    def __init__(self) -> None:
        self.recorded: list[tuple[str, str, dict[str, float]]] = []

    def predict_proba(self, X: Any) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.eye(2, dtype=np.float32)[X.reshape(-1)]

    def predict_evaluation_outputs(self, X: Any) -> dict[str, np.ndarray]:
        labels = 1 - X.reshape(-1)
        return {"initial": np.eye(2, dtype=np.float32)[labels]}

    def record_evaluation_metrics(
        self,
        *,
        split: str,
        output: str,
        metrics: dict[str, float],
    ) -> None:
        self.recorded.append((split, output, dict(metrics)))


def test_native_inductive_evaluation_computes_primary_and_named_outputs() -> None:
    method = _NamedMethod()
    selected = InductiveEvaluationSplit(
        X=np.array([0, 1], dtype=np.int64),
        y_true=np.array([0, 1], dtype=np.int64),
    )

    result = evaluate_inductive_method(
        method=method,
        split_provider=lambda _name: selected,
        report_splits=["test"],
        metrics=["accuracy"],
    )

    assert result == {
        "test": {"accuracy": 1.0},
        "test_initial": {"accuracy": 0.0},
    }
    assert method.recorded == [("test", "initial", {"accuracy": 0.0})]


def test_native_inductive_evaluation_exposes_reported_and_terminal_metric_sets() -> None:
    class Method:
        def predict_proba(self, X: Any) -> np.ndarray:
            return np.eye(2, dtype=np.float32)[X.reshape(-1)]

        @staticmethod
        def evaluation_metric_sets() -> dict[str, dict[str, Any]]:
            return {
                "terminal": {
                    "test": {
                        "accuracy": 0.75,
                        "role": "terminal_checkpoint",
                    }
                },
                "reported": {
                    "test": {
                        "accuracy": 0.8,
                        "policy": "median_last_20_checkpoints",
                        "selection_uses_test": False,
                    }
                },
            }

    selected = InductiveEvaluationSplit(
        X=np.array([0, 1], dtype=np.int64),
        y_true=np.array([0, 1], dtype=np.int64),
    )
    result = evaluate_inductive_method(
        method=Method(),
        split_provider=lambda _name: selected,
        report_splits=["test"],
        metrics=["accuracy"],
    )

    assert result["test"] == {"accuracy": 1.0}
    assert result["terminal"]["test"]["accuracy"] == 0.75
    assert result["reported"]["test"] == {
        "accuracy": 0.8,
        "policy": "median_last_20_checkpoints",
        "selection_uses_test": False,
    }


def test_native_split_provider_selects_labels_and_views_from_declared_reference() -> None:
    class LabelStore:
        @staticmethod
        def has(name: str) -> bool:
            return name == "labels.y"

        @staticmethod
        def get(name: str) -> np.ndarray:
            assert name == "labels.y"
            return np.array([1, 1, 0, 0], dtype=np.int64)

    preprocess = SimpleNamespace(
        dataset=SimpleNamespace(
            train=SimpleNamespace(
                X=np.arange(8, dtype=np.float32).reshape(4, 2),
                y=np.zeros(4, dtype=np.int64),
            ),
            test=None,
        ),
        train_artifacts=LabelStore(),
        test_artifacts=None,
    )
    sampling = SimpleNamespace(
        is_graph=lambda: False,
        indices={"val": np.array([1, 3], dtype=np.int64)},
        refs={"val": "train"},
    )
    views = SimpleNamespace(
        views={
            "left": SimpleNamespace(
                train=SimpleNamespace(X=np.arange(12).reshape(4, 3)),
                test=None,
            )
        }
    )
    provider = make_inductive_split_provider(
        preprocess=preprocess,
        sampling=sampling,
        views=views,
    )

    selected = provider("val")

    np.testing.assert_array_equal(selected.X, preprocess.dataset.train.X[[1, 3]])
    np.testing.assert_array_equal(selected.y_true, [1, 0])
    np.testing.assert_array_equal(selected.views["left"]["X"], [[3, 4, 5], [9, 10, 11]])
    assert provider("val") is selected


class _DatasetMethod:
    info = MethodInfo(
        method_id="dataset",
        name="Dataset",
        prediction_input="dataset",
        evaluation_reference_splits=("train_labeled",),
    )

    def __init__(self) -> None:
        self.payload: InductiveDataset | None = None

    def predict_proba(self, data: InductiveDataset) -> np.ndarray:
        self.payload = data
        return np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)


def test_native_dataset_prediction_builds_declared_references() -> None:
    method = _DatasetMethod()
    splits = {
        "test": InductiveEvaluationSplit(
            X=np.zeros((2, 1), dtype=np.float32),
            y_true=np.array([0, 1], dtype=np.int64),
            views={"left": {"X": np.zeros((2, 2), dtype=np.float32)}},
        ),
        "train_labeled": InductiveEvaluationSplit(
            X=np.ones((3, 1), dtype=np.float32),
            y_true=np.array([0, 1, 0], dtype=np.int64),
            views={"left": {"X": np.ones((3, 2), dtype=np.float32)}},
        ),
    }

    result = evaluate_inductive_method(
        method=method,
        split_provider=splits.__getitem__,
        report_splits=["test"],
        metrics=["accuracy"],
    )

    assert result == {"test": {"accuracy": 1.0}}
    assert method.payload is not None
    assert method.payload.y_l is None
    assert method.payload.meta["evaluation_split"] == "test"
    references = method.payload.meta["evaluation_reference_splits"]
    assert set(references) == {"train_labeled"}
    np.testing.assert_array_equal(references["train_labeled"].y_l, [0, 1, 0])


def test_native_runtime_converts_numpy_predictions_inputs_to_fitted_torch_device() -> None:
    torch = pytest.importorskip("torch")

    class TorchMethod:
        info = MethodInfo(method_id="torch", name="Torch")

        def __init__(self) -> None:
            self.seen = None

        def predict_proba(self, X):
            self.seen = X
            return torch.eye(2, device=X.device)[X.to(torch.int64).reshape(-1)]

    method = TorchMethod()
    runtime = MethodEvaluationRuntime.from_features(torch.zeros(1))
    selected = InductiveEvaluationSplit(
        X=np.array([0, 1], dtype=np.int64),
        y_true=np.array([0, 1], dtype=np.int64),
    )
    result = evaluate_inductive_method(
        method=method,
        split_provider=lambda _name: selected,
        report_splits=["test"],
        metrics=["accuracy"],
        runtime=runtime,
    )

    assert result == {"test": {"accuracy": 1.0}}
    assert isinstance(method.seen, torch.Tensor)
    assert method.seen.device == runtime.device

    with pytest.raises(EvaluationError) as caught:
        evaluate_inductive_method(
            method=method,
            split_provider=lambda _name: selected,
            report_splits=["test"],
            metrics=["accuracy"],
            runtime=runtime,
            strict=True,
        )
    assert caught.value.kind == "torch_required"


def test_native_runtime_does_not_guess_device_from_method_attributes() -> None:
    class Method:
        device = "cpu"

        def predict_proba(self, X: Any) -> np.ndarray:
            assert isinstance(X, np.ndarray)
            return np.eye(2, dtype=np.float32)[X.reshape(-1)]

    selected = InductiveEvaluationSplit(
        X=np.array([0, 1], dtype=np.int64),
        y_true=np.array([0, 1], dtype=np.int64),
    )

    result = evaluate_inductive_method(
        method=Method(),
        split_provider=lambda _name: selected,
        report_splits=["test"],
        metrics=["accuracy"],
    )

    assert result == {"test": {"accuracy": 1.0}}


def test_native_runtime_rejects_device_without_torch_backend() -> None:
    with pytest.raises(ValueError, match="requires backend='torch'"):
        MethodEvaluationRuntime(backend="numpy", device="cpu")


def test_native_transductive_evaluation_passes_only_fit_data_to_method() -> None:
    fit = SimpleNamespace(meta={"public": True})
    data = SimpleNamespace(
        fit=fit,
        evaluation=SimpleNamespace(
            y_true=np.array([0, 1, 0, 1], dtype=np.int64),
            masks={
                "val_mask": np.array([False, False, True, False]),
                "test_mask": np.array([False, False, False, True]),
            },
        ),
    )

    class Method:
        def predict_proba(self, received):
            assert received is fit
            assert "y_true" not in received.meta
            return np.array(
                [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],
                dtype=np.float32,
            )

    result = evaluate_transductive_method(
        method=Method(),
        data=data,
        report_splits=["val", "test"],
        metrics=["accuracy"],
        declared_masks=data.evaluation.masks,
    )

    assert result == {"val": {"accuracy": 1.0}, "test": {"accuracy": 1.0}}


def test_native_evaluation_rejects_prediction_population_mismatch() -> None:
    class Method:
        def predict_proba(self, _X):
            return np.zeros((1, 2), dtype=np.float32)

    selected = InductiveEvaluationSplit(
        X=np.zeros((2, 1), dtype=np.float32),
        y_true=np.array([0, 1], dtype=np.int64),
    )
    with pytest.raises(EvaluationError) as caught:
        evaluate_inductive_method(
            method=Method(),
            split_provider=lambda _name: selected,
            report_splits=["test"],
            metrics=["accuracy"],
        )
    assert caught.value.kind == "shape"
