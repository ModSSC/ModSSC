from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pytest

import modssc.evaluation.runtime as runtime
from modssc.evaluation import EvaluationError, InductiveEvaluationSplit, MethodEvaluationRuntime


class _EmptyStore:
    @staticmethod
    def has(_name: str) -> bool:
        return False


def _preprocess(*, test=True):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train=SimpleNamespace(X=np.arange(8).reshape(4, 2), y=np.array([0, 1, 0, 1])),
            test=(
                SimpleNamespace(X=np.arange(6).reshape(3, 2), y=np.array([1, 0, 1]))
                if test
                else None
            ),
        ),
        train_artifacts=_EmptyStore(),
        test_artifacts=_EmptyStore(),
    )


def _sampling(*, reference="train", graph=False):
    return SimpleNamespace(
        is_graph=lambda: graph,
        indices={"test": np.array([0, 1])},
        refs={"test": reference},
    )


def test_runtime_and_split_selection_validation() -> None:
    with pytest.raises(ValueError, match="backend must"):
        MethodEvaluationRuntime(backend="jax")
    explicit = MethodEvaluationRuntime.from_features(np.ones((1, 1)), backend="numpy")
    assert explicit == MethodEvaluationRuntime(backend="numpy")

    with pytest.raises(EvaluationError, match="graph sampling"):
        runtime._select_inductive_split(
            preprocess=_preprocess(), sampling=_sampling(graph=True), views=None, split="test"
        )
    missing = _sampling()
    missing.indices = {}
    with pytest.raises(EvaluationError, match="unknown evaluation split"):
        runtime._select_inductive_split(
            preprocess=_preprocess(), sampling=missing, views=None, split="test"
        )
    with pytest.raises(EvaluationError, match="no test split"):
        runtime._select_inductive_split(
            preprocess=_preprocess(test=False),
            sampling=_sampling(reference="test"),
            views=None,
            split="test",
        )
    with pytest.raises(EvaluationError, match="unknown reference"):
        runtime._select_inductive_split(
            preprocess=_preprocess(),
            sampling=_sampling(reference="validation"),
            views=None,
            split="test",
        )

    selected = runtime._select_inductive_split(
        preprocess=_preprocess(),
        sampling=_sampling(reference="test"),
        views=None,
        split="test",
    )
    np.testing.assert_array_equal(selected.y_true, [1, 0])


def test_split_selection_rejects_missing_test_view() -> None:
    views = SimpleNamespace(
        views={"left": SimpleNamespace(train=SimpleNamespace(X=np.ones((4, 1))), test=None)}
    )
    with pytest.raises(EvaluationError, match="view 'left'"):
        runtime._select_inductive_split(
            preprocess=_preprocess(),
            sampling=_sampling(reference="test"),
            views=views,
            split="test",
        )


def test_backend_container_helpers_cover_nested_values() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.ones(1)
    assert runtime._array_backend_flags(tensor) == (True, False)
    assert runtime._array_backend_flags({"a": tensor, "b": np.ones(1)}) == (True, True)
    assert runtime._array_backend_flags(["unknown", tensor, np.ones(1)]) == (True, True)
    assert runtime._array_backend_flags(()) == (False, False)
    assert runtime._array_backend_flags(np.ones(1)) == (False, True)
    assert runtime._array_backend_flags("unknown") == (False, False)
    assert runtime._is_torch_container(tensor)
    assert runtime._is_torch_container({"x": tensor})
    assert not runtime._is_torch_container({"x": tensor, "meta": np.ones(1)})
    assert not runtime._is_torch_container([])

    assert runtime._first_torch_device(None) is None
    assert runtime._first_torch_device(np.ones(1)) is None
    assert runtime._first_torch_device({"empty": "x", "tensor": tensor}) == tensor.device
    assert runtime._first_torch_device({"empty": "x"}) is None
    assert runtime._first_torch_device(["x", tensor]) == tensor.device
    assert runtime._first_torch_device(["x"]) is None
    assert runtime._first_torch_device(tensor) == tensor.device


def test_smart_torch_materialization_covers_all_conversion_paths() -> None:
    torch = pytest.importorskip("torch")
    assert runtime._smart_to_torch(None, "cpu") is None
    mapped = runtime._smart_to_torch({"x": np.ones((1, 1), dtype=np.float64)}, "cpu")
    assert mapped["x"].dtype == torch.float32
    tensor = torch.ones(1)
    assert runtime._smart_to_torch(tensor, "cpu").device.type == "cpu"
    assert runtime._smart_to_torch(np.array([255], dtype=np.uint8), "cpu").item() == 1.0
    assert runtime._smart_to_torch(np.array([1.0], dtype=np.float64), "cpu").dtype == torch.float32

    torch_runtime = MethodEvaluationRuntime(backend="torch", device=None)
    assert (
        runtime._materialize_for_runtime(
            tensor, runtime=torch_runtime, strict=True, context="features"
        )
        is tensor
    )
    assert (
        runtime._materialize_for_runtime(
            tensor,
            runtime=MethodEvaluationRuntime(backend="torch", device=torch.device("cpu")),
            strict=True,
            context="features",
        ).device.type
        == "cpu"
    )
    assert runtime._materialize_for_runtime(
        np.ones(1), runtime=MethodEvaluationRuntime(), strict=True, context="features"
    ).shape == (1,)
    assert (
        runtime._materialize_views(
            None, runtime=MethodEvaluationRuntime(), strict=False, context="test"
        )
        is None
    )


def test_public_runtime_and_method_contract_validation() -> None:
    configured = MethodEvaluationRuntime(backend="numpy")
    assert (
        runtime._public_method_runtime(SimpleNamespace(evaluation_runtime_=configured))
        is configured
    )
    with pytest.raises(EvaluationError, match="must be a MethodEvaluationRuntime"):
        runtime._public_method_runtime(SimpleNamespace(evaluation_runtime_="numpy"))

    with pytest.raises(EvaluationError, match="prediction_input"):
        runtime._prediction_input_contract(
            SimpleNamespace(info=SimpleNamespace(prediction_input="raw"))
        )
    assert runtime._evaluation_reference_splits(
        SimpleNamespace(evaluation_reference_splits=lambda: ("train",))
    ) == ("train",)
    assert runtime._evaluation_reference_splits(
        SimpleNamespace(info=SimpleNamespace(evaluation_reference_splits=("test",)))
    ) == ("test",)
    for declared in (["test"], ("",), ("test", "test")):
        with pytest.raises(EvaluationError, match="evaluation_reference_splits"):
            runtime._evaluation_reference_splits(
                SimpleNamespace(evaluation_reference_splits=declared)
            )


def test_dataset_prediction_requires_views() -> None:
    method = SimpleNamespace(info=SimpleNamespace(prediction_input="dataset"))
    split = InductiveEvaluationSplit(X=np.ones((1, 1)), y_true=np.array([0]))
    with pytest.raises(EvaluationError, match="requires views"):
        runtime._prediction_payload(
            method=method,
            split_name="test",
            split=split,
            split_provider=lambda _name: split,
            runtime=MethodEvaluationRuntime(),
            strict=False,
        )
    assert runtime._prediction_distribution(np.array([], dtype=np.int64)) == {}


class _DuplicateMapping(Mapping):
    def __getitem__(self, key):
        return 1

    def __iter__(self):
        return iter(("same",))

    def __len__(self):
        return 1

    def items(self):
        return [("same", 1), ("same", 2)]


@pytest.mark.parametrize(
    ("method", "message"),
    [
        (SimpleNamespace(predict_evaluation_outputs=1), "must be callable"),
        (SimpleNamespace(predict_evaluation_outputs=lambda _payload: []), "return a mapping"),
        (SimpleNamespace(predict_evaluation_outputs=lambda _payload: {1: 2}), "non-empty strings"),
        (
            SimpleNamespace(predict_evaluation_outputs=lambda _payload: _DuplicateMapping()),
            "duplicate",
        ),
    ],
)
def test_additional_output_contract_validation(method, message: str) -> None:
    with pytest.raises(EvaluationError, match=message):
        runtime._additional_outputs(method, object())


def test_primary_prediction_contracts() -> None:
    with pytest.raises(EvaluationError, match="predict_evaluation_proba"):
        runtime._primary_evaluation_scores(SimpleNamespace(predict_evaluation_proba=1), object())
    assert (
        runtime._primary_evaluation_scores(
            SimpleNamespace(predict_evaluation_proba=lambda _payload: "primary"), object()
        )
        == "primary"
    )
    with pytest.raises(EvaluationError, match="predict_proba"):
        runtime._primary_evaluation_scores(SimpleNamespace(), object())


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (1, "must be callable"),
        (lambda: [], "return a mapping"),
        (lambda: {1: {}}, "names must be"),
        (lambda: {"reported": []}, "must be a mapping"),
        (lambda: {"reported": {1: 2}}, "invalid key"),
    ],
)
def test_reported_metric_set_contract_validation(provider, message: str) -> None:
    with pytest.raises(EvaluationError, match=message):
        runtime._reported_metric_sets(SimpleNamespace(evaluation_metric_sets=provider))


def _split() -> InductiveEvaluationSplit:
    return InductiveEvaluationSplit(X=np.array([0, 1]), y_true=np.array([0, 1]))


class _ScoresMethod:
    def predict_proba(self, X):
        return np.eye(2)[np.asarray(X).reshape(-1)]


def test_inductive_evaluation_validates_top_level_contracts() -> None:
    for splits, message in [([""], "non-empty"), (["test", "test"], "unique")]:
        with pytest.raises(EvaluationError, match=message):
            runtime.evaluate_inductive_method(
                method=_ScoresMethod(),
                split_provider=lambda _name: _split(),
                report_splits=splits,
                metrics=["accuracy"],
            )
    with pytest.raises(EvaluationError, match="runtime must"):
        runtime.evaluate_inductive_method(
            method=_ScoresMethod(),
            split_provider=lambda _name: _split(),
            report_splits=["test"],
            metrics=["accuracy"],
            runtime="numpy",
        )
    with pytest.raises(EvaluationError, match="split_provider"):
        runtime.evaluate_inductive_method(
            method=_ScoresMethod(),
            split_provider=lambda _name: object(),
            report_splits=["test"],
            metrics=["accuracy"],
        )


def test_inductive_evaluation_detects_output_recorder_and_result_collisions() -> None:
    class WithOutput(_ScoresMethod):
        def predict_evaluation_outputs(self, X):
            return {"aux": self.predict_proba(X)}

    with pytest.raises(EvaluationError, match="result collision"):
        runtime.evaluate_inductive_method(
            method=WithOutput(),
            split_provider=lambda _name: _split(),
            report_splits=["test", "test_aux"],
            metrics=["accuracy"],
        )

    class CrossSplitOutput(_ScoresMethod):
        def predict_evaluation_outputs(self, X):
            return {"aux": self.predict_proba(X)} if int(np.asarray(X)[0]) == 0 else {}

    splits = {
        "test_aux": InductiveEvaluationSplit(X=np.array([1, 0]), y_true=np.array([1, 0])),
        "test": InductiveEvaluationSplit(X=np.array([0, 1]), y_true=np.array([0, 1])),
    }
    with pytest.raises(EvaluationError, match="output collision"):
        runtime.evaluate_inductive_method(
            method=CrossSplitOutput(),
            split_provider=splits.__getitem__,
            report_splits=["test_aux", "test"],
            metrics=["accuracy"],
        )

    method = WithOutput()
    method.record_evaluation_metrics = 1
    with pytest.raises(EvaluationError, match="record_evaluation_metrics"):
        runtime.evaluate_inductive_method(
            method=method,
            split_provider=lambda _name: _split(),
            report_splits=["test"],
            metrics=["accuracy"],
        )

    reported = _ScoresMethod()
    reported.evaluation_metric_sets = lambda: {"test": {"accuracy": 0.5}}
    with pytest.raises(EvaluationError, match="metric-set collision"):
        runtime.evaluate_inductive_method(
            method=reported,
            split_provider=lambda _name: _split(),
            report_splits=["test"],
            metrics=["accuracy"],
        )


def _node_data():
    return SimpleNamespace(
        fit=object(),
        evaluation=SimpleNamespace(
            y_true=np.array([0, 1]),
            masks={"test_mask": np.array([True, False])},
        ),
    )


def test_transductive_evaluation_validates_every_boundary() -> None:
    for splits, message in [([""], "non-empty"), (["test", "test"], "unique")]:
        with pytest.raises(EvaluationError, match=message):
            runtime.evaluate_transductive_method(
                method=SimpleNamespace(predict_proba=lambda _data: np.array([0, 1])),
                data=_node_data(),
                report_splits=splits,
                metrics=["accuracy"],
            )
    with pytest.raises(EvaluationError, match="PreparedNodeData"):
        runtime.evaluate_transductive_method(
            method=object(), data=object(), report_splits=[], metrics=[]
        )
    with pytest.raises(EvaluationError, match="predict_proba"):
        runtime.evaluate_transductive_method(
            method=object(), data=_node_data(), report_splits=[], metrics=[]
        )
    with pytest.raises(EvaluationError, match="one value per node"):
        runtime.evaluate_transductive_method(
            method=SimpleNamespace(predict_proba=lambda _data: np.array([0])),
            data=_node_data(),
            report_splits=[],
            metrics=[],
        )


@pytest.mark.parametrize(
    ("masks", "declared", "message"),
    [
        ({}, None, "missing mask"),
        ({"test_mask": np.ones((1, 1), dtype=bool)}, None, "mask size mismatch"),
        ({"test_mask": np.array([True, False])}, {}, "runtime mask differs"),
        (
            {"test_mask": np.array([True, False])},
            {"test_mask": np.array([False, True])},
            "runtime mask differs",
        ),
    ],
)
def test_transductive_mask_contracts(masks, declared, message: str) -> None:
    data = _node_data()
    data.evaluation.masks = masks
    with pytest.raises(EvaluationError, match=message):
        runtime.evaluate_transductive_method(
            method=SimpleNamespace(predict_proba=lambda _data: np.array([0, 1])),
            data=data,
            report_splits=["test"],
            metrics=["accuracy"],
            declared_masks=declared,
        )


def test_transductive_evaluation_accepts_runtime_masks_without_declared_copy() -> None:
    assert runtime.evaluate_transductive_method(
        method=SimpleNamespace(predict_proba=lambda _data: np.array([0, 1])),
        data=_node_data(),
        report_splits=["test"],
        metrics=["accuracy"],
        declared_masks=None,
    ) == {"test": {"accuracy": 1.0}}
