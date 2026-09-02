from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import modssc.inductive.execution as execution
from modssc.capabilities import CapabilityIssue, CompatibilityReport, IncompatiblePipelineError
from modssc.inductive.base import MethodInfo
from modssc.inductive.model_binding import ModelBindingError
from modssc.runtime.method_spec import MethodSpecError
from tests.inductive.test_execution import _preprocess, _sampling


def test_torch_dependency_error_is_native(monkeypatch) -> None:
    monkeypatch.setattr(
        execution.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("torch")),
    )
    with pytest.raises(execution.InductiveExecutionError, match="requires dependency"):
        execution._torch_module()


def test_execution_container_helpers_cover_nested_backends() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.ones(1)
    assert execution._array_backend_flags(tensor) == (True, False)
    assert execution._array_backend_flags({"x": tensor}) == (True, False)
    assert execution._array_backend_flags([tensor, "unknown"]) == (True, True)
    assert execution._array_backend_flags((np.ones(1),)) == (False, True)
    assert execution._is_torch_container({"x": tensor})
    assert not execution._is_torch_container({"x": tensor, "meta": np.ones(1)})

    assert execution._torch_container_device({"x": tensor}) == tensor.device
    assert execution._torch_container_device({"nested": ["x", tensor]}) == tensor.device
    assert execution._torch_container_device(["x"]) is None
    assert execution._torch_container_device("x") is None
    assert execution._feature_tensor(None) is None
    assert execution._feature_tensor(np.ones(1)).shape == (1,)
    assert execution._feature_tensor({"x": tensor}) is tensor
    assert execution._feature_tensor({"meta": 1}) is None


def test_leading_size_and_dtype_descriptors_handle_unknown_shapes() -> None:
    class NoShape:
        shape = None

    class EmptyShape:
        shape = ()

    class BadShape:
        shape = ("not-an-int",)

    class UnlistableShape:
        dtype = "custom"
        shape = 1

    assert execution._leading_size(None) is None
    assert execution._leading_size({"x": NoShape()}) is None
    assert execution._leading_size(EmptyShape()) is None
    assert execution._leading_size(BadShape()) is None
    assert execution._dtype_descriptor(None) is None
    assert execution._dtype_descriptor(SimpleNamespace(dtype=None, shape=None)) is None
    assert execution._dtype_descriptor(UnlistableShape()) == {"dtype": "custom", "shape": None}


def test_smart_torch_conversion_covers_mapping_move_and_uint8() -> None:
    torch = pytest.importorskip("torch")
    assert execution._smart_to_torch(None, "cpu") is None
    assert execution._smart_to_torch({"x": np.ones(1)}, "cpu")["x"].device.type == "cpu"
    cpu = torch.ones(1)
    assert execution._smart_to_torch(cpu, cpu.device) is cpu
    assert execution._smart_to_torch(cpu, torch.device("meta")).device.type == "meta"
    assert execution._smart_to_torch(np.array([255], dtype=np.uint8), "cpu").item() == 1.0


def test_label_materialization_fallback_object_and_torch_paths() -> None:
    torch = pytest.importorskip("torch")
    preprocess = _preprocess()
    preprocess.train_artifacts.set("labels.y", np.array([0, 1]))
    indices = np.array([5], dtype=np.int64)
    with pytest.raises(execution.InductiveExecutionError, match="shorter"):
        execution._labels_for_backend(preprocess, np.ones((1, 2)), indices, strict=True)
    np.testing.assert_array_equal(
        execution._labels_for_backend(preprocess, np.ones((1, 2)), indices, strict=False), [1]
    )

    preprocess.train_artifacts.set("labels.y", np.array([0, None] + [1] * 8, dtype=object))
    np.testing.assert_array_equal(
        execution._labels_for_backend(preprocess, np.ones((2, 2)), np.array([0, 1]), strict=False),
        [0, -1],
    )

    preprocess.train_artifacts.set("labels.y", torch.arange(10))
    cpu_subset = execution._labels_for_backend(
        preprocess,
        np.ones((1, 2)),
        np.array([2]),
        strict=False,
    )
    assert cpu_subset.device.type == "cpu"
    moved = execution._labels_for_backend(
        preprocess,
        torch.ones((1, 2), device="meta"),
        np.array([2]),
        strict=False,
    )
    assert moved.device.type == "meta"

    preprocess.train_artifacts.data.clear()
    preprocess = replace(
        preprocess,
        dataset=replace(
            preprocess.dataset,
            train=replace(preprocess.dataset.train, y=torch.tensor([0, 1])),
        ),
    )
    untouched = execution._labels_for_backend(
        preprocess, torch.ones((1, 2)), np.array([5]), strict=False
    )
    assert untouched.shape == (2,)
    preprocess = replace(
        preprocess,
        dataset=replace(
            preprocess.dataset,
            train=replace(preprocess.dataset.train, y=np.array([0])),
        ),
    )
    np.testing.assert_array_equal(
        execution._labels_for_backend(preprocess, np.ones((1, 2)), np.array([5]), strict=False),
        [0],
    )


def test_strict_tensor_validation_covers_absent_shape_dtype_and_alignment() -> None:
    torch = pytest.importorskip("torch")
    execution._validate_tensor_contract("missing", {"meta": 1})
    with pytest.raises(execution.InductiveExecutionError, match="at least 2D"):
        execution._validate_tensor_contract("X", np.ones(2))
    with pytest.raises(execution.InductiveExecutionError, match="floating torch"):
        execution._validate_tensor_contract("X", torch.ones((2, 2), dtype=torch.int64))
    execution._validate_tensor_contract("X", np.ones((2, 2)))

    with pytest.raises(execution.InductiveExecutionError, match="torch-backed"):
        execution._validate_strict_inputs(
            X_l=np.ones((2, 2)),
            y_l=np.array([0, 1]),
            X_u=None,
            X_u_w=None,
            X_u_s=None,
            X_u_s_1=None,
            requires_torch=True,
        )
    with pytest.raises(execution.InductiveExecutionError, match="row mismatch"):
        execution._validate_strict_inputs(
            X_l=np.ones((2, 2)),
            y_l=np.array([0]),
            X_u=None,
            X_u_w=None,
            X_u_s=None,
            X_u_s_1=None,
            requires_torch=False,
        )


def _views():
    dataset = SimpleNamespace(train=SimpleNamespace(X=np.arange(20).reshape(10, 2)))
    return SimpleNamespace(views={"left": dataset})


def test_view_materialization_covers_strict_and_conversion_paths() -> None:
    torch = pytest.importorskip("torch")
    indices_l = np.array([0, 1])
    indices_u = np.array([2, 3])
    payload = execution._views_for_backend(
        _views(),
        labeled_indices=indices_l,
        unlabeled_indices=indices_u,
        backend_reference=np.ones((2, 2)),
        strict=True,
    )
    assert isinstance(payload["left"]["X_l"], np.ndarray)
    with pytest.raises(execution.InductiveExecutionError, match="torch-backed"):
        execution._views_for_backend(
            _views(),
            labeled_indices=indices_l,
            unlabeled_indices=indices_u,
            backend_reference=torch.ones((2, 2)),
            strict=True,
        )
    converted = execution._views_for_backend(
        _views(),
        labeled_indices=indices_l,
        unlabeled_indices=indices_u,
        backend_reference=torch.ones((2, 2)),
        strict=False,
    )
    assert isinstance(converted["left"]["X_l"], torch.Tensor)

    torch_views = SimpleNamespace(
        views={"left": SimpleNamespace(train=SimpleNamespace(X=torch.arange(20).reshape(10, 2)))}
    )
    strict_payload = execution._views_for_backend(
        torch_views,
        labeled_indices=indices_l,
        unlabeled_indices=indices_u,
        backend_reference=torch.ones((2, 2)),
        strict=True,
    )
    assert isinstance(strict_payload["left"]["X_l"], torch.Tensor)


def test_during_fit_split_validation_and_torch_conversion() -> None:
    torch = pytest.importorskip("torch")
    preprocess = _preprocess()
    sampling = _sampling()
    with pytest.raises(execution.InductiveExecutionError, match="absent"):
        execution._evaluation_splits_for_backend(
            preprocess,
            sampling,
            splits=("missing",),
            backend_reference=np.ones((1, 2)),
            strict=False,
        )

    sampling.indices["train_eval"] = np.array([0, 1])
    sampling.refs["train_eval"] = "train"
    train = execution._evaluation_splits_for_backend(
        preprocess,
        sampling,
        splits=("train_eval",),
        backend_reference=np.ones((1, 2)),
        strict=False,
    )
    np.testing.assert_array_equal(train["train_eval"]["y"], [0, 1])

    sampling.indices["bad"] = np.array([0])
    sampling.refs["bad"] = "other"
    with pytest.raises(execution.InductiveExecutionError, match="no source"):
        execution._evaluation_splits_for_backend(
            preprocess,
            sampling,
            splits=("bad",),
            backend_reference=np.ones((1, 2)),
            strict=False,
        )
    with pytest.raises(execution.InductiveExecutionError, match="not torch-backed"):
        execution._evaluation_splits_for_backend(
            preprocess,
            sampling,
            splits=("test",),
            backend_reference=torch.ones((1, 2)),
            strict=True,
        )
    converted = execution._evaluation_splits_for_backend(
        preprocess,
        sampling,
        splits=("test",),
        backend_reference=torch.ones((1, 2)),
        strict=False,
    )
    assert isinstance(converted["test"]["X"], torch.Tensor)
    assert isinstance(converted["test"]["y"], torch.Tensor)

    torch_preprocess = replace(
        preprocess,
        dataset=replace(
            preprocess.dataset,
            test=replace(
                preprocess.dataset.test,
                X=torch.arange(8, dtype=torch.float32).reshape(4, 2),
                y=torch.tensor([1, 0, 1, 0]),
            ),
        ),
    )
    strict_split = execution._evaluation_splits_for_backend(
        torch_preprocess,
        sampling,
        splits=("test",),
        backend_reference=torch.ones((1, 2)),
        strict=True,
    )
    assert isinstance(strict_split["test"]["X"], torch.Tensor)


def test_method_metadata_helpers_cover_callable_and_fallbacks() -> None:
    assert (
        execution._partition_artifact_sha256(
            {
                "policy": {"partition_artifact_sha256": None},
                "partition_artifact_sha256": "legacy",
            }
        )
        == "legacy"
    )
    assert (
        execution._unlabeled_index_space(SimpleNamespace(unlabeled_index_space=lambda: "local"))
        == "local"
    )
    assert (
        execution._method_backend(
            execution.InductiveExecutionConfig(method_id="x"),
            SimpleNamespace(_backend=None),
            SimpleNamespace(backend="torch"),
        )
        == "torch"
    )
    assert (
        execution._method_backend(execution.InductiveExecutionConfig(method_id="x"), object(), None)
        is None
    )


def test_build_method_translates_native_spec_and_binding_errors(monkeypatch) -> None:
    class Method:
        def __init__(self, spec=None):
            self.spec = spec

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: Method)
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: MethodInfo(method_id="x", name="X"),
    )
    monkeypatch.setattr(
        execution,
        "build_method_spec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(MethodSpecError("method_spec", "bad")),
    )
    with pytest.raises(execution.InductiveExecutionError, match="bad"):
        execution._build_method(
            execution.InductiveExecutionConfig(method_id="x"), X_l=np.ones((1, 2)), y_l=[0]
        )

    monkeypatch.setattr(execution, "build_method_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        execution,
        "bind_model_to_spec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelBindingError("model_config", "bad model")
        ),
    )
    with pytest.raises(execution.InductiveExecutionError, match="bad model"):
        execution._build_method(
            execution.InductiveExecutionConfig(method_id="x"), X_l=np.ones((1, 2)), y_l=[0]
        )


def test_prepare_inputs_strict_and_declared_torch_backend(monkeypatch) -> None:
    inputs = execution.InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling())
    with pytest.raises(execution.InductiveExecutionError, match="auto"):
        execution._prepare_inputs(
            inputs,
            execution.InductiveExecutionConfig(
                method_id="x", params={"backend": "auto"}, strict=True
            ),
        )

    seen = []
    monkeypatch.setattr(execution, "_torch_module", lambda: seen.append(True))
    prepared = execution._prepare_inputs(
        inputs,
        execution.InductiveExecutionConfig(method_id="x", params={"backend": "torch"}, strict=True),
    )
    assert seen == [True]
    assert prepared.X_l.shape == (2, 2)


def test_dataset_assembly_covers_missing_unlabeled_and_graph_guards() -> None:
    inputs = execution.InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling())
    config = execution.InductiveExecutionConfig(method_id="x")
    prepared = execution._prepare_inputs(inputs, config)
    without_unlabeled = replace(prepared, X_u=None)
    data = execution._assemble_inductive_dataset(
        inputs,
        config,
        method=SimpleNamespace(info=MethodInfo("x", "X")),
        prepared=without_unlabeled,
    )
    assert "idx_u" not in data.meta

    unknown_source = replace(prepared, X_train=object())
    with pytest.raises(execution.InductiveExecutionError, match="population size"):
        execution._assemble_inductive_dataset(
            inputs,
            config,
            method=SimpleNamespace(info=MethodInfo("x", "X")),
            prepared=unknown_source,
        )

    for graph, message in [
        (SimpleNamespace(n_nodes=True, meta={}), "integer n_nodes"),
        (SimpleNamespace(n_nodes=1, meta={}), "fewer nodes"),
    ]:
        with pytest.raises(execution.InductiveExecutionError, match=message):
            execution._assemble_inductive_dataset(
                replace(inputs, graph=graph),
                config,
                method=SimpleNamespace(info=MethodInfo("x", "X")),
                prepared=prepared,
            )
    valid_graph = SimpleNamespace(n_nodes=10, meta={})
    graph_data = execution._assemble_inductive_dataset(
        replace(inputs, graph=valid_graph),
        config,
        method=SimpleNamespace(info=MethodInfo("x", "X")),
        prepared=prepared,
    )
    assert "graph_fingerprint" not in graph_data.meta

    prepared_public = execution.prepare_inductive_dataset(
        inputs, config, method=SimpleNamespace(info=MethodInfo("x", "X"))
    )
    assert prepared_public.X_l.shape == (2, 2)


def test_execute_translates_capability_and_fitted_state_errors(monkeypatch) -> None:
    class Method:
        info = MethodInfo(method_id="x", name="X")

        def fit(self, data, *, device, seed):
            return self

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: Method)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: Method.info)
    report = CompatibilityReport(
        method_id="x",
        issues=(CapabilityIssue(code="test", message="incompatible"),),
    )
    monkeypatch.setattr(
        execution,
        "validate_consumed_input_capabilities",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(IncompatiblePipelineError(report)),
    )
    with pytest.raises(execution.InductiveExecutionError, match="incompatible"):
        execution.execute_inductive_method(
            execution.InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            execution.InductiveExecutionConfig(method_id="x"),
        )

    class LockedMethod:
        __slots__ = ()
        info = MethodInfo(method_id="x", name="X")

        def fit(self, data, *, device, seed):
            return self

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: LockedMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: LockedMethod.info)
    monkeypatch.setattr(
        execution,
        "validate_consumed_input_capabilities",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: {}),
    )
    with pytest.raises(execution.InductiveExecutionError, match="evaluation_runtime"):
        execution.execute_inductive_method(
            execution.InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            execution.InductiveExecutionConfig(method_id="x"),
        )
