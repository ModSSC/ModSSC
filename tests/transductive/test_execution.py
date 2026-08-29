from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

import modssc.transductive.execution as execution
from modssc.capabilities import MethodCapabilities
from modssc.data_loader.types import LoadedDataset, Split
from modssc.graph.artifacts import GraphArtifact, NodeDataset
from modssc.runtime.contracts import (
    ExecutionContractError,
    InputRoleRequirement,
    MethodExecutionContract,
)
from modssc.runtime.execution import ExecutionContext, RunIdentity
from modssc.runtime.outcome import MethodExecutionOutcome, MethodNotEvaluableError
from modssc.transductive import (
    DeviceSpec,
    TransductiveExecutionConfig,
    TransductiveExecutionError,
    TransductiveExecutionInput,
    execute_transductive_method,
)
from modssc.transductive.base import MethodInfo


@dataclass(frozen=True)
class _Spec:
    backend: str = "numpy"


class _CapturingMethod:
    info = MethodInfo(method_id="capture", name="Capture", supports_gpu=True)
    captured_data: ClassVar[NodeDataset | None] = None

    def __init__(self, spec: _Spec | None = None) -> None:
        self.spec = spec or _Spec()
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, data: NodeDataset, *, device: str | None, seed: int) -> _CapturingMethod:
        assert device == "cpu"
        assert seed == 7
        assert "y_true" not in data.meta
        assert "val_mask" not in data.masks
        assert "test_mask" not in data.masks
        type(self).captured_data = data
        self.diagnostics_ = {"fit": "ok"}
        return self

    def execution_resolution(self) -> dict[str, Any]:
        return {"backend": "runtime-public", "runtime_token": "native"}


def _dataset() -> LoadedDataset:
    return LoadedDataset(
        train=Split(
            X=np.arange(8, dtype=np.float32).reshape(4, 2),
            y=np.array([0, 1, 0, 1]),
        ),
        test=Split(
            X=np.arange(4, dtype=np.float32).reshape(2, 2),
            y=np.array([1, 0]),
        ),
    )


def _graph(*, n_nodes: int = 6) -> GraphArtifact:
    return GraphArtifact(
        n_nodes=n_nodes,
        edge_index=np.array([[0, 1, 4], [1, 2, 5]], dtype=np.int64),
    )


def _masks() -> dict[str, np.ndarray]:
    return {
        "train_mask": np.array([True, True, True, False, False, False]),
        "val_mask": np.array([False, False, False, True, False, False]),
        "test_mask": np.array([False, False, False, False, True, True]),
        "labeled_mask": np.array([True, True, False, False, False, False]),
        "unlabeled_mask": np.array([False, False, True, False, False, False]),
    }


def _inputs() -> TransductiveExecutionInput:
    return TransductiveExecutionInput(
        dataset=_dataset(),
        graph=_graph(),
        masks=_masks(),
    )


def test_native_execution_owns_fit_data_and_public_runtime_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: _CapturingMethod.info,
    )

    result = execute_transductive_method(
        _inputs(),
        TransductiveExecutionConfig(
            method_id="capture",
            device=DeviceSpec(device="cpu", dtype="float32"),
            seed=7,
            use_test_split=True,
            expected_labeled_count=2,
        ),
    )

    assert result.data.fit is _CapturingMethod.captured_data
    np.testing.assert_array_equal(result.data.fit.y, [0, 1, -1, -1, -1, -1])
    np.testing.assert_array_equal(result.data.evaluation.y_true, [0, 1, 0, 1, 1, 0])
    assert result.backend == "runtime-public"
    assert result.resolved_device == "cpu"
    assert result.resolution["runtime_token"] == "native"
    assert result.resolution["diagnostics"] == {"fit": "ok"}
    assert result.resolution["dtypes"]["X"] == {
        "dtype": "float32",
        "shape": [6, 2],
    }
    assert result.resolution["normalization"]["strict_contract_validated"] is False


def test_native_execution_runs_tsvm_without_a_graph() -> None:
    result = execute_transductive_method(
        TransductiveExecutionInput(
            dataset=_dataset(),
            graph=None,
            masks=_masks(),
        ),
        TransductiveExecutionConfig(
            method_id="tsvm",
            params={"max_iter": 1, "epochs_per_iter": 1, "batch_size": 8},
            device=DeviceSpec(device="cpu", dtype="float32"),
            seed=7,
            use_test_split=True,
            expected_labeled_count=2,
        ),
    )

    assert result.data.fit.graph is None
    assert result.resolution["pipeline_capabilities"]["has_graph"] is False
    assert result.resolution["pipeline_capabilities"]["representation"] == "dense"
    assert result.method.predict_proba(result.data.fit).shape == (6, 2)


def test_native_execution_rejects_multiclass_tsvm_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = LoadedDataset(
        train=Split(
            X=np.arange(8, dtype=np.float32).reshape(4, 2),
            y=np.array([0, 1, 2, 0]),
        ),
        test=Split(
            X=np.arange(4, dtype=np.float32).reshape(2, 2),
            y=np.array([1, 2]),
        ),
    )
    masks = {
        "train_mask": np.array([True, True, True, True, False, False]),
        "val_mask": np.array([False, False, False, False, True, False]),
        "test_mask": np.array([False, False, False, False, False, True]),
        "labeled_mask": np.array([True, True, True, False, False, False]),
        "unlabeled_mask": np.array([False, False, False, True, False, False]),
    }
    _CapturingMethod.captured_data = None
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            TransductiveExecutionInput(dataset=dataset, graph=None, masks=masks),
            TransductiveExecutionConfig(
                method_id="tsvm",
                device=DeviceSpec(device="cpu"),
                seed=7,
                use_test_split=True,
                expected_labeled_count=3,
            ),
        )

    assert caught.value.kind == "capability"
    assert "E_CAPABILITY_CLASS_COUNT" in str(caught.value)
    assert _CapturingMethod.captured_data is None


def test_native_execution_rejects_graph_requiring_method_by_capability_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingMethod.captured_data = None
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: _CapturingMethod.info)

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            TransductiveExecutionInput(
                dataset=_dataset(),
                graph=None,
                masks=_masks(),
            ),
            TransductiveExecutionConfig(
                method_id="capture",
                device=DeviceSpec(device="cpu"),
                seed=7,
                use_test_split=True,
            ),
        )

    assert caught.value.kind == "capability"
    assert "E_CAPABILITY_GRAPH" in str(caught.value)
    assert _CapturingMethod.captured_data is None


def test_native_execution_enforces_the_generic_scientific_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingMethod.captured_data = None
    expected = MethodNotEvaluableError(
        MethodExecutionOutcome(
            status="not_evaluable",
            reason="declared gate",
            diagnostics={"converged": False},
        )
    )

    def enforce(method: object) -> None:
        assert _CapturingMethod.captured_data is not None
        assert isinstance(method, _CapturingMethod)
        raise expected

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: _CapturingMethod.info)
    monkeypatch.setattr(execution, "enforce_method_execution", enforce)

    with pytest.raises(MethodNotEvaluableError) as raised:
        execute_transductive_method(
            _inputs(),
            TransductiveExecutionConfig(
                method_id="capture",
                device=DeviceSpec(device="cpu", dtype="float32"),
                seed=7,
                use_test_split=True,
            ),
        )
    assert raised.value is expected


def test_native_execution_rejects_auto_device_in_strict_mode_before_method_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: _CapturingMethod.info,
    )
    monkeypatch.setattr(
        execution,
        "get_method_class",
        lambda _method_id: pytest.fail("method class lookup must not occur"),
    )

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            _inputs(),
            TransductiveExecutionConfig(
                method_id="capture",
                device=DeviceSpec(device="auto"),
                strict=True,
            ),
        )

    assert caught.value.kind == "auto_backend"
    assert caught.value.code == "E_TRANSDUCTIVE_AUTO_FORBIDDEN"


def test_native_execution_rejects_configured_augmentation_before_method_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: pytest.fail("method lookup must not occur"),
    )
    inputs = TransductiveExecutionInput(
        dataset=_dataset(),
        graph=_graph(),
        masks=_masks(),
        augmentation_configured=True,
    )

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            inputs,
            TransductiveExecutionConfig(method_id="unused"),
        )

    assert caught.value.kind == "augmentation_contract"
    assert caught.value.code == "E_TRANSDUCTIVE_AUGMENTATION_UNSUPPORTED"


def _execution_context(tmp_path, *, resume_policy: str = "auto") -> ExecutionContext:
    return ExecutionContext(
        identity=RunIdentity(config_sha256="0" * 64, seed=7),
        output_dir=tmp_path,
        resume_policy=resume_policy,
    )


def test_native_execution_derives_required_checkpointing_from_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: _CapturingMethod.info)
    inputs = TransductiveExecutionInput(
        dataset=_dataset(),
        graph=_graph(),
        masks=_masks(),
        execution_context=_execution_context(tmp_path),
    )

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            inputs,
            TransductiveExecutionConfig(
                method_id="capture",
                seed=7,
                use_test_split=True,
            ),
        )

    assert caught.value.kind == "capability"
    assert caught.value.code == "E_TRANSDUCTIVE_CAPABILITY"
    assert "E_CAPABILITY_CHECKPOINTING" in str(caught.value)


def test_native_execution_reports_checkpointing_from_exact_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class CheckpointMethod(_CapturingMethod):
        info = MethodInfo(
            method_id="checkpoint",
            name="Checkpoint",
            supports_gpu=True,
            capabilities=MethodCapabilities(
                regime="transductive",
                requires_unlabeled=True,
                requires_graph=True,
                supports_checkpointing=True,
            ),
        )

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: CheckpointMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: CheckpointMethod.info)
    inputs = TransductiveExecutionInput(
        dataset=_dataset(),
        graph=_graph(),
        masks=_masks(),
        execution_context=_execution_context(tmp_path),
    )

    result = execute_transductive_method(
        inputs,
        TransductiveExecutionConfig(
            method_id="checkpoint",
            seed=7,
            use_test_split=True,
        ),
    )

    assert result.resolution["pipeline_capabilities"]["checkpointing_required"] is True


def test_native_execution_rejects_auto_method_backend_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: _CapturingMethod.info,
    )

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            _inputs(),
            TransductiveExecutionConfig(
                method_id="capture",
                params={"backend": "auto"},
                strict=True,
            ),
        )

    assert caught.value.kind == "auto_backend"


def test_native_execution_closes_torch_dependency_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: _CapturingMethod.info,
    )

    def missing_torch(name: str) -> Any:
        assert name == "torch"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(execution.importlib, "import_module", missing_torch)

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            _inputs(),
            TransductiveExecutionConfig(
                method_id="capture",
                params={"backend": "torch"},
            ),
        )

    assert caught.value.kind == "dependency_missing"
    assert caught.value.code == "E_TRANSDUCTIVE_DEPENDENCY_MISSING"


def test_native_execution_translates_method_spec_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoSpecMethod:
        info = MethodInfo(method_id="no_spec", name="No spec")

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: NoSpecMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: NoSpecMethod.info)

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            _inputs(),
            TransductiveExecutionConfig(method_id="no_spec"),
        )

    assert caught.value.kind == "method_spec"
    assert caught.value.code == "E_TRANSDUCTIVE_METHOD_SPEC"


def test_native_execution_preserves_method_introspection_error_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UninspectableMethod:
        info = MethodInfo(method_id="uninspectable", name="Uninspectable")

        def __init__(self) -> None:
            raise RuntimeError("constructor unavailable")

    monkeypatch.setattr(
        execution,
        "get_method_class",
        lambda _method_id: UninspectableMethod,
    )
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: UninspectableMethod.info,
    )

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            _inputs(),
            TransductiveExecutionConfig(method_id="uninspectable", strict=True),
        )

    assert caught.value.kind == "method_introspection"
    assert caught.value.code == "E_TRANSDUCTIVE_METHOD_INTROSPECTION"


def test_native_execution_translates_data_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: _CapturingMethod.info,
    )
    invalid = TransductiveExecutionInput(
        dataset=_dataset(),
        graph=GraphArtifact(
            n_nodes=5,
            edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        ),
        masks=_masks(),
    )

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            invalid,
            TransductiveExecutionConfig(method_id="capture", use_test_split=True),
        )

    assert caught.value.kind == "data_contract"
    assert caught.value.code == "E_TRANSDUCTIVE_DATA_CONTRACT"


def test_native_execution_rejects_invalid_public_resolution_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidResolutionMethod(_CapturingMethod):
        execution_resolution = {"backend": "private-shape"}

    monkeypatch.setattr(
        execution,
        "get_method_class",
        lambda _method_id: InvalidResolutionMethod,
    )
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: InvalidResolutionMethod.info,
    )

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            _inputs(),
            TransductiveExecutionConfig(
                method_id="capture",
                seed=7,
                use_test_split=True,
            ),
        )

    assert caught.value.kind == "method_contract"


def test_native_execution_helper_resolution_covers_cpu_auto_and_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    automatic = TransductiveExecutionConfig(
        method_id="capture",
        device=DeviceSpec(device="auto"),
    )
    assert execution._resolve_method_device(automatic, supports_gpu=False) == "cpu"

    monkeypatch.setattr(execution, "resolve_device_name", lambda _requested: "mps")
    assert execution._resolve_method_device(automatic, supports_gpu=True) == "mps"
    assert execution._requested_backend(automatic, spec=object()) is None


def test_dtype_descriptor_handles_none_and_unserializable_shape() -> None:
    assert execution._dtype_descriptor(None) is None

    class InvalidShape:
        def __iter__(self):
            raise TypeError("shape unavailable")

    value = type("Value", (), {"dtype": "custom", "shape": InvalidShape()})()
    assert execution._dtype_descriptor(value) == {"dtype": "custom", "shape": None}
    assert execution._dtype_descriptor(object()) is None


def test_public_method_resolution_handles_absent_and_nonmapping_providers() -> None:
    assert execution._public_method_resolution(object()) == {}

    class Invalid:
        def execution_resolution(self):
            return ["not", "a", "mapping"]

    with pytest.raises(TransductiveExecutionError) as caught:
        execution._public_method_resolution(Invalid())
    assert caught.value.kind == "method_contract"


def test_native_execution_omits_nonmapping_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyDiagnosticsMethod(_CapturingMethod):
        def fit(self, data: NodeDataset, *, device: str | None, seed: int):
            super().fit(data, device=device, seed=seed)
            self.diagnostics_ = "legacy"
            return self

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: LegacyDiagnosticsMethod)
    monkeypatch.setattr(
        execution,
        "get_method_info",
        lambda _method_id: LegacyDiagnosticsMethod.info,
    )

    result = execute_transductive_method(
        _inputs(),
        TransductiveExecutionConfig(
            method_id="capture",
            device=DeviceSpec(device="cpu"),
            seed=7,
            use_test_split=True,
        ),
    )

    assert "diagnostics" not in result.resolution


def test_incompatible_input_contract_prevents_transductive_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingMethod.captured_data = None
    contract = MethodExecutionContract(
        base=_CapturingMethod.info.capabilities,
        inputs=(InputRoleRequirement("fit.X", ranks=frozenset({3})),),
        source="test",
    )
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: _CapturingMethod.info)
    monkeypatch.setattr(
        execution,
        "resolve_method_execution_contract",
        lambda *_args, **_kwargs: contract,
    )

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            _inputs(),
            TransductiveExecutionConfig(
                method_id="capture",
                device=DeviceSpec(device="cpu"),
                seed=7,
                use_test_split=True,
            ),
        )

    assert caught.value.kind == "execution_contract"
    assert isinstance(caught.value.__cause__, ExecutionContractError)
    assert {issue.code for issue in caught.value.__cause__.report.issues} == {"E_INPUT_RANK"}
    assert _CapturingMethod.captured_data is None


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_invalid_transductive_execution_contract_is_closed_before_fit(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    _CapturingMethod.captured_data = None
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: _CapturingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: _CapturingMethod.info)

    def invalid_contract(**_kwargs):
        raise error_type("malformed contract")

    monkeypatch.setattr(execution, "_resolve_execution_contract", invalid_contract)

    with pytest.raises(TransductiveExecutionError) as caught:
        execute_transductive_method(
            _inputs(),
            TransductiveExecutionConfig(
                method_id="capture",
                device=DeviceSpec(device="cpu"),
                seed=7,
                use_test_split=True,
            ),
        )

    assert caught.value.kind == "execution_contract"
    assert isinstance(caught.value.__cause__, error_type)
    assert _CapturingMethod.captured_data is None
