from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from modssc.capabilities import MethodCapabilities
from modssc.inductive.registry import available_methods as available_inductive_methods
from modssc.runtime import pipeline
from modssc.runtime.method_spec import MethodSpecError
from modssc.runtime.pipeline import (
    MaterializedPipeline,
    MethodResolutionRequest,
    MethodRuntimeResolution,
    PipelineResolutionError,
    PipelineResolutionRequest,
    resolve_method,
    resolve_pipeline,
    validate_materialized_pipeline,
)
from modssc.supervised.errors import UnknownBackendError, UnknownClassifierError
from modssc.transductive.registry import available_methods as available_transductive_methods


@dataclass(frozen=True)
class _BackendSpec:
    backend: str = "numpy"


@dataclass(frozen=True)
class _PlainSpec:
    alpha: float = 1.0


@dataclass(frozen=True)
class _ClassifierSpec:
    classifier_id: str = "knn"
    classifier_backend: str = "numpy"


@dataclass(frozen=True)
class _NestedSpec:
    payload: Any = None


class _BackendMethod:
    def __init__(self, spec: _BackendSpec | None = None) -> None:
        self.spec = spec or _BackendSpec()


class _PlainMethod:
    def __init__(self, spec: _PlainSpec | None = None) -> None:
        self.spec = spec or _PlainSpec()


class _ClassifierMethod:
    def __init__(self, spec: _ClassifierSpec | None = None) -> None:
        self.spec = spec or _ClassifierSpec()


class _NestedMethod:
    def __init__(self, spec: _NestedSpec | None = None) -> None:
        self.spec = spec or _NestedSpec()


class _NoSpecMethod:
    pass


class _Sampling:
    indices = {"train_unlabeled": np.array([1, 2], dtype=np.int64)}
    masks: dict[str, Any] = {}

    def is_graph(self) -> bool:
        return False


def _install_method(
    monkeypatch: pytest.MonkeyPatch,
    *,
    regime: str,
    method_class: type[Any],
    capabilities: MethodCapabilities,
    supports_gpu: bool = True,
    required_extra: str | None = None,
) -> None:
    info = SimpleNamespace(
        capabilities=capabilities,
        supports_gpu=supports_gpu,
        required_extra=required_extra,
    )
    if regime == "inductive":
        monkeypatch.setattr(pipeline, "get_inductive_method_class", lambda _method_id: method_class)
        monkeypatch.setattr(pipeline, "get_inductive_method_info", lambda _method_id: info)
        monkeypatch.setattr(
            pipeline,
            "get_transductive_method_class",
            lambda _method_id: pytest.fail("transductive registry must not be queried"),
        )
        monkeypatch.setattr(
            pipeline,
            "get_transductive_method_info",
            lambda _method_id: pytest.fail("transductive registry must not be queried"),
        )
    else:
        monkeypatch.setattr(
            pipeline,
            "get_inductive_method_class",
            lambda _method_id: pytest.fail("inductive registry must not be queried"),
        )
        monkeypatch.setattr(
            pipeline,
            "get_inductive_method_info",
            lambda _method_id: pytest.fail("inductive registry must not be queried"),
        )
        monkeypatch.setattr(pipeline, "get_transductive_method_class", lambda _id: method_class)
        monkeypatch.setattr(pipeline, "get_transductive_method_info", lambda _id: info)


def _materialized(*, has_graph: bool = False) -> MaterializedPipeline:
    return MaterializedPipeline(
        modality="tabular",
        primary_input=np.ones((4, 2), dtype=np.float32),
        sampling=_Sampling(),
        has_graph=has_graph,
    )


def test_resolve_pipeline_owns_registry_backend_device_and_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="inductive",
        method_class=_BackendMethod,
        capabilities=MethodCapabilities(
            regime="inductive",
            requires_unlabeled=True,
            backends=frozenset({"numpy", "torch"}),
        ),
        required_extra="inductive-torch",
    )
    request = PipelineResolutionRequest(
        method=MethodResolutionRequest(
            regime="inductive",
            method_id="configurable",
            params={"backend": "torch"},
            requested_device="cpu",
            strict=True,
            preprocess_step_ids=("core.to_torch",),
            model_configured=True,
        ),
        pipeline=_materialized(),
    )

    result = resolve_pipeline(request)

    assert result.method.requires_torch is True
    assert result.method.requested_backend == "torch"
    assert result.method.resolved_device == "cpu"
    assert result.method.required_extra == "inductive-torch"
    assert result.resolved_backend == "torch"
    assert result.pipeline_capabilities.classifier_outputs == frozenset(
        {"predictions", "scores", "logits"}
    )
    assert result.compatibility.compatible
    json.dumps(result.to_dict(), allow_nan=False)


def test_strict_torch_inductive_contract_requires_native_conversion_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="inductive",
        method_class=_BackendMethod,
        capabilities=MethodCapabilities(
            regime="inductive",
            requires_unlabeled=True,
            backends=frozenset({"numpy", "torch"}),
        ),
    )

    with pytest.raises(PipelineResolutionError) as raised:
        resolve_method(
            MethodResolutionRequest(
                regime="inductive",
                method_id="configurable",
                params={"backend": "torch"},
                strict=True,
                preprocess_step_ids=(),
            )
        )

    assert raised.value.kind == "torch_preprocess"
    assert raised.value.code == "E_PIPELINE_TORCH_PREPROCESS_REQUIRED"


def test_strict_method_spec_backend_must_be_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="transductive",
        method_class=_BackendMethod,
        capabilities=MethodCapabilities(regime="transductive", requires_graph=True),
    )

    with pytest.raises(PipelineResolutionError) as raised:
        resolve_method(
            MethodResolutionRequest(
                regime="transductive",
                method_id="graph_method",
                params={},
                strict=True,
            )
        )

    assert raised.value.kind == "backend_required"
    assert raised.value.code == "E_PIPELINE_BACKEND_REQUIRED"


def test_device_auto_policy_is_resolved_natively(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_method(
        monkeypatch,
        regime="transductive",
        method_class=_PlainMethod,
        capabilities=MethodCapabilities(regime="transductive"),
        supports_gpu=False,
    )

    resolved = resolve_method(
        MethodResolutionRequest(
            regime="transductive",
            method_id="cpu_method",
            requested_device="auto",
        )
    )
    assert resolved.resolved_device == "cpu"

    with pytest.raises(PipelineResolutionError) as raised:
        resolve_method(
            MethodResolutionRequest(
                regime="transductive",
                method_id="cpu_method",
                requested_device="auto",
                strict=True,
            )
        )
    assert raised.value.kind == "auto_device"


def test_capability_failure_is_exposed_as_one_native_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="transductive",
        method_class=_BackendMethod,
        capabilities=MethodCapabilities(
            regime="transductive",
            requires_unlabeled=True,
            requires_graph=True,
            backends=frozenset({"numpy"}),
        ),
    )

    with pytest.raises(PipelineResolutionError) as raised:
        resolve_pipeline(
            PipelineResolutionRequest(
                method=MethodResolutionRequest(
                    regime="transductive",
                    method_id="graph_method",
                    params={"backend": "numpy"},
                    strict=True,
                ),
                pipeline=_materialized(has_graph=False),
            )
        )

    assert raised.value.kind == "capability"
    assert raised.value.report is not None
    assert [issue.code for issue in raised.value.report.issues] == ["E_CAPABILITY_GRAPH"]


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"regime": "invalid"}, ValueError),
        ({"method_id": ""}, ValueError),
        ({"params": []}, TypeError),
        ({"requested_device": ""}, ValueError),
        ({"dtype": ""}, ValueError),
        ({"preprocess_step_ids": "core.to_torch"}, TypeError),
        ({"preprocess_step_ids": ("",)}, ValueError),
        ({"model_classifier_id": ""}, ValueError),
        ({"model_classifier_backend": ""}, ValueError),
    ],
)
def test_method_request_rejects_invalid_public_inputs(
    kwargs: dict[str, Any],
    error_type: type[Exception],
) -> None:
    values: dict[str, Any] = {
        "regime": "inductive",
        "method_id": "method",
    }
    values.update(kwargs)
    with pytest.raises(error_type):
        MethodResolutionRequest(**values)


@pytest.mark.parametrize("field_name", ["view_count", "strong_augmentation_count"])
@pytest.mark.parametrize("invalid", [True, -1, 1.5])
def test_materialized_pipeline_rejects_invalid_counts(field_name: str, invalid: Any) -> None:
    values = {
        "modality": "tabular",
        "primary_input": np.ones((1, 1), dtype=np.float32),
        "sampling": _Sampling(),
        field_name: invalid,
    }
    with pytest.raises(ValueError, match="non-negative integer"):
        MaterializedPipeline(**values)


def test_resolution_request_validates_nested_contract_types() -> None:
    method = MethodResolutionRequest(regime="inductive", method_id="method")
    materialized = _materialized()
    with pytest.raises(TypeError, match="MethodResolutionRequest"):
        PipelineResolutionRequest(method=object(), pipeline=materialized)
    with pytest.raises(TypeError, match="MaterializedPipeline"):
        PipelineResolutionRequest(method=method, pipeline=object())


def test_method_lookup_and_capability_contract_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "get_inductive_method_class",
        lambda _method_id: (_ for _ in ()).throw(ValueError("unknown")),
    )
    with pytest.raises(PipelineResolutionError) as lookup_error:
        resolve_method(MethodResolutionRequest(regime="inductive", method_id="missing"))
    assert lookup_error.value.kind == "method_lookup"

    monkeypatch.setattr(pipeline, "get_inductive_method_class", lambda _method_id: _PlainMethod)
    monkeypatch.setattr(
        pipeline,
        "get_inductive_method_info",
        lambda _method_id: SimpleNamespace(
            capabilities=object(), supports_gpu=False, required_extra=None
        ),
    )
    with pytest.raises(PipelineResolutionError) as contract_error:
        resolve_method(MethodResolutionRequest(regime="inductive", method_id="invalid"))
    assert contract_error.value.kind == "method_lookup"


def test_method_spec_introspection_and_construction_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="transductive",
        method_class=_BackendMethod,
        capabilities=MethodCapabilities(regime="transductive"),
    )
    monkeypatch.setattr(
        pipeline,
        "method_spec_has_field",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MethodSpecError("method_introspection", "cannot inspect")
        ),
    )
    with pytest.raises(PipelineResolutionError) as introspection_error:
        resolve_method(MethodResolutionRequest(regime="transductive", method_id="method"))
    assert introspection_error.value.kind == "method_introspection"

    monkeypatch.setattr(pipeline, "method_spec_has_field", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        pipeline,
        "build_method_spec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MethodSpecError("method_spec", "cannot build")
        ),
    )
    with pytest.raises(PipelineResolutionError) as spec_error:
        resolve_method(MethodResolutionRequest(regime="transductive", method_id="method"))
    assert spec_error.value.kind == "method_spec"


def test_default_and_auto_method_backends_are_resolved_natively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="transductive",
        method_class=_BackendMethod,
        capabilities=MethodCapabilities(regime="transductive"),
    )
    default = resolve_method(MethodResolutionRequest(regime="transductive", method_id="method"))
    assert default.requested_backend is None
    assert default.resolved_backend == "numpy"

    automatic = resolve_method(
        MethodResolutionRequest(
            regime="transductive",
            method_id="method",
            params={"backend": "auto"},
        )
    )
    assert automatic.requested_backend == "auto"
    assert automatic.resolved_backend is None

    with pytest.raises(PipelineResolutionError) as strict_auto:
        resolve_method(
            MethodResolutionRequest(
                regime="transductive",
                method_id="method",
                params={"backend": "auto"},
                strict=True,
            )
        )
    assert strict_auto.value.kind == "auto_backend"


def test_classifier_backend_and_torch_only_contract_resolve_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="inductive",
        method_class=_ClassifierMethod,
        capabilities=MethodCapabilities(
            regime="inductive",
            backends=frozenset({"numpy", "torch"}),
        ),
    )
    classifier = resolve_method(
        MethodResolutionRequest(
            regime="inductive",
            method_id="method",
            params={"classifier_backend": "numpy"},
            model_classifier_backend="torch",
        )
    )
    assert classifier.classifier_backend == "numpy"
    assert classifier.requires_torch is False

    _install_method(
        monkeypatch,
        regime="inductive",
        method_class=_PlainMethod,
        capabilities=MethodCapabilities(
            regime="inductive",
            backends=frozenset({"numpy", "torch"}),
        ),
    )
    model_classifier = resolve_method(
        MethodResolutionRequest(
            regime="inductive",
            method_id="method",
            model_classifier_backend="torch",
        )
    )
    assert model_classifier.classifier_backend == "torch"
    assert model_classifier.requires_torch is True
    assert model_classifier.resolved_backend == "torch"

    _install_method(
        monkeypatch,
        regime="inductive",
        method_class=_ClassifierMethod,
        capabilities=MethodCapabilities(
            regime="inductive",
            backends=frozenset({"numpy", "torch"}),
        ),
    )
    monkeypatch.setattr("modssc.supervised.api.has_module", lambda _module: False)
    automatic = resolve_method(
        MethodResolutionRequest(
            regime="inductive",
            method_id="method",
            params={"classifier_backend": "auto"},
        )
    )
    assert automatic.classifier_backend == "numpy"
    with pytest.raises(PipelineResolutionError) as strict_auto:
        resolve_method(
            MethodResolutionRequest(
                regime="inductive",
                method_id="method",
                params={"classifier_backend": "auto"},
                strict=True,
            )
        )
    assert strict_auto.value.kind == "auto_backend"

    _install_method(
        monkeypatch,
        regime="inductive",
        method_class=_PlainMethod,
        capabilities=MethodCapabilities(
            regime="inductive",
            backends=frozenset({"torch"}),
        ),
    )
    torch_only = resolve_method(MethodResolutionRequest(regime="inductive", method_id="torch_only"))
    assert torch_only.requires_torch is True
    assert torch_only.resolved_backend == "torch"


def test_default_classic_classifier_dependency_is_read_from_native_spec() -> None:
    resolved = resolve_method(MethodResolutionRequest(regime="inductive", method_id="supervised"))

    assert resolved.required_extra is None
    assert resolved.required_extras == ()
    assert resolved.classifier_backend == "numpy"
    assert resolved.resolved_backend == "numpy"
    assert resolved.to_dict()["required_extras"] == []


def test_nested_classifier_specs_collect_and_deduplicate_required_extras() -> None:
    resolved = resolve_method(
        MethodResolutionRequest(
            regime="inductive",
            method_id="democratic_co_learning",
            params={
                "classifier_specs": (
                    {
                        "classifier_id": "image_pretrained",
                        "classifier_backend": "torch",
                    },
                    _ClassifierSpec(classifier_id="mlp", classifier_backend="torch"),
                    {"classifier_id": "image_cnn", "classifier_backend": "torch"},
                    {
                        "classifier_id": "image_pretrained",
                        "classifier_backend": "torch",
                    },
                )
            },
        )
    )

    assert resolved.required_extras == ("supervised-torch", "vision")
    assert resolved.classifier_backend == "torch"
    assert resolved.requires_torch is True


def test_model_classifier_collects_vision_extra_without_importing_backend() -> None:
    resolved = resolve_method(
        MethodResolutionRequest(
            regime="inductive",
            method_id="fixmatch",
            model_classifier_id="image_pretrained",
            model_classifier_backend="torch",
            model_configured=True,
        )
    )

    assert resolved.required_extras == ("vision",)
    assert resolved.classifier_backend == "torch"
    assert resolved.requires_torch is True


def test_classifier_auto_uses_construction_policy_for_dependency_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "modssc.supervised.api.has_module",
        lambda module: module == "sklearn",
    )

    resolved = resolve_method(
        MethodResolutionRequest(
            regime="inductive",
            method_id="supervised",
            params={"classifier_backend": "auto"},
        )
    )

    assert resolved.required_extras == ("sklearn",)
    assert resolved.classifier_backend == "sklearn"
    assert resolved.resolved_backend == "sklearn"


@pytest.mark.parametrize(
    ("classifier_spec", "cause_type"),
    [
        (
            {"classifier_id": "does_not_exist", "classifier_backend": "numpy"},
            UnknownClassifierError,
        ),
        (
            {"classifier_id": "knn", "classifier_backend": "does_not_exist"},
            UnknownBackendError,
        ),
    ],
)
def test_method_spec_classifier_lookup_errors_use_pipeline_boundary(
    classifier_spec: dict[str, str],
    cause_type: type[Exception],
) -> None:
    with pytest.raises(PipelineResolutionError) as raised:
        resolve_method(
            MethodResolutionRequest(
                regime="inductive",
                method_id="democratic_co_learning",
                params={"classifier_specs": (classifier_spec,)},
            )
        )

    assert raised.value.kind == "method_spec"
    assert raised.value.code == "E_PIPELINE_METHOD_SPEC"
    assert isinstance(raised.value.__cause__, cause_type)


@pytest.mark.parametrize("classifier_specs", ["knn", (object(),)])
def test_invalid_nested_classifier_declarations_use_pipeline_boundary(
    classifier_specs: object,
) -> None:
    with pytest.raises(PipelineResolutionError) as raised:
        resolve_method(
            MethodResolutionRequest(
                regime="inductive",
                method_id="democratic_co_learning",
                params={"classifier_specs": classifier_specs},
            )
        )

    assert raised.value.kind == "method_spec"
    assert isinstance(raised.value.__cause__, ValueError)


def test_incomplete_recursive_classifier_declaration_uses_pipeline_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="inductive",
        method_class=_NestedMethod,
        capabilities=MethodCapabilities(regime="inductive"),
    )

    with pytest.raises(PipelineResolutionError) as raised:
        resolve_method(
            MethodResolutionRequest(
                regime="inductive",
                method_id="nested",
                params={"payload": {"classifier_id": "knn"}},
            )
        )

    assert raised.value.kind == "method_spec"
    assert isinstance(raised.value.__cause__, ValueError)


@pytest.mark.parametrize("classifier_backend", ["torch", None])
def test_unregistered_custom_model_requires_explicit_software_dependencies(
    classifier_backend: str | None,
) -> None:
    resolved = resolve_method(
        MethodResolutionRequest(
            regime="inductive",
            method_id="fixmatch",
            model_classifier_id="custom.model.factory",
            model_classifier_backend=classifier_backend,
            model_configured=True,
        )
    )

    # Custom factories declare their distributions through run.software_dependencies.
    assert resolved.required_extras == ()
    assert resolved.classifier_backend == classifier_backend
    assert resolved.requires_torch is True


def test_method_without_dataclass_spec_is_allowed_only_without_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="transductive",
        method_class=_NoSpecMethod,
        capabilities=MethodCapabilities(regime="transductive"),
        supports_gpu=False,
    )

    resolved = resolve_method(MethodResolutionRequest(regime="transductive", method_id="extension"))
    assert resolved.required_extras == ()

    with pytest.raises(PipelineResolutionError) as raised:
        resolve_method(
            MethodResolutionRequest(
                regime="transductive",
                method_id="extension",
                params={"alpha": 1.0},
            )
        )
    assert raised.value.kind == "method_spec"


_REGISTERED_METHODS = [
    *(("inductive", method_id) for method_id in available_inductive_methods()),
    *(("transductive", method_id) for method_id in available_transductive_methods()),
]


@pytest.mark.parametrize(("regime", "method_id"), _REGISTERED_METHODS)
def test_all_registered_methods_resolve_default_dependency_facts(
    regime: str,
    method_id: str,
) -> None:
    resolved = resolve_method(
        MethodResolutionRequest(regime=regime, method_id=method_id)  # type: ignore[arg-type]
    )

    if resolved.required_extra is not None:
        assert resolved.required_extra in resolved.required_extras


def test_gpu_capable_auto_device_uses_native_device_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_method(
        monkeypatch,
        regime="transductive",
        method_class=_PlainMethod,
        capabilities=MethodCapabilities(regime="transductive"),
        supports_gpu=True,
    )
    monkeypatch.setattr(pipeline, "resolve_device_name", lambda _requested: "cuda")

    resolved = resolve_method(
        MethodResolutionRequest(
            regime="transductive",
            method_id="gpu_method",
            requested_device="auto",
        )
    )

    assert resolved.resolved_device == "cuda"


def test_public_functions_reject_wrong_contract_types() -> None:
    with pytest.raises(TypeError, match="MethodResolutionRequest"):
        resolve_method(object())
    with pytest.raises(TypeError, match="PipelineResolutionRequest"):
        resolve_pipeline(object())

    fake_method = object()
    with pytest.raises(TypeError, match="MethodRuntimeResolution"):
        validate_materialized_pipeline(fake_method, _materialized())

    method = MethodRuntimeResolution(
        regime="inductive",
        method_id="method",
        supports_gpu=False,
        required_extra=None,
        capabilities=MethodCapabilities(regime="inductive"),
        requested_device="cpu",
        resolved_device="cpu",
        requested_backend=None,
        resolved_backend="numpy",
        classifier_backend=None,
        requires_torch=False,
        model_configured=False,
        dtype="float32",
        strict=False,
        preprocess_step_ids=(),
    )
    with pytest.raises(TypeError, match="MaterializedPipeline"):
        validate_materialized_pipeline(method, object())
