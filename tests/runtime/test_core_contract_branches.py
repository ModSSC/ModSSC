from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from modssc import capabilities as capabilities_module
from modssc.capabilities import (
    materialize_consumed_input_capabilities,
    materialize_pipeline_capabilities,
)
from modssc.cli._utils import ensure_mapping, json_dumps
from modssc.runtime import pipeline as pipeline_module
from modssc.runtime import software as software_module
from modssc.runtime.method_spec import MethodSpecError, build_method_spec, method_spec_has_field
from modssc.runtime.pipeline import MethodResolutionRequest, PipelineResolutionError
from modssc.runtime.software import (
    SoftwareManifest,
    SoftwareProvenanceError,
    attach_software_manifest,
    collect_software_manifest,
    normalize_distribution_name,
    requirement_distribution_name,
    resolve_required_distributions,
    software_identity_payload,
)


class _Sparse:
    shape = (2, 2)
    nnz = 1

    def tocsr(self) -> _Sparse:
        return self


class _TorchLike:
    __module__ = "torch.fake"
    shape = (2, 2)


class _ShapeOnly:
    shape = (2, 2)


class _ScalarShape:
    shape: tuple[()] = ()


class _BadShape:
    shape = (object(),)


class _NoLength:
    def __len__(self) -> int:
        raise TypeError("no length")


class _TorchMask:
    __module__ = "torch.fake"

    def detach(self) -> _TorchMask:
        return self

    def cpu(self) -> _TorchMask:
        return self

    def numpy(self) -> np.ndarray:
        return np.array([False, True])


def test_capability_materializers_cover_every_native_input_shape() -> None:
    representation_cases = [
        ({"input_ids": [1]}, "text", "tokens"),
        ({"x": np.ones((1, 2))}, "tabular", "dense"),
        ({"other": np.ones((1, 2))}, "tabular", "structured"),
        (_Sparse(), "tabular", "sparse"),
        (np.array(["hello"], dtype=object), "text", "text"),
        (np.array(["hello"], dtype="U"), "vision", "objects"),
        (_TorchLike(), "vision", "dense"),
        (["a.txt"], "text", "text"),
        (("a.png",), "vision", "paths"),
        (_ShapeOnly(), "tabular", "dense"),
        (object(), "tabular", "objects"),
    ]
    for value, modality, expected in representation_cases:
        assert capabilities_module._representation_of(value, modality=modality) == expected

    assert capabilities_module._backend_of({"x": _TorchLike()}) == "torch"
    assert capabilities_module._backend_of({"a": np.ones(1), "b": np.zeros(1)}) == "numpy"
    assert capabilities_module._backend_of({"a": np.ones(1), "b": _TorchLike()}) is None
    assert capabilities_module._backend_of(_Sparse()) == "numpy"
    assert capabilities_module._backend_of(object()) is None

    assert not capabilities_module._has_rows(None)
    assert capabilities_module._has_rows({"x": [1]})
    assert capabilities_module._has_rows(_ScalarShape())
    assert capabilities_module._has_rows(_BadShape())
    assert not capabilities_module._has_rows([])
    assert capabilities_module._has_rows(_NoLength())
    assert capabilities_module._mask_has_values(_TorchMask())
    assert not capabilities_module._mask_has_values(None)


def test_consumed_capability_validation_errors_and_view_fallbacks() -> None:
    with pytest.raises(TypeError, match="expose X_l"):
        materialize_consumed_input_capabilities(
            regime="inductive", modality="tabular", consumed_input=object()
        )
    with pytest.raises(TypeError, match="views must be a mapping"):
        materialize_consumed_input_capabilities(
            regime="inductive",
            modality="tabular",
            consumed_input=SimpleNamespace(X_l=np.ones((1, 1)), views=[]),
        )

    inductive = SimpleNamespace(
        X_l=np.ones((1, 2)),
        X_u=None,
        X_u_w=None,
        X_u_s=None,
        X_u_s_1=None,
        views={"X_u_s1": np.empty((0, 2)), "X_u_s_2": np.ones((1, 2)), "scientific": 1},
        graph=None,
    )
    result = materialize_consumed_input_capabilities(
        regime="inductive", modality="tabular", consumed_input=inductive
    )
    assert result.strong_augmentation_count == 1
    assert result.view_count == 1

    no_reserved_view = SimpleNamespace(
        X_l=np.ones((1, 2)),
        X_u=None,
        X_u_w=None,
        X_u_s=None,
        X_u_s_1=None,
        views={"scientific": object()},
        graph=None,
    )
    assert (
        materialize_consumed_input_capabilities(
            regime="inductive", modality="tabular", consumed_input=no_reserved_view
        ).strong_augmentation_count
        == 0
    )

    with pytest.raises(TypeError, match=r"expose X \(or fit.X\)"):
        materialize_consumed_input_capabilities(
            regime="transductive", modality="graph", consumed_input=object()
        )
    with pytest.raises(TypeError, match="masks must be a mapping"):
        materialize_consumed_input_capabilities(
            regime="transductive",
            modality="graph",
            consumed_input=SimpleNamespace(X=np.ones((2, 1)), masks=[]),
        )
    transductive = SimpleNamespace(
        fit=SimpleNamespace(
            X=np.ones((2, 1)),
            masks={"unlabeled": np.array([False, True])},
            graph=object(),
        )
    )
    result = materialize_consumed_input_capabilities(
        regime="transductive", modality="graph", consumed_input=transductive
    )
    assert result.has_unlabeled and result.has_graph

    with pytest.raises(ValueError, match="regime"):
        materialize_consumed_input_capabilities(
            regime="online",  # type: ignore[arg-type]
            modality="tabular",
            consumed_input=inductive,
        )


class _Sampling:
    def __init__(self, *, graph: bool, present: bool) -> None:
        self._graph = graph
        self.masks = {"unlabeled": np.array([present])}
        self.indices = {
            "train_unlabeled": np.array([1], dtype=np.int64)
            if present
            else np.array([], dtype=np.int64)
        }

    def is_graph(self) -> bool:
        return self._graph


def test_legacy_materializer_covers_graph_indices_and_backend_resolution() -> None:
    graph = materialize_pipeline_capabilities(
        regime="transductive",
        modality="graph",
        primary_input=object(),
        sampling=_Sampling(graph=True, present=True),
        configured_backend="AUTO",
        requires_torch=True,
    )
    assert graph.has_unlabeled and graph.backend == "torch"
    assert "logits" in graph.classifier_outputs

    indices = materialize_pipeline_capabilities(
        regime="inductive",
        modality="tabular",
        primary_input=np.ones((1, 1)),
        sampling=_Sampling(graph=False, present=False),
        configured_backend=None,
        model_configured=True,
    )
    assert not indices.has_unlabeled and indices.backend == "numpy"


def test_cli_json_and_mapping_helpers_cover_success_and_failure() -> None:
    assert json_dumps({"values": frozenset({"b", "a"})}, indent=None) == '{"values": ["a", "b"]}'
    with pytest.raises(TypeError, match="not JSON serializable"):
        json_dumps({"value": object()})
    assert ensure_mapping(None, message="bad") == {}
    assert ensure_mapping({"x": 1}, message="bad") == {"x": 1}
    with pytest.raises(Exception) as caught:
        ensure_mapping([], message="mapping required")
    assert getattr(caught.value, "exit_code", None) == 2


@dataclass(frozen=True)
class _Spec:
    backend: str = "numpy"
    value: int = 1


class _Method:
    def __init__(self) -> None:
        self.spec = _Spec()


class _BrokenMethod:
    def __init__(self) -> None:
        raise RuntimeError("broken")


def test_method_spec_remaining_introspection_and_override_paths() -> None:
    with pytest.raises(ValueError, match="field_name"):
        method_spec_has_field(_Method, "")
    assert not method_spec_has_field(_BrokenMethod, "backend")
    assert build_method_spec(_Method, {"value": 4}) == _Spec(value=4)
    with pytest.raises(MethodSpecError, match="dataclass spec"):
        build_method_spec(_BrokenMethod, require_spec=True)


def test_pipeline_private_resolution_failure_and_cycle_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = MethodResolutionRequest(
        regime="inductive",
        method_id="dummy",
        params={},
    )
    monkeypatch.setattr(pipeline_module, "method_spec_has_field", lambda *args, **kwargs: True)

    def fail_spec(*args: Any, **kwargs: Any) -> None:
        raise MethodSpecError("method_spec", "no spec")

    monkeypatch.setattr(pipeline_module, "build_method_spec", fail_spec)
    with pytest.raises(PipelineResolutionError, match="no spec"):
        pipeline_module._method_backend(request, method_class=_Method)

    cyclic: dict[str, Any] = {}
    cyclic["child"] = cyclic
    assert pipeline_module._classifier_references(cyclic) == ()

    assert pipeline_module._materialized_method_spec(request, _BrokenMethod) is None


@pytest.mark.parametrize("value", [None, "", " bad name! "])
def test_distribution_name_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(SoftwareProvenanceError):
        normalize_distribution_name(value)


def test_software_validation_and_collection_edge_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SoftwareProvenanceError, match="requirements must be strings"):
        requirement_distribution_name(1)  # type: ignore[arg-type]
    with pytest.raises(SoftwareProvenanceError, match="invalid requirement"):
        requirement_distribution_name(" @@@")
    with pytest.raises(SoftwareProvenanceError, match="extra names"):
        resolve_required_distributions(extras=[""])
    with pytest.raises(SoftwareProvenanceError, match="unknown optional"):
        resolve_required_distributions(extras=["missing"], optional_dependencies={})

    with pytest.raises(SoftwareProvenanceError, match="duplicates"):
        SoftwareManifest(("NumPy", "numpy"), {"numpy": "1"})
    with pytest.raises(SoftwareProvenanceError, match="duplicate version"):
        SoftwareManifest(("numpy",), {"NumPy": "1", "numpy": "1"})
    for version in ("", 1):
        with pytest.raises(SoftwareProvenanceError, match="non-empty string or null"):
            SoftwareManifest(("numpy",), {"numpy": version})  # type: ignore[dict-item]

    bad_manifest_values: list[Any] = [
        [],
        {"schema_version": 1},
        {"schema_version": 2, "required_distributions": [], "versions": {}},
        {"schema_version": 1, "required_distributions": [1], "versions": {}},
        {"schema_version": 1, "required_distributions": [], "versions": []},
    ]
    for value in bad_manifest_values:
        with pytest.raises(SoftwareProvenanceError):
            SoftwareManifest.from_dict(value)

    monkeypatch.setattr(
        software_module.metadata,
        "version",
        lambda name: (
            "9.9" if name == "ok" else (_ for _ in ()).throw(metadata.PackageNotFoundError(name))
        ),
    )
    assert software_module._installed_version("ok") == "9.9"
    assert software_module._installed_version("missing") is None

    def missing_version(_name: str) -> str:
        raise metadata.PackageNotFoundError

    manifest = collect_software_manifest(["numpy"], version_getter=missing_version)
    assert manifest.missing_versions == ("numpy",)
    with pytest.raises(TypeError, match="runtime_versions"):
        attach_software_manifest([], required_distributions=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="runtime_versions"):
        software_identity_payload([])  # type: ignore[arg-type]

    legacy = software_identity_payload({"numpy": "2", "torch": None})
    assert legacy["software_manifest"]["versions"] == {"numpy": "2", "torch": None}
