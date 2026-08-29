from __future__ import annotations

import json

import pytest

from modssc.graph.specs import GraphBuilderSpec
from modssc.runtime import dependencies
from modssc.runtime.dependencies import (
    PipelineDependencyError,
    PipelineDependencyRequest,
    resolve_pipeline_dependencies,
)


def test_native_dependency_resolution_uses_component_registries_and_is_stable() -> None:
    graph_spec = {"scheme": "knn", "k": 3, "backend": "faiss"}
    request = PipelineDependencyRequest(
        dataset_id=" cifar10 ",
        preprocess_step_ids=(" core.to_torch ", "vision.openclip", "core.to_torch"),
        method_required_extra=" inductive-torch ",
        classifier_id=" image_pretrained ",
        classifier_backend=" torch ",
        graph_spec=graph_spec,
    )
    graph_spec["backend"] = "numpy"

    result = resolve_pipeline_dependencies(request)

    assert request.dataset_id == "cifar10"
    assert request.preprocess_step_ids == (
        "core.to_torch",
        "vision.openclip",
        "core.to_torch",
    )
    assert request.graph_spec == {"scheme": "knn", "k": 3, "backend": "faiss"}
    assert result.extras == (
        "graph-faiss",
        "inductive-torch",
        "preprocess-vision",
        "vision",
    )
    assert result.resolved_graph_backend == "faiss"
    assert result.to_dict() == {
        "extras": list(result.extras),
        "resolved_graph_backend": "faiss",
    }
    json.dumps(result.to_dict(), allow_nan=False)


def test_auto_graph_backend_is_resolved_by_native_graph_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_backends: list[str] = []

    def resolve_backend(spec: GraphBuilderSpec) -> str:
        seen_backends.append(spec.backend)
        return "sklearn"

    monkeypatch.setattr(dependencies, "resolve_graph_backend", resolve_backend)

    result = resolve_pipeline_dependencies(
        PipelineDependencyRequest(
            dataset_id="toy",
            graph_spec={"scheme": "knn", "k": 2, "backend": "auto"},
        )
    )

    assert seen_backends == ["auto"]
    assert result.extras == ("sklearn",)
    assert result.resolved_graph_backend == "sklearn"


def test_native_components_without_optional_extras_stay_empty() -> None:
    no_graph = resolve_pipeline_dependencies(
        PipelineDependencyRequest(
            dataset_id="toy",
            preprocess_step_ids=("core.to_numpy",),
            classifier_id="knn",
            classifier_backend="numpy",
        )
    )
    numpy_graph = resolve_pipeline_dependencies(
        PipelineDependencyRequest(
            dataset_id="toy",
            graph_spec={"scheme": "knn", "k": 2, "backend": "numpy"},
        )
    )

    assert no_graph.extras == ()
    assert no_graph.resolved_graph_backend is None
    assert numpy_graph.extras == ()
    assert numpy_graph.resolved_graph_backend == "numpy"


def test_annoy_graph_backend_resolves_its_native_extra() -> None:
    result = resolve_pipeline_dependencies(
        PipelineDependencyRequest(
            dataset_id="toy",
            graph_spec={
                "scheme": "knn",
                "metric": "euclidean",
                "k": 10,
                "backend": "annoy",
                "include_self_in_knn": True,
                "annoy_query_k": 30,
            },
        )
    )

    assert result.extras == ("graph-annoy",)
    assert result.resolved_graph_backend == "annoy"


def test_pre_resolved_method_extras_are_normalized_and_deduplicated() -> None:
    request = PipelineDependencyRequest(
        dataset_id="toy",
        method_required_extra=" sklearn ",
        method_required_extras=(" vision ", "sklearn", "vision"),
    )

    result = resolve_pipeline_dependencies(request)

    assert request.method_required_extras == ("vision", "sklearn", "vision")
    assert result.extras == ("sklearn", "vision")


def test_direct_classifier_auto_uses_native_construction_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "modssc.supervised.api.has_module",
        lambda module: module == "sklearn",
    )

    result = resolve_pipeline_dependencies(
        PipelineDependencyRequest(
            dataset_id="toy",
            classifier_id="knn",
            classifier_backend="auto",
        )
    )

    assert result.extras == ("sklearn",)


def test_unregistered_direct_classifier_is_left_to_explicit_software_dependencies() -> None:
    result = resolve_pipeline_dependencies(
        PipelineDependencyRequest(
            dataset_id="toy",
            classifier_id="custom.model.factory",
            classifier_backend="torch",
        )
    )

    assert result.extras == ()


@pytest.mark.parametrize(
    "dependency_request",
    [
        PipelineDependencyRequest(dataset_id="unknown-dataset"),
        PipelineDependencyRequest(dataset_id="toy", preprocess_step_ids=("unknown-step",)),
        PipelineDependencyRequest(
            dataset_id="toy",
            classifier_id="mlp",
            classifier_backend="unknown-backend",
        ),
        PipelineDependencyRequest(
            dataset_id="toy",
            graph_spec={"scheme": "knn", "k": 2, "backend": "unknown-backend"},
        ),
        PipelineDependencyRequest(
            dataset_id="toy",
            graph_spec={"scheme": "epsilon", "radius": 1.0, "backend": "faiss"},
        ),
    ],
)
def test_native_lookup_and_graph_contract_errors_share_one_boundary(
    dependency_request: PipelineDependencyRequest,
) -> None:
    with pytest.raises(PipelineDependencyError, match="native pipeline dependencies") as raised:
        resolve_pipeline_dependencies(dependency_request)

    assert raised.value.__cause__ is not None


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"dataset_id": " "}, ValueError),
        ({"dataset_id": "toy", "preprocess_step_ids": "core.to_numpy"}, TypeError),
        ({"dataset_id": "toy", "preprocess_step_ids": ("",)}, ValueError),
        ({"dataset_id": "toy", "method_required_extra": 1}, ValueError),
        ({"dataset_id": "toy", "method_required_extras": "vision"}, TypeError),
        ({"dataset_id": "toy", "method_required_extras": (None,)}, ValueError),
        ({"dataset_id": "toy", "graph_spec": []}, TypeError),
    ],
)
def test_dependency_request_rejects_ambiguous_identifiers(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        PipelineDependencyRequest(**kwargs)  # type: ignore[arg-type]


def test_resolver_rejects_non_request_values() -> None:
    with pytest.raises(TypeError, match="PipelineDependencyRequest"):
        resolve_pipeline_dependencies(object())  # type: ignore[arg-type]
