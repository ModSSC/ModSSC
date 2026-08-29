"""Native dependency declarations for a materialized ModSSC pipeline.

The benchmark runner supplies identifiers parsed from configuration.  Native
registries own the mapping from those identifiers to optional dependency
extras, including environment-dependent graph backend resolution.  Expansion
of extras into Python distribution requirements remains an installation
metadata concern and is intentionally handled by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from modssc.data_loader import dataset_info
from modssc.graph.construction.builder import resolve_graph_backend
from modssc.graph.errors import GraphError
from modssc.graph.specs import GraphBuilderSpec, graph_backend_required_extra
from modssc.preprocess import step_info
from modssc.supervised.api import resolve_classifier_backend_spec
from modssc.supervised.errors import SupervisedError
from modssc.supervised.registry import list_classifiers


class PipelineDependencyError(ValueError):
    """Raised when a native component dependency cannot be resolved."""


def _optional_name(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string or None")
    return value.strip()


@dataclass(frozen=True)
class PipelineDependencyRequest:
    """Identifiers needed to discover dependencies of one selected pipeline."""

    dataset_id: str
    preprocess_step_ids: Sequence[str] = ()
    method_required_extra: str | None = None
    method_required_extras: Sequence[str] = ()
    classifier_id: str | None = None
    classifier_backend: str | None = None
    graph_spec: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        object.__setattr__(self, "dataset_id", self.dataset_id.strip())
        if isinstance(self.preprocess_step_ids, str):
            raise TypeError("preprocess_step_ids must be a sequence of step identifiers")
        steps: list[str] = []
        for step_id in self.preprocess_step_ids:
            if not isinstance(step_id, str) or not step_id.strip():
                raise ValueError("preprocess_step_ids items must be non-empty strings")
            steps.append(step_id.strip())
        object.__setattr__(self, "preprocess_step_ids", tuple(steps))
        object.__setattr__(
            self,
            "method_required_extra",
            _optional_name(self.method_required_extra, field_name="method_required_extra"),
        )
        if isinstance(self.method_required_extras, str):
            raise TypeError("method_required_extras must be a sequence of extra identifiers")
        method_extras: list[str] = []
        for extra in self.method_required_extras:
            normalized = _optional_name(extra, field_name="method_required_extras item")
            if normalized is None:
                raise ValueError("method_required_extras items must be non-empty strings")
            method_extras.append(normalized)
        object.__setattr__(self, "method_required_extras", tuple(method_extras))
        object.__setattr__(
            self,
            "classifier_id",
            _optional_name(self.classifier_id, field_name="classifier_id"),
        )
        object.__setattr__(
            self,
            "classifier_backend",
            _optional_name(self.classifier_backend, field_name="classifier_backend"),
        )
        if self.graph_spec is not None:
            if not isinstance(self.graph_spec, Mapping):
                raise TypeError("graph_spec must be a mapping or None")
            object.__setattr__(self, "graph_spec", dict(self.graph_spec))


@dataclass(frozen=True)
class PipelineDependencyResolution:
    """Stable optional extras and the graph backend that selected them."""

    extras: tuple[str, ...] = field(default_factory=tuple)
    resolved_graph_backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "extras": list(self.extras),
            "resolved_graph_backend": self.resolved_graph_backend,
        }


def resolve_pipeline_dependencies(
    request: PipelineDependencyRequest,
) -> PipelineDependencyResolution:
    """Resolve component declarations without exposing native registries to runners."""

    if not isinstance(request, PipelineDependencyRequest):
        raise TypeError("request must be a PipelineDependencyRequest")

    extras: set[str] = set()
    resolved_graph_backend: str | None = None
    try:
        dataset_extra = dataset_info(request.dataset_id).required_extra
        if dataset_extra:
            extras.add(str(dataset_extra))

        for step_id in request.preprocess_step_ids:
            step_extra = step_info(step_id).get("required_extra")
            if step_extra:
                extras.add(str(step_extra))

        if request.method_required_extra:
            extras.add(request.method_required_extra)
        extras.update(request.method_required_extras)

        if (
            request.classifier_id is not None
            and request.classifier_backend is not None
            and request.classifier_id in set(list_classifiers())
        ):
            classifier_extra = resolve_classifier_backend_spec(
                request.classifier_id,
                backend=request.classifier_backend,
            ).required_extra
            if classifier_extra:
                extras.add(str(classifier_extra))

        if request.graph_spec is not None:
            graph = GraphBuilderSpec.from_dict(dict(request.graph_spec))
            graph.validate()
            resolved_graph_backend = resolve_graph_backend(graph)
            graph_extra = graph_backend_required_extra(resolved_graph_backend)
            if graph_extra:
                extras.add(str(graph_extra))
    except (
        TypeError,
        ValueError,
        RuntimeError,
        ImportError,
        GraphError,
        SupervisedError,
    ) as exc:
        raise PipelineDependencyError(
            f"unable to resolve native pipeline dependencies: {exc}"
        ) from exc

    return PipelineDependencyResolution(
        extras=tuple(sorted(extras)),
        resolved_graph_backend=resolved_graph_backend,
    )


__all__ = [
    "PipelineDependencyError",
    "PipelineDependencyRequest",
    "PipelineDependencyResolution",
    "resolve_pipeline_dependencies",
]
