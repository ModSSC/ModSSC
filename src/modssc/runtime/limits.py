"""Native resource-limit resolution and effective-config materialization."""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Any


class ResourceLimitError(ValueError):
    """Raised when a resource-limit contract cannot be resolved safely."""


@dataclass(frozen=True)
class ResourceLimits:
    profile: str | None = None
    max_preprocess_batch_size: int | None = None
    max_method_batch_size: int | None = None
    max_method_sup_batch_size: int | None = None
    max_graph_chunk_size: int | None = None
    max_train_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(frozen=True)
class ResolvedResourceLimits:
    profile: str | None
    max_preprocess_batch_size: int | None
    max_method_batch_size: int | None
    max_method_sup_batch_size: int | None
    max_graph_chunk_size: int | None
    max_train_samples: int | None
    max_test_samples: int | None


_LIMIT_PRESETS: dict[str, dict[str, int]] = {
    "v100": {
        "max_preprocess_batch_size": 32,
        "max_method_batch_size": 128,
        "max_method_sup_batch_size": 64,
        "max_graph_chunk_size": 512,
    },
    "h100": {
        "max_preprocess_batch_size": 64,
        "max_method_batch_size": 512,
        "max_method_sup_batch_size": 256,
        "max_graph_chunk_size": 1024,
    },
}


def _detect_profile() -> str | None:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not getattr(cuda, "is_available", lambda: False)():
        return None
    try:
        props = cuda.get_device_properties(0)
    except Exception:
        return None
    name = str(getattr(props, "name", "")).lower()
    if "h100" in name:
        return "h100"
    if "v100" in name:
        return "v100"
    total_memory = float(getattr(props, "total_memory", 0.0))
    if total_memory <= 0:
        return None
    return "h100" if total_memory / (1024**3) >= 60 else "v100"


def resolve_resource_limits(
    limits: ResourceLimits | None,
    *,
    strict: bool = False,
) -> ResolvedResourceLimits | None:
    """Resolve an explicit or hardware-profile resource contract."""

    if limits is None:
        return None
    if not isinstance(limits, ResourceLimits):
        raise TypeError("limits must be ResourceLimits or None")
    profile = limits.profile.lower() if limits.profile else None
    if profile not in {None, "auto", "v100", "h100"}:
        raise ResourceLimitError("profile must be auto, v100, h100, or None")
    if strict and profile == "auto":
        raise ResourceLimitError("profile='auto' is forbidden in strict execution")
    resolved_profile = (_detect_profile() or "v100") if profile == "auto" else profile
    defaults = _LIMIT_PRESETS.get(resolved_profile or "", {})
    resolved = ResolvedResourceLimits(
        profile=resolved_profile,
        max_preprocess_batch_size=(
            limits.max_preprocess_batch_size
            if limits.max_preprocess_batch_size is not None
            else defaults.get("max_preprocess_batch_size")
        ),
        max_method_batch_size=(
            limits.max_method_batch_size
            if limits.max_method_batch_size is not None
            else defaults.get("max_method_batch_size")
        ),
        max_method_sup_batch_size=(
            limits.max_method_sup_batch_size
            if limits.max_method_sup_batch_size is not None
            else defaults.get("max_method_sup_batch_size")
        ),
        max_graph_chunk_size=(
            limits.max_graph_chunk_size
            if limits.max_graph_chunk_size is not None
            else defaults.get("max_graph_chunk_size")
        ),
        max_train_samples=limits.max_train_samples,
        max_test_samples=limits.max_test_samples,
    )
    if all(
        value is None
        for value in (
            resolved.max_preprocess_batch_size,
            resolved.max_method_batch_size,
            resolved.max_method_sup_batch_size,
            resolved.max_graph_chunk_size,
            resolved.max_train_samples,
            resolved.max_test_samples,
        )
    ):
        return None
    return resolved


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp_key(
    container: dict[str, Any],
    *,
    key: str,
    limit: int | None,
    path: str,
    changes: list[str],
    set_if_missing: bool = False,
) -> None:
    if limit is None:
        return
    if key not in container:
        if set_if_missing:
            container[key] = int(limit)
            changes.append(f"{path}.{key}: set to {int(limit)}")
        return
    current = _coerce_int(container.get(key))
    if current is None:
        return
    updated = min(current, int(limit))
    if updated != current:
        container[key] = updated
        changes.append(f"{path}.{key}: {current} -> {updated}")


def _clamp_preprocess_steps(
    plan: dict[str, Any],
    *,
    limit: int | None,
    path: str,
    changes: list[str],
) -> None:
    steps = plan.get("steps")
    if limit is None or not isinstance(steps, list):
        return
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not isinstance(step.get("params"), dict):
            continue
        step_id = step.get("id") or step.get("step_id") or str(index)
        _clamp_key(
            step["params"],
            key="batch_size",
            limit=limit,
            path=f"{path}.steps[{step_id}].params",
            changes=changes,
        )


def _mapping_child(parent: dict[str, Any], key: str) -> dict[str, Any]:
    child = parent.get(key)
    if not isinstance(child, dict):
        child = {}
        parent[key] = child
    return child


def apply_resource_limits(
    config: dict[str, Any],
    *,
    limits: ResourceLimits | None,
    strict: bool = False,
) -> tuple[dict[str, Any], list[str], ResolvedResourceLimits | None]:
    """Return an effective config with one native resource contract applied."""

    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    effective = copy.deepcopy(config)
    resolved = resolve_resource_limits(limits, strict=strict)
    if resolved is None:
        return effective, [], None
    changes: list[str] = []

    dataset = effective.get("dataset")
    if isinstance(dataset, dict):
        options = _mapping_child(dataset, "options")
        _clamp_key(
            options,
            key="max_train_samples",
            limit=resolved.max_train_samples,
            path="dataset.options",
            changes=changes,
            set_if_missing=True,
        )
        _clamp_key(
            options,
            key="max_test_samples",
            limit=resolved.max_test_samples,
            path="dataset.options",
            changes=changes,
            set_if_missing=True,
        )

    preprocess = effective.get("preprocess")
    if isinstance(preprocess, dict) and isinstance(preprocess.get("plan"), dict):
        _clamp_preprocess_steps(
            preprocess["plan"],
            limit=resolved.max_preprocess_batch_size,
            path="preprocess.plan",
            changes=changes,
        )

    views = effective.get("views")
    views_plan = views.get("plan") if isinstance(views, dict) else None
    view_items = views_plan.get("views") if isinstance(views_plan, dict) else None
    if isinstance(view_items, list):
        for index, view in enumerate(view_items):
            if not isinstance(view, dict) or not isinstance(view.get("preprocess"), dict):
                continue
            view_name = view.get("name") or str(index)
            _clamp_preprocess_steps(
                view["preprocess"],
                limit=resolved.max_preprocess_batch_size,
                path=f"views.plan.views[{view_name}].preprocess",
                changes=changes,
            )

    method = effective.get("method")
    if isinstance(method, dict):
        params = method.get("params")
        if isinstance(params, dict):
            _clamp_key(
                params,
                key="batch_size",
                limit=resolved.max_method_batch_size,
                path="method.params",
                changes=changes,
            )
            _clamp_key(
                params,
                key="sup_batch_size",
                limit=resolved.max_method_sup_batch_size,
                path="method.params",
                changes=changes,
            )
            classifier_params = params.get("classifier_params")
            if isinstance(classifier_params, dict):
                _clamp_key(
                    classifier_params,
                    key="batch_size",
                    limit=resolved.max_method_batch_size,
                    path="method.params.classifier_params",
                    changes=changes,
                )
        model = method.get("model")
        classifier_params = model.get("classifier_params") if isinstance(model, dict) else None
        if isinstance(classifier_params, dict):
            _clamp_key(
                classifier_params,
                key="batch_size",
                limit=resolved.max_method_batch_size,
                path="method.model.classifier_params",
                changes=changes,
            )

    graph = effective.get("graph")
    graph_spec = graph.get("spec") if isinstance(graph, dict) else None
    if isinstance(graph_spec, dict):
        _clamp_key(
            graph_spec,
            key="chunk_size",
            limit=resolved.max_graph_chunk_size,
            path="graph.spec",
            changes=changes,
            set_if_missing=True,
        )
    return effective, changes, resolved


__all__ = [
    "ResolvedResourceLimits",
    "ResourceLimitError",
    "ResourceLimits",
    "apply_resource_limits",
    "resolve_resource_limits",
]
