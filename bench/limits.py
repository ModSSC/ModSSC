from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import BenchConfigError, LimitsConfig


@dataclass(frozen=True)
class ResolvedLimits:
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
        import importlib

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

    total_mem = float(getattr(props, "total_memory", 0.0))
    if total_mem <= 0:
        return None
    total_gb = total_mem / (1024**3)
    if total_gb >= 60:
        return "h100"
    return "v100"


def resolve_limits(cfg: LimitsConfig | None, *, strict: bool = False) -> ResolvedLimits | None:
    if cfg is None:
        return None

    profile = cfg.profile.lower() if cfg.profile else None
    if strict and profile == "auto":
        raise BenchConfigError(
            "limits.profile='auto' is forbidden when run.benchmark_mode=true",
            code="E_BENCH_AUTO_FORBIDDEN",
        )
    resolved_profile = None
    defaults: dict[str, int] = {}

    if profile:
        resolved_profile = _detect_profile() or "v100" if profile == "auto" else profile
        defaults = _LIMIT_PRESETS.get(resolved_profile, {})

    resolved = ResolvedLimits(
        profile=resolved_profile,
        max_preprocess_batch_size=(
            cfg.max_preprocess_batch_size
            if cfg.max_preprocess_batch_size is not None
            else defaults.get("max_preprocess_batch_size")
        ),
        max_method_batch_size=(
            cfg.max_method_batch_size
            if cfg.max_method_batch_size is not None
            else defaults.get("max_method_batch_size")
        ),
        max_method_sup_batch_size=(
            cfg.max_method_sup_batch_size
            if cfg.max_method_sup_batch_size is not None
            else defaults.get("max_method_sup_batch_size")
        ),
        max_graph_chunk_size=(
            cfg.max_graph_chunk_size
            if cfg.max_graph_chunk_size is not None
            else defaults.get("max_graph_chunk_size")
        ),
        max_train_samples=cfg.max_train_samples,
        max_test_samples=cfg.max_test_samples,
    )

    if (
        resolved.max_preprocess_batch_size is None
        and resolved.max_method_batch_size is None
        and resolved.max_method_sup_batch_size is None
        and resolved.max_graph_chunk_size is None
        and resolved.max_train_samples is None
        and resolved.max_test_samples is None
    ):
        return None

    return resolved


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def _clamp_value(value: Any, limit: int) -> tuple[int | Any, bool]:
    current = _coerce_int(value)
    if current is None:
        return value, False
    updated = min(int(current), int(limit))
    return updated, updated != current


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
    old = container.get(key)
    new, changed = _clamp_value(old, int(limit))
    if changed:
        container[key] = new
        changes.append(f"{path}.{key}: {old} -> {new}")


def _clamp_preprocess_steps(
    plan: dict[str, Any],
    *,
    limit: int | None,
    path: str,
    changes: list[str],
) -> None:
    if limit is None:
        return
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            continue
        step_id = step.get("id") or step.get("step_id") or str(idx)
        _clamp_key(
            params,
            key="batch_size",
            limit=limit,
            path=f"{path}.steps[{step_id}].params",
            changes=changes,
            set_if_missing=False,
        )


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    child = parent.get(key)
    if not isinstance(child, dict):
        child = {}
        parent[key] = child
    return child


def apply_limits(
    raw: dict[str, Any], *, limits: LimitsConfig | None, strict: bool = False
) -> tuple[dict[str, Any], list[str], ResolvedLimits | None]:
    resolved = resolve_limits(limits, strict=strict)
    if resolved is None:
        return raw, [], None

    changes: list[str] = []

    dataset = raw.get("dataset")
    if isinstance(dataset, dict):
        options = _ensure_dict(dataset, "options")
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

    preprocess = raw.get("preprocess")
    if isinstance(preprocess, dict):
        plan = preprocess.get("plan")
        if isinstance(plan, dict):
            _clamp_preprocess_steps(
                plan,
                limit=resolved.max_preprocess_batch_size,
                path="preprocess.plan",
                changes=changes,
            )

    views = raw.get("views")
    if isinstance(views, dict):
        views_plan = views.get("plan")
        if isinstance(views_plan, dict):
            view_list = views_plan.get("views")
            if isinstance(view_list, list):
                for idx, view in enumerate(view_list):
                    if not isinstance(view, dict):
                        continue
                    view_name = view.get("name") or str(idx)
                    view_pre = view.get("preprocess")
                    if isinstance(view_pre, dict):
                        _clamp_preprocess_steps(
                            view_pre,
                            limit=resolved.max_preprocess_batch_size,
                            path=f"views.plan.views[{view_name}].preprocess",
                            changes=changes,
                        )

    method = raw.get("method")
    if isinstance(method, dict):
        params = method.get("params")
        if isinstance(params, dict):
            _clamp_key(
                params,
                key="batch_size",
                limit=resolved.max_method_batch_size,
                path="method.params",
                changes=changes,
                set_if_missing=False,
            )
            _clamp_key(
                params,
                key="sup_batch_size",
                limit=resolved.max_method_sup_batch_size,
                path="method.params",
                changes=changes,
                set_if_missing=False,
            )
            clf_params = params.get("classifier_params")
            if isinstance(clf_params, dict):
                _clamp_key(
                    clf_params,
                    key="batch_size",
                    limit=resolved.max_method_batch_size,
                    path="method.params.classifier_params",
                    changes=changes,
                    set_if_missing=False,
                )

        model = method.get("model")
        if isinstance(model, dict):
            model_params = model.get("classifier_params")
            if isinstance(model_params, dict):
                _clamp_key(
                    model_params,
                    key="batch_size",
                    limit=resolved.max_method_batch_size,
                    path="method.model.classifier_params",
                    changes=changes,
                    set_if_missing=False,
                )

    graph = raw.get("graph")
    if isinstance(graph, dict):
        spec = graph.get("spec")
        if isinstance(spec, dict):
            _clamp_key(
                spec,
                key="chunk_size",
                limit=resolved.max_graph_chunk_size,
                path="graph.spec",
                changes=changes,
                set_if_missing=True,
            )

    return raw, changes, resolved
