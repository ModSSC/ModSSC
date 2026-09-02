"""Thin YAML adapter for native ModSSC resource limits."""

from __future__ import annotations

from typing import Any

from modssc.runtime.limits import (
    ResolvedResourceLimits,
    ResourceLimitError,
    ResourceLimits,
    apply_resource_limits,
    resolve_resource_limits,
)

from .schema import BenchConfigError, LimitsConfig

ResolvedLimits = ResolvedResourceLimits


def _native_limits(config: LimitsConfig | None) -> ResourceLimits | None:
    if config is None:
        return None
    return ResourceLimits(
        profile=config.profile,
        max_preprocess_batch_size=config.max_preprocess_batch_size,
        max_method_batch_size=config.max_method_batch_size,
        max_method_sup_batch_size=config.max_method_sup_batch_size,
        max_graph_chunk_size=config.max_graph_chunk_size,
        max_train_samples=config.max_train_samples,
        max_test_samples=config.max_test_samples,
    )


def resolve_limits(
    config: LimitsConfig | None,
    *,
    strict: bool = False,
) -> ResolvedLimits | None:
    try:
        return resolve_resource_limits(_native_limits(config), strict=strict)
    except ResourceLimitError as exc:
        code = "E_BENCH_AUTO_FORBIDDEN" if "auto" in str(exc) else "E_BENCH_CONFIG"
        raise BenchConfigError(str(exc), code=code) from exc


def apply_limits(
    raw: dict[str, Any],
    *,
    limits: LimitsConfig | None,
    strict: bool = False,
) -> tuple[dict[str, Any], list[str], ResolvedLimits | None]:
    try:
        return apply_resource_limits(raw, limits=_native_limits(limits), strict=strict)
    except ResourceLimitError as exc:
        code = "E_BENCH_AUTO_FORBIDDEN" if "auto" in str(exc) else "E_BENCH_CONFIG"
        raise BenchConfigError(str(exc), code=code) from exc


__all__ = ["ResolvedLimits", "apply_limits", "resolve_limits"]
