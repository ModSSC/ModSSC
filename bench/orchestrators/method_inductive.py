from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from modssc.inductive import (
    DeviceSpec,
    InductiveExecutionConfig,
    InductiveExecutionError,
    InductiveExecutionInput,
    ModelBuildConfig,
    execute_inductive_method,
)

from ..errors import BenchRuntimeError
from ..schema import MethodConfig

_LOGGER = logging.getLogger(__name__)

_BENCH_ERROR_CODES = {
    "graph_sampling": "E_BENCH_GRAPH_SAMPLING_INVALID",
    "auto_backend": "E_BENCH_AUTO_FORBIDDEN",
    "dependency_missing": "E_BENCH_DEPENDENCY_MISSING",
    "labels_contract": "E_BENCH_LABELS_CONTRACT",
    "evaluation_split": "E_BENCH_EVAL_SPLIT_INVALID",
    "torch_required": "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
    "shape": "E_BENCH_SHAPE_CONTRACT",
    "dtype": "E_BENCH_DTYPE_CONTRACT",
    "method_contract": "E_BENCH_METHOD_CONTRACT",
    "method_introspection": "E_BENCH_METHOD_INTROSPECTION",
    "method_spec": "E_BENCH_METHOD_SPEC",
    "model_config": "E_BENCH_MODEL_CONFIG",
    "capability": "E_BENCH_CAPABILITY",
    "graph_contract": "E_BENCH_GRAPH_CONTRACT",
    "execution_contract": "E_BENCH_EXECUTION_CONTRACT",
}


def _native_model_config(model: Any | None) -> ModelBuildConfig | None:
    if model is None:
        return None
    return ModelBuildConfig(
        factory=model.factory,
        params=dict(model.params),
        classifier_id=model.classifier_id,
        classifier_backend=model.classifier_backend,
        classifier_params=dict(model.classifier_params),
        ema=model.ema,
    )


def _native_execution_config(
    cfg: MethodConfig,
    *,
    seed: int,
    strict: bool,
    requires_torch: bool,
    during_fit_splits: list[str] | None,
) -> InductiveExecutionConfig:
    return InductiveExecutionConfig(
        method_id=cfg.method_id,
        device=DeviceSpec(device=cfg.device.device, dtype=cfg.device.dtype),
        params=dict(cfg.params),
        model=_native_model_config(cfg.model),
        seed=int(seed),
        strict=bool(strict),
        requires_torch=bool(requires_torch),
        during_fit_splits=tuple(during_fit_splits or ()),
    )


def _bench_execution_error(exc: InductiveExecutionError) -> BenchRuntimeError:
    return BenchRuntimeError(_BENCH_ERROR_CODES[exc.kind], str(exc))


def run(
    inputs: InductiveExecutionInput,
    *,
    during_fit_splits: list[str] | None = None,
    cfg: MethodConfig,
    seed: int,
    strict: bool = False,
    requires_torch: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Adapt one validated benchmark method block to the native inductive API."""

    start = perf_counter()
    model_ref = cfg.model.factory or cfg.model.classifier_id if cfg.model is not None else None
    _LOGGER.info(
        "Inductive method start: id=%s seed=%s device=%s dtype=%s model=%s strict=%s",
        cfg.method_id,
        int(seed),
        cfg.device.device,
        cfg.device.dtype,
        model_ref,
        bool(strict),
    )
    _LOGGER.debug("Inductive method params: %s", dict(cfg.params))

    config = _native_execution_config(
        cfg,
        seed=seed,
        strict=strict,
        requires_torch=requires_torch,
        during_fit_splits=during_fit_splits,
    )
    try:
        result = execute_inductive_method(inputs, config)
    except InductiveExecutionError as exc:
        raise _bench_execution_error(exc) from exc

    _LOGGER.info(
        "Inductive method done: id=%s duration_s=%.3f",
        cfg.method_id,
        perf_counter() - start,
    )
    return result.method, dict(result.resolution)
