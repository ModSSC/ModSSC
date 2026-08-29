from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from modssc.transductive import (
    DeviceSpec,
    PreparedNodeData,
    TransductiveExecutionConfig,
    TransductiveExecutionError,
    TransductiveExecutionInput,
    execute_transductive_method,
)

from ..errors import BenchRuntimeError
from ..schema import MethodConfig

_LOGGER = logging.getLogger(__name__)

_BENCH_ERROR_CODES = {
    "auto_backend": "E_BENCH_AUTO_FORBIDDEN",
    "dependency_missing": "E_BENCH_DEPENDENCY_MISSING",
    "method_contract": "E_BENCH_METHOD_CONTRACT",
    "method_introspection": "E_BENCH_METHOD_INTROSPECTION",
    "method_spec": "E_BENCH_METHOD_SPEC",
    "data_contract": "E_BENCH_MASK_CONTRACT",
    "augmentation_contract": "E_BENCH_AUGMENTATION_UNSUPPORTED",
    "capability": "E_BENCH_CAPABILITY",
    "execution_contract": "E_BENCH_EXECUTION_CONTRACT",
}


def _native_execution_config(
    cfg: MethodConfig,
    *,
    seed: int,
    use_test_split: bool,
    expected_labeled_count: int | None,
    strict: bool,
) -> TransductiveExecutionConfig:
    return TransductiveExecutionConfig(
        method_id=cfg.method_id,
        device=DeviceSpec(device=cfg.device.device, dtype=cfg.device.dtype),
        params=dict(cfg.params),
        seed=int(seed),
        strict=bool(strict),
        use_test_split=bool(use_test_split),
        expected_labeled_count=expected_labeled_count,
    )


def _bench_execution_error(exc: TransductiveExecutionError) -> BenchRuntimeError:
    return BenchRuntimeError(_BENCH_ERROR_CODES[exc.kind], str(exc))


def run(
    inputs: TransductiveExecutionInput,
    *,
    cfg: MethodConfig,
    seed: int,
    use_test_split: bool,
    expected_labeled_count: int | None = None,
    strict: bool = False,
) -> tuple[Any, PreparedNodeData, dict[str, Any]]:
    """Adapt one validated benchmark method block to the native transductive API."""

    start = perf_counter()
    _LOGGER.info(
        "Transductive method start: id=%s seed=%s device=%s dtype=%s use_test=%s strict=%s",
        cfg.method_id,
        int(seed),
        cfg.device.device,
        cfg.device.dtype,
        bool(use_test_split),
        bool(strict),
    )
    _LOGGER.debug("Transductive method params: %s", dict(cfg.params))

    config = _native_execution_config(
        cfg,
        seed=seed,
        use_test_split=use_test_split,
        expected_labeled_count=expected_labeled_count,
        strict=strict,
    )
    try:
        result = execute_transductive_method(inputs, config)
    except TransductiveExecutionError as exc:
        raise _bench_execution_error(exc) from exc

    fit_data = result.data.fit
    n_train = int(inputs.dataset.train.X.shape[0])
    n_test = int(fit_data.X.shape[0]) - n_train
    labeled_count = int(fit_data.masks["labeled_mask"].sum())
    train_all_mask = fit_data.masks.get("train_all_mask")
    train_count = int(train_all_mask.sum()) if train_all_mask is not None else None
    _LOGGER.info(
        "Transductive data: n_nodes=%s n_train=%s n_test=%s labeled=%s train=%s",
        int(fit_data.X.shape[0]),
        n_train,
        n_test,
        labeled_count,
        train_count,
    )
    _LOGGER.info(
        "Transductive method done: id=%s device=%s backend=%s duration_s=%.3f",
        cfg.method_id,
        result.resolved_device,
        result.backend,
        perf_counter() - start,
    )
    return result.method, result.data, dict(result.resolution)
