from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_augmentation.api import build_strategy
from modssc.data_augmentation.plan import AugmentationPlan, StepConfig
from modssc.data_augmentation.types import AugmentationContext
from modssc.data_augmentation.utils import is_torch_tensor

_LOGGER = logging.getLogger(__name__)


def _plan_from_dict(obj: Mapping[str, Any], *, modality: str | None) -> AugmentationPlan:
    steps_raw = obj.get("steps", [])
    if not isinstance(steps_raw, list):
        raise ValueError("augmentation.steps must be a list")
    steps: list[StepConfig] = []
    for item in steps_raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each augmentation step must be a mapping")
        op_id = str(item.get("id") or item.get("op_id") or "")
        if not op_id:
            raise ValueError("Each augmentation step must define 'id'")
        params = item.get("params", {}) or {}
        if not isinstance(params, Mapping):
            raise ValueError(f"params for op {op_id!r} must be a mapping")
        steps.append(StepConfig(op_id=op_id, params=dict(params)))
    return AugmentationPlan(steps=tuple(steps), modality=modality)


def run(
    X_u: Any,
    *,
    weak_plan: Mapping[str, Any],
    strong_plan: Mapping[str, Any],
    seed: int,
    mode: str,
    modality: str | None,
    sample_ids: np.ndarray | None = None,
) -> tuple[Any, Any]:
    start = perf_counter()
    if mode != "fixed":
        raise ValueError("Only augmentation.mode='fixed' is supported")

    if X_u is None:
        _LOGGER.info("Augmentation skipped: X_u is None")
        return None, None

    X_u_arr = X_u
    if hasattr(X_u_arr, "shape") and int(X_u_arr.shape[0]) == 0:
        _LOGGER.info("Augmentation skipped: empty unlabeled split")
        return X_u_arr, X_u_arr

    weak = _plan_from_dict(weak_plan, modality=modality)
    strong = _plan_from_dict(strong_plan, modality=modality)
    _LOGGER.info(
        "Augmentation start: mode=%s modality=%s seed=%s n_samples=%s",
        mode,
        modality,
        int(seed),
        int(X_u_arr.shape[0]),
    )
    _LOGGER.debug(
        "Augmentation steps: weak=%s strong=%s",
        [step.op_id for step in weak.steps],
        [step.op_id for step in strong.steps],
    )
    strategy = build_strategy(weak=weak, strong=strong)

    if sample_ids is None:
        sample_ids = np.arange(int(X_u_arr.shape[0]), dtype=np.int64)

    n_samples = int(X_u_arr.shape[0])
    out_w: list[Any] = []
    out_s: list[Any] = []
    out_w_arr: Any | None = None
    out_s_arr: Any | None = None

    def _ctx(i: int) -> AugmentationContext:
        return AugmentationContext(
            seed=int(seed),
            sample_id=int(sample_ids[i]),
            epoch=0,
            modality=modality,
        )

    # Seed first sample to decide whether we can preallocate.
    sample0 = X_u_arr[0]
    xw0, xs0 = strategy.apply(sample0, ctx=_ctx(0))

    if is_torch_tensor(xw0) and is_torch_tensor(xs0):
        try:
            out_w_arr = xw0.new_empty((n_samples,) + tuple(xw0.shape))
            out_s_arr = xs0.new_empty((n_samples,) + tuple(xs0.shape))
            out_w_arr[0] = xw0
            out_s_arr[0] = xs0
        except Exception:
            out_w = [xw0]
            out_s = [xs0]
            out_w_arr = None
            out_s_arr = None
    elif isinstance(xw0, np.ndarray) and isinstance(xs0, np.ndarray):
        try:
            out_w_arr = np.empty((n_samples,) + xw0.shape, dtype=xw0.dtype)
            out_s_arr = np.empty((n_samples,) + xs0.shape, dtype=xs0.dtype)
            out_w_arr[0] = xw0
            out_s_arr[0] = xs0
        except Exception:
            out_w = [xw0]
            out_s = [xs0]
            out_w_arr = None
            out_s_arr = None
    else:
        out_w = [xw0]
        out_s = [xs0]

    for i in range(1, n_samples):
        sample = X_u_arr[i]
        xw, xs = strategy.apply(sample, ctx=_ctx(i))
        if out_w_arr is not None and out_s_arr is not None:
            if tuple(xw.shape) != tuple(out_w_arr.shape[1:]) or tuple(xs.shape) != tuple(
                out_s_arr.shape[1:]
            ):
                if is_torch_tensor(out_w_arr):
                    out_w = [out_w_arr[j].clone() for j in range(i)]
                    out_s = [out_s_arr[j].clone() for j in range(i)]
                else:
                    out_w = [out_w_arr[j].copy() for j in range(i)]
                    out_s = [out_s_arr[j].copy() for j in range(i)]
                out_w_arr = None
                out_s_arr = None
                out_w.append(xw)
                out_s.append(xs)
            else:
                out_w_arr[i] = xw
                out_s_arr[i] = xs
        else:
            out_w.append(xw)
            out_s.append(xs)

    if out_w_arr is not None and out_s_arr is not None:
        _LOGGER.info("Augmentation done: duration_s=%.3f", perf_counter() - start)
        _LOGGER.debug(
            "Augmentation output shapes: weak=%s strong=%s",
            tuple(out_w_arr.shape),
            tuple(out_s_arr.shape),
        )
        return out_w_arr, out_s_arr

    if is_torch_tensor(X_u_arr):
        import importlib

        torch = importlib.import_module("torch")
        out_w_t = torch.stack(out_w, dim=0)
        out_s_t = torch.stack(out_s, dim=0)
        _LOGGER.info("Augmentation done: duration_s=%.3f", perf_counter() - start)
        _LOGGER.debug(
            "Augmentation output shapes: weak=%s strong=%s",
            tuple(out_w_t.shape),
            tuple(out_s_t.shape),
        )
        return out_w_t, out_s_t

    out_w_np = np.stack(out_w, axis=0)
    out_s_np = np.stack(out_s, axis=0)
    _LOGGER.info("Augmentation done: duration_s=%.3f", perf_counter() - start)
    _LOGGER.debug(
        "Augmentation output shapes: weak=%s strong=%s",
        tuple(out_w_np.shape),
        tuple(out_s_np.shape),
    )
    return out_w_np, out_s_np
