from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable, Mapping
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_augmentation.utils import is_torch_tensor
from modssc.evaluation import compute_metrics, labels_1d, predict_labels, to_numpy
from modssc.inductive.types import InductiveDataset
from modssc.preprocess.types import PreprocessResult
from modssc.sampling.result import SamplingResult
from modssc.views.types import ViewsResult

from ..errors import BenchRuntimeError
from .slicing import select_rows

_LOGGER = logging.getLogger(__name__)


def _split_data(
    pre: PreprocessResult,
    sampling: SamplingResult,
    *,
    split: str,
) -> tuple[Any, Any]:
    if sampling.is_graph():
        raise BenchRuntimeError(
            "E_BENCH_EVAL_SPLIT_INVALID",
            "inductive evaluation does not support graph sampling",
        )

    idx = np.asarray(sampling.indices[split], dtype=np.int64)
    ref = sampling.refs.get(split, "train")

    if ref == "train":
        base = pre.dataset.train
    else:
        if pre.dataset.test is None:
            raise BenchRuntimeError(
                "E_BENCH_EVAL_SPLIT_INVALID",
                "requested test split but dataset has no test split",
            )
        base = pre.dataset.test

    X = base.X
    y = _labels_for_split(pre, ref, base)
    return (
        select_rows(X, idx, context=f"evaluation.{split}.X"),
        select_rows(y, idx, context=f"evaluation.{split}.y"),
    )


def _labels_for_split(pre: PreprocessResult, ref: str, base: Any) -> Any:
    store = pre.train_artifacts if ref == "train" else pre.test_artifacts
    if store is not None and store.has("labels.y"):
        return store.get("labels.y")
    return base.y


def _smart_to_torch(x: Any, device: Any) -> Any:
    if x is None:
        return None

    if isinstance(x, dict):
        return {k: _smart_to_torch(v, device) for k, v in x.items()}

    torch = importlib.import_module("torch")

    if is_torch_tensor(x):
        if hasattr(x, "to"):
            return x.to(device)
        return x

    x_np = np.asarray(x)
    if x_np.dtype == np.uint8:
        return torch.tensor(x_np, device=device, dtype=torch.float32).div_(255.0)
    dtype = torch.float32 if x_np.dtype == np.float64 else None
    return torch.as_tensor(x_np, device=device, dtype=dtype)


def _array_backend_flags(x: Any) -> tuple[bool, bool]:
    if is_torch_tensor(x):
        return True, False
    if isinstance(x, Mapping):
        has_torch = False
        has_numpy = False
        for value in x.values():
            child_torch, child_numpy = _array_backend_flags(value)
            has_torch = has_torch or child_torch
            has_numpy = has_numpy or child_numpy
        return has_torch, has_numpy
    if isinstance(x, (list, tuple, set)):
        has_torch = False
        has_numpy = False
        for value in x:
            child_torch, child_numpy = _array_backend_flags(value)
            has_torch = has_torch or child_torch
            has_numpy = has_numpy or child_numpy
        return has_torch, has_numpy
    if isinstance(x, np.ndarray):
        return False, True
    return False, False


def _is_torch_container(x: Any) -> bool:
    if is_torch_tensor(x):
        return True
    if isinstance(x, Mapping):
        has_torch, has_numpy = _array_backend_flags(x)
        return has_torch and not has_numpy
    return False


def _views_for_split(
    views: ViewsResult,
    *,
    split: str,
    sampling: SamplingResult,
    backend_ref: Any,
    strict: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    idx = np.asarray(sampling.indices[split], dtype=np.int64)
    split_ref = sampling.refs.get(split, "train")
    device = _first_torch_device(backend_ref)
    use_torch = device is not None or is_torch_tensor(backend_ref)

    for name, ds in views.views.items():
        base = ds.train if split_ref == "train" else ds.test
        if base is None:
            raise BenchRuntimeError(
                "E_BENCH_EVAL_SPLIT_INVALID",
                f"requested test split but view '{name}' has no test split",
            )
        X = select_rows(base.X, idx, context=f"evaluation.views[{name}].{split}")
        if use_torch:
            if strict and not _is_torch_container(X):
                raise BenchRuntimeError(
                    "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
                    f"view '{name}' is not torch-backed in benchmark_mode",
                )
            if not strict and device is not None:
                X = _smart_to_torch(X, device)
        out[name] = {"X": X}
    return out


def _first_torch_device(obj: Any) -> Any | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            dev = _first_torch_device(v)
            if dev is not None:
                return dev
        return None
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            dev = _first_torch_device(v)
            if dev is not None:
                return dev
        return None
    params = getattr(obj, "parameters", None)
    if callable(params):
        first_param = next(params(), None)
        if first_param is not None and hasattr(first_param, "device"):
            return first_param.device
    buffers = getattr(obj, "buffers", None)
    if callable(buffers):
        first_buffer = next(buffers(), None)
        if first_buffer is not None and hasattr(first_buffer, "device"):
            return first_buffer.device
    if is_torch_tensor(obj):
        return getattr(obj, "device", None)
    dev_attr = getattr(obj, "device", None)
    if isinstance(dev_attr, str):
        return dev_attr
    if dev_attr is not None and type(dev_attr).__name__ == "device":
        return dev_attr
    return None


def _infer_method_device(method: Any) -> Any | None:
    for attr in ("_svm", "_bundle", "_model", "_clf", "_clf1", "_clf2"):
        candidate = getattr(method, attr, None)
        dev = _first_torch_device(candidate)
        if dev is not None:
            return dev
        for nested in ("_model", "model", "_bundle", "_svm", "_clf", "_clf1", "_clf2"):
            dev = _first_torch_device(getattr(candidate, nested, None))
            if dev is not None:
                return dev
    return None


def evaluate_inductive(
    *,
    method: Any,
    pre: PreprocessResult,
    sampling: SamplingResult,
    report_splits: Iterable[str],
    metrics: Iterable[str],
    method_id: str,
    views: ViewsResult | None,
    strict: bool = False,
) -> dict[str, dict[str, float]]:
    start = perf_counter()
    _LOGGER.info(
        "Evaluation (inductive): splits=%s metrics=%s strict=%s",
        list(report_splits),
        list(metrics),
        bool(strict),
    )
    results: dict[str, dict[str, float]] = {}
    for split in report_splits:
        X, y = _split_data(pre, sampling, split=split)

        m_dev_spec = getattr(method, "device", None)
        dest_dev = getattr(m_dev_spec, "device", None) or (
            m_dev_spec if isinstance(m_dev_spec, str) else None
        )

        if dest_dev is None and getattr(method, "_backend", None) == "torch":
            dest_dev = _infer_method_device(method)

        if dest_dev is not None:
            if strict and not _is_torch_container(X):
                raise BenchRuntimeError(
                    "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
                    f"evaluation split '{split}' is not torch-backed in benchmark_mode",
                )
            if not strict:
                X = _smart_to_torch(X, dest_dev)

        if method_id == "co_training":
            if views is None:
                raise BenchRuntimeError(
                    "E_BENCH_EVAL_CONTRACT",
                    "co_training requires views for evaluation",
                )
            views_payload = _views_for_split(
                views,
                split=split,
                sampling=sampling,
                backend_ref=X,
                strict=strict,
            )
            data = InductiveDataset(X_l=X, y_l=y, views=views_payload)
            scores = method.predict_proba(data)
        else:
            scores = method.predict_proba(X)

        scores_np = to_numpy(scores)
        y_true = labels_1d(y)
        y_pred = predict_labels(scores_np)
        results[split] = compute_metrics(y_true, y_pred, metrics)

    _LOGGER.info("Evaluation (inductive) done: duration_s=%.3f", perf_counter() - start)
    return results


def evaluate_transductive(
    *,
    method: Any,
    data: Any,
    report_splits: Iterable[str],
    metrics: Iterable[str],
    masks: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    start = perf_counter()
    _LOGGER.info(
        "Evaluation (transductive): splits=%s metrics=%s", list(report_splits), list(metrics)
    )
    scores = method.predict_proba(data)
    scores_np = to_numpy(scores)
    y_true_raw = data.y
    meta = getattr(data, "meta", None)
    if isinstance(meta, Mapping) and "y_true" in meta:
        y_true_raw = meta["y_true"]
    y_true = labels_1d(y_true_raw)
    y_pred_all = predict_labels(scores_np)

    results: dict[str, dict[str, float]] = {}
    for split in report_splits:
        key = f"{split}_mask"
        if key not in masks:
            raise BenchRuntimeError("E_BENCH_EVAL_CONTRACT", f"missing mask for split '{split}'")
        mask = np.asarray(masks[key], dtype=bool)
        if mask.shape[0] != y_true.shape[0]:
            raise BenchRuntimeError(
                "E_BENCH_EVAL_CONTRACT",
                f"mask size mismatch for split '{split}': {mask.shape[0]} vs {y_true.shape[0]}",
            )
        results[split] = compute_metrics(y_true[mask], y_pred_all[mask], metrics)

    _LOGGER.info("Evaluation (transductive) done: duration_s=%.3f", perf_counter() - start)
    return results
