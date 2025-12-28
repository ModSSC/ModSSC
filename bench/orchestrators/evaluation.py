from __future__ import annotations

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

_LOGGER = logging.getLogger(__name__)


def _split_data(
    pre: PreprocessResult,
    sampling: SamplingResult,
    *,
    split: str,
) -> tuple[Any, Any]:
    if sampling.is_graph():
        raise ValueError("Inductive evaluation does not support graph sampling")

    idx = np.asarray(sampling.indices[split], dtype=np.int64)
    ref = sampling.refs.get(split, "train")

    if ref == "train":
        base = pre.dataset.train
    else:
        if pre.dataset.test is None:
            raise ValueError("Requested test split but dataset has no test")
        base = pre.dataset.test

    X = base.X
    y = _labels_for_split(pre, ref, base)
    return _select_rows(X, idx), _select_rows(y, idx)


def _labels_for_split(pre: PreprocessResult, ref: str, base: Any) -> Any:
    store = pre.train_artifacts if ref == "train" else pre.test_artifacts
    if store is not None and store.has("labels.y"):
        return store.get("labels.y")
    return base.y


def _select_rows(X: Any, idx: np.ndarray) -> Any:
    if is_torch_tensor(X):
        import importlib

        torch = importlib.import_module("torch")
        return X[torch.as_tensor(idx, device=X.device, dtype=torch.long)]
    return X[idx]


def _to_torch_like(x: Any, ref: Any) -> Any:
    if is_torch_tensor(x):
        return x
    import importlib

    torch = importlib.import_module("torch")
    return torch.as_tensor(np.asarray(x), device=ref.device, dtype=ref.dtype)


def _views_for_split(
    views: ViewsResult,
    *,
    split: str,
    sampling: SamplingResult,
    backend_ref: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    idx = np.asarray(sampling.indices[split], dtype=np.int64)
    split_ref = sampling.refs.get(split, "train")
    use_torch = is_torch_tensor(backend_ref)
    for name, ds in views.views.items():
        base = ds.train if split_ref == "train" else ds.test
        if base is None:
            raise ValueError("Requested test split but view has no test split")
        X = _select_rows(base.X, idx)
        if use_torch:
            X = _to_torch_like(X, backend_ref)
        out[name] = {"X": X}
    return out


def evaluate_inductive(
    *,
    method: Any,
    pre: PreprocessResult,
    sampling: SamplingResult,
    report_splits: Iterable[str],
    metrics: Iterable[str],
    method_id: str,
    views: ViewsResult | None,
) -> dict[str, dict[str, float]]:
    start = perf_counter()
    _LOGGER.info("Evaluation (inductive): splits=%s metrics=%s", list(report_splits), list(metrics))
    results: dict[str, dict[str, float]] = {}
    for split in report_splits:
        X, y = _split_data(pre, sampling, split=split)
        if method_id == "co_training":
            if views is None:
                raise ValueError("co_training requires views for evaluation")
            views_payload = _views_for_split(views, split=split, sampling=sampling, backend_ref=X)
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
            raise ValueError(f"Missing mask for split {split!r}")
        mask = masks[key]
        results[split] = compute_metrics(y_true[mask], y_pred_all[mask], metrics)
    _LOGGER.info("Evaluation (transductive) done: duration_s=%.3f", perf_counter() - start)
    return results
