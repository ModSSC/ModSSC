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
    if isinstance(X, dict):
        new_X = {}
        # Special handling for Graph data
        if "edge_index" in X:
            try:
                import torch
                from torch_geometric.utils import subgraph
            except ImportError:
                pass
            else:
                # Align subset device with edge_index to avoid mismatch in subgraph
                edge_index_in = X["edge_index"]
                device = edge_index_in.device if hasattr(edge_index_in, "device") else "cpu"

                subset = torch.as_tensor(idx, dtype=torch.long, device=device)

                # relabel_nodes=True ensures indices map to 0..len(subset)-1
                # which corresponds to the sliced feature matrices
                edge_index, _ = subgraph(subset, edge_index_in, relabel_nodes=True)
                new_X["edge_index"] = edge_index

        for k, v in X.items():
            if k == "edge_index":
                continue
            new_X[k] = _select_rows(v, idx)
        return new_X

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
    device = _first_torch_device(backend_ref)
    use_torch = device is not None or is_torch_tensor(backend_ref)

    for name, ds in views.views.items():
        base = ds.train if split_ref == "train" else ds.test
        if base is None:
            raise ValueError("Requested test split but view has no test split")
        X = _select_rows(base.X, idx)
        if use_torch and device is not None:
            X = _smart_to_torch(X, device)
        out[name] = {"X": X}
    return out


def _smart_to_torch(x: Any, device: Any) -> Any:
    """Converts numpy to torch, scaling uint8 to [0,1]."""
    if x is None:
        return None

    if isinstance(x, dict):
        return {k: _smart_to_torch(v, device) for k, v in x.items()}

    import importlib

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
    if is_torch_tensor(obj):
        return getattr(obj, "device", None)
    return None


def _infer_method_device(method: Any) -> Any | None:
    for attr in ("_svm", "_bundle", "_model", "_clf1", "_clf2"):
        dev = _first_torch_device(getattr(method, attr, None))
        if dev is not None:
            return dev
    for val in getattr(method, "__dict__", {}).values():
        dev = _first_torch_device(val)
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
) -> dict[str, dict[str, float]]:
    start = perf_counter()
    _LOGGER.info("Evaluation (inductive): splits=%s metrics=%s", list(report_splits), list(metrics))
    results: dict[str, dict[str, float]] = {}
    for split in report_splits:
        X, y = _split_data(pre, sampling, split=split)

        # JIT Convert if method has device spec (indicates torch/deep method)
        m_dev_spec = getattr(method, "device", None)
        dest_dev = getattr(m_dev_spec, "device", None) or (
            m_dev_spec if isinstance(m_dev_spec, str) else None
        )

        # Fallback inspection for torch models if device attr missing
        if dest_dev is None and hasattr(method, "_bundle") and method._bundle is not None:
            try:
                mdl = method._bundle.model
                import torch

                if isinstance(mdl, torch.nn.Module):
                    p = next(mdl.parameters(), None)
                    if p is not None:
                        dest_dev = p.device
            except Exception as e:
                _LOGGER.debug("Failed to infer device from model: %s", e)

        if dest_dev is None and getattr(method, "_backend", None) == "torch":
            dest_dev = _infer_method_device(method)

        if dest_dev is not None:
            X = _smart_to_torch(X, dest_dev)

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
