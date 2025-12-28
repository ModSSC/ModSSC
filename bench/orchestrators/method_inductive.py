from __future__ import annotations

import logging
from dataclasses import is_dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_augmentation.utils import is_torch_tensor
from modssc.inductive.deep import build_torch_bundle_from_classifier
from modssc.inductive.registry import get_method_class, get_method_info
from modssc.inductive.types import DeviceSpec, InductiveDataset
from modssc.preprocess.types import PreprocessResult
from modssc.sampling.result import SamplingResult
from modssc.views.types import ViewsResult

from ..schema import MethodConfig
from ..utils.import_tools import load_object

_LOGGER = logging.getLogger(__name__)


def _indices_for(X: Any, idx: np.ndarray):
    if is_torch_tensor(X):
        import importlib

        torch = importlib.import_module("torch")
        return torch.as_tensor(idx, device=X.device, dtype=torch.long)
    return idx


def _select_rows(X: Any, idx: np.ndarray):
    if X is None:
        return None
    return X[_indices_for(X, idx)]


def _labels_for_backend(pre: PreprocessResult, X_l: Any, idx: np.ndarray) -> Any:
    labels = None
    if pre.train_artifacts.has("labels.y"):
        labels = pre.train_artifacts.get("labels.y")

    if is_torch_tensor(X_l):
        if labels is not None and is_torch_tensor(labels):
            return labels[_indices_for(labels, idx)]
        import importlib

        torch = importlib.import_module("torch")
        y = np.asarray(pre.dataset.train.y)
        return torch.as_tensor(y[idx], device=X_l.device, dtype=torch.int64)

    if labels is not None and not is_torch_tensor(labels):
        return np.asarray(labels)[idx]

    y = np.asarray(pre.dataset.train.y)
    return y[idx]


def _to_torch_like(x: Any, ref: Any) -> Any:
    if is_torch_tensor(x):
        return x
    import importlib

    torch = importlib.import_module("torch")
    return torch.as_tensor(np.asarray(x), device=ref.device, dtype=ref.dtype)


def _build_views(
    views: ViewsResult,
    *,
    idx_l: np.ndarray,
    idx_u: np.ndarray,
    ref: Any,
) -> dict[str, Any]:
    use_torch = is_torch_tensor(ref)
    out: dict[str, Any] = {}
    for name, ds in views.views.items():
        X_train = ds.train.X
        X_l = _select_rows(X_train, idx_l)
        X_u = _select_rows(X_train, idx_u)
        if use_torch:
            X_l = _to_torch_like(X_l, ref)
            X_u = _to_torch_like(X_u, ref)
        out[name] = {"X_l": X_l, "X_u": X_u}
    return out


def _build_spec(method_cls: type[Any], params: dict[str, Any]) -> Any:
    if not params:
        return None
    spec = None
    try:
        inst = method_cls()
        spec = getattr(inst, "spec", None)
    except Exception:
        spec = None

    if spec is not None and is_dataclass(spec):
        return replace(spec, **params)

    raise ValueError("Method params provided but no dataclass spec is available")


def _default_spec(method_cls: type[Any]) -> Any:
    try:
        inst = method_cls()
    except Exception:
        return None
    spec = getattr(inst, "spec", None)
    if spec is not None and is_dataclass(spec):
        return spec
    return None


def _infer_num_classes(y: Any) -> int:
    if is_torch_tensor(y):
        import importlib

        torch = importlib.import_module("torch")
        return int(torch.unique(y).numel())
    y_arr = np.asarray(y)
    return int(np.unique(y_arr).size)


def _inject_model_bundle(
    spec: Any,
    model_cfg: Any,
    *,
    X_l: Any,
    y_l: Any,
    method_id: str,
    seed: int,
) -> Any:
    if model_cfg is None:
        return spec
    if spec is None:
        raise ValueError("method.model is set but no dataclass spec is available")
    if not hasattr(spec, "model_bundle"):
        raise ValueError("method.model is set but the method spec has no model_bundle field")
    if getattr(model_cfg, "factory", None):
        factory = load_object(model_cfg.factory)
        bundle = factory(**dict(model_cfg.params))
        return replace(spec, model_bundle=bundle)

    if not is_torch_tensor(X_l):
        raise ValueError("Torch model bundle requires torch.Tensor inputs (use core.to_torch).")
    num_classes = _infer_num_classes(y_l)
    ema = model_cfg.ema
    if ema is None:
        ema = str(method_id) == "mean_teacher"
    bundle = build_torch_bundle_from_classifier(
        classifier_id=model_cfg.classifier_id,
        classifier_backend=model_cfg.classifier_backend,
        classifier_params=model_cfg.classifier_params,
        sample=X_l,
        num_classes=num_classes,
        seed=int(seed),
        ema=bool(ema),
    )
    return replace(spec, model_bundle=bundle)


def run(
    pre: PreprocessResult,
    sampling: SamplingResult,
    *,
    views: ViewsResult | None,
    X_u_w: Any | None,
    X_u_s: Any | None,
    cfg: MethodConfig,
    seed: int,
) -> Any:
    start = perf_counter()
    if sampling.is_graph():
        raise ValueError("Inductive method cannot run on graph sampling result")

    model_ref = None
    if cfg.model is not None:
        model_ref = cfg.model.factory or cfg.model.classifier_id
    _LOGGER.info(
        "Inductive method start: id=%s seed=%s device=%s dtype=%s model=%s",
        cfg.method_id,
        int(seed),
        cfg.device.device,
        cfg.device.dtype,
        model_ref,
    )
    _LOGGER.debug("Inductive method params: %s", dict(cfg.params))
    method_cls = get_method_class(cfg.method_id)
    _ = get_method_info(cfg.method_id)

    idx_l = np.asarray(sampling.indices["train_labeled"], dtype=np.int64)
    idx_u = np.asarray(sampling.indices["train_unlabeled"], dtype=np.int64)
    _LOGGER.info("Inductive method data: n_labeled=%s n_unlabeled=%s", idx_l.size, idx_u.size)

    X_train = pre.dataset.train.X
    X_l = _select_rows(X_train, idx_l)
    X_u = _select_rows(X_train, idx_u)
    y_l = _labels_for_backend(pre, X_l, idx_l)

    views_payload = _build_views(views, idx_l=idx_l, idx_u=idx_u, ref=X_l) if views else None

    meta: dict[str, Any] = {
        "dataset_fingerprint": pre.dataset.meta.get("dataset_fingerprint"),
        "split_fingerprint": sampling.split_fingerprint,
        "preprocess_fingerprint": pre.preprocess_fingerprint,
    }
    if X_u is not None:
        meta["idx_u"] = _indices_for(X_u, idx_u)
        meta["ulb_size"] = int(X_train.shape[0])

    data = InductiveDataset(
        X_l=X_l,
        y_l=y_l,
        X_u=X_u,
        X_u_w=X_u_w,
        X_u_s=X_u_s,
        views=views_payload,
        meta=meta,
    )

    spec = _build_spec(method_cls, cfg.params) if cfg.params else None
    if spec is None and cfg.model is not None:
        spec = _default_spec(method_cls)
    spec = _inject_model_bundle(
        spec, cfg.model, X_l=X_l, y_l=y_l, method_id=cfg.method_id, seed=seed
    )

    method = method_cls(spec) if spec is not None else method_cls()
    device = DeviceSpec(device=cfg.device.device, dtype=cfg.device.dtype)
    method.fit(data, device=device, seed=int(seed))
    _LOGGER.info(
        "Inductive method done: id=%s duration_s=%.3f", cfg.method_id, perf_counter() - start
    )
    return method
