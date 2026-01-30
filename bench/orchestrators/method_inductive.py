from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_augmentation.utils import is_torch_tensor
from modssc.device import resolve_device_name
from modssc.inductive.deep import build_torch_bundle_from_classifier
from modssc.inductive.registry import get_method_class, get_method_info
from modssc.inductive.types import DeviceSpec, InductiveDataset
from modssc.preprocess.types import PreprocessResult
from modssc.sampling.result import SamplingResult
from modssc.views.types import ViewsResult

from ..schema import MethodConfig
from ..utils.import_tools import load_object

_LOGGER = logging.getLogger(__name__)


def _is_torch_container(x: Any) -> bool:
    if is_torch_tensor(x):
        return True
    if isinstance(x, dict):
        return any(is_torch_tensor(v) for v in x.values())
    return False


def _indices_for(X: Any, idx: np.ndarray):
    if _is_torch_container(X):
        import importlib

        torch = importlib.import_module("torch")
        device = X["x"].device if isinstance(X, dict) and "x" in X else getattr(X, "device", "cpu")
        return torch.as_tensor(idx, device=device, dtype=torch.long)
    return idx


def _select_rows(X: Any, idx: np.ndarray):
    if X is None:
        return None

    if isinstance(X, dict):
        out = {}
        # Special handling for Graph
        try:
            import importlib

            torch = importlib.import_module("torch")
            from torch_geometric.utils import subgraph

            if "edge_index" in X:
                ei = X["edge_index"]
                idx_t = torch.as_tensor(idx, device=ei.device, dtype=torch.long)
                sub_ei, _ = subgraph(idx_t, ei, relabel_nodes=True)
                out["edge_index"] = sub_ei
        except Exception:
            pass

        for k, v in X.items():
            if k == "edge_index":
                if "edge_index" not in out:  # fallback
                    out[k] = v
                continue

            if is_torch_tensor(v):
                import importlib

                torch = importlib.import_module("torch")
                # Heuristic: if dimension 0 looks like it matches the index space
                if v.ndim > 0 and v.shape[0] > idx.max():
                    out[k] = v[torch.as_tensor(idx, device=v.device, dtype=torch.long)]
                else:
                    out[k] = v
            elif isinstance(v, (np.ndarray, list)):
                v_arr = np.array(v)
                if v_arr.ndim > 0 and v_arr.shape[0] > idx.max():
                    out[k] = v_arr[idx]
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    if is_torch_tensor(X):
        import importlib

        torch = importlib.import_module("torch")
        return X[torch.as_tensor(idx, device=X.device, dtype=torch.long)]

    return X[idx]


def _labels_for_backend(pre: PreprocessResult, X_l: Any, idx: np.ndarray) -> Any:
    labels = pre.train_artifacts.get("labels.y") if pre.train_artifacts.has("labels.y") else None
    source = labels if labels is not None else pre.dataset.train.y
    idx_max = int(idx.max()) if idx.size else -1

    def _len0(obj: Any) -> int | None:
        try:
            return int(obj.shape[0])
        except Exception:
            return None

    # If cached labels are shorter than needed, fall back to dataset labels.
    if labels is not None and idx_max >= 0:
        src_len = _len0(labels)
        if src_len is not None and src_len <= idx_max:
            source = pre.dataset.train.y

    if is_torch_tensor(source):
        y_sub = source
        if y_sub is not None and getattr(y_sub, "ndim", 0) > 0 and y_sub.shape[0] > idx_max:
            y_sub = y_sub[_indices_for(y_sub, idx)]
        if _is_torch_container(X_l):
            if isinstance(X_l, dict) and "x" in X_l:
                device = getattr(X_l["x"], "device", "cpu")
            else:
                device = getattr(X_l, "device", "cpu")
            if hasattr(y_sub, "device") and y_sub.device != device:
                y_sub = y_sub.to(device)
        return y_sub

    y_arr = np.asarray(source)
    if y_arr.dtype == np.object_:
        y_arr = np.array([-1 if v is None else v for v in y_arr.tolist()], dtype=np.int64)
    else:
        try:
            y_arr = y_arr.astype(np.int64, copy=False)
        except Exception:
            y_arr = y_arr.astype(np.int64)

    if y_arr.ndim > 0 and y_arr.shape[0] > idx_max:
        y_arr = y_arr[idx]

    if _is_torch_container(X_l):
        import importlib

        torch = importlib.import_module("torch")
        if isinstance(X_l, dict) and "x" in X_l:
            device = getattr(X_l["x"], "device", "cpu")
        else:
            device = getattr(X_l, "device", "cpu")
        return torch.as_tensor(y_arr, device=device, dtype=torch.int64)
    return y_arr


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
    use_torch = _is_torch_container(ref)

    def _get_dev(obj):
        if isinstance(obj, dict) and "x" in obj:
            return obj["x"].device
        return getattr(obj, "device", None)

    device = _get_dev(ref) if use_torch else None

    out: dict[str, Any] = {}
    for name, ds in views.views.items():
        X_train = ds.train.X
        X_l = _select_rows(X_train, idx_l)
        X_u = _select_rows(X_train, idx_u)
        if use_torch:
            X_l = _smart_to_torch(X_l, device)
            X_u = _smart_to_torch(X_u, device)
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

    def _make_bundle(
        *,
        seed_offset: int,
        sample_override: Any | None = None,
        classifier_id: str | None = None,
        classifier_backend: str | None = None,
        classifier_params: dict[str, Any] | None = None,
        ema: bool | None = None,
    ) -> Any:
        has_override = any(
            value is not None
            for value in (classifier_id, classifier_backend, classifier_params, ema)
        )
        if getattr(model_cfg, "factory", None):
            if has_override:
                raise ValueError(
                    "method.model.factory cannot be combined with classifier overrides"
                )
            factory = load_object(model_cfg.factory)
            return factory(**dict(model_cfg.params))
        sample = sample_override if sample_override is not None else X_l
        if not is_torch_tensor(sample):
            # Lazy conversion for uint8 storage capability
            import importlib

            torch = importlib.import_module("torch")
            # Assuming NCHW layout for vision if not specified otherwise, or relying on bundle to handle it.
            # Convert to float32 for model compat
            if isinstance(sample, dict) and "x" in sample:
                sample = torch.as_tensor(sample["x"])
            else:
                sample = torch.as_tensor(sample)

            if sample.dtype == torch.uint8:
                sample = sample.to(dtype=torch.float32).div(255.0)

        num_classes = _infer_num_classes(y_l)
        local_classifier_id = classifier_id or model_cfg.classifier_id
        local_classifier_backend = classifier_backend or model_cfg.classifier_backend
        local_classifier_params = (
            model_cfg.classifier_params if classifier_params is None else classifier_params
        )
        if local_classifier_id is None:
            raise ValueError("classifier_id must be provided for torch model bundles")
        if ema is None:
            ema = model_cfg.ema
        if ema is None:
            ema = str(method_id) == "mean_teacher"
        return build_torch_bundle_from_classifier(
            classifier_id=local_classifier_id,
            classifier_backend=local_classifier_backend,
            classifier_params=local_classifier_params,
            sample=sample,
            num_classes=num_classes,
            seed=int(seed) + int(seed_offset),
            ema=bool(ema),
        )

    if hasattr(spec, "model_bundle"):
        return replace(spec, model_bundle=_make_bundle(seed_offset=0))
    if hasattr(spec, "teacher_bundle") and hasattr(spec, "student_bundle"):
        return replace(
            spec,
            student_bundle=_make_bundle(seed_offset=0),
            teacher_bundle=_make_bundle(seed_offset=1),
        )
    if hasattr(spec, "model_bundle_1") and hasattr(spec, "model_bundle_2"):
        return replace(
            spec,
            model_bundle_1=_make_bundle(seed_offset=0),
            model_bundle_2=_make_bundle(seed_offset=1),
        )
    if hasattr(spec, "pretrain_bundle") and hasattr(spec, "finetune_bundle"):
        return replace(
            spec,
            pretrain_bundle=_make_bundle(seed_offset=0),
            finetune_bundle=_make_bundle(seed_offset=1),
        )
    if hasattr(spec, "shared_bundle") and hasattr(spec, "head_bundles"):
        if getattr(model_cfg, "factory", None):
            raise ValueError("TriNet does not support model.factory; use classifier config.")
        shared_bundle = _make_bundle(seed_offset=0)
        if not is_torch_tensor(X_l):
            raise ValueError("TriNet requires torch.Tensor inputs (use core.to_torch).")
        sample = X_l[:1] if getattr(X_l, "ndim", 0) > 0 and int(X_l.shape[0]) > 1 else X_l
        output = shared_bundle.model(sample)
        head_sample = None
        if is_torch_tensor(output):
            head_sample = output.detach()
        elif isinstance(output, Mapping):
            for key in ("feat", "features", "embedding", "proj", "projection", "z", "logits"):
                candidate = output.get(key)
                if is_torch_tensor(candidate):
                    head_sample = candidate.detach()
                    break
        elif isinstance(output, tuple) and output and is_torch_tensor(output[0]):
            head_sample = output[0].detach()
        if head_sample is None:
            raise ValueError(
                "TriNet head construction requires shared model to return a torch.Tensor."
            )
        head_classifier_id = model_cfg.classifier_id
        if head_classifier_id not in {"mlp", "logreg"}:
            head_classifier_id = "logreg"
        head_bundles = tuple(
            _make_bundle(
                seed_offset=1 + idx,
                sample_override=head_sample,
                classifier_id=head_classifier_id,
                classifier_backend="torch",
                classifier_params=model_cfg.classifier_params,
                ema=False,
            )
            for idx in range(3)
        )
        return replace(
            spec,
            shared_bundle=shared_bundle,
            head_bundles=head_bundles,
        )
    raise ValueError("method.model is set but the method spec has no model_bundle field")


def _smart_to_torch(x: Any, device: Any) -> Any:
    """Converts numpy to torch, scaling uint8 to [0,1]."""
    if x is None:
        return None

    if isinstance(x, dict):
        return {k: _smart_to_torch(v, device) for k, v in x.items()}

    if is_torch_tensor(x):
        try:
            import importlib

            torch = importlib.import_module("torch")
            if isinstance(x, torch.Tensor) and device is not None and x.device != device:
                return x.to(device)
        except Exception:
            _LOGGER.debug(
                "Failed to move tensor to device %s in _smart_to_torch; returning original tensor.",
                device,
                exc_info=True,
            )
        return x

    import importlib

    torch = importlib.import_module("torch")
    x_np = np.asarray(x)

    if x_np.dtype == np.uint8:
        return torch.tensor(x_np, device=device, dtype=torch.float32).div_(255.0)

    dtype = torch.float32 if x_np.dtype == np.float64 else None
    return torch.as_tensor(x_np, device=device, dtype=dtype)


def run(
    pre: PreprocessResult,
    sampling: SamplingResult,
    *,
    views: ViewsResult | None,
    X_u_w: Any | None,
    X_u_s: Any | None,
    X_u_s_1: Any | None,
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

    # Check for multi-GPU availability
    try:
        import torch

        if torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
            if n_devices > 1:
                _LOGGER.info("Multi-GPU Environment: %d CUDA devices detected.", n_devices)
                for i in range(n_devices):
                    try:
                        name = torch.cuda.get_device_name(i)
                        _LOGGER.info("  Device %d: %s", i, name)
                    except Exception:
                        pass
                _LOGGER.info(
                    "Multi-GPU Use: NOTE - Unless specified, training defaults to 'cuda:0'."
                )
    except ImportError:
        pass

    _LOGGER.debug("Inductive method params: %s", dict(cfg.params))
    method_cls = get_method_class(cfg.method_id)
    _ = get_method_info(cfg.method_id)

    idx_l = np.asarray(sampling.indices["train_labeled"], dtype=np.int64)
    idx_u = np.asarray(sampling.indices["train_unlabeled"], dtype=np.int64)
    _LOGGER.info("Inductive method data: n_labeled=%s n_unlabeled=%s", idx_l.size, idx_u.size)

    X_train = pre.dataset.train.X
    X_l = _select_rows(X_train, idx_l)
    X_u = _select_rows(X_train, idx_u)

    # JIT Conversion to Torch if method requires it (inferred by missing to_torch in pre)
    if not _is_torch_container(X_l):
        target_device = resolve_device_name(cfg.device.device)
        X_l = _smart_to_torch(X_l, target_device)
        if X_u is not None:
            X_u = _smart_to_torch(X_u, target_device)
        if X_u_w is not None:
            X_u_w = _smart_to_torch(X_u_w, target_device)
        if X_u_s is not None:
            X_u_s = _smart_to_torch(X_u_s, target_device)
        if X_u_s_1 is not None:
            X_u_s_1 = _smart_to_torch(X_u_s_1, target_device)
    else:
        # If features are already torch, ensure augmented views are on the same device.
        if isinstance(X_l, dict) and "x" in X_l:
            target_device = getattr(X_l["x"], "device", "cpu")
        else:
            target_device = getattr(X_l, "device", "cpu")
        X_u_w = _smart_to_torch(X_u_w, target_device)
        X_u_s = _smart_to_torch(X_u_s, target_device)
        X_u_s_1 = _smart_to_torch(X_u_s_1, target_device)

    y_l = _labels_for_backend(pre, X_l, idx_l)

    if X_u_s_1 is not None and _is_torch_container(X_l) and not is_torch_tensor(X_u_s_1):
        if isinstance(X_l, dict) and "x" in X_l:
            target_device = getattr(X_l["x"], "device", "cpu")
        else:
            target_device = getattr(X_l, "device", "cpu")
        X_u_s_1 = _smart_to_torch(X_u_s_1, target_device)

    views_payload = _build_views(views, idx_l=idx_l, idx_u=idx_u, ref=X_l) if views else None
    if X_u_s_1 is not None:
        if cfg.method_id == "comatch":
            if views_payload:
                _LOGGER.debug("CoMatch ignores view plan to attach X_u_s_1.")
            views_payload = {"X_u_s_1": X_u_s_1}
        elif views_payload is None:
            views_payload = {"X_u_s_1": X_u_s_1}
        else:
            _LOGGER.debug("Skipping X_u_s_1 injection because views payload is non-empty.")

    meta: dict[str, Any] = {
        "dataset_fingerprint": pre.dataset.meta.get("dataset_fingerprint"),
        "split_fingerprint": sampling.split_fingerprint,
        "preprocess_fingerprint": pre.preprocess_fingerprint,
    }
    if X_u is not None:
        meta["idx_u"] = _indices_for(X_u, idx_u)
        if isinstance(X_train, dict) and "x" in X_train:
            meta["ulb_size"] = int(X_train["x"].shape[0])
        else:
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
