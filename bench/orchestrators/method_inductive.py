from __future__ import annotations

import importlib
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

from ..errors import BenchRuntimeError
from ..schema import MethodConfig
from ..utils.import_tools import load_object
from .slicing import select_rows

_LOGGER = logging.getLogger(__name__)


def _torch_module() -> Any:
    return importlib.import_module("torch")


def _array_backend_flags(x: Any) -> tuple[bool, bool]:
    if is_torch_tensor(x):
        return True, False
    if isinstance(x, dict):
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
            if not child_torch and not child_numpy:
                has_numpy = True
        return has_torch, has_numpy
    if isinstance(x, np.ndarray):
        return False, True
    return False, False


def _is_torch_container(x: Any) -> bool:
    if is_torch_tensor(x):
        return True
    if isinstance(x, dict):
        has_torch, has_numpy = _array_backend_flags(x)
        return has_torch and not has_numpy
    return False


def _torch_container_device(x: Any) -> Any:
    if is_torch_tensor(x):
        return x.device
    if isinstance(x, dict):
        if "x" in x and is_torch_tensor(x["x"]):
            return x["x"].device
        for value in x.values():
            dev = _torch_container_device(value)
            if dev is not None:
                return dev
    if isinstance(x, (list, tuple, set)):
        for value in x:
            dev = _torch_container_device(value)
            if dev is not None:
                return dev
    return None


def _feature_tensor(x: Any) -> Any | None:
    if x is None:
        return None
    if is_torch_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Mapping) and "x" in x:
        return x.get("x")
    return None


def _len0(obj: Any) -> int | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping) and "x" in obj:
        obj = obj.get("x")
    shape = getattr(obj, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def _indices_for(X: Any, idx: np.ndarray):
    if _is_torch_container(X):
        torch = _torch_module()
        device = _torch_container_device(X) or "cpu"
        return torch.as_tensor(idx, device=device, dtype=torch.long)
    return idx


def _labels_for_backend(pre: PreprocessResult, X_l: Any, idx: np.ndarray, *, strict: bool) -> Any:
    labels = pre.train_artifacts.get("labels.y") if pre.train_artifacts.has("labels.y") else None
    source = labels if labels is not None else pre.dataset.train.y
    idx_max = int(idx.max()) if idx.size else -1

    if labels is not None and idx_max >= 0:
        src_len = _len0(labels)
        if src_len is not None and src_len <= idx_max:
            if strict:
                raise BenchRuntimeError(
                    "E_BENCH_LABELS_CONTRACT",
                    "preprocess labels.y is shorter than required labeled indices; no strict fallback",
                )
            source = pre.dataset.train.y

    if is_torch_tensor(source):
        y_sub = source
        if y_sub is not None and getattr(y_sub, "ndim", 0) > 0 and y_sub.shape[0] > idx_max:
            y_sub = y_sub[_indices_for(y_sub, idx)]
        if _is_torch_container(X_l):
            device = _torch_container_device(X_l) or "cpu"
            if hasattr(y_sub, "device") and y_sub.device != device:
                y_sub = y_sub.to(device)
        return y_sub

    y_arr = np.asarray(source)
    if y_arr.dtype == np.object_:
        y_arr = np.array([-1 if v is None else v for v in y_arr.tolist()], dtype=np.int64)
    else:
        y_arr = y_arr.astype(np.int64, copy=False)

    if y_arr.ndim > 0 and y_arr.shape[0] > idx_max:
        y_arr = y_arr[idx]

    if _is_torch_container(X_l):
        torch = _torch_module()
        device = _torch_container_device(X_l) or "cpu"
        return torch.as_tensor(y_arr, device=device, dtype=torch.int64)
    return y_arr


def _smart_to_torch(x: Any, device: Any) -> Any:
    if x is None:
        return None

    if isinstance(x, dict):
        return {k: _smart_to_torch(v, device) for k, v in x.items()}

    if is_torch_tensor(x):
        if device is not None and getattr(x, "device", None) != device and hasattr(x, "to"):
            return x.to(device)
        return x

    torch = _torch_module()
    x_np = np.asarray(x)
    return torch.as_tensor(x_np, device=device)


def _validate_strict_tensor_contract(name: str, value: Any) -> None:
    tensor = _feature_tensor(value)
    if tensor is None:
        return
    ndim = getattr(tensor, "ndim", None)
    if isinstance(ndim, int) and ndim < 2:
        raise BenchRuntimeError(
            "E_BENCH_SHAPE_CONTRACT",
            f"{name} must be at least 2D, got ndim={ndim}",
        )
    if is_torch_tensor(tensor):
        torch = _torch_module()
        if not torch.is_floating_point(tensor):
            raise BenchRuntimeError(
                "E_BENCH_DTYPE_CONTRACT",
                f"{name} must be floating torch tensor in benchmark_mode; got dtype={tensor.dtype}",
            )


def _validate_strict_inputs(
    *,
    X_l: Any,
    y_l: Any,
    X_u: Any | None,
    X_u_w: Any | None,
    X_u_s: Any | None,
    X_u_s_1: Any | None,
    requires_torch: bool,
) -> None:
    for name, value in (
        ("X_l", X_l),
        ("X_u", X_u),
        ("X_u_w", X_u_w),
        ("X_u_s", X_u_s),
        ("X_u_s_1", X_u_s_1),
    ):
        if value is None:
            continue
        if requires_torch and not _is_torch_container(value):
            raise BenchRuntimeError(
                "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
                f"{name} must be torch-backed in benchmark_mode (declare conversion in preprocess)",
            )
        _validate_strict_tensor_contract(name, value)

    n_x = _len0(X_l)
    n_y = _len0(y_l)
    if n_x is not None and n_y is not None and int(n_x) != int(n_y):
        raise BenchRuntimeError(
            "E_BENCH_SHAPE_CONTRACT",
            f"X_l/y_l row mismatch: X_l={n_x} y_l={n_y}",
        )


def _build_views(
    views: ViewsResult,
    *,
    idx_l: np.ndarray,
    idx_u: np.ndarray,
    ref: Any,
    strict: bool,
) -> dict[str, Any]:
    use_torch = _is_torch_container(ref)
    device = _torch_container_device(ref) if use_torch else None

    out: dict[str, Any] = {}
    for name, ds in views.views.items():
        X_train = ds.train.X
        X_l = select_rows(X_train, idx_l, context=f"method_inductive.views[{name}].labeled")
        X_u = select_rows(X_train, idx_u, context=f"method_inductive.views[{name}].unlabeled")
        if use_torch:
            if strict and (not _is_torch_container(X_l) or not _is_torch_container(X_u)):
                raise BenchRuntimeError(
                    "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
                    f"view '{name}' must be torch-backed in benchmark_mode",
                )
            if not strict:
                X_l = _smart_to_torch(X_l, device)
                X_u = _smart_to_torch(X_u, device)
        out[name] = {"X_l": X_l, "X_u": X_u}
    return out


def _default_spec(method_cls: type[Any], *, strict: bool) -> Any:
    try:
        inst = method_cls()
    except (TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError) as exc:
        if strict:
            raise BenchRuntimeError(
                "E_BENCH_METHOD_INTROSPECTION",
                f"failed to instantiate method for spec introspection: {exc}",
            ) from exc
        return None
    spec = getattr(inst, "spec", None)
    if spec is not None and is_dataclass(spec):
        return spec
    return None


def _build_spec(method_cls: type[Any], params: dict[str, Any], *, strict: bool) -> Any:
    if not params:
        return None
    spec = _default_spec(method_cls, strict=strict)
    if spec is not None and is_dataclass(spec):
        return replace(spec, **params)
    raise BenchRuntimeError(
        "E_BENCH_METHOD_SPEC",
        "method.params were provided but no dataclass spec is available",
    )


def _infer_num_classes(y: Any) -> int:
    if is_torch_tensor(y):
        torch = _torch_module()
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
    strict: bool,
) -> Any:
    if model_cfg is None:
        return spec
    if spec is None:
        raise BenchRuntimeError(
            "E_BENCH_METHOD_SPEC",
            "method.model is set but no dataclass spec is available",
        )

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
                raise BenchRuntimeError(
                    "E_BENCH_MODEL_CONFIG",
                    "method.model.factory cannot be combined with classifier overrides",
                )
            factory = load_object(model_cfg.factory)
            return factory(**dict(model_cfg.params))

        sample = sample_override if sample_override is not None else X_l
        if isinstance(sample, Mapping) and "x" in sample:
            sample = sample["x"]

        if not is_torch_tensor(sample):
            if strict:
                raise BenchRuntimeError(
                    "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
                    "model bundle requires torch sample; declare conversion in preprocess",
                )
            sample = _smart_to_torch(sample, device=None)

        if strict:
            torch = _torch_module()
            if not torch.is_floating_point(sample):
                raise BenchRuntimeError(
                    "E_BENCH_DTYPE_CONTRACT",
                    f"model bundle sample must be floating tensor in benchmark_mode; got {sample.dtype}",
                )

        num_classes = _infer_num_classes(y_l)
        local_classifier_id = classifier_id or model_cfg.classifier_id
        local_classifier_backend = classifier_backend or model_cfg.classifier_backend
        local_classifier_params = (
            model_cfg.classifier_params if classifier_params is None else classifier_params
        )
        if local_classifier_id is None:
            raise BenchRuntimeError(
                "E_BENCH_MODEL_CONFIG",
                "classifier_id must be provided for torch model bundles",
            )
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
            raise BenchRuntimeError(
                "E_BENCH_MODEL_CONFIG",
                "TriNet does not support model.factory; use classifier config",
            )
        shared_bundle = _make_bundle(seed_offset=0)
        if not is_torch_tensor(X_l) and not (isinstance(X_l, Mapping) and "x" in X_l):
            raise BenchRuntimeError(
                "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
                "TriNet requires torch.Tensor inputs (use core.to_torch)",
            )

        if isinstance(X_l, Mapping):
            sample = select_rows(
                X_l,
                np.array([0], dtype=np.int64),
                context="method_inductive.trinet_sample",
            )
        else:
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
            raise BenchRuntimeError(
                "E_BENCH_MODEL_CONFIG",
                "TriNet head construction requires shared model to return a torch.Tensor",
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
        return replace(spec, shared_bundle=shared_bundle, head_bundles=head_bundles)

    raise BenchRuntimeError(
        "E_BENCH_MODEL_CONFIG",
        "method.model is set but method spec has no model bundle field",
    )


def _dtype_descriptor(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = value
    if isinstance(value, Mapping) and "x" in value:
        payload = value["x"]
    dtype = getattr(payload, "dtype", None)
    shape = getattr(payload, "shape", None)
    out: dict[str, Any] = {}
    if dtype is not None:
        out["dtype"] = str(dtype)
    if shape is not None:
        try:
            out["shape"] = list(shape)
        except TypeError:
            out["shape"] = None
    return out or None


def _resolve_method_backend(cfg: MethodConfig, method: Any, spec: Any) -> str | None:
    backend = cfg.params.get("backend")
    if backend is None and spec is not None and hasattr(spec, "backend"):
        backend = spec.backend
    if backend is None:
        backend = getattr(method, "_backend", None)
    return str(backend) if backend is not None else None


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
    strict: bool = False,
    requires_torch: bool = False,
) -> tuple[Any, dict[str, Any]]:
    start = perf_counter()
    if sampling.is_graph():
        raise BenchRuntimeError(
            "E_BENCH_GRAPH_SAMPLING_INVALID",
            "inductive method cannot run on graph sampling result",
        )

    model_ref = None
    if cfg.model is not None:
        model_ref = cfg.model.factory or cfg.model.classifier_id
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
    method_cls = get_method_class(cfg.method_id)
    _ = get_method_info(cfg.method_id)
    requested_backend = cfg.params.get("backend")
    if strict and isinstance(requested_backend, str) and requested_backend.lower() == "auto":
        raise BenchRuntimeError(
            "E_BENCH_AUTO_FORBIDDEN",
            "method.params.backend='auto' is forbidden when run.benchmark_mode=true",
        )
    if isinstance(requested_backend, str) and requested_backend.lower() == "torch":
        try:
            importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise BenchRuntimeError(
                "E_BENCH_DEPENDENCY_MISSING",
                "method backend 'torch' requires dependency 'torch'",
            ) from exc

    idx_l = np.asarray(sampling.indices["train_labeled"], dtype=np.int64)
    idx_u = np.asarray(sampling.indices["train_unlabeled"], dtype=np.int64)
    _LOGGER.info("Inductive method data: n_labeled=%s n_unlabeled=%s", idx_l.size, idx_u.size)

    X_train = pre.dataset.train.X
    X_l = select_rows(X_train, idx_l, context="method_inductive.train_labeled")
    X_u = select_rows(X_train, idx_u, context="method_inductive.train_unlabeled")

    if not strict:
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
            target_device = _torch_container_device(X_l) or "cpu"
            X_u_w = _smart_to_torch(X_u_w, target_device)
            X_u_s = _smart_to_torch(X_u_s, target_device)
            X_u_s_1 = _smart_to_torch(X_u_s_1, target_device)

    y_l = _labels_for_backend(pre, X_l, idx_l, strict=strict)

    if strict:
        _validate_strict_inputs(
            X_l=X_l,
            y_l=y_l,
            X_u=X_u,
            X_u_w=X_u_w,
            X_u_s=X_u_s,
            X_u_s_1=X_u_s_1,
            requires_torch=requires_torch,
        )

    views_payload = (
        _build_views(views, idx_l=idx_l, idx_u=idx_u, ref=X_l, strict=strict) if views else None
    )
    if X_u_s_1 is not None:
        if cfg.method_id == "comatch":
            if views_payload:
                _LOGGER.debug("CoMatch ignores view plan to attach X_u_s_1")
            views_payload = {"X_u_s_1": X_u_s_1}
        elif views_payload is None:
            views_payload = {"X_u_s_1": X_u_s_1}
        else:
            _LOGGER.debug("Skipping X_u_s_1 injection because views payload is non-empty")

    meta: dict[str, Any] = {
        "dataset_fingerprint": pre.dataset.meta.get("dataset_fingerprint"),
        "split_fingerprint": sampling.split_fingerprint,
        "preprocess_fingerprint": pre.preprocess_fingerprint,
    }
    if X_u is not None:
        meta["idx_u"] = _indices_for(X_u, idx_u)
        if isinstance(X_train, Mapping) and "x" in X_train:
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

    spec = _build_spec(method_cls, cfg.params, strict=strict) if cfg.params else None
    if spec is None and cfg.model is not None:
        spec = _default_spec(method_cls, strict=strict)
    spec = _inject_model_bundle(
        spec,
        cfg.model,
        X_l=X_l,
        y_l=y_l,
        method_id=cfg.method_id,
        seed=seed,
        strict=strict,
    )

    method = method_cls(spec) if spec is not None else method_cls()
    device = DeviceSpec(device=cfg.device.device, dtype=cfg.device.dtype)
    method.fit(data, device=device, seed=int(seed))

    method_resolution = {
        "backend": _resolve_method_backend(cfg, method, spec),
        "classifier_backend": (
            cfg.model.classifier_backend
            if cfg.model is not None
            else getattr(method, "_classifier_backend", None)
        ),
        "dtypes": {
            "X_l": _dtype_descriptor(X_l),
            "y_l": _dtype_descriptor(y_l),
            "X_u": _dtype_descriptor(X_u),
            "X_u_w": _dtype_descriptor(X_u_w),
            "X_u_s": _dtype_descriptor(X_u_s),
            "X_u_s_1": _dtype_descriptor(X_u_s_1),
        },
        "normalization": {
            "implicit_method_conversion": False,
            "strict_contract_validated": bool(strict),
        },
    }

    _LOGGER.info(
        "Inductive method done: id=%s duration_s=%.3f", cfg.method_id, perf_counter() - start
    )
    return method, method_resolution
