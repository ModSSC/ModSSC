from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from modssc.device import resolve_device_name
from modssc.graph.artifacts import GraphArtifact, NodeDataset
from modssc.transductive.registry import get_method_class, get_method_info

from ..errors import BenchRuntimeError
from ..schema import MethodConfig

_LOGGER = logging.getLogger(__name__)


def _resolve_method_device(device: str | None, *, supports_gpu: bool, strict: bool) -> str | None:
    if device is None:
        return None
    if strict and device == "auto":
        raise BenchRuntimeError(
            "E_BENCH_AUTO_FORBIDDEN",
            "method.device.device='auto' is forbidden when run.benchmark_mode=true",
        )
    if device != "auto":
        return device
    if not supports_gpu:
        return "cpu"
    return resolve_device_name(device)


def _default_spec(method_cls: type[Any], *, strict: bool) -> Any:
    try:
        inst = method_cls()
    except (TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError) as exc:
        if strict:
            raise BenchRuntimeError(
                "E_BENCH_METHOD_INTROSPECTION",
                f"failed to instantiate transductive method for spec introspection: {exc}",
            ) from exc
        return None
    spec = getattr(inst, "spec", None)
    if spec is not None and is_dataclass(spec):
        return spec
    return None


def _build_spec(method_cls: type[Any], params: dict[str, Any], *, strict: bool) -> Any:
    spec = _default_spec(method_cls, strict=strict)
    if not params:
        return spec
    if spec is not None and is_dataclass(spec):
        return replace(spec, **params)
    raise BenchRuntimeError(
        "E_BENCH_METHOD_SPEC",
        "method.params were provided but no dataclass spec is available",
    )


def _resolve_method_backend(*, cfg: MethodConfig, spec: Any, strict: bool) -> str | None:
    backend = cfg.params.get("backend")
    if backend is None and spec is not None and hasattr(spec, "backend"):
        backend = spec.backend
    if backend is None:
        return None
    backend_s = str(backend)
    if strict and backend_s.lower() == "auto":
        raise BenchRuntimeError(
            "E_BENCH_AUTO_FORBIDDEN",
            "method backend 'auto' is forbidden when run.benchmark_mode=true",
        )
    return backend_s


def _ensure_backend_dependencies(backend: str | None) -> None:
    if backend is None:
        return
    if backend.lower() == "torch":
        try:
            importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise BenchRuntimeError(
                "E_BENCH_DEPENDENCY_MISSING",
                "method backend 'torch' requires dependency 'torch'",
            ) from exc


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _combine_splits(train: Any, test: Any | None) -> np.ndarray:
    if test is None:
        return _to_numpy(train)
    return np.concatenate([_to_numpy(train), _to_numpy(test)], axis=0)


def _mask_from_indices(n: int, idx: np.ndarray, *, offset: int = 0) -> np.ndarray:
    m = np.zeros((n,), dtype=bool)
    if idx.size == 0:
        return m
    m[idx + offset] = True
    return m


def _build_masks_from_indices(
    *,
    n_train: int,
    n_test: int | None,
    indices: Mapping[str, np.ndarray],
    refs: Mapping[str, str],
) -> dict[str, np.ndarray]:
    n_total = n_train + (n_test or 0)

    def _offset(name: str) -> int:
        ref = refs.get(name, "train")
        return 0 if ref == "train" else n_train

    train_mask = _mask_from_indices(n_total, indices["train"], offset=_offset("train"))
    val_mask = _mask_from_indices(n_total, indices["val"], offset=_offset("val"))
    test_mask = _mask_from_indices(n_total, indices["test"], offset=_offset("test"))

    labeled_mask = _mask_from_indices(
        n_total, indices["train_labeled"], offset=_offset("train_labeled")
    )
    unlabeled_mask = _mask_from_indices(
        n_total, indices["train_unlabeled"], offset=_offset("train_unlabeled")
    )

    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "unlabeled_mask": unlabeled_mask,
        "labeled_mask": labeled_mask,
    }


def graph_from_dataset(dataset: Any, n_nodes: int) -> GraphArtifact:
    edges = dataset.train.edges
    if isinstance(edges, GraphArtifact):
        return edges

    edge_weight = None
    edge_index = None
    if isinstance(edges, Mapping):
        edge_index = edges.get("edge_index")
        edge_weight = edges.get("edge_weight")
    else:
        edge_index = edges

    edge_index = np.asarray(edge_index)
    if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    return GraphArtifact(
        n_nodes=int(n_nodes),
        edge_index=edge_index,
        edge_weight=None if edge_weight is None else np.asarray(edge_weight),
        directed=True,
        meta={},
    )


def _dtype_descriptor(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    out: dict[str, Any] = {}
    if dtype is not None:
        out["dtype"] = str(dtype)
    if shape is not None:
        try:
            out["shape"] = list(shape)
        except TypeError:
            out["shape"] = None
    return out or None


def run(
    *,
    dataset: Any,
    graph: GraphArtifact,
    masks: dict[str, np.ndarray],
    cfg: MethodConfig,
    seed: int,
    use_test_split: bool,
    expected_labeled_count: int | None = None,
    strict: bool = False,
) -> tuple[Any, NodeDataset, dict[str, Any]]:
    start = perf_counter()
    method_info = get_method_info(cfg.method_id)
    resolved_device = _resolve_method_device(
        cfg.device.device,
        supports_gpu=method_info.supports_gpu,
        strict=strict,
    )

    method_cls = get_method_class(cfg.method_id)
    spec = _build_spec(method_cls, cfg.params, strict=strict)
    resolved_backend = _resolve_method_backend(cfg=cfg, spec=spec, strict=strict)
    _ensure_backend_dependencies(resolved_backend)

    _LOGGER.info(
        "Transductive method start: id=%s seed=%s device=%s resolved_device=%s backend=%s use_test=%s strict=%s",
        cfg.method_id,
        int(seed),
        cfg.device.device,
        resolved_device,
        resolved_backend,
        bool(use_test_split),
        bool(strict),
    )
    _LOGGER.debug("Transductive method params: %s", dict(cfg.params))

    X_train = dataset.train.X
    y_train = dataset.train.y
    X_test = dataset.test.X if use_test_split and dataset.test is not None else None
    y_test = dataset.test.y if use_test_split and dataset.test is not None else None

    X_all = _combine_splits(X_train, X_test)
    y_all = _combine_splits(y_train, y_test)
    n_test = 0 if X_test is None else int(X_test.shape[0])
    _LOGGER.info(
        "Transductive data: n_nodes=%s n_train=%s n_test=%s",
        int(X_all.shape[0]),
        int(X_train.shape[0]),
        n_test,
    )

    labeled_mask = np.asarray(masks.get("labeled_mask", masks.get("train_mask")), dtype=bool)
    labeled_count = int(labeled_mask.sum())
    if expected_labeled_count is not None and labeled_count != int(expected_labeled_count):
        raise BenchRuntimeError(
            "E_BENCH_MASK_CONTRACT",
            "transductive labeled mask mismatch: expected "
            f"{int(expected_labeled_count)} from sampling stats, got {labeled_count}",
        )

    train_all_mask = masks.get("train_mask")
    train_count = (
        int(np.asarray(train_all_mask, dtype=bool).sum()) if train_all_mask is not None else None
    )
    _LOGGER.info(
        "Transductive masks: labeled_count=%s train_count=%s",
        labeled_count,
        train_count,
    )

    y_obs = _to_numpy(y_all).astype(np.int64, copy=True)
    if y_obs.ndim == 1 and y_obs.shape[0] == labeled_mask.shape[0]:
        unlabeled_mask = masks.get("unlabeled_mask")
        if unlabeled_mask is None:
            y_obs[~labeled_mask] = -1
        else:
            unlabeled_mask = np.asarray(unlabeled_mask, dtype=bool)
            if unlabeled_mask.shape != y_obs.shape:
                y_obs[~labeled_mask] = -1
            else:
                y_obs[unlabeled_mask] = -1

    data = NodeDataset(
        X=X_all,
        y=y_obs,
        graph=graph,
        masks={
            "train_mask": labeled_mask,
            "val_mask": masks["val_mask"],
            "test_mask": masks["test_mask"],
            "unlabeled_mask": masks["unlabeled_mask"],
            "labeled_mask": labeled_mask,
            **(
                {"train_all_mask": np.asarray(train_all_mask, dtype=bool)}
                if train_all_mask is not None
                else {}
            ),
        },
        meta={"y_true": _to_numpy(y_all), "y_obs": y_obs},
    )

    method = method_cls(spec) if spec is not None else method_cls()
    method.fit(data, device=resolved_device, seed=int(seed))

    resolved_backend_final = resolved_backend
    if resolved_backend_final is None:
        maybe_backend = getattr(method, "_backend", None)
        resolved_backend_final = str(maybe_backend) if maybe_backend is not None else None

    method_resolution = {
        "backend": resolved_backend_final,
        "classifier_backend": None,
        "dtypes": {
            "X": _dtype_descriptor(X_all),
            "y_true": _dtype_descriptor(_to_numpy(y_all)),
            "y_obs": _dtype_descriptor(y_obs),
        },
        "normalization": {
            "implicit_method_conversion": False,
            "strict_contract_validated": bool(strict),
        },
    }

    _LOGGER.info(
        "Transductive method done: id=%s duration_s=%.3f",
        cfg.method_id,
        perf_counter() - start,
    )
    return method, data, method_resolution
