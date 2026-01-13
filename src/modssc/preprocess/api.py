from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset, Split
from modssc.device import resolve_device_name
from modssc.preprocess.cache import CacheManager
from modssc.preprocess.errors import PreprocessCacheError, PreprocessValidationError
from modssc.preprocess.fingerprint import derive_seed, fingerprint
from modssc.preprocess.plan import PreprocessPlan
from modssc.preprocess.registry import StepRegistry, default_step_registry
from modssc.preprocess.store import ArtifactStore
from modssc.preprocess.types import PreprocessResult, ResolvedPlan, ResolvedStep, SkippedStep

logger = logging.getLogger(__name__)


_IMPLICIT_CONSUMES: dict[str, tuple[str, ...]] = {
    "graph.node2vec": ("raw.X", "raw.y"),
    "graph.dgi": ("raw.X",),
}


_MAX_ESTIMATE_ITEMS = 1024
_MAX_ESTIMATE_DEPTH = 2


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(s) for s in shape)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_torch() -> Any | None:
    try:
        import importlib

        return importlib.import_module("torch")
    except Exception:
        return None


def _estimate_collection_bytes(
    values: Any, *, max_items: int, depth: int, total_len: int | None
) -> tuple[int, int, bool]:
    size = 0
    unknown = 0
    approx = False
    count = 0
    for item in values:
        item_size, item_unknown, item_approx = _estimate_bytes(
            item, max_items=max_items, depth=depth
        )
        size += item_size
        unknown += item_unknown
        approx = approx or item_approx
        count += 1
        if count >= max_items:
            break
    if total_len is not None and total_len > count:
        approx = True
        if unknown == 0 and count > 0:
            size = int(size * (total_len / count))
        else:
            unknown += total_len - count
    elif total_len is None and count >= max_items:
        approx = True
    return size, unknown, approx


def _estimate_bytes(
    value: Any, *, max_items: int = _MAX_ESTIMATE_ITEMS, depth: int = 0
) -> tuple[int, int, bool]:
    if value is None:
        return 0, 0, False

    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return int(nbytes), 0, False
        except Exception:
            pass

    if isinstance(value, (bytes, bytearray, memoryview)):
        return int(len(value)), 0, False
    if isinstance(value, str):
        return len(value.encode("utf-8")), 0, False

    torch = _get_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.element_size() * value.nelement()), 0, False

    data = getattr(value, "data", None)
    indices = getattr(value, "indices", None)
    indptr = getattr(value, "indptr", None)
    if data is not None and indices is not None and indptr is not None:
        try:
            return int(data.nbytes + indices.nbytes + indptr.nbytes), 0, False
        except Exception:
            pass

    if depth >= _MAX_ESTIMATE_DEPTH:
        return 0, 1, False

    if isinstance(value, Mapping):
        return _estimate_collection_bytes(
            value.values(), max_items=max_items, depth=depth + 1, total_len=len(value)
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _estimate_collection_bytes(
            value, max_items=max_items, depth=depth + 1, total_len=len(value)
        )

    return 0, 1, False


def _format_bytes(size: int) -> str:
    size_f = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_f < 1024.0:
            if unit == "B":
                return f"{int(size_f)} B"
            return f"{size_f:.1f} {unit}"
        size_f /= 1024.0
    return f"{size_f:.1f} PB"


def _format_size_estimate(size: int, unknown: int, approx: bool) -> str:
    if size == 0 and unknown > 0:
        return "unknown"
    label = _format_bytes(size)
    extras: list[str] = []
    if approx:
        extras.append("approx")
    if unknown:
        extras.append(f"unknown={unknown}")
    if extras:
        return f"{label} ({', '.join(extras)})"
    return label


def _split_data_size(
    store: ArtifactStore | None, *, output_key: str | None = None
) -> tuple[int, int, bool]:
    if store is None:
        return 0, 0, False
    keys: list[str] = []
    x_key = None
    if output_key and store.has(output_key):
        x_key = output_key
    else:
        x_key = "features.X" if store.has("features.X") else "raw.X"
    y_key = "labels.y" if store.has("labels.y") else "raw.y"
    if store.has(x_key):
        keys.append(x_key)
    if store.has(y_key) and y_key not in keys:
        keys.append(y_key)
    if store.has("graph.edge_index") and "graph.edge_index" not in keys:
        keys.append("graph.edge_index")
    if store.has("graph.edge_weight") and "graph.edge_weight" not in keys:
        keys.append("graph.edge_weight")
    for key in store.data:
        if key.startswith("graph.mask."):
            keys.append(key)
    total = 0
    unknown = 0
    approx = False
    for key in keys:
        size, key_unknown, key_approx = _estimate_bytes(store.get(key))
        total += size
        unknown += key_unknown
        approx = approx or key_approx
    return total, unknown, approx


def _format_split_size(store: ArtifactStore | None, *, output_key: str | None = None) -> str:
    if store is None:
        return "n/a"
    size, unknown, approx = _split_data_size(store, output_key=output_key)
    return _format_size_estimate(size, unknown, approx)


def _device_from_outputs(torch: Any, outputs: dict[str, Any] | None) -> str | None:
    if torch is None or outputs is None:
        return None
    for value in outputs.values():
        if isinstance(value, torch.Tensor):
            return str(value.device)
    return None


def _device_hint(step_params: dict[str, Any], step_obj: Any) -> str | None:
    if isinstance(step_params, dict):
        device = step_params.get("device")
        if device is not None:
            return str(device)
    device = getattr(step_obj, "device", None)
    if device is not None:
        return str(device)
    return None


def _normalize_device_name(device: str | None, torch: Any | None) -> str | None:
    if device is None:
        return None
    if device == "auto":
        return resolve_device_name(device, torch=torch)
    return device


def _gpu_model_for_device(torch: Any, device: str) -> str | None:
    try:
        torch_device = torch.device(device)
    except Exception:
        torch_device = None
    if torch_device is not None and torch_device.type == "cuda":
        if not torch.cuda.is_available():
            return None
        index = torch_device.index
        if index is None:
            index = torch.cuda.current_device()
        try:
            return str(torch.cuda.get_device_name(index))
        except Exception:
            return None
    if torch_device is not None and torch_device.type == "mps":
        return "Apple MPS"
    if device.startswith("mps"):
        return "Apple MPS"
    return None


def _maybe_log_gpu_info(
    step_params: dict[str, Any],
    step_obj: Any,
    *,
    produced_train: dict[str, Any] | None,
    produced_test: dict[str, Any] | None,
    use_device_hint: bool,
) -> bool:
    torch = _get_torch()
    if torch is None:
        return False
    device = _device_from_outputs(torch, produced_train) or _device_from_outputs(
        torch, produced_test
    )
    if device is None and use_device_hint:
        device = _device_hint(step_params, step_obj)
    device = _normalize_device_name(device, torch)
    if device is None or not (device.startswith("cuda") or device.startswith("mps")):
        return False
    model = _gpu_model_for_device(torch, device)
    if model:
        logger.info("Preprocess GPU device: device=%s model=%s", device, model)
    else:
        logger.info("Preprocess GPU device: device=%s", device)
    return True


def _maybe_warn_nonfinite(name: str, value: Any, *, max_elems: int = 1_000_000) -> None:
    if not isinstance(value, np.ndarray):
        return
    if int(value.size) > max_elems:
        return
    if not np.isfinite(value).all():
        logger.warning("Non-finite values detected in %s", name)


def _cache_outputs_complete(outputs: dict[str, Any], produces: tuple[str, ...]) -> bool:
    expected = {str(k) for k in produces}
    return expected.issubset(outputs.keys())


def _dataset_fingerprint(dataset: LoadedDataset) -> str:
    fp = dataset.meta.get("dataset_fingerprint") if hasattr(dataset, "meta") else None
    if isinstance(fp, str) and fp:
        return fp

    # Fallback: structural fingerprint only (no full data hashing).
    train_x = getattr(dataset.train, "X", None)
    train_y = getattr(dataset.train, "y", None)
    payload = {
        "modality": dataset.meta.get("modality") if hasattr(dataset, "meta") else None,
        "train": {
            "x_shape": getattr(train_x, "shape", None),
            "y_shape": getattr(train_y, "shape", None),
            "has_edges": dataset.train.edges is not None,
            "has_masks": dataset.train.masks is not None,
        },
        "has_test": dataset.test is not None,
    }
    return fingerprint(payload, prefix="dataset:")


def _initial_store(split: Split) -> ArtifactStore:
    store = ArtifactStore()
    store.set("raw.X", split.X)
    store.set("raw.y", split.y)
    if split.edges is not None:
        # Convention: store graph edges as graph.edge_index (weights, if any, are separate).
        store.set("graph.edge_index", split.edges)
    if split.masks is not None:
        for k, v in split.masks.items():
            store.set(f"graph.mask.{k}", v)
    return store


def _final_keep_keys(
    steps: tuple[ResolvedStep, ...], *, output_key: str, initial_keys: set[str]
) -> set[str]:
    produced = {k for step in steps for k in step.spec.produces}
    keep = {output_key}
    if output_key == "raw.X" or output_key not in initial_keys and output_key not in produced:
        keep.add("raw.X")
    if "labels.y" in produced:
        keep.add("labels.y")
    else:
        keep.add("raw.y")
    if "graph.edge_weight" in produced:
        keep.add("graph.edge_weight")
        keep.add("graph.edge_index")
    elif "graph.edge_index" in produced:
        keep.add("graph.edge_index")
    return keep


def _build_purge_keep_sets(
    steps: tuple[ResolvedStep, ...], *, output_key: str, initial_keys: set[str]
) -> list[set[str]]:
    required = _final_keep_keys(steps, output_key=output_key, initial_keys=initial_keys)
    keep_sets: list[set[str]] = [set() for _ in steps]
    for i in range(len(steps) - 1, -1, -1):
        keep_sets[i] = set(required)
        step = steps[i]
        required.update(step.spec.consumes)
        required.update(_IMPLICIT_CONSUMES.get(step.step_id, ()))
    return keep_sets


def _purge_store(store: ArtifactStore, *, keep: set[str]) -> None:
    if not keep:
        store.data = {}
        return
    store.data = {k: v for k, v in store.data.items() if k in keep}


def resolve_plan(
    dataset: LoadedDataset,
    plan: PreprocessPlan,
    *,
    registry: StepRegistry | None = None,
) -> ResolvedPlan:
    reg = registry or default_step_registry()
    modality = str(dataset.meta.get("modality", "")) if hasattr(dataset, "meta") else ""

    # Track which fields are available as we walk through the plan.
    fields = set(_initial_store(dataset.train).keys())
    if dataset.test is not None:
        fields |= set(_initial_store(dataset.test).keys())

    resolved: list[ResolvedStep] = []
    skipped: list[SkippedStep] = []

    for i, step_cfg in enumerate(plan.steps):
        if not step_cfg.enabled:
            skipped.append(SkippedStep(step_id=step_cfg.step_id, reason="disabled", index=i))
            continue

        spec = reg.spec(step_cfg.step_id)
        allowed = step_cfg.modalities or spec.modalities
        if allowed and modality and modality not in allowed:
            skipped.append(
                SkippedStep(
                    step_id=step_cfg.step_id,
                    reason=f"modality {modality!r} not in {list(allowed)!r}",
                    index=i,
                )
            )
            continue

        required = step_cfg.requires_fields
        if required and any(k not in fields for k in required):
            missing = [k for k in required if k not in fields]
            skipped.append(
                SkippedStep(
                    step_id=step_cfg.step_id,
                    reason=f"missing required fields: {missing}",
                    index=i,
                )
            )
            continue

        resolved.append(
            ResolvedStep(step_id=step_cfg.step_id, params=dict(step_cfg.params), index=i, spec=spec)
        )
        # Conservative: assume it produces its declared fields.
        for k in spec.produces:
            fields.add(k)

    resolved_fp = fingerprint(
        {
            "plan_fp": plan.fingerprint(),
            "modality": modality,
            "steps": [
                {"id": s.step_id, "params": dict(s.params), "index": s.index, "kind": s.spec.kind}
                for s in resolved
            ],
        },
        prefix="resolved_plan:",
    )
    return ResolvedPlan(steps=tuple(resolved), skipped=tuple(skipped), fingerprint=resolved_fp)


def preprocess(
    dataset: LoadedDataset,
    plan: PreprocessPlan,
    *,
    seed: int = 0,
    fit_indices: np.ndarray | None = None,
    cache: bool = True,
    cache_dir: str | None = None,
    purge_unused_artifacts: bool = False,
    registry: StepRegistry | None = None,
) -> PreprocessResult:
    """Run preprocessing.

    Parameters
    - dataset: LoadedDataset from modssc.data_loader
    - plan: PreprocessPlan
    - seed: master seed for deterministic steps
    - fit_indices: optional indices (relative to train split) used by fittable steps
    - cache: enable step-level cache
    - cache_dir: optional override of preprocessing cache directory
    - purge_unused_artifacts: drop artifacts not needed by downstream steps
    """
    start = perf_counter()
    reg = registry or default_step_registry()
    resolved = resolve_plan(dataset, plan, registry=reg)
    dataset_fp = _dataset_fingerprint(dataset)

    fit_fp = "fit:none"
    if fit_indices is not None:
        fit_arr = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
        fit_hash = hashlib.sha256(fit_arr.tobytes()).hexdigest()
        fit_fp = f"fit:{fit_hash}"

    preprocess_fp = fingerprint(
        {
            "dataset_fp": dataset_fp,
            "resolved_plan_fp": resolved.fingerprint,
            "fit_fp": fit_fp,
            "seed": int(seed),
        },
        prefix="preprocess:",
    )

    cm = None
    if cache:
        cm = CacheManager.for_dataset(dataset_fp)
        if cache_dir is not None:
            cm.root = Path(cache_dir).expanduser().resolve()

    train_store = _initial_store(dataset.train)
    test_store = _initial_store(dataset.test) if dataset.test is not None else None
    purge_keep_sets = None
    if purge_unused_artifacts:
        purge_keep_sets = _build_purge_keep_sets(
            resolved.steps, output_key=plan.output_key, initial_keys=set(train_store)
        )
        logger.info(
            "Preprocess purge enabled: retaining minimal artifacts per step (steps=%s)",
            len(purge_keep_sets),
        )
        if purge_keep_sets and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Preprocess purge keep keys (step 0): %s", sorted(purge_keep_sets[0]))

    logger.info(
        "Preprocess start: dataset_fp=%s steps=%s output_key=%s seed=%s cache=%s",
        dataset_fp,
        [s.step_id for s in resolved.steps],
        plan.output_key,
        seed,
        bool(cache),
    )
    logger.debug(
        "Preprocess input shapes: train_X=%s train_y=%s test_X=%s test_y=%s",
        _shape_of(dataset.train.X),
        _shape_of(dataset.train.y),
        _shape_of(dataset.test.X) if dataset.test is not None else None,
        _shape_of(dataset.test.y) if dataset.test is not None else None,
    )
    logger.info(
        "Preprocess input size: train_data=%s test_data=%s",
        _format_split_size(train_store, output_key=plan.output_key),
        _format_split_size(test_store, output_key=plan.output_key),
    )

    prov_train = {k: f"{dataset_fp}:{k}" for k in train_store}
    prov_test = {k: f"{dataset_fp}:{k}" for k in (test_store if test_store else [])}

    gpu_logged = False
    for _step_num, step in enumerate(resolved.steps):
        step_id = step.step_id
        spec = step.spec
        derived = derive_seed(seed, step_id=step_id, step_index=step.index)
        rng = np.random.default_rng(derived)
        step_start = perf_counter()
        logger.debug(
            "Preprocess step start: id=%s index=%s kind=%s params=%s",
            step_id,
            step.index,
            spec.kind,
            dict(step.params),
        )

        # Compute step fingerprint with input provenance.
        inputs_train = {k: prov_train.get(k) for k in spec.consumes if k in prov_train}
        inputs_test = (
            {k: prov_test.get(k) for k in spec.consumes if k in prov_test} if test_store else {}
        )
        step_fp = fingerprint(
            {
                "dataset_fp": dataset_fp,
                "step_id": step_id,
                "index": step.index,
                "params": dict(step.params),
                "kind": spec.kind,
                "seed": int(derived),
                "fit_fp": fit_fp if spec.kind == "fittable" else None,
                "inputs_train": inputs_train,
                "inputs_test": inputs_test,
            },
            prefix="step:",
        )

        step_obj = reg.instantiate(step_id, params=dict(step.params))

        # Fit if needed.
        if spec.kind == "fittable":
            if fit_indices is None:
                raise PreprocessValidationError(
                    f"Step {step_id!r} is fittable but fit_indices is None."
                )
            if not hasattr(step_obj, "fit"):
                raise PreprocessValidationError(
                    f"Step {step_id!r} declared fittable but has no fit()."
                )
            step_obj.fit(train_store, fit_indices=np.asarray(fit_indices, dtype=np.int64), rng=rng)

        # Load from cache if available, otherwise compute and save.
        produced_train: dict[str, Any] | None = None
        train_from_cache = False
        if cm is not None and cm.has_step_outputs(step_fp, split="train"):
            try:
                produced_train = cm.load_step_outputs(step_fingerprint=step_fp, split="train")
                if not _cache_outputs_complete(produced_train, spec.produces):
                    raise PreprocessCacheError(
                        f"Incomplete cached outputs for step {step_id!r} (train)"
                    )
                train_from_cache = True
            except PreprocessCacheError as e:
                logger.warning("Preprocess cache miss for %s (train): %s", step_id, e)
                produced_train = None
        if produced_train is None:
            produced_train = step_obj.transform(train_store, rng=rng)
            if not isinstance(produced_train, dict):
                raise PreprocessValidationError(
                    f"Step {step_id!r} must return a dict of produced artifacts."
                )
            if cm is not None:
                cm.save_step_outputs(
                    step_fingerprint=step_fp,
                    split="train",
                    produced=produced_train,
                    manifest={
                        "step_id": step_id,
                        "index": step.index,
                        "params": dict(step.params),
                        "kind": spec.kind,
                        "required_extra": spec.required_extra,
                        "consumes": list(spec.consumes),
                        "produces": list(spec.produces),
                        "inputs_train": inputs_train,
                        "fit_fp": fit_fp if spec.kind == "fittable" else None,
                        "seed": int(derived),
                    },
                )

        for k, v in produced_train.items():
            train_store.set(k, v)
            prov_train[k] = step_fp
        logger.debug(
            "Preprocess step train outputs: id=%s keys=%s duration_s=%.3f",
            step_id,
            sorted(produced_train.keys()),
            perf_counter() - step_start,
        )

        produced_test: dict[str, Any] | None = None
        test_from_cache = False
        if test_store is not None:
            if cm is not None and cm.has_step_outputs(step_fp, split="test"):
                try:
                    produced_test = cm.load_step_outputs(step_fingerprint=step_fp, split="test")
                    if not _cache_outputs_complete(produced_test, spec.produces):
                        raise PreprocessCacheError(
                            f"Incomplete cached outputs for step {step_id!r} (test)"
                        )
                    test_from_cache = True
                except PreprocessCacheError as e:
                    logger.warning("Preprocess cache miss for %s (test): %s", step_id, e)
                    produced_test = None
            if produced_test is None:
                produced_test = step_obj.transform(test_store, rng=rng)
                if not isinstance(produced_test, dict):
                    raise PreprocessValidationError(
                        f"Step {step_id!r} must return a dict of produced artifacts."
                    )
                if cm is not None:
                    cm.save_step_outputs(
                        step_fingerprint=step_fp,
                        split="test",
                        produced=produced_test,
                        manifest={
                            "step_id": step_id,
                            "index": step.index,
                            "params": dict(step.params),
                            "kind": spec.kind,
                            "required_extra": spec.required_extra,
                            "consumes": list(spec.consumes),
                            "produces": list(spec.produces),
                            "inputs_test": inputs_test,
                            "fit_fp": fit_fp if spec.kind == "fittable" else None,
                            "seed": int(derived),
                        },
                    )

            for k, v in produced_test.items():
                test_store.set(k, v)
                prov_test[k] = step_fp
            logger.debug(
                "Preprocess step test outputs: id=%s keys=%s",
                step_id,
                sorted(produced_test.keys()),
            )

        if purge_keep_sets is not None:
            keep = purge_keep_sets[_step_num]
            _purge_store(train_store, keep=keep)
            if test_store is not None:
                _purge_store(test_store, keep=keep)

        if not gpu_logged:
            computed = not train_from_cache
            if test_store is not None:
                computed = computed or not test_from_cache
            gpu_logged = _maybe_log_gpu_info(
                step.params,
                step_obj,
                produced_train=produced_train,
                produced_test=produced_test,
                use_device_hint=computed,
            )

        step_duration = perf_counter() - step_start
        logger.info(
            "Preprocess step done: id=%s index=%s duration_s=%.3f train_data=%s test_data=%s",
            step_id,
            step.index,
            step_duration,
            _format_split_size(train_store, output_key=plan.output_key),
            _format_split_size(test_store, output_key=plan.output_key),
        )

    # Choose final X for downstream training.
    out_key = plan.output_key
    X_train = train_store.get(out_key, train_store.require("raw.X"))
    y_train = train_store.get("labels.y", train_store.require("raw.y"))

    edges_train = train_store.get("graph.edge_index", dataset.train.edges)
    if train_store.has("graph.edge_weight"):
        edges_train = {
            "edge_index": train_store.get("graph.edge_index"),
            "edge_weight": train_store.get("graph.edge_weight"),
        }

    train_out = Split(X=X_train, y=y_train, edges=edges_train, masks=dataset.train.masks)

    test_out = None
    if dataset.test is not None and test_store is not None:
        X_test = test_store.get(out_key, test_store.require("raw.X"))
        y_test = test_store.get("labels.y", test_store.require("raw.y"))
        edges_test = test_store.get("graph.edge_index", dataset.test.edges)
        if test_store.has("graph.edge_weight"):
            edges_test = {
                "edge_index": test_store.get("graph.edge_index"),
                "edge_weight": test_store.get("graph.edge_weight"),
            }
        test_out = Split(X=X_test, y=y_test, edges=edges_test, masks=dataset.test.masks)

    meta = dict(dataset.meta)
    meta.update(
        {
            "preprocess_fingerprint": preprocess_fp,
            "preprocess_plan_fingerprint": resolved.fingerprint,
            "preprocess_fit_fingerprint": fit_fp,
        }
    )
    if cm is not None:
        meta["preprocess_cache_dir"] = str(cm.dataset_dir())

    out_dataset = LoadedDataset(train=train_out, test=test_out, meta=meta)
    _maybe_warn_nonfinite("train.X", train_out.X)
    if test_out is not None:
        _maybe_warn_nonfinite("test.X", test_out.X)

    logger.info(
        "Preprocess done: dataset_fp=%s duration_s=%.3f train_X=%s test_X=%s",
        dataset_fp,
        perf_counter() - start,
        _shape_of(train_out.X),
        _shape_of(test_out.X) if test_out is not None else None,
    )
    return PreprocessResult(
        dataset=out_dataset,
        plan=resolved,
        preprocess_fingerprint=preprocess_fp,
        train_artifacts=train_store,
        test_artifacts=test_store,
        cache_dir=str(cm.dataset_dir()) if cm is not None else None,
    )


def fit_transform(*args: Any, **kwargs: Any) -> PreprocessResult:
    """Alias for preprocess()."""
    return preprocess(*args, **kwargs)
