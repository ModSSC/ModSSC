from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset, Split
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
}


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(s) for s in shape)
    except Exception:
        return None


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

    prov_train = {k: f"{dataset_fp}:{k}" for k in train_store}
    prov_test = {k: f"{dataset_fp}:{k}" for k in (test_store if test_store else [])}

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
        if cm is not None and cm.has_step_outputs(step_fp, split="train"):
            try:
                produced_train = cm.load_step_outputs(step_fingerprint=step_fp, split="train")
                if not _cache_outputs_complete(produced_train, spec.produces):
                    raise PreprocessCacheError(
                        f"Incomplete cached outputs for step {step_id!r} (train)"
                    )
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

        if test_store is not None:
            produced_test: dict[str, Any] | None = None
            if cm is not None and cm.has_step_outputs(step_fp, split="test"):
                try:
                    produced_test = cm.load_step_outputs(step_fingerprint=step_fp, split="test")
                    if not _cache_outputs_complete(produced_test, spec.produces):
                        raise PreprocessCacheError(
                            f"Incomplete cached outputs for step {step_id!r} (test)"
                        )
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
