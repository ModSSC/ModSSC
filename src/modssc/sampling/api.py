from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from platformdirs import user_cache_dir

from modssc.sampling.errors import MissingDatasetFingerprintError
from modssc.sampling.fingerprint import derive_seed, stable_hash
from modssc.sampling.imbalance import apply_imbalance
from modssc.sampling.labeling import select_labeled
from modssc.sampling.plan import HoldoutSplitSpec, SamplingPlan
from modssc.sampling.result import SamplingResult
from modssc.sampling.splitters import make_holdout_split, make_kfold_split
from modssc.sampling.stats import build_graph_stats, build_inductive_stats
from modssc.sampling.storage import load_split as _load_split
from modssc.sampling.storage import save_split as _save_split

SPLIT_CACHE_ENV = "MODSSC_SPLIT_CACHE_DIR"
SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


def default_split_cache_dir() -> Path:
    override = os.environ.get(SPLIT_CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()

    # Heuristic: if running in a dev repo (pyproject.toml exists in parents),
    # default to a local "cache" folder at the repo root.
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent / "cache" / "splits"

    return Path(user_cache_dir("modssc")) / "splits"


def split_dir_for(
    *, dataset_fingerprint: str, split_fingerprint: str, root: Path | None = None
) -> Path:
    base = (root or default_split_cache_dir()).expanduser().resolve()
    return base / dataset_fingerprint / split_fingerprint


def save_split(result: SamplingResult, out_dir: Path, *, overwrite: bool = False) -> Path:
    return _save_split(result, out_dir, overwrite=overwrite)


def load_split(dir_path: Path) -> SamplingResult:
    return _load_split(dir_path)


def sample(
    dataset: Any,
    *,
    plan: SamplingPlan,
    seed: int,
    dataset_fingerprint: str | None = None,
    dataset_id: str | None = None,
    cache_root: Path | None = None,
    save: bool = False,
    overwrite: bool = False,
) -> tuple[SamplingResult, Path | None]:
    """Sample a canonical dataset into a reproducible experimental split.

    Returns (result, path). Path is not None if save=True.
    """
    start = perf_counter()
    ds_fp = _resolve_dataset_fingerprint(dataset, dataset_fingerprint)

    seed_split = derive_seed(seed, "split")
    seed_label = derive_seed(seed, "labeling")
    seed_imb = derive_seed(seed, "imbalance")

    plan_dict = plan.as_dict()
    split_fingerprint = stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_fingerprint": ds_fp,
            "plan": plan_dict,
            "seed": int(seed),
        }
    )

    created_at = datetime.now(timezone.utc).isoformat()

    # detect graph
    is_graph = (
        getattr(getattr(dataset, "train", None), "edges", None) is not None
        or getattr(getattr(dataset, "train", None), "masks", None) is not None
    )
    logger.info(
        "Sampling start: dataset_id=%s dataset_fingerprint=%s seed=%s graph=%s split=%s",
        dataset_id,
        ds_fp,
        seed,
        bool(is_graph),
        plan.split.kind,
    )
    logger.debug("Sampling plan: %s", plan_dict)

    if is_graph:
        result = _sample_graph(
            dataset,
            plan=plan,
            seed_split=seed_split,
            seed_label=seed_label,
            seed_imb=seed_imb,
            dataset_fingerprint=ds_fp,
            split_fingerprint=split_fingerprint,
            created_at=created_at,
            plan_dict=plan_dict,
        )
    else:
        result = _sample_inductive(
            dataset,
            plan=plan,
            seed_split=seed_split,
            seed_label=seed_label,
            seed_imb=seed_imb,
            dataset_fingerprint=ds_fp,
            split_fingerprint=split_fingerprint,
            created_at=created_at,
            plan_dict=plan_dict,
        )

    out_path: Path | None = None
    if save:
        out_path = split_dir_for(
            dataset_fingerprint=ds_fp, split_fingerprint=split_fingerprint, root=cache_root
        )
        save_split(result, out_path, overwrite=overwrite)
    duration = perf_counter() - start
    logger.info(
        "Sampling done: train=%s val=%s test=%s labeled=%s unlabeled=%s duration_s=%.3f",
        int(result.train_idx.shape[0]),
        int(result.val_idx.shape[0]),
        int(result.test_idx.shape[0]),
        int(result.labeled_idx.shape[0]),
        int(result.unlabeled_idx.shape[0]),
        duration,
    )
    logger.debug("Sampling stats: %s", dict(result.stats))
    _warn_on_sampling_stats(result)
    return result, out_path


def _warn_on_sampling_stats(result: SamplingResult) -> None:
    if result.is_graph():
        stats = result.stats
        labeled = stats.get("labeled_class_dist", {})
        classes = labeled.get("classes", {})
        if isinstance(classes, dict):
            missing = [k for k, v in classes.items() if int(v) == 0]
            if missing:
                logger.warning("Sampling labeled classes missing: %s", missing)
        return

    stats = result.stats
    labeled = stats.get("train_labeled", {})
    train = stats.get("train", {})
    if isinstance(labeled, dict) and isinstance(train, dict):
        train_classes = train.get("classes", {}) if isinstance(train, dict) else {}
        labeled_classes = labeled.get("classes", {}) if isinstance(labeled, dict) else {}
        if isinstance(train_classes, dict) and isinstance(labeled_classes, dict):
            missing = [k for k in train_classes if int(labeled_classes.get(k, 0)) == 0]
            if missing:
                logger.warning("Sampling labeled classes missing: %s", missing)

    if int(result.train_idx.shape[0]) == 0 or int(result.labeled_idx.shape[0]) == 0:
        logger.warning("Sampling produced empty train or labeled split")


# ----------------------------
# internal
# ----------------------------


def _resolve_dataset_fingerprint(dataset: Any, provided: str | None) -> str:
    if provided:
        return str(provided)
    meta = getattr(dataset, "meta", None)
    if isinstance(meta, dict):
        if "dataset_fingerprint" in meta:
            return str(meta["dataset_fingerprint"])
        if "fingerprint" in meta:
            return str(meta["fingerprint"])
    raise MissingDatasetFingerprintError


def _sample_inductive(
    dataset: Any,
    *,
    plan: SamplingPlan,
    seed_split: int,
    seed_label: int,
    seed_imb: int,
    dataset_fingerprint: str,
    split_fingerprint: str,
    created_at: str,
    plan_dict: dict[str, Any],
) -> SamplingResult:
    y_train = np.asarray(dataset.train.y)
    n_train = int(y_train.shape[0])

    has_official_test = getattr(dataset, "test", None) is not None
    y_test = None
    n_test = None
    if has_official_test:
        y_test = np.asarray(dataset.test.y)
        n_test = int(y_test.shape[0])

    # Split on train indices only if official test is respected
    if has_official_test and plan.policy.respect_official_test:
        if not plan.policy.allow_override_official:
            test_idx = np.arange(n_test or 0, dtype=np.int64)
            test_ref = "test"
            pool_y = y_train
            pool_n = n_train
            if isinstance(plan.split, HoldoutSplitSpec):
                rng = np.random.default_rng(seed_split)
                # ignore test_fraction, split only val from train
                parts = make_holdout_split(
                    n_samples=pool_n,
                    y=pool_y,
                    test_fraction=0.0,
                    val_fraction=float(plan.split.val_fraction),
                    stratify=bool(plan.split.stratify),
                    rng=rng,
                )
            else:
                rng = np.random.default_rng(seed_split)
                parts = make_kfold_split(
                    n_samples=pool_n,
                    y=pool_y,
                    k=int(plan.split.k),
                    fold=int(plan.split.fold),
                    stratify=bool(plan.split.stratify),
                    shuffle=bool(plan.split.shuffle),
                    val_fraction=0.0,
                    rng=rng,
                )
                # in this mode, fold acts as val
                parts = {
                    "train": parts["train"],
                    "val": parts["test"],
                    "test": np.asarray([], dtype=np.int64),
                }
            train_idx = parts["train"]
            val_idx = parts["val"]
        else:
            raise NotImplementedError("override_official is not implemented in the current API")
    else:
        rng = np.random.default_rng(seed_split)
        if isinstance(plan.split, HoldoutSplitSpec):
            parts = make_holdout_split(
                n_samples=n_train,
                y=y_train,
                test_fraction=float(plan.split.test_fraction),
                val_fraction=float(plan.split.val_fraction),
                stratify=bool(plan.split.stratify),
                rng=rng,
            )
        else:
            parts = make_kfold_split(
                n_samples=n_train,
                y=y_train,
                k=int(plan.split.k),
                fold=int(plan.split.fold),
                stratify=bool(plan.split.stratify),
                shuffle=bool(plan.split.shuffle),
                val_fraction=float(plan.split.val_fraction),
                rng=rng,
            )
        train_idx = parts["train"]
        val_idx = parts["val"]
        test_idx = parts["test"]
        test_ref = "train"

    # apply imbalance to train if requested
    rng_imb = np.random.default_rng(seed_imb)
    if plan.imbalance.apply_to == "train":
        train_idx_adj = apply_imbalance(idx=train_idx, y=y_train, spec=plan.imbalance, rng=rng_imb)
    else:
        train_idx_adj = train_idx

    rng_lab = np.random.default_rng(seed_label)
    labeled = select_labeled(train_idx=train_idx_adj, y=y_train, spec=plan.labeling, rng=rng_lab)

    if plan.imbalance.apply_to == "labeled":
        labeled_adj = apply_imbalance(idx=labeled, y=y_train, spec=plan.imbalance, rng=rng_imb)
        labeled = labeled_adj

    unlabeled = np.setdiff1d(train_idx_adj, labeled, assume_unique=False)

    # Ensure train indices reflect imbalance apply_to=train
    train_idx_final = np.sort(train_idx_adj)

    indices = {
        "train": train_idx_final,
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
        "train_labeled": np.sort(labeled),
        "train_unlabeled": np.sort(unlabeled),
    }
    refs = {
        "train": "train",
        "val": "train",
        "test": test_ref,
        "train_labeled": "train",
        "train_unlabeled": "train",
    }

    policy_info = {
        "respect_official_test": bool(plan.policy.respect_official_test),
        "has_official_test": bool(has_official_test),
        "test_ref": test_ref,
    }
    stats = build_inductive_stats(
        y_train=y_train,
        train_idx=indices["train"],
        val_idx=indices["val"],
        test_ref=test_ref,
        y_test=y_test,
        test_idx=indices["test"],
        labeled_idx=indices["train_labeled"],
        unlabeled_idx=indices["train_unlabeled"],
        policy=policy_info,
    )

    result = SamplingResult(
        schema_version=SCHEMA_VERSION,
        created_at=created_at,
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_fingerprint,
        plan=plan_dict,
        indices=indices,
        refs=refs,
        masks={},
        stats=stats,
    )
    result.validate(n_train=n_train, n_test=n_test, n_nodes=None)
    return result


def _sample_graph(
    dataset: Any,
    *,
    plan: SamplingPlan,
    seed_split: int,
    seed_label: int,
    seed_imb: int,
    dataset_fingerprint: str,
    split_fingerprint: str,
    created_at: str,
    plan_dict: dict[str, Any],
) -> SamplingResult:
    y = np.asarray(dataset.train.y)
    n_nodes = int(y.shape[0])

    rng_split = np.random.default_rng(seed_split)

    official = getattr(dataset.train, "masks", None)
    use_official = (
        bool(plan.policy.use_official_graph_masks)
        and isinstance(official, dict)
        and {"train", "val", "test"}.issubset(set(official.keys()))
    )

    if use_official:
        train_mask = np.asarray(official["train"], dtype=bool)
        val_mask = np.asarray(official["val"], dtype=bool)
        test_mask = np.asarray(official["test"], dtype=bool)
    else:
        # generate node splits
        if isinstance(plan.split, HoldoutSplitSpec):
            parts = make_holdout_split(
                n_samples=n_nodes,
                y=y,
                test_fraction=float(plan.split.test_fraction),
                val_fraction=float(plan.split.val_fraction),
                stratify=bool(plan.split.stratify),
                rng=rng_split,
            )
        else:
            parts = make_kfold_split(
                n_samples=n_nodes,
                y=y,
                k=int(plan.split.k),
                fold=int(plan.split.fold),
                stratify=bool(plan.split.stratify),
                shuffle=bool(plan.split.shuffle),
                val_fraction=float(plan.split.val_fraction),
                rng=rng_split,
            )
        train_mask = _idx_to_mask(n_nodes, parts["train"])
        val_mask = _idx_to_mask(n_nodes, parts["val"])
        test_mask = _idx_to_mask(n_nodes, parts["test"])

    # labeling happens inside train_mask
    train_idx = np.where(train_mask)[0].astype(np.int64)

    rng_lab = np.random.default_rng(seed_label)
    labeled_idx = select_labeled(train_idx=train_idx, y=y, spec=plan.labeling, rng=rng_lab)

    rng_imb = np.random.default_rng(seed_imb)
    if plan.imbalance.apply_to == "labeled":
        labeled_idx = apply_imbalance(idx=labeled_idx, y=y, spec=plan.imbalance, rng=rng_imb)

    labeled_mask = _idx_to_mask(n_nodes, labeled_idx)
    unlabeled_mask = train_mask & ~labeled_mask

    masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
        "labeled": labeled_mask,
        "unlabeled": unlabeled_mask,
    }

    stats = build_graph_stats(masks=masks, y=y, labeled_idx=labeled_idx)

    result = SamplingResult(
        schema_version=SCHEMA_VERSION,
        created_at=created_at,
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_fingerprint,
        plan=plan_dict,
        indices={},
        refs={},
        masks=masks,
        stats=stats,
    )
    result.validate(n_train=0, n_test=None, n_nodes=n_nodes)
    return result


def _idx_to_mask(n: int, idx: np.ndarray) -> np.ndarray:
    m = np.zeros((n,), dtype=bool)
    if idx.size:
        m[idx] = True
    return m
