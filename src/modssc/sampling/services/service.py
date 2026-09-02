from __future__ import annotations

import logging
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from platformdirs import user_cache_dir

from modssc.runtime.paths import default_local_cache_subdir
from modssc.sampling.errors import MissingDatasetFingerprintError, SamplingValidationError
from modssc.sampling.fingerprint import stable_hash
from modssc.sampling.imbalance import apply_imbalance
from modssc.sampling.labeling import select_labeled
from modssc.sampling.partition_artifact import load_ordered_partition
from modssc.sampling.plan import HoldoutSplitSpec, PartitionSpec, SamplingPlan
from modssc.sampling.result import SamplingResult
from modssc.sampling.splitters import make_holdout_split, make_kfold_split
from modssc.sampling.stats import build_graph_stats, build_inductive_stats
from modssc.sampling.storage import load_split as _load_split
from modssc.sampling.storage import save_split as _save_split

SPLIT_CACHE_ENV = "MODSSC_SPLIT_CACHE_DIR"
CACHE_ROOT_ENV = "MODSSC_CACHE_ROOT"
logger = logging.getLogger(__name__)


def default_split_cache_dir() -> Path:
    override = os.environ.get(SPLIT_CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()

    root_override = os.environ.get(CACHE_ROOT_ENV)
    if root_override:
        return Path(root_override).expanduser().resolve() / "splits"

    local = default_local_cache_subdir("splits")
    if local is not None:
        return local

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

    component_seeds = plan.component_seeds.resolve(seed)
    seed_partition = component_seeds["partition"]
    seed_split = component_seeds["split"]
    seed_label = component_seeds["labeling"]
    seed_imb = component_seeds["imbalance"]

    plan_dict = plan.as_dict()
    schema_version = plan.fingerprint_schema_version()
    split_fingerprint = stable_hash(
        {
            "schema_version": schema_version,
            "dataset_fingerprint": ds_fp,
            "plan": plan.fingerprint_payload(),
            "seed": int(seed),
        }
    )

    created_at = datetime.now(UTC).isoformat()

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
    logger.debug("Sampling component seeds: %s", component_seeds)

    if is_graph:
        if plan.partition != PartitionSpec():
            raise ValueError("sampling.partition is not supported for graph datasets")
        result = _sample_graph(
            dataset,
            plan=plan,
            run_seed=int(seed),
            seed_split=seed_split,
            seed_label=seed_label,
            seed_imb=seed_imb,
            dataset_fingerprint=ds_fp,
            split_fingerprint=split_fingerprint,
            created_at=created_at,
            plan_dict=plan_dict,
            schema_version=schema_version,
        )
    else:
        result = _sample_inductive(
            dataset,
            plan=plan,
            run_seed=int(seed),
            seed_partition=seed_partition,
            seed_split=seed_split,
            seed_label=seed_label,
            seed_imb=seed_imb,
            dataset_fingerprint=ds_fp,
            split_fingerprint=split_fingerprint,
            created_at=created_at,
            plan_dict=plan_dict,
            schema_version=schema_version,
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
    run_seed: int,
    seed_partition: int,
    seed_split: int,
    seed_label: int,
    seed_imb: int,
    dataset_fingerprint: str,
    split_fingerprint: str,
    created_at: str,
    plan_dict: dict[str, Any],
    schema_version: int,
) -> SamplingResult:
    y_train = np.asarray(dataset.train.y)
    n_train = int(y_train.shape[0])

    has_official_test = getattr(dataset, "test", None) is not None
    y_test = None
    n_test = None
    if has_official_test:
        y_test = np.asarray(dataset.test.y)
        n_test = int(y_test.shape[0])

    artifact = plan.partition.ordered_indices_artifact
    if artifact is not None:
        if plan.imbalance.kind != "none":
            raise ValueError("sampling.imbalance must be 'none' with an ordered partition artifact")
        indices = load_ordered_partition(
            spec=artifact,
            run_seed=run_seed,
            y_train=y_train,
            n_test=n_test,
            dataset_fingerprint=dataset_fingerprint,
        )
        refs = {
            "train": "train",
            "val": "train",
            "test": artifact.test_ref,
            "train_labeled": "train",
            "train_unlabeled": "train",
        }
        policy_info = {
            "respect_official_test": bool(plan.policy.respect_official_test),
            "allow_override_official": bool(plan.policy.allow_override_official),
            "has_official_test": bool(has_official_test),
            "official_test_ignored": artifact.test_ref != "test",
            "test_ref": artifact.test_ref,
            "partition_source_n": n_train,
            "partition_selected_n": int(indices["train"].size),
            "partition_artifact_sha256": artifact.sha256,
            "unlabeled_pool": artifact.unlabeled_pool,
            "ordered_indices": True,
        }
        stats = build_inductive_stats(
            y_train=y_train,
            train_idx=indices["train"],
            val_idx=indices["val"],
            test_ref=artifact.test_ref,
            y_test=y_test,
            test_idx=indices["test"],
            labeled_idx=indices["train_labeled"],
            unlabeled_idx=indices["train_unlabeled"],
            policy=policy_info,
        )
        result = SamplingResult(
            schema_version=schema_version,
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

    partition_idx = _select_partition_indices(
        n_samples=n_train,
        y=y_train,
        spec=plan.partition,
        rng=np.random.default_rng(seed_partition),
    )
    pool_y = y_train[partition_idx]
    pool_n = int(partition_idx.shape[0])

    # Split on train indices only if official test is respected
    if has_official_test and plan.policy.respect_official_test:
        if not plan.policy.allow_override_official:
            test_idx = np.arange(n_test or 0, dtype=np.int64)
            test_ref = "test"
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
                    shuffle=bool(plan.split.shuffle),
                    test_size=0,
                    val_size=plan.split.val_size,
                    holdout_from=plan.split.holdout_from,
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
            train_idx = partition_idx[parts["train"]]
            val_idx = partition_idx[parts["val"]]
        else:
            rng = np.random.default_rng(seed_split)
            if isinstance(plan.split, HoldoutSplitSpec):
                parts = make_holdout_split(
                    n_samples=pool_n,
                    y=pool_y,
                    test_fraction=float(plan.split.test_fraction),
                    val_fraction=float(plan.split.val_fraction),
                    stratify=bool(plan.split.stratify),
                    rng=rng,
                    shuffle=bool(plan.split.shuffle),
                    test_size=plan.split.test_size,
                    val_size=plan.split.val_size,
                    holdout_from=plan.split.holdout_from,
                )
            else:
                parts = make_kfold_split(
                    n_samples=pool_n,
                    y=pool_y,
                    k=int(plan.split.k),
                    fold=int(plan.split.fold),
                    stratify=bool(plan.split.stratify),
                    shuffle=bool(plan.split.shuffle),
                    val_fraction=float(plan.split.val_fraction),
                    rng=rng,
                )
            train_idx = partition_idx[parts["train"]]
            val_idx = partition_idx[parts["val"]]
            test_idx = partition_idx[parts["test"]]
            test_ref = "train"
    else:
        rng = np.random.default_rng(seed_split)
        if isinstance(plan.split, HoldoutSplitSpec):
            parts = make_holdout_split(
                n_samples=pool_n,
                y=pool_y,
                test_fraction=float(plan.split.test_fraction),
                val_fraction=float(plan.split.val_fraction),
                stratify=bool(plan.split.stratify),
                rng=rng,
                shuffle=bool(plan.split.shuffle),
                test_size=plan.split.test_size,
                val_size=plan.split.val_size,
                holdout_from=plan.split.holdout_from,
            )
        else:
            parts = make_kfold_split(
                n_samples=pool_n,
                y=pool_y,
                k=int(plan.split.k),
                fold=int(plan.split.fold),
                stratify=bool(plan.split.stratify),
                shuffle=bool(plan.split.shuffle),
                val_fraction=float(plan.split.val_fraction),
                rng=rng,
            )
        train_idx = partition_idx[parts["train"]]
        val_idx = partition_idx[parts["val"]]
        test_idx = partition_idx[parts["test"]]
        test_ref = "train"

    # apply imbalance to train if requested
    rng_imb = np.random.default_rng(seed_imb)
    if plan.imbalance.kind == "none":
        train_idx_adj = train_idx
    elif plan.imbalance.apply_to == "train":
        train_idx_adj = apply_imbalance(idx=train_idx, y=y_train, spec=plan.imbalance, rng=rng_imb)
    else:
        train_idx_adj = train_idx

    rng_lab = _labeling_rng(seed_label, plan=plan)
    labeling_pool = partition_idx if plan.labeling.selection_scope == "partition" else train_idx_adj
    labeled = select_labeled(
        train_idx=labeling_pool,
        y=y_train,
        spec=plan.labeling,
        rng=rng_lab,
        run_seed=run_seed,
    )
    if np.setdiff1d(labeled, train_idx_adj).size:
        raise SamplingValidationError(
            "labeling.selection_scope='partition' selected an example outside the final "
            "train split; change the split seed/protocol or select labels from 'train'"
        )

    if plan.imbalance.kind != "none" and plan.imbalance.apply_to == "labeled":
        labeled_adj = apply_imbalance(idx=labeled, y=y_train, spec=plan.imbalance, rng=rng_imb)
        labeled = labeled_adj

    if plan.labeling.unlabeled_pool == "includes_labeled":
        unlabeled = np.asarray(train_idx_adj, dtype=np.int64).copy()
    else:
        unlabeled = np.setdiff1d(train_idx_adj, labeled, assume_unique=False)

    # Ensure train indices reflect imbalance apply_to=train
    train_idx_final = np.asarray(train_idx_adj, dtype=np.int64)

    indices = {
        "train": train_idx_final,
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
        "train_labeled": np.sort(labeled),
        "train_unlabeled": np.asarray(unlabeled, dtype=np.int64),
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
        "allow_override_official": bool(plan.policy.allow_override_official),
        "has_official_test": bool(has_official_test),
        "official_test_ignored": bool(
            has_official_test
            and plan.policy.respect_official_test
            and plan.policy.allow_override_official
        ),
        "test_ref": test_ref,
        "partition_source_n": n_train,
        "partition_selected_n": pool_n,
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
        schema_version=schema_version,
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


def _select_partition_indices(
    *,
    n_samples: int,
    y: np.ndarray,
    spec: PartitionSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.arange(n_samples, dtype=np.int64)
    if spec.ordering == "class_balanced_stream":
        indices = _class_balanced_stream_order(y)
    if spec.max_samples is None or spec.max_samples >= n_samples:
        return indices
    if spec.shuffle:
        indices = rng.permutation(indices)
    selected = indices[: spec.max_samples]
    if spec.ordering == "canonical":
        return np.sort(selected)
    return np.asarray(selected, dtype=np.int64)


def _class_balanced_stream_order(y: np.ndarray) -> np.ndarray:
    """Interleave a canonical stream while compensating class imbalance.

    The operation is dataset- and method-agnostic. It is useful for protocols
    that require a deterministic class-balanced stream before slicing exact
    train/validation pools.
    """

    labels = np.asarray(y)
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if classes.size == 0:
        return np.asarray([], dtype=np.int64)
    queues = [deque() for _ in range(int(classes.size))]
    positions = np.zeros(int(classes.size), dtype=np.int64)
    target = counts.astype(np.float64) / float(counts.max())
    ordered: list[int] = []
    for index, class_position in enumerate(inverse.tolist()):
        queues[int(class_position)].append(index)
        while True:
            denominator = max(int(positions.max()), 1)
            selected_class = int(np.argmax(target - positions / denominator))
            if not queues[selected_class]:
                break
            ordered.append(int(queues[selected_class].popleft()))
            positions[selected_class] += 1
    for queue in queues:
        ordered.extend(int(index) for index in queue)
    return np.asarray(ordered, dtype=np.int64)


def _labeling_rng(
    seed: int,
    *,
    plan: SamplingPlan,
) -> np.random.Generator | np.random.RandomState:
    if plan.labeling.rng_backend == "legacy_random_state":
        return np.random.RandomState(seed)
    return np.random.default_rng(seed)


def _sample_graph(
    dataset: Any,
    *,
    plan: SamplingPlan,
    run_seed: int,
    seed_split: int,
    seed_label: int,
    seed_imb: int,
    dataset_fingerprint: str,
    split_fingerprint: str,
    created_at: str,
    plan_dict: dict[str, Any],
    schema_version: int,
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
                shuffle=bool(plan.split.shuffle),
                test_size=plan.split.test_size,
                val_size=plan.split.val_size,
                holdout_from=plan.split.holdout_from,
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

    rng_lab = _labeling_rng(seed_label, plan=plan)
    labeled_idx = select_labeled(
        train_idx=train_idx,
        y=y,
        spec=plan.labeling,
        rng=rng_lab,
        run_seed=run_seed,
    )

    rng_imb = np.random.default_rng(seed_imb)
    if plan.imbalance.apply_to == "labeled":
        labeled_idx = apply_imbalance(idx=labeled_idx, y=y, spec=plan.imbalance, rng=rng_imb)

    labeled_mask = _idx_to_mask(n_nodes, labeled_idx)
    unlabeled_mask = (
        train_mask.copy()
        if plan.labeling.unlabeled_pool == "includes_labeled"
        else train_mask & ~labeled_mask
    )

    masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
        "labeled": labeled_mask,
        "unlabeled": unlabeled_mask,
    }

    stats = build_graph_stats(masks=masks, y=y, labeled_idx=labeled_idx)

    result = SamplingResult(
        schema_version=schema_version,
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
