from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bench.utils.hashing import derive_seed
from bench.utils.io import atomic_write_json


class HistoricalAcceptanceError(ValueError):
    """Raised when a historical seed sweep is incomplete or malformed."""


@dataclass(frozen=True)
class HistoricalProtocol:
    protocol_id: str
    method_id: str
    dataset_id: str
    dataset_fingerprint: str
    dataset_content_sha256: str
    expected_seeds: tuple[int, ...]
    target_error: float
    margin_absolute: float
    critical_unknowns: tuple[str, ...]
    secondary_target_errors: tuple[tuple[str, float], ...] = ()


_PROTOCOLS = {
    "self-training": HistoricalProtocol(
        protocol_id="paper:li-zhou-2005-setred-table3-wine-self-training",
        method_id="self_training",
        dataset_id="wine",
        dataset_fingerprint=("984da35a8465017e8fb1881e20932fc1ac8036e6736e425caeba76db467bb543"),
        dataset_content_sha256=("88e42e2b23ebb7dc5c3de3d4b258016178b7ab4591c54b5265cfd1d01a03f99a"),
        expected_seeds=tuple(range(1, 51)),
        target_error=0.079,
        margin_absolute=0.02,
        critical_unknowns=(
            "historical split seeds and rounding are unavailable",
            "nearest-neighbour scaling and confidence details are under-specified",
            "pool, quota, replenishment, and tie-breaking details are under-specified",
        ),
    ),
    "self-training-v2": HistoricalProtocol(
        protocol_id=("paper:li-zhou-2005-setred-table3-wine-self-training-confirmation-v2"),
        method_id="self_training",
        dataset_id="wine",
        dataset_fingerprint=("984da35a8465017e8fb1881e20932fc1ac8036e6736e425caeba76db467bb543"),
        dataset_content_sha256=("88e42e2b23ebb7dc5c3de3d4b258016178b7ab4591c54b5265cfd1d01a03f99a"),
        expected_seeds=tuple(range(51, 101)),
        target_error=0.079,
        margin_absolute=0.02,
        critical_unknowns=(
            "historical split seeds and rounding are unavailable",
            "the published pool size and exact confidence formula are unavailable",
            "dynamic labeled-only min-max scaling is a post-audit historical reconstruction",
        ),
    ),
    "co-training": HistoricalProtocol(
        protocol_id="paper:blum-mitchell-1998-webkb-course-table2",
        method_id="co_training",
        dataset_id="webkb_course_cotraining",
        dataset_fingerprint=("5a1d45139e2a1ccb17abf374fb6ec17dc7d0bb3f9ff7caf08935d7731bb80683"),
        dataset_content_sha256=("894e2f310924fd66239632029db7738b8e1fcd330ffb86cb201cf6937ed9a264"),
        expected_seeds=tuple(range(1, 6)),
        target_error=0.050,
        margin_absolute=0.02,
        critical_unknowns=(
            "the five historical split seeds are unavailable",
            "historical HTML and anchor-text tokenization is under-specified",
            "naive-Bayes smoothing, conflict, and tie-breaking details are under-specified",
        ),
        secondary_target_errors=(("fulltext", 0.062), ("inlinks", 0.116)),
    ),
    "co-training-v2": HistoricalProtocol(
        protocol_id="paper:blum-mitchell-1998-webkb-course-confirmation-v2",
        method_id="co_training",
        dataset_id="webkb_course_cotraining",
        dataset_fingerprint=("5a1d45139e2a1ccb17abf374fb6ec17dc7d0bb3f9ff7caf08935d7731bb80683"),
        dataset_content_sha256=("894e2f310924fd66239632029db7738b8e1fcd330ffb86cb201cf6937ed9a264"),
        expected_seeds=tuple(range(6, 11)),
        target_error=0.050,
        margin_absolute=0.02,
        critical_unknowns=(
            "the five historical split seeds are unavailable",
            "the exact 1998 tokenizer and feature selector are unavailable",
            "dynamic top-2000 mutual information and the Craven score are sourced reconstructions",
        ),
        secondary_target_errors=(("fulltext", 0.062), ("inlinks", 0.116)),
    ),
    "co-training-nigam": HistoricalProtocol(
        protocol_id="paper:nigam-ghani2000-webkb-table2",
        method_id="co_training",
        dataset_id="webkb_course_cotraining",
        dataset_fingerprint=("5a1d45139e2a1ccb17abf374fb6ec17dc7d0bb3f9ff7caf08935d7731bb80683"),
        dataset_content_sha256=("894e2f310924fd66239632029db7738b8e1fcd330ffb86cb201cf6937ed9a264"),
        expected_seeds=tuple(range(21, 31)),
        target_error=0.054,
        margin_absolute=0.02,
        critical_unknowns=(
            "the ten historical split seeds are unavailable",
            "historical HTML and anchor-text tokenization is under-specified",
            "the historical cross-view collision policy is unpublished; the ordered "
            "multiset policy is a reconstruction",
        ),
        secondary_target_errors=(("nb12", 0.130), ("nb788", 0.033)),
    ),
}

_FALLBACK_T_975 = {
    5: 2.7764451051977987,
    10: 2.2621571628540993,
    50: 2.009575234489209,
}


def _student_t_critical_95(n: int) -> tuple[float, str]:
    try:
        from scipy.stats import t
    except ImportError:
        try:
            return _FALLBACK_T_975[n], "prefixed_student_t"
        except KeyError as exc:
            raise HistoricalAcceptanceError(
                f"no dependency-free Student t quantile is registered for n={n}"
            ) from exc
    return float(t.ppf(0.975, df=n - 1)), "scipy.stats.t"


def _mapping(value: Any, *, field: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalAcceptanceError(f"{path}: {field} must be a mapping")
    return value


def _locked_value(
    values: Mapping[str, Any],
    *,
    key: str,
    expected: str,
    field: str,
    path: Path,
) -> str:
    value = values.get(key)
    if value != expected:
        raise HistoricalAcceptanceError(f"{path}: {field} must equal {expected!r}, got {value!r}")
    return expected


def _hex_digest(value: Any, *, length: int, field: str, path: Path) -> str:
    if not (
        isinstance(value, str)
        and len(value) == length
        and all(character.lower() in "0123456789abcdef" for character in value)
    ):
        raise HistoricalAcceptanceError(f"{path}: {field} must be a {length}-character hex digest")
    return value.lower()


def _environment_version(value: Any, *, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HistoricalAcceptanceError(f"{path}: versions.{field} must be a non-empty string")
    return value


def _metric_accuracy(metrics: Mapping[str, Any], *, split: str, path: Path) -> float:
    field = f"metrics.{split}"
    split_metrics = _mapping(metrics.get(split), field=field, path=path)
    accuracy = split_metrics.get("accuracy")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, int | float)
        or not math.isfinite(float(accuracy))
        or not 0.0 <= float(accuracy) <= 1.0
    ):
        raise HistoricalAcceptanceError(f"{path}: {field}.accuracy must be finite and in [0, 1]")
    return float(accuracy)


def _locked_subset(
    actual: Any,
    expected: Any,
    *,
    field: str,
    path: Path,
) -> None:
    """Require an exact value or recursively lock selected mapping keys."""

    if isinstance(expected, Mapping):
        values = _mapping(actual, field=field, path=path)
        for key, expected_value in expected.items():
            if key not in values:
                raise HistoricalAcceptanceError(f"{path}: {field}.{key} is required")
            _locked_subset(
                values[key],
                expected_value,
                field=f"{field}.{key}",
                path=path,
            )
        return
    if actual != expected:
        raise HistoricalAcceptanceError(f"{path}: {field} must equal {expected!r}, got {actual!r}")


def _critical_config(protocol: HistoricalProtocol, *, seed: int) -> dict[str, Any]:
    common = {
        "run": {
            "seed": seed,
            "seeds": None,
            "model_seed": None,
            "seeded_sections": ["sampling", "preprocess"],
            "fail_fast": True,
            "benchmark_mode": False,
        },
        "dataset": {"id": protocol.dataset_id, "download": False, "options": {}},
        "evaluation": {
            "split_for_model_selection": None,
            "report_splits": ["test"],
            "metrics": ["accuracy", "macro_f1"],
        },
        "graph": None,
        "augmentation": None,
        "search": None,
    }
    if protocol.method_id == "self_training":
        confirmation_v2 = protocol.protocol_id.endswith("-confirmation-v2")
        preprocess_steps = [
            {"id": "labels.encode"},
            {"id": "core.ensure_2d"},
        ]
        if not confirmation_v2:
            preprocess_steps.append({"id": "tabular.standard_scaler"})
        preprocess_steps.append({"id": "core.to_numpy"})
        method_params = {
            "classifier_id": "knn",
            "classifier_backend": "numpy",
            "classifier_params": {
                "k": 1,
                "metric": "euclidean",
                "weights": "uniform",
            },
            "max_iter": 40,
            "confidence_threshold": None,
            "max_new_labels": None,
            "min_new_labels": 1,
            "use_group_propagation": False,
            "selection_strategy": "li_zhou_2005_1nn_distance",
            "paper_pool_size_unspecified": 75 if confirmation_v2 else None,
            "paper_candidates_per_class_unspecified": 1,
            "paper_distance_confidence_unspecified": (
                "nearest_neighbor_distance" if confirmation_v2 else "margin"
            ),
        }
        if confirmation_v2:
            method_params["paper_feature_scaling_unspecified"] = "dynamic_labeled_minmax"
        common.update(
            {
                "sampling": {
                    "seed": seed,
                    "plan": {
                        "split": {
                            "kind": "holdout",
                            "test_fraction": 0.25,
                            "val_fraction": 0.0,
                            "stratify": True,
                            "shuffle": True,
                        },
                        "labeling": {
                            "mode": "fraction",
                            "value": 0.1,
                            "strategy": "proportional",
                            "min_per_class": 1,
                            "per_class": False,
                            "fixed_indices": None,
                        },
                        "imbalance": {"kind": "none"},
                        "policy": {
                            "respect_official_test": True,
                            "use_official_graph_masks": True,
                            "allow_override_official": False,
                        },
                    },
                },
                "preprocess": {
                    "seed": seed,
                    "fit_on": "train_labeled",
                    "cache": True,
                    "plan": {
                        "output_key": "features.X",
                        "steps": preprocess_steps,
                    },
                },
                "method": {
                    "kind": "inductive",
                    "method_id": "self_training",
                    "profile": protocol.protocol_id,
                    "model": None,
                    "device": {"device": "cpu", "dtype": "float32"},
                    "params": method_params,
                },
                "views": None,
            }
        )
        return common

    co_v2 = protocol.protocol_id == "paper:blum-mitchell-1998-webkb-course-confirmation-v2"
    co_nigam = protocol.protocol_id == "paper:nigam-ghani2000-webkb-table2"
    method_params = {
        "classifier_id": "multinomial_nb",
        "classifier_backend": "sklearn",
        "classifier_params": {"alpha": 1.0, "fit_prior": True},
        "view_keys": ["fulltext", "inlinks"],
        "protocol": (
            "shared_pool_exhaustive_multiset"
            if co_nigam
            else "fixed_pool_binary_feature_selection"
            if co_v2
            else "fixed_pool_binary"
        ),
        "p": 1,
        "n": 3,
        "u": 75,
        "k": 0 if co_nigam else 30,
        "positive_label": 1,
        "negative_label": 0,
        "confidence_threshold": None,
    }
    if co_v2:
        method_params.update(
            {
                "dynamic_feature_selection": "mutual_information_presence",
                "feature_selection_max_features": 2000,
                "selection_score": "craven_1998_normalized_nb",
            }
        )
    elif co_nigam:
        method_params.update(
            {
                "dynamic_feature_selection": "none",
                "feature_selection_max_features": None,
                "selection_score": "posterior_probability",
            }
        )
    vectorizer_params = {"dense": True, "strip_html": True}
    if co_v2 or co_nigam:
        vectorizer_params["min_df"] = 1

    labeling = {
        "mode": "count",
        "value": 12,
        "strategy": "proportional",
        "min_per_class": 1,
        "per_class": False,
        "fixed_indices": None,
    }
    if co_nigam:
        labeling.update(
            {
                "strategy": "random",
                "min_per_class": 0,
                "class_counts": {"0": 9, "1": 3},
                "selection_order": "permutation",
            }
        )

    common["run"]["seeded_sections"] = ["sampling", "preprocess", "views"]
    common.update(
        {
            "sampling": {
                "seed": seed,
                "plan": {
                    "split": {
                        "kind": "holdout",
                        "test_fraction": 263 / 1051,
                        "val_fraction": 0.0,
                        "stratify": not (co_v2 or co_nigam),
                        "shuffle": True,
                    },
                    "labeling": labeling,
                    "imbalance": {"kind": "none"},
                    "policy": {
                        "respect_official_test": True,
                        "use_official_graph_masks": True,
                        "allow_override_official": False,
                    },
                },
            },
            "preprocess": {
                "seed": seed,
                "fit_on": "train",
                "cache": True,
                "plan": {
                    "output_key": "features.X",
                    "steps": [
                        {"id": "labels.encode"},
                        {"id": "core.copy_raw"},
                        {"id": "core.to_numpy"},
                    ],
                },
            },
            "method": {
                "kind": "inductive",
                "method_id": "co_training",
                "profile": protocol.protocol_id,
                "model": None,
                "device": {"device": "cpu", "dtype": "float32"},
                "params": method_params,
            },
            "views": {
                "seed": seed,
                "plan": {
                    "views": [
                        {
                            "name": "fulltext",
                            "input_columns": {"mode": "indices", "indices": [0]},
                            "preprocess": {
                                "output_key": "features.X",
                                "steps": [
                                    {"id": "text.ensure_strings"},
                                    {
                                        "id": "text.count_vectorizer",
                                        "params": vectorizer_params,
                                    },
                                ],
                            },
                        },
                        {
                            "name": "inlinks",
                            "input_columns": {"mode": "indices", "indices": [1]},
                            "preprocess": {
                                "output_key": "features.X",
                                "steps": [
                                    {"id": "text.ensure_strings"},
                                    {
                                        "id": "text.count_vectorizer",
                                        "params": vectorizer_params,
                                    },
                                ],
                            },
                        },
                    ]
                },
            },
        }
    )
    return common


def _expected_sampling_contract(protocol: HistoricalProtocol, *, seed: int) -> dict[str, Any]:
    if protocol.method_id == "self_training":
        stats = {
            "train": {"n": 134, "classes": {"1": 44, "2": 54, "3": 36}},
            "val": {"n": 0, "classes": {}},
            "test": {"n": 44, "classes": {"1": 15, "2": 17, "3": 12}},
            "train_labeled": {"n": 13, "classes": {"1": 4, "2": 5, "3": 4}},
            "train_unlabeled": {"n": 121},
        }
    elif protocol.protocol_id == "paper:nigam-ghani2000-webkb-table2":
        stats = {
            "train": {"n": 788},
            "val": {"n": 0},
            "test": {"n": 263},
            "train_labeled": {"n": 12, "classes": {"0": 9, "1": 3}},
            "train_unlabeled": {"n": 776},
        }
    elif protocol.protocol_id.endswith("-confirmation-v2"):
        # The random, non-stratified v2 split legitimately changes class counts
        # with the seed. Lock only structural sizes here; the exact indices are
        # authenticated and deterministically replayed below.
        stats = {
            "train": {"n": 788},
            "val": {"n": 0},
            "test": {"n": 263},
            "train_labeled": {"n": 12},
            "train_unlabeled": {"n": 776},
        }
    else:
        stats = {
            "train": {"n": 788, "classes": {"0": 616, "1": 172}},
            "val": {"n": 0, "classes": {}},
            "test": {"n": 263, "classes": {"0": 205, "1": 58}},
            "train_labeled": {"n": 12, "classes": {"0": 9, "1": 3}},
            "train_unlabeled": {"n": 776},
        }
    return {"seed": seed, "stats": stats}


def _co_v2_sampling_plan() -> dict[str, Any]:
    """Canonical expanded sampling plan participating in the split fingerprint."""

    return {
        "split": {
            "kind": "holdout",
            "test_fraction": 263 / 1051,
            "val_fraction": 0.0,
            "stratify": False,
            "shuffle": True,
        },
        "labeling": {
            "mode": "count",
            "value": 12,
            "per_class": False,
            "min_per_class": 1,
            "strategy": "proportional",
            "fixed_indices": None,
            "selection_order": "choice",
        },
        "imbalance": {
            "kind": "none",
            "apply_to": "train",
            "max_per_class": None,
            "alpha": None,
            "min_per_class": 1,
        },
        "policy": {
            "respect_official_test": True,
            "use_official_graph_masks": True,
            "allow_override_official": False,
            "merge_official_splits": False,
        },
    }


def _co_v2_split_fingerprint(*, seed: int) -> str:
    payload = {
        "schema_version": 1,
        "dataset_fingerprint": _PROTOCOLS["co-training-v2"].dataset_fingerprint,
        "plan": _co_v2_sampling_plan(),
        "seed": int(seed),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _co_v2_expected_indices(*, seed: int) -> dict[str, np.ndarray]:
    """Replay the v2 split from its seed against the canonical WebKB row order."""

    from modssc.sampling.fingerprint import derive_seed
    from modssc.sampling.labeling import select_labeled
    from modssc.sampling.plan import LabelingSpec
    from modssc.sampling.splitters import make_holdout_split

    labels = np.concatenate([np.zeros((821,), dtype=np.int64), np.ones((230,), dtype=np.int64)])
    parts = make_holdout_split(
        n_samples=1051,
        y=labels,
        test_fraction=263 / 1051,
        val_fraction=0.0,
        stratify=False,
        rng=np.random.default_rng(derive_seed(seed, "split")),
    )
    train = np.asarray(parts["train"], dtype=np.int64)
    labeled = select_labeled(
        train_idx=train,
        y=labels,
        spec=LabelingSpec(
            mode="count",
            value=12,
            per_class=False,
            min_per_class=1,
            strategy="proportional",
            fixed_indices=None,
            selection_order="choice",
        ),
        rng=np.random.default_rng(derive_seed(seed, "labeling")),
        run_seed=seed,
    )
    return {
        "train": np.sort(train),
        "val": np.sort(np.asarray(parts["val"], dtype=np.int64)),
        "test": np.sort(np.asarray(parts["test"], dtype=np.int64)),
        "train_labeled": np.sort(labeled),
        "train_unlabeled": np.setdiff1d(train, labeled, assume_unique=True),
    }


def _co_nigam_sampling_plan() -> dict[str, Any]:
    """Canonical Nigam--Ghani sampling plan participating in its split fingerprint."""

    return {
        "split": {
            "kind": "holdout",
            "test_fraction": 263 / 1051,
            "val_fraction": 0.0,
            "stratify": False,
            "shuffle": True,
        },
        "labeling": {
            "mode": "count",
            "value": 12,
            "per_class": False,
            "min_per_class": 0,
            "strategy": "random",
            "fixed_indices": None,
            "selection_order": "permutation",
            "class_counts": {"0": 9, "1": 3},
        },
        "imbalance": {
            "kind": "none",
            "apply_to": "train",
            "max_per_class": None,
            "alpha": None,
            "min_per_class": 1,
        },
        "policy": {
            "respect_official_test": True,
            "use_official_graph_masks": True,
            "allow_override_official": False,
            "merge_official_splits": False,
        },
    }


def _co_nigam_split_fingerprint(*, seed: int) -> str:
    payload = {
        "schema_version": 1,
        "dataset_fingerprint": _PROTOCOLS["co-training-nigam"].dataset_fingerprint,
        "plan": _co_nigam_sampling_plan(),
        "seed": int(seed),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _co_nigam_expected_indices(*, seed: int) -> dict[str, np.ndarray]:
    """Replay the Nigam--Ghani split from its seed against canonical WebKB order."""

    from modssc.sampling.fingerprint import derive_seed
    from modssc.sampling.labeling import select_labeled
    from modssc.sampling.plan import LabelingSpec
    from modssc.sampling.splitters import make_holdout_split

    labels = np.concatenate([np.zeros((821,), dtype=np.int64), np.ones((230,), dtype=np.int64)])
    parts = make_holdout_split(
        n_samples=1051,
        y=labels,
        test_fraction=263 / 1051,
        val_fraction=0.0,
        stratify=False,
        rng=np.random.default_rng(derive_seed(seed, "split")),
    )
    train = np.asarray(parts["train"], dtype=np.int64)
    labeled = select_labeled(
        train_idx=train,
        y=labels,
        spec=LabelingSpec(
            mode="count",
            value=12,
            per_class=False,
            min_per_class=0,
            strategy="random",
            fixed_indices=None,
            class_counts={"0": 9, "1": 3},
            selection_order="permutation",
        ),
        rng=np.random.default_rng(derive_seed(seed, "labeling")),
        run_seed=seed,
    )
    return {
        "train": np.sort(train),
        "val": np.sort(np.asarray(parts["val"], dtype=np.int64)),
        "test": np.sort(np.asarray(parts["test"], dtype=np.int64)),
        "train_labeled": np.sort(labeled),
        "train_unlabeled": np.setdiff1d(train, labeled, assume_unique=True),
    }


def _validate_co_v2_replay(
    *,
    replay_root: Path,
    run_json_path: Path,
    seed: int,
    split_fingerprint: str,
) -> None:
    split_path = replay_root / "split.json"
    try:
        split_values = _mapping(
            json.loads(split_path.read_text(encoding="utf-8")),
            field="replay split",
            path=split_path,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalAcceptanceError(
            f"{run_json_path}: cannot read deterministic replay split"
        ) from exc
    expected_plan = _co_v2_sampling_plan()
    _locked_subset(
        split_values,
        {
            "schema_version": 1,
            "dataset_fingerprint": _PROTOCOLS["co-training-v2"].dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "mode": "inductive",
            "plan": expected_plan,
            "refs": {
                "train": "train",
                "val": "train",
                "test": "train",
                "train_labeled": "train",
                "train_unlabeled": "train",
            },
        },
        field="replay split",
        path=split_path,
    )
    if split_values.get("plan") != expected_plan:
        raise HistoricalAcceptanceError(f"{run_json_path}: Co-Training v2 replay plan is not exact")
    deterministic_fingerprint = _co_v2_split_fingerprint(seed=seed)
    if split_fingerprint != deterministic_fingerprint:
        raise HistoricalAcceptanceError(
            f"{run_json_path}: Co-Training v2 split fingerprint is not deterministic for seed {seed}"
        )

    arrays_path = replay_root / "arrays.npz"
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "idx__train",
                "idx__val",
                "idx__test",
                "idx__train_labeled",
                "idx__train_unlabeled",
            }:
                raise HistoricalAcceptanceError(
                    f"{run_json_path}: Co-Training v2 replay arrays have unexpected keys"
                )
            observed = {
                key.removeprefix("idx__"): np.asarray(archive[key]) for key in archive.files
            }
    except HistoricalAcceptanceError:
        raise
    except (OSError, ValueError) as exc:
        raise HistoricalAcceptanceError(
            f"{run_json_path}: cannot read Co-Training v2 replay arrays"
        ) from exc

    expected = _co_v2_expected_indices(seed=seed)
    for name, expected_indices in expected.items():
        indices = observed[name]
        if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
            raise HistoricalAcceptanceError(
                f"{run_json_path}: Co-Training v2 {name} indices must be a 1D integer array"
            )
        if not np.array_equal(indices, expected_indices):
            raise HistoricalAcceptanceError(
                f"{run_json_path}: Co-Training v2 {name} indices do not replay from seed {seed}"
            )


def _validate_co_nigam_replay(
    *,
    replay_root: Path,
    run_json_path: Path,
    seed: int,
    split_fingerprint: str,
) -> None:
    split_path = replay_root / "split.json"
    try:
        split_values = _mapping(
            json.loads(split_path.read_text(encoding="utf-8")),
            field="replay split",
            path=split_path,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalAcceptanceError(
            f"{run_json_path}: cannot read deterministic Nigam-Ghani replay split"
        ) from exc
    expected_plan = _co_nigam_sampling_plan()
    _locked_subset(
        split_values,
        {
            "schema_version": 1,
            "dataset_fingerprint": _PROTOCOLS["co-training-nigam"].dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "mode": "inductive",
            "plan": expected_plan,
            "refs": {
                "train": "train",
                "val": "train",
                "test": "train",
                "train_labeled": "train",
                "train_unlabeled": "train",
            },
        },
        field="replay split",
        path=split_path,
    )
    if split_values.get("plan") != expected_plan:
        raise HistoricalAcceptanceError(
            f"{run_json_path}: Nigam-Ghani Co-Training replay plan is not exact"
        )
    deterministic_fingerprint = _co_nigam_split_fingerprint(seed=seed)
    if split_fingerprint != deterministic_fingerprint:
        raise HistoricalAcceptanceError(
            f"{run_json_path}: Nigam-Ghani Co-Training split fingerprint is not "
            f"deterministic for seed {seed}"
        )

    arrays_path = replay_root / "arrays.npz"
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "idx__train",
                "idx__val",
                "idx__test",
                "idx__train_labeled",
                "idx__train_unlabeled",
            }:
                raise HistoricalAcceptanceError(
                    f"{run_json_path}: Nigam-Ghani Co-Training replay arrays have unexpected keys"
                )
            observed = {
                key.removeprefix("idx__"): np.asarray(archive[key]) for key in archive.files
            }
    except HistoricalAcceptanceError:
        raise
    except (OSError, ValueError) as exc:
        raise HistoricalAcceptanceError(
            f"{run_json_path}: cannot read Nigam-Ghani Co-Training replay arrays"
        ) from exc

    expected = _co_nigam_expected_indices(seed=seed)
    for name, expected_indices in expected.items():
        indices = observed[name]
        if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
            raise HistoricalAcceptanceError(
                f"{run_json_path}: Nigam-Ghani Co-Training {name} indices must be a "
                "1D integer array"
            )
        if not np.array_equal(indices, expected_indices):
            raise HistoricalAcceptanceError(
                f"{run_json_path}: Nigam-Ghani Co-Training {name} indices do not replay "
                f"from seed {seed}"
            )


def _validate_replay(
    *,
    run_json_path: Path,
    sampling: Mapping[str, Any],
    protocol: HistoricalProtocol,
    split_fingerprint: str,
    seed: int | None = None,
) -> str:
    replay = _mapping(sampling.get("replay"), field="artifacts.sampling.replay", path=run_json_path)
    _locked_subset(
        replay,
        {
            "format": "modssc.sampling.storage.v1",
            "path": "sampling_split",
            "manifest": "MANIFEST.json",
        },
        field="artifacts.sampling.replay",
        path=run_json_path,
    )
    expected_manifest_sha = _hex_digest(
        replay.get("manifest_sha256"),
        length=64,
        field="artifacts.sampling.replay.manifest_sha256",
        path=run_json_path,
    )
    replay_root = (run_json_path.parent / "sampling_split").resolve()
    try:
        replay_root.relative_to(run_json_path.parent.resolve())
    except ValueError as exc:  # pragma: no cover - fixed literal guarded above
        raise HistoricalAcceptanceError(
            f"{run_json_path}: replay path escapes the run directory"
        ) from exc
    manifest_path = replay_root / "MANIFEST.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalAcceptanceError(
            f"{run_json_path}: cannot authenticate replay manifest"
        ) from exc
    observed_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_manifest_sha != expected_manifest_sha:
        raise HistoricalAcceptanceError(
            f"{run_json_path}: replay manifest SHA-256 does not match run.json"
        )
    manifest_values = _mapping(manifest, field="replay manifest", path=manifest_path)
    _locked_subset(
        manifest_values,
        {
            "format": "modssc.sampling.storage.v1",
            "schema_version": 1,
            "dataset_fingerprint": protocol.dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
        },
        field="replay manifest",
        path=manifest_path,
    )
    files = _mapping(
        manifest_values.get("files"), field="replay manifest.files", path=manifest_path
    )
    if set(files) != {"arrays.npz", "split.json"}:
        raise HistoricalAcceptanceError(
            f"{manifest_path}: replay manifest.files must contain arrays.npz and split.json"
        )
    for name in ("arrays.npz", "split.json"):
        record = _mapping(files[name], field=f"replay manifest.files.{name}", path=manifest_path)
        expected_sha = _hex_digest(
            record.get("sha256"),
            length=64,
            field=f"replay manifest.files.{name}.sha256",
            path=manifest_path,
        )
        try:
            observed_sha = hashlib.sha256((replay_root / name).read_bytes()).hexdigest()
        except OSError as exc:
            raise HistoricalAcceptanceError(
                f"{run_json_path}: cannot authenticate replay file {name}"
            ) from exc
        if observed_sha != expected_sha:
            raise HistoricalAcceptanceError(f"{run_json_path}: replay file {name} SHA-256 mismatch")
    if protocol.protocol_id == "paper:blum-mitchell-1998-webkb-course-confirmation-v2":
        if seed is None:  # pragma: no cover - guarded by the run contract
            raise HistoricalAcceptanceError(
                f"{run_json_path}: Co-Training v2 replay requires its run seed"
            )
        _validate_co_v2_replay(
            replay_root=replay_root,
            run_json_path=run_json_path,
            seed=seed,
            split_fingerprint=split_fingerprint,
        )
    elif protocol.protocol_id == "paper:nigam-ghani2000-webkb-table2":
        if seed is None:  # pragma: no cover - guarded by the run contract
            raise HistoricalAcceptanceError(
                f"{run_json_path}: Nigam-Ghani Co-Training replay requires its run seed"
            )
        _validate_co_nigam_replay(
            replay_root=replay_root,
            run_json_path=run_json_path,
            seed=seed,
            split_fingerprint=split_fingerprint,
        )
    return expected_manifest_sha


def _int_field(values: Mapping[str, Any], key: str, *, field: str, path: Path) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HistoricalAcceptanceError(f"{path}: {field}.{key} must be an integer")
    return int(value)


def _list_field(values: Mapping[str, Any], key: str, *, field: str, path: Path) -> list[Any]:
    value = values.get(key)
    if not isinstance(value, list):
        raise HistoricalAcceptanceError(f"{path}: {field}.{key} must be a list")
    return value


def _validate_self_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    path: Path,
    protocol: HistoricalProtocol | None = None,
    expected_seed: int | None = None,
) -> None:
    protocol = protocol or _PROTOCOLS["self-training"]
    confirmation_v2 = protocol.protocol_id.endswith("-confirmation-v2")
    field = "artifacts.method.diagnostics"
    _locked_subset(
        diagnostics,
        {
            "protocol": "li_zhou_2005_1nn_distance",
            "initial_labeled_size": 13,
            "initial_unlabeled_count": 121,
        },
        field=field,
        path=path,
    )
    if expected_seed is not None:
        _locked_subset(
            diagnostics,
            {"seed": expected_seed},
            field=field,
            path=path,
        )
    if confirmation_v2:
        _locked_subset(
            diagnostics,
            {
                "selection_parameters": {
                    "paper_pool_size_unspecified": 75,
                    "paper_candidates_per_class_unspecified": 1,
                    "paper_distance_confidence_unspecified": "nearest_neighbor_distance",
                    "paper_feature_scaling_unspecified": "dynamic_labeled_minmax",
                }
            },
            field=field,
            path=path,
        )
    n_iter = _int_field(diagnostics, "n_iter", field=field, path=path)
    final_labeled = _int_field(diagnostics, "final_labeled_size", field=field, path=path)
    remaining = _int_field(diagnostics, "remaining_unlabeled_count", field=field, path=path)
    added = _int_field(diagnostics, "pseudo_labels_added", field=field, path=path)
    if not 0 <= n_iter <= 40:
        raise HistoricalAcceptanceError(f"{path}: Self-Training n_iter must be in [0, 40]")
    if final_labeled != 13 + added or remaining != 121 - added or not 0 <= added <= 121:
        raise HistoricalAcceptanceError(f"{path}: Self-Training final L/U sizes are inconsistent")
    trace = _list_field(diagnostics, "round_trace", field=field, path=path)
    if len(trace) != n_iter:
        raise HistoricalAcceptanceError(f"{path}: Self-Training round_trace length is inconsistent")

    labeled_before = 13
    unlabeled_before = 121
    accepted_total = 0
    previously_accepted: set[int] = set()
    previous_pool: set[int] | None = None
    previous_round_accepted: set[int] = set()
    for iteration, raw_round in enumerate(trace):
        round_values = _mapping(
            raw_round,
            field=f"{field}.round_trace[{iteration}]",
            path=path,
        )
        round_field = f"{field}.round_trace[{iteration}]"
        _locked_subset(
            round_values,
            {
                "iteration": iteration,
                "labeled_before": labeled_before,
                "unlabeled_before": unlabeled_before,
            },
            field=round_field,
            path=path,
        )
        accepted_indices = _list_field(
            round_values, "accepted_indices", field=round_field, path=path
        )
        accepted_labels = _list_field(round_values, "accepted_labels", field=round_field, path=path)
        candidates = _list_field(round_values, "candidate_indices", field=round_field, path=path)
        candidate_labels = _list_field(
            round_values, "candidate_labels", field=round_field, path=path
        )
        pool = _list_field(round_values, "pool_indices", field=round_field, path=path)
        if not (
            len(accepted_indices)
            == len(accepted_labels)
            == len(candidates)
            == len(candidate_labels)
            <= 3
        ):
            raise HistoricalAcceptanceError(f"{path}: Self-Training round quotas are inconsistent")
        if accepted_indices != candidates or accepted_labels != candidate_labels:
            raise HistoricalAcceptanceError(
                f"{path}: Self-Training paper candidates must all be accepted"
            )
        expected_pool_size = min(75, unlabeled_before) if confirmation_v2 else unlabeled_before
        pool_set = {int(value) for value in pool}
        accepted_set = {int(value) for value in accepted_indices}
        if (
            len(pool) != expected_pool_size
            or len(pool_set) != len(pool)
            or any(value < 0 or value >= 121 for value in pool_set)
            or len(accepted_set) != len(accepted_indices)
            or not accepted_set.issubset(pool_set)
            or previously_accepted.intersection(pool_set)
        ):
            raise HistoricalAcceptanceError(f"{path}: Self-Training pool trace is inconsistent")
        if confirmation_v2 and previous_pool is not None:
            retained = previous_pool - previous_round_accepted
            if not retained.issubset(pool_set):
                raise HistoricalAcceptanceError(
                    f"{path}: Self-Training v2 persistent pool was not retained"
                )
        accepted_count = len(accepted_indices)
        labeled_after = _int_field(round_values, "labeled_after", field=round_field, path=path)
        remaining_after = _int_field(
            round_values, "remaining_unlabeled", field=round_field, path=path
        )
        if labeled_after != labeled_before + accepted_count:
            raise HistoricalAcceptanceError(
                f"{path}: Self-Training labeled trajectory is inconsistent"
            )
        if remaining_after != unlabeled_before - accepted_count:
            raise HistoricalAcceptanceError(
                f"{path}: Self-Training unlabeled trajectory is inconsistent"
            )
        labeled_before = labeled_after
        unlabeled_before = remaining_after
        accepted_total += accepted_count
        previously_accepted.update(accepted_set)
        previous_pool = pool_set
        previous_round_accepted = accepted_set
    if accepted_total != added or labeled_before != final_labeled or unlabeled_before != remaining:
        raise HistoricalAcceptanceError(f"{path}: Self-Training aggregate trace is inconsistent")


def _validate_co_feature_diagnostic(
    values: Mapping[str, Any],
    *,
    count_key: str,
    digest_key: str,
    maximum_key: str,
    field: str,
    path: Path,
) -> None:
    count = _int_field(values, count_key, field=field, path=path)
    if not 1 <= count <= 2000:
        raise HistoricalAcceptanceError(f"{path}: {field}.{count_key} must be in [1, 2000]")
    _hex_digest(values.get(digest_key), length=64, field=f"{field}.{digest_key}", path=path)
    maximum = values.get(maximum_key)
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int | float)
        or not math.isfinite(float(maximum))
        or float(maximum) < 0.0
    ):
        raise HistoricalAcceptanceError(
            f"{path}: {field}.{maximum_key} must be finite and non-negative"
        )


def _validate_co_nigam_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    path: Path,
    expected_seed: int | None,
) -> None:
    field = "artifacts.method.diagnostics"
    expected = {
        "protocol": "shared_pool_exhaustive_multiset",
        "p": 1,
        "n": 3,
        "u": 75,
        "k": 0,
        "negative_label": 0,
        "positive_label": 1,
        "shared_labeled_multiset": True,
        "overlap_policy": "ordered_multiset_view1_then_view2",
        "views_select_from_same_pre_round_pool": True,
        "selection_score_space": "posterior_probability",
        "combination_score_space": "summed_log_probability",
        "probability_underflow_safe": True,
        "unique_pseudo_labeled_examples": 776,
        "remaining_unlabeled_count": 0,
        "remaining_unlabeled_indices": [],
        "initial_labeled_size": 12,
        "initial_unlabeled_count": 776,
        "initial_class_counts": {"0": 9, "1": 3},
        "termination": "unlabeled_exhausted",
        "addition_policy": "ordered_multiset_view1_then_view2",
        "paper_confidence": "posterior_probability",
        "ranking_space": "log_posterior_probability",
        "word_likelihood_smoothing": "add_one",
        "class_prior_smoothing": "add_one",
        "dynamic_feature_selection": "none",
        "selection_diagnostics_scope": "training_and_pseudo_labels_only",
        "test_metrics_used_for_protocol_selection": False,
        "supervised_controls": {
            "nb12_training_size": 12,
            "nb788_training_size": 788,
            "feature_space": "concatenated_namespaced_views",
            "class_prior_smoothing": "add_one",
            "test_metrics_used_for_protocol_selection": False,
        },
    }
    if expected_seed is not None:
        expected["seed"] = expected_seed
    _locked_subset(diagnostics, expected, field=field, path=path)

    initial_pool = _list_field(diagnostics, "initial_pool_indices", field=field, path=path)
    if (
        len(initial_pool) != 75
        or any(isinstance(value, bool) or not isinstance(value, int) for value in initial_pool)
        or len(set(initial_pool)) != 75
        or any(value < 0 or value >= 776 for value in initial_pool)
    ):
        raise HistoricalAcceptanceError(
            f"{path}: Nigam-Ghani Co-Training initial pool must contain 75 unique indices"
        )

    trace = _list_field(diagnostics, "round_trace", field=field, path=path)
    n_iter = _int_field(diagnostics, "n_iter", field=field, path=path)
    if n_iter != len(trace) or not 97 <= n_iter <= 194:
        raise HistoricalAcceptanceError(
            f"{path}: Nigam-Ghani Co-Training n_iter must match a 97--194 round_trace"
        )

    proposal_total1 = _int_field(
        diagnostics, "pseudo_label_proposals_view1", field=field, path=path
    )
    proposal_total2 = _int_field(
        diagnostics, "pseudo_label_proposals_view2", field=field, path=path
    )
    added_total = _int_field(diagnostics, "pseudo_labels_added_to_shared_l", field=field, path=path)
    received_total1 = _int_field(
        diagnostics, "pseudo_labels_received_by_view1", field=field, path=path
    )
    received_total2 = _int_field(
        diagnostics, "pseudo_labels_received_by_view2", field=field, path=path
    )
    final_labeled = _int_field(diagnostics, "final_labeled_size", field=field, path=path)
    overlap_total = _int_field(diagnostics, "overlap_count", field=field, path=path)
    duplicate_total = _int_field(
        diagnostics, "duplicate_multiset_additions", field=field, path=path
    )
    same_label_total = _int_field(diagnostics, "same_label_overlap_count", field=field, path=path)
    conflicting_total = _int_field(diagnostics, "conflicting_overlap_count", field=field, path=path)
    for key, value in (
        ("pseudo_label_proposals_view1", proposal_total1),
        ("pseudo_label_proposals_view2", proposal_total2),
        ("pseudo_labels_added_to_shared_l", added_total),
        ("pseudo_labels_received_by_view1", received_total1),
        ("pseudo_labels_received_by_view2", received_total2),
        ("final_labeled_size", final_labeled),
        ("overlap_count", overlap_total),
        ("duplicate_multiset_additions", duplicate_total),
        ("same_label_overlap_count", same_label_total),
        ("conflicting_overlap_count", conflicting_total),
    ):
        if value < 0:
            raise HistoricalAcceptanceError(f"{path}: {field}.{key} must be non-negative")

    current_pool = [int(value) for value in initial_pool]
    seen_in_pool = set(current_pool)
    promoted: set[int] = set()
    replenished_total = 0
    traced_proposals1 = 0
    traced_proposals2 = 0
    traced_additions = 0
    traced_overlap = 0
    traced_same_label = 0
    traced_conflicts = 0
    training_size = 12
    for index, raw_round in enumerate(trace):
        round_values = _mapping(
            raw_round,
            field=f"{field}.round_trace[{index}]",
            path=path,
        )
        round_field = f"{field}.round_trace[{index}]"
        expected_replenished_count = min(8, 701 - replenished_total)
        _locked_subset(
            round_values,
            {
                "round": index + 1,
                "round_status": "completed",
                "overlap_policy": "ordered_multiset_view1_then_view2",
                "requested_replenishment_count": 8,
                "pool_size_before": len(current_pool),
                "training_size_view1_before": training_size,
                "training_size_view2_before": training_size,
                "reservoir_remaining": 701 - replenished_total - expected_replenished_count,
            },
            field=round_field,
            path=path,
        )
        pool_before = _list_field(round_values, "pool_indices_before", field=round_field, path=path)
        if pool_before != current_pool:
            raise HistoricalAcceptanceError(
                f"{path}: Nigam-Ghani Co-Training pool trajectory is inconsistent"
            )

        selected1 = _list_field(round_values, "selected_by_view1", field=round_field, path=path)
        selected2 = _list_field(round_values, "selected_by_view2", field=round_field, path=path)
        expected_quota = min(4, len(current_pool))
        if len(selected1) != expected_quota or len(selected2) != expected_quota:
            raise HistoricalAcceptanceError(
                f"{path}: Nigam-Ghani Co-Training round quotas must match the same pre-round pool"
            )

        selected_indices_by_view: list[list[int]] = []
        selected_labels_by_view: list[list[int]] = []
        for view_name, selected in (("view1", selected1), ("view2", selected2)):
            view_indices: list[int] = []
            view_labels: list[int] = []
            for position, raw_selection in enumerate(selected):
                selection = _mapping(
                    raw_selection,
                    field=f"{round_field}.selected_by_{view_name}[{position}]",
                    path=path,
                )
                pool_position = _int_field(
                    selection,
                    "pool_position",
                    field=f"{round_field}.selected_by_{view_name}[{position}]",
                    path=path,
                )
                unlabeled_index = _int_field(
                    selection,
                    "unlabeled_index",
                    field=f"{round_field}.selected_by_{view_name}[{position}]",
                    path=path,
                )
                label = _int_field(
                    selection,
                    "label",
                    field=f"{round_field}.selected_by_{view_name}[{position}]",
                    path=path,
                )
                confidence = selection.get("confidence")
                if (
                    pool_position < 0
                    or pool_position >= len(current_pool)
                    or current_pool[pool_position] != unlabeled_index
                    or label not in {0, 1}
                    or isinstance(confidence, bool)
                    or not isinstance(confidence, int | float)
                    or not math.isfinite(float(confidence))
                    or not 0.0 <= float(confidence) <= 1.0
                ):
                    raise HistoricalAcceptanceError(
                        f"{path}: Nigam-Ghani Co-Training selection trace is inconsistent"
                    )
                view_indices.append(unlabeled_index)
                view_labels.append(label)
            if len(set(view_indices)) != expected_quota or view_labels != (
                [1] + [0] * (expected_quota - 1)
            ):
                raise HistoricalAcceptanceError(
                    f"{path}: Nigam-Ghani Co-Training positive/negative quota is inconsistent"
                )
            selected_indices_by_view.append(view_indices)
            selected_labels_by_view.append(view_labels)

        indices1, indices2 = selected_indices_by_view
        labels1, labels2 = selected_labels_by_view
        selected_indices = [*indices1, *indices2]
        selected_labels = [*labels1, *labels2]
        additions = _list_field(round_values, "multiset_additions", field=round_field, path=path)
        if len(additions) != len(selected_indices):
            raise HistoricalAcceptanceError(
                f"{path}: Nigam-Ghani Co-Training multiset additions must preserve every proposal"
            )
        for position, raw_addition in enumerate(additions):
            _locked_subset(
                raw_addition,
                {
                    "proposal_order": position,
                    "source_view": "view1" if position < len(selected1) else "view2",
                    "unlabeled_index": selected_indices[position],
                    "label": selected_labels[position],
                },
                field=f"{round_field}.multiset_additions[{position}]",
                path=path,
            )

        removed = _list_field(round_values, "removed_indices", field=round_field, path=path)
        overlap = _list_field(round_values, "overlap_indices", field=round_field, path=path)
        conflicts = _list_field(
            round_values, "conflicting_overlap_indices", field=round_field, path=path
        )
        labels_by_index1 = dict(zip(indices1, labels1, strict=True))
        labels_by_index2 = dict(zip(indices2, labels2, strict=True))
        overlap_set = set(indices1).intersection(indices2)
        expected_overlap = [value for value in current_pool if value in overlap_set]
        expected_conflicts = [
            value
            for value in expected_overlap
            if labels_by_index1[value] != labels_by_index2[value]
        ]
        selected_set = set(selected_indices)
        expected_removed = [value for value in current_pool if value in selected_set]
        if (
            overlap != expected_overlap
            or conflicts != expected_conflicts
            or removed != expected_removed
            or promoted.intersection(removed)
        ):
            raise HistoricalAcceptanceError(
                f"{path}: Nigam-Ghani Co-Training same-pool overlap/removal trace is inconsistent"
            )

        addition_count = len(additions)
        removed_count = len(removed)
        overlap_count = len(overlap)
        conflict_count = len(conflicts)
        same_label_count = overlap_count - conflict_count
        _locked_subset(
            round_values,
            {
                "proposal_count_view1": len(selected1),
                "proposal_count_view2": len(selected2),
                "multiset_addition_count": addition_count,
                "unique_removed_count": removed_count,
                "duplicate_multiset_addition_count": addition_count - removed_count,
                "same_label_overlap_count": same_label_count,
                "conflicting_overlap_count": conflict_count,
                "training_size_view1_after": training_size + addition_count,
                "training_size_view2_after": training_size + addition_count,
            },
            field=round_field,
            path=path,
        )

        replenished = _list_field(round_values, "replenished_indices", field=round_field, path=path)
        if (
            len(replenished) != expected_replenished_count
            or any(isinstance(value, bool) or not isinstance(value, int) for value in replenished)
            or len(set(replenished)) != len(replenished)
            or seen_in_pool.intersection(replenished)
            or any(value < 0 or value >= 776 for value in replenished)
        ):
            raise HistoricalAcceptanceError(
                f"{path}: Nigam-Ghani Co-Training replenishment trace is inconsistent"
            )
        removed_set = set(removed)
        expected_pool_after = [
            *[value for value in current_pool if value not in removed_set],
            *replenished,
        ]
        pool_after = _list_field(round_values, "pool_indices_after", field=round_field, path=path)
        _locked_subset(
            round_values,
            {
                "pool_size_after": len(expected_pool_after),
                "pool_growth": expected_replenished_count - removed_count,
            },
            field=round_field,
            path=path,
        )
        if pool_after != expected_pool_after:
            raise HistoricalAcceptanceError(
                f"{path}: Nigam-Ghani Co-Training replenished pool is inconsistent"
            )

        promoted.update(removed)
        seen_in_pool.update(replenished)
        replenished_total += len(replenished)
        traced_proposals1 += len(selected1)
        traced_proposals2 += len(selected2)
        traced_additions += addition_count
        traced_overlap += overlap_count
        traced_same_label += same_label_count
        traced_conflicts += conflict_count
        training_size += addition_count
        current_pool = expected_pool_after

    if (
        promoted != set(range(776))
        or seen_in_pool != set(range(776))
        or replenished_total != 701
        or current_pool
    ):
        raise HistoricalAcceptanceError(
            f"{path}: Nigam-Ghani Co-Training exhaustion trace is inconsistent"
        )
    if proposal_total1 != traced_proposals1 or proposal_total2 != traced_proposals2:
        raise HistoricalAcceptanceError(
            f"{path}: Nigam-Ghani Co-Training pseudo_label_proposals_view1/view2 are inconsistent"
        )
    if added_total != traced_additions or added_total != proposal_total1 + proposal_total2:
        raise HistoricalAcceptanceError(
            f"{path}: Nigam-Ghani Co-Training pseudo_labels_added_to_shared_l is inconsistent"
        )
    if received_total1 != added_total or received_total2 != added_total:
        raise HistoricalAcceptanceError(
            f"{path}: Nigam-Ghani Co-Training pseudo_labels_received_by_view1/view2 are inconsistent"
        )
    if final_labeled != 12 + added_total or final_labeled != training_size:
        raise HistoricalAcceptanceError(
            f"{path}: Nigam-Ghani Co-Training final_labeled_size is inconsistent"
        )
    if (
        overlap_total != traced_overlap
        or duplicate_total != traced_overlap
        or same_label_total != traced_same_label
        or conflicting_total != traced_conflicts
        or overlap_total != same_label_total + conflicting_total
        or overlap_total != added_total - 776
    ):
        raise HistoricalAcceptanceError(
            f"{path}: Nigam-Ghani Co-Training overlap and duplicate aggregate fields are inconsistent"
        )


def _validate_co_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    path: Path,
    protocol: HistoricalProtocol | None = None,
    expected_seed: int | None = None,
) -> None:
    protocol = protocol or _PROTOCOLS["co-training"]
    if protocol.protocol_id == "paper:nigam-ghani2000-webkb-table2":
        _validate_co_nigam_diagnostics(
            diagnostics,
            path=path,
            expected_seed=expected_seed,
        )
        return
    co_v2 = protocol.protocol_id == "paper:blum-mitchell-1998-webkb-course-confirmation-v2"
    field = "artifacts.method.diagnostics"
    _locked_subset(
        diagnostics,
        {
            "protocol": ("fixed_pool_binary_feature_selection" if co_v2 else "fixed_pool_binary"),
            "p": 1,
            "n": 3,
            "u": 75,
            "k": 30,
            "negative_label": 0,
            "positive_label": 1,
            "n_iter": 30,
            "shared_labeled_multiset": True,
            "overlap_policy": "ordered_multiset_view1_then_view2",
            "selection_score_space": ("craven_1998_normalized_nb" if co_v2 else "log_probability"),
            "combination_score_space": "summed_log_probability",
            "probability_underflow_safe": True,
            "pseudo_labels_added_to_shared_l": 240,
            "pseudo_labels_received_by_view1": 240,
            "pseudo_labels_received_by_view2": 240,
            "final_labeled_size": 252,
        },
        field=field,
        path=path,
    )
    if co_v2:
        _locked_subset(
            diagnostics,
            {
                "seed": expected_seed,
                "dynamic_feature_selection": "mutual_information_presence",
                "feature_selection_max_features": 2000,
                "selection_diagnostics_scope": "training_and_pseudo_labels_only",
                "test_metrics_used_for_protocol_selection": False,
            },
            field=field,
            path=path,
        )
        _validate_co_feature_diagnostic(
            diagnostics,
            count_key="final_feature_count_view1",
            digest_key="final_features_sha256_view1",
            maximum_key="final_maximum_mutual_information_view1",
            field=field,
            path=path,
        )
        _validate_co_feature_diagnostic(
            diagnostics,
            count_key="final_feature_count_view2",
            digest_key="final_features_sha256_view2",
            maximum_key="final_maximum_mutual_information_view2",
            field=field,
            path=path,
        )
    initial_pool = _list_field(diagnostics, "initial_pool_indices", field=field, path=path)
    if len(initial_pool) != 75 or len(set(initial_pool)) != 75:
        raise HistoricalAcceptanceError(f"{path}: Co-Training initial pool must contain 75 indices")
    unique_promoted = _int_field(
        diagnostics, "unique_pseudo_labeled_examples", field=field, path=path
    )
    remaining = _int_field(diagnostics, "remaining_unlabeled_count", field=field, path=path)
    if not 0 <= unique_promoted <= 240 or remaining != 776 - unique_promoted:
        raise HistoricalAcceptanceError(f"{path}: Co-Training remaining U size is inconsistent")
    trace = _list_field(diagnostics, "round_trace", field=field, path=path)
    if len(trace) != 30:
        raise HistoricalAcceptanceError(f"{path}: Co-Training round_trace must contain 30 rounds")

    unique_removed: set[int] = set()
    for index, raw_round in enumerate(trace):
        round_values = _mapping(
            raw_round,
            field=f"{field}.round_trace[{index}]",
            path=path,
        )
        round_field = f"{field}.round_trace[{index}]"
        _locked_subset(
            round_values,
            {
                "round": index + 1,
                "round_status": "completed",
                "overlap_policy": "ordered_multiset_view1_then_view2",
                "requested_replenishment_count": 8,
                "training_size_view1_before": 12 + 8 * index,
                "training_size_view1_after": 20 + 8 * index,
                "training_size_view2_before": 12 + 8 * index,
                "training_size_view2_after": 20 + 8 * index,
            },
            field=round_field,
            path=path,
        )
        if co_v2:
            _locked_subset(
                round_values,
                {"feature_selection": "mutual_information_presence"},
                field=round_field,
                path=path,
            )
            _validate_co_feature_diagnostic(
                round_values,
                count_key="selected_feature_count_view1",
                digest_key="selected_features_sha256_view1",
                maximum_key="maximum_mutual_information_view1",
                field=round_field,
                path=path,
            )
            _validate_co_feature_diagnostic(
                round_values,
                count_key="selected_feature_count_view2",
                digest_key="selected_features_sha256_view2",
                maximum_key="maximum_mutual_information_view2",
                field=round_field,
                path=path,
            )
        selected1 = _list_field(round_values, "selected_by_view1", field=round_field, path=path)
        selected2 = _list_field(round_values, "selected_by_view2", field=round_field, path=path)
        additions = _list_field(round_values, "multiset_additions", field=round_field, path=path)
        removed = _list_field(round_values, "removed_indices", field=round_field, path=path)
        replenished = _list_field(round_values, "replenished_indices", field=round_field, path=path)
        pool_before = _list_field(round_values, "pool_indices_before", field=round_field, path=path)
        pool_after = _list_field(round_values, "pool_indices_after", field=round_field, path=path)
        if len(selected1) != 4 or len(selected2) != 4 or len(additions) != 8:
            raise HistoricalAcceptanceError(f"{path}: Co-Training round quotas must be 4+4")
        for selected in (selected1, selected2):
            labels = [
                _mapping(item, field=f"{round_field}.selection", path=path).get("label")
                for item in selected
            ]
            if labels.count(1) != 1 or labels.count(0) != 3:
                raise HistoricalAcceptanceError(
                    f"{path}: Co-Training positive/negative quota is inconsistent"
                )
        if len(set(removed)) != len(removed) or not 4 <= len(removed) <= 8:
            raise HistoricalAcceptanceError(f"{path}: Co-Training removed set is inconsistent")
        if len(replenished) != 8:
            raise HistoricalAcceptanceError(f"{path}: Co-Training must replenish eight examples")
        before_size = _int_field(round_values, "pool_size_before", field=round_field, path=path)
        after_size = _int_field(round_values, "pool_size_after", field=round_field, path=path)
        growth = _int_field(round_values, "pool_growth", field=round_field, path=path)
        if before_size != len(pool_before) or after_size != len(pool_after):
            raise HistoricalAcceptanceError(f"{path}: Co-Training pool sizes are inconsistent")
        if after_size != before_size - len(removed) + 8 or growth != after_size - before_size:
            raise HistoricalAcceptanceError(
                f"{path}: Co-Training replenishment trace is inconsistent"
            )
        unique_removed.update(int(value) for value in removed)
    if len(unique_removed) != unique_promoted:
        raise HistoricalAcceptanceError(f"{path}: Co-Training unique promotions are inconsistent")


def _validate_run_contract(
    root: Mapping[str, Any],
    *,
    path: Path,
    protocol: HistoricalProtocol,
    seed: int,
    sampling: Mapping[str, Any],
    split_fingerprint: str,
    method: Mapping[str, Any],
) -> dict[str, str]:
    _locked_subset(
        root.get("protocol"),
        {
            "kind": "inductive",
            "report_splits": ["test"],
            "split_for_model_selection": None,
            "use_test_split": False,
        },
        field="protocol",
        path=path,
    )
    config = _mapping(root.get("config"), field="config", path=path)
    critical_config = _critical_config(protocol, seed=seed)
    _locked_subset(config, critical_config, field="config", path=path)
    hashes = _mapping(root.get("hashes"), field="hashes", path=path)
    config_hash = _hex_digest(
        hashes.get("config_hash"), length=64, field="hashes.config_hash", path=path
    )
    effective_hash = _hex_digest(
        hashes.get("effective_config_hash"),
        length=64,
        field="hashes.effective_config_hash",
        path=path,
    )
    if config_hash != effective_hash:
        raise HistoricalAcceptanceError(f"{path}: effective config hash differs from config hash")
    _locked_subset(
        sampling,
        _expected_sampling_contract(protocol, seed=seed),
        field="artifacts.sampling",
        path=path,
    )
    replay_sha = _validate_replay(
        run_json_path=path,
        sampling=sampling,
        protocol=protocol,
        split_fingerprint=split_fingerprint,
        seed=seed,
    )
    diagnostics = _mapping(
        method.get("diagnostics"), field="artifacts.method.diagnostics", path=path
    )
    if protocol.method_id == "self_training":
        _validate_self_diagnostics(
            diagnostics,
            path=path,
            protocol=protocol,
            expected_seed=derive_seed(seed, "method"),
        )
    else:
        _validate_co_diagnostics(
            diagnostics,
            path=path,
            protocol=protocol,
            expected_seed=derive_seed(seed, "method"),
        )
    contract_payload = {
        "critical_config": critical_config,
        "protocol": root["protocol"],
        "sampling_stats": sampling["stats"],
        "replay_manifest_sha256": replay_sha,
        "diagnostics": diagnostics,
    }
    contract_sha = hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "config_hash": config_hash,
        "replay_manifest_sha256": replay_sha,
        "run_contract_sha256": contract_sha,
    }


def _load_run(
    path: Path,
    *,
    protocol: HistoricalProtocol,
    expected_git_sha: str,
) -> tuple[int, dict[str, float], dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalAcceptanceError(f"cannot read run.json: {path}") from exc
    root = _mapping(payload, field="run.json", path=path)
    run = _mapping(root.get("run"), field="run", path=path)
    if run.get("status") != "success":
        raise HistoricalAcceptanceError(f"{path}: run.status must equal 'success'")
    if root.get("error") is not None:
        raise HistoricalAcceptanceError(f"{path}: a successful run must not contain an error")
    seed = run.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise HistoricalAcceptanceError(f"{path}: run.seed must be an integer")
    metrics = _mapping(root.get("metrics"), field="metrics", path=path)
    accuracies = {"test": _metric_accuracy(metrics, split="test", path=path)}
    for view_name, _target_error in protocol.secondary_target_errors:
        split = f"test_{view_name}"
        accuracies[split] = _metric_accuracy(metrics, split=split, path=path)

    artifacts = _mapping(root.get("artifacts"), field="artifacts", path=path)
    method = _mapping(artifacts.get("method"), field="artifacts.method", path=path)
    _locked_value(
        method,
        key="id",
        expected=protocol.method_id,
        field="artifacts.method.id",
        path=path,
    )
    _locked_value(
        method,
        key="profile",
        expected=protocol.protocol_id,
        field="artifacts.method.profile",
        path=path,
    )
    dataset = _mapping(artifacts.get("dataset"), field="artifacts.dataset", path=path)
    _locked_value(
        dataset,
        key="id",
        expected=protocol.dataset_id,
        field="artifacts.dataset.id",
        path=path,
    )
    _locked_value(
        dataset,
        key="fingerprint",
        expected=protocol.dataset_fingerprint,
        field="artifacts.dataset.fingerprint",
        path=path,
    )
    _locked_value(
        dataset,
        key="content_sha256",
        expected=protocol.dataset_content_sha256,
        field="artifacts.dataset.content_sha256",
        path=path,
    )

    versions = _mapping(root.get("versions"), field="versions", path=path)
    if versions.get("git_dirty") is not False:
        raise HistoricalAcceptanceError(f"{path}: versions.git_dirty must be false")
    git_sha = _hex_digest(versions.get("git_sha"), length=40, field="versions.git_sha", path=path)
    if git_sha != expected_git_sha:
        raise HistoricalAcceptanceError(
            f"{path}: versions.git_sha must equal approved commit {expected_git_sha}"
        )
    git_diff_sha256 = _hex_digest(
        versions.get("git_diff_sha256"),
        length=64,
        field="versions.git_diff_sha256",
        path=path,
    )
    environment = {
        key: _environment_version(versions.get(key), field=key, path=path)
        for key in ("python", "numpy", "scikit_learn")
    }
    sampling = _mapping(artifacts.get("sampling"), field="artifacts.sampling", path=path)
    split_fingerprint = _hex_digest(
        sampling.get("split_fingerprint"),
        length=64,
        field="artifacts.sampling.split_fingerprint",
        path=path,
    )
    contract = _validate_run_contract(
        root,
        path=path,
        protocol=protocol,
        seed=int(seed),
        sampling=sampling,
        split_fingerprint=split_fingerprint,
        method=method,
    )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = {"path": str(path.resolve()), "sha256": digest}
    provenance = {
        "git_sha": git_sha,
        "git_diff_sha256": git_diff_sha256,
        "environment": environment,
        "split_fingerprint": split_fingerprint,
        **contract,
    }
    return int(seed), accuracies, source, provenance


def _error_summary(
    errors: Sequence[float],
    *,
    target_error: float,
    student_t_critical: float,
    student_t_critical_source: str,
) -> dict[str, Any]:
    n = len(errors)
    mean = math.fsum(errors) / n
    sample_variance = math.fsum((value - mean) ** 2 for value in errors) / (n - 1)
    sample_std = math.sqrt(sample_variance)
    half_width = student_t_critical * sample_std / math.sqrt(n)
    ci_low = mean - half_width
    ci_high = mean + half_width
    absolute_difference = abs(mean - target_error)
    target_in_ci95 = (
        ci_low <= target_error <= ci_high
        or math.isclose(target_error, ci_low, rel_tol=0.0, abs_tol=1e-12)
        or math.isclose(target_error, ci_high, rel_tol=0.0, abs_tol=1e-12)
    )
    return {
        "n": n,
        "mean_error": mean,
        "sample_std_error": sample_std,
        "std_ddof": 1,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "student_t_critical": student_t_critical,
        "student_t_critical_source": student_t_critical_source,
        "target_error": target_error,
        "absolute_difference": absolute_difference,
        "target_in_ci95": target_in_ci95,
    }


def discover_run_jsons(
    *, sweep_root: Path | None, run_json_paths: Sequence[Path] | None
) -> list[Path]:
    if (sweep_root is None) == (run_json_paths is None):
        raise HistoricalAcceptanceError("provide exactly one of sweep_root or run_json_paths")
    if sweep_root is not None:
        if not sweep_root.is_dir():
            raise HistoricalAcceptanceError(f"sweep root is not a directory: {sweep_root}")
        paths = sorted(sweep_root.rglob("run.json"))
    else:
        paths = [Path(path) for path in run_json_paths or ()]
    if not paths:
        raise HistoricalAcceptanceError("no run.json was found")
    return paths


def evaluate_historical_runs(
    *,
    protocol_name: str,
    run_json_paths: Sequence[Path],
    expected_git_sha: str,
) -> dict[str, Any]:
    try:
        protocol = _PROTOCOLS[protocol_name]
    except KeyError as exc:
        raise HistoricalAcceptanceError(f"unknown protocol: {protocol_name}") from exc
    expected_git_sha = _hex_digest(
        expected_git_sha,
        length=40,
        field="expected_git_sha",
        path=Path("acceptance arguments"),
    )

    by_seed: dict[int, tuple[dict[str, float], dict[str, Any], dict[str, Any]]] = {}
    duplicate_seeds: set[int] = set()
    for path in run_json_paths:
        seed, accuracies, source, provenance = _load_run(
            Path(path),
            protocol=protocol,
            expected_git_sha=expected_git_sha,
        )
        if seed in by_seed:
            duplicate_seeds.add(seed)
        else:
            by_seed[seed] = (accuracies, source, provenance)
    if duplicate_seeds:
        rendered = ", ".join(str(seed) for seed in sorted(duplicate_seeds))
        raise HistoricalAcceptanceError(f"duplicate seeds: {rendered}")

    expected = set(protocol.expected_seeds)
    actual = set(by_seed)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise HistoricalAcceptanceError(
            f"seed set mismatch; missing={missing}, unexpected={unexpected}"
        )

    reference_provenance = by_seed[protocol.expected_seeds[0]][2]
    split_seeds: dict[str, list[int]] = {}
    for seed in protocol.expected_seeds:
        provenance = by_seed[seed][2]
        if provenance["git_diff_sha256"] != reference_provenance["git_diff_sha256"]:
            raise HistoricalAcceptanceError("versions.git_diff_sha256 differs between runs")
        if provenance["environment"] != reference_provenance["environment"]:
            raise HistoricalAcceptanceError(
                "Python/numpy/scikit-learn environment differs between runs"
            )
        split_seeds.setdefault(provenance["split_fingerprint"], []).append(seed)
    duplicate_splits = {
        fingerprint: seeds for fingerprint, seeds in split_seeds.items() if len(seeds) > 1
    }
    if duplicate_splits:
        rendered = ", ".join(
            f"{fingerprint} (seeds {seeds})"
            for fingerprint, seeds in sorted(duplicate_splits.items())
        )
        raise HistoricalAcceptanceError(f"duplicate split fingerprints: {rendered}")

    records = []
    errors = []
    for seed in protocol.expected_seeds:
        accuracies, source, provenance = by_seed[seed]
        accuracy = accuracies["test"]
        error = 1.0 - accuracy
        errors.append(error)
        record = {
            "seed": seed,
            "test_accuracy": accuracy,
            "test_error": error,
            "split_fingerprint": provenance["split_fingerprint"],
            "run_json": source,
        }
        for view_name, _target_error in protocol.secondary_target_errors:
            view_accuracy = accuracies[f"test_{view_name}"]
            record[f"test_{view_name}_accuracy"] = view_accuracy
            record[f"test_{view_name}_error"] = 1.0 - view_accuracy
        records.append(record)

    n = len(errors)
    critical, critical_source = _student_t_critical_95(n)
    primary_summary = _error_summary(
        errors,
        target_error=protocol.target_error,
        student_t_critical=critical,
        student_t_critical_source=critical_source,
    )
    within_margin = primary_summary["absolute_difference"] <= protocol.margin_absolute
    numeric_status = (
        "numeric_matched"
        if within_margin and primary_summary["target_in_ci95"]
        else "numeric_not_matched"
    )

    secondary_diagnostics = {}
    for view_name, target_error in protocol.secondary_target_errors:
        view_errors = [
            1.0 - by_seed[seed][0][f"test_{view_name}"] for seed in protocol.expected_seeds
        ]
        secondary_diagnostics[view_name] = {
            "metric": f"test_{view_name}.error:one_minus_accuracy",
            **_error_summary(
                view_errors,
                target_error=target_error,
                student_t_critical=critical,
                student_t_critical_source=critical_source,
            ),
        }
        secondary_diagnostics[view_name]["margin_absolute"] = protocol.margin_absolute
        secondary_diagnostics[view_name]["within_margin"] = (
            secondary_diagnostics[view_name]["absolute_difference"] <= protocol.margin_absolute
        )

    secondary_controls_passed = all(
        bool(diagnostic["within_margin"] and diagnostic["target_in_ci95"])
        for diagnostic in secondary_diagnostics.values()
    )
    protocol_diagnostics_passed = secondary_controls_passed
    result_status = (
        "replicated_paper_approx"
        if numeric_status == "numeric_matched" and protocol_diagnostics_passed
        else "failed_replication"
    )

    sealed_provenance = {
        "schema_version": 1,
        "profile": protocol.protocol_id,
        "method": {"id": protocol.method_id, "profile": protocol.protocol_id},
        "dataset": {
            "id": protocol.dataset_id,
            "fingerprint": protocol.dataset_fingerprint,
            "content_sha256": protocol.dataset_content_sha256,
        },
        "code": {
            "git_dirty": False,
            "git_sha": expected_git_sha,
            "git_diff_sha256": reference_provenance["git_diff_sha256"],
        },
        "environment": dict(reference_provenance["environment"]),
        "runs": [
            {
                "seed": record["seed"],
                "split_fingerprint": record["split_fingerprint"],
                "run_json_sha256": record["run_json"]["sha256"],
                "config_hash": by_seed[record["seed"]][2]["config_hash"],
                "replay_manifest_sha256": by_seed[record["seed"]][2]["replay_manifest_sha256"],
                "run_contract_sha256": by_seed[record["seed"]][2]["run_contract_sha256"],
            }
            for record in records
        ],
    }
    seal_input = json.dumps(
        sealed_provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    sealed_provenance["seal_sha256"] = hashlib.sha256(seal_input).hexdigest()

    report = {
        "schema_version": 2,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "name": protocol_name,
            "profile": protocol.protocol_id,
            "method_id": protocol.method_id,
            "dataset_id": protocol.dataset_id,
            "dataset_fingerprint": protocol.dataset_fingerprint,
            "dataset_content_sha256": protocol.dataset_content_sha256,
            "expected_seeds": list(protocol.expected_seeds),
            "target_error": protocol.target_error,
            "margin_absolute": protocol.margin_absolute,
        },
        "completeness": {
            "expected": n,
            "successful": n,
            "duplicate_seeds": [],
            "missing_seeds": [],
            "unexpected_seeds": [],
            "complete": True,
        },
        "statistics": {
            "metric": "test.error:one_minus_accuracy",
            **primary_summary,
            "margin_absolute": protocol.margin_absolute,
            "within_margin": within_margin,
        },
        "numeric_status": numeric_status,
        "protocol_diagnostics_passed": protocol_diagnostics_passed,
        "result_status": result_status,
        "scientific_status": "paper_approx",
        "scientific_status_reason": (
            "Numerical equivalence cannot remove the pre-registered historical protocol unknowns."
        ),
        "critical_unknowns": list(protocol.critical_unknowns),
        "sealed_provenance": sealed_provenance,
        "runs": records,
    }
    if secondary_diagnostics:
        report["secondary_diagnostics"] = secondary_diagnostics
    return report


_TSV_FIELDS = (
    "profile",
    "n",
    "mean_error",
    "sample_std_error",
    "ci95_low",
    "ci95_high",
    "target_error",
    "absolute_difference",
    "margin_absolute",
    "within_margin",
    "target_in_ci95",
    "numeric_status",
    "protocol_diagnostics_passed",
    "result_status",
    "scientific_status",
)


def _summary_tsv(report: Mapping[str, Any]) -> str:
    protocol = _mapping(report.get("protocol"), field="protocol", path=Path("report"))
    statistics = _mapping(report.get("statistics"), field="statistics", path=Path("report"))
    row = {
        "profile": protocol["profile"],
        "n": statistics["n"],
        "mean_error": statistics["mean_error"],
        "sample_std_error": statistics["sample_std_error"],
        "ci95_low": statistics["ci95_low"],
        "ci95_high": statistics["ci95_high"],
        "target_error": statistics["target_error"],
        "absolute_difference": statistics["absolute_difference"],
        "margin_absolute": statistics["margin_absolute"],
        "within_margin": statistics["within_margin"],
        "target_in_ci95": statistics["target_in_ci95"],
        "numeric_status": report["numeric_status"],
        "protocol_diagnostics_passed": report["protocol_diagnostics_passed"],
        "result_status": report["result_status"],
        "scientific_status": report["scientific_status"],
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_TSV_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_report(*, report: Mapping[str, Any], output_json: Path, output_tsv: Path) -> None:
    if output_json.resolve() == output_tsv.resolve():
        raise HistoricalAcceptanceError("JSON and TSV outputs must use different paths")
    atomic_write_json(output_json, dict(report))
    _atomic_write_text(output_tsv, _summary_tsv(report))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a complete historical Self-Training or Co-Training seed sweep."
    )
    parser.add_argument("--protocol", choices=sorted(_PROTOCOLS), required=True)
    parser.add_argument(
        "--expected-git-sha",
        required=True,
        help="Approved immutable 40-character Git commit for every run.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sweep-root", type=Path)
    source.add_argument("--run-json", type=Path, action="append", dest="run_json_paths")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = discover_run_jsons(
            sweep_root=args.sweep_root,
            run_json_paths=args.run_json_paths,
        )
        report = evaluate_historical_runs(
            protocol_name=args.protocol,
            run_json_paths=paths,
            expected_git_sha=args.expected_git_sha,
        )
        write_report(
            report=report,
            output_json=args.output_json,
            output_tsv=args.output_tsv,
        )
    except HistoricalAcceptanceError as exc:
        print(f"historical acceptance failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"{report['scientific_status']} ({report['result_status']}; "
        f"{report['numeric_status']}): {args.output_json}"
    )
    return 0 if report["result_status"] == "replicated_paper_approx" else 3


if __name__ == "__main__":  # pragma: no cover - exercised via the module entry point
    raise SystemExit(main())
