from __future__ import annotations

import builtins
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bench.campaign.acceptance import historical as historical_acceptance
from bench.campaign.acceptance.historical import (
    HistoricalAcceptanceError,
    _atomic_write_text,
    _student_t_critical_95,
    discover_run_jsons,
    main,
    write_report,
)
from bench.campaign.acceptance.historical import (
    evaluate_historical_runs as _evaluate_historical_runs,
)

_GIT_SHA = "a" * 40
_GIT_DIFF_SHA256 = "b" * 64


def evaluate_historical_runs(
    *,
    protocol_name: str,
    run_json_paths: list[Path],
    expected_git_sha: str = _GIT_SHA,
) -> dict[str, object]:
    return _evaluate_historical_runs(
        protocol_name=protocol_name,
        run_json_paths=run_json_paths,
        expected_git_sha=expected_git_sha,
    )


def _self_diagnostics() -> dict[str, object]:
    return {
        "protocol": "li_zhou_2005_1nn_distance",
        "seed": 123,
        "n_iter": 1,
        "initial_labeled_size": 13,
        "initial_unlabeled_count": 121,
        "final_labeled_size": 14,
        "remaining_unlabeled_count": 120,
        "pseudo_labels_added": 1,
        "selection_parameters": {},
        "round_trace": [
            {
                "iteration": 0,
                "pool_indices": list(range(121)),
                "candidate_indices": [0],
                "candidate_labels": [0],
                "accepted_indices": [0],
                "accepted_labels": [0],
                "labeled_before": 13,
                "labeled_after": 14,
                "unlabeled_before": 121,
                "remaining_unlabeled": 120,
            }
        ],
    }


def _self_v2_diagnostics() -> dict[str, object]:
    return {
        "protocol": "li_zhou_2005_1nn_distance",
        "seed": 51,
        "n_iter": 2,
        "initial_labeled_size": 13,
        "initial_unlabeled_count": 121,
        "final_labeled_size": 15,
        "remaining_unlabeled_count": 119,
        "pseudo_labels_added": 2,
        "selection_parameters": {
            "paper_pool_size_unspecified": 75,
            "paper_candidates_per_class_unspecified": 1,
            "paper_distance_confidence_unspecified": "nearest_neighbor_distance",
            "paper_feature_scaling_unspecified": "dynamic_labeled_minmax",
        },
        "round_trace": [
            {
                "iteration": 0,
                "pool_indices": list(range(75)),
                "candidate_indices": [0],
                "candidate_labels": [0],
                "accepted_indices": [0],
                "accepted_labels": [0],
                "labeled_before": 13,
                "labeled_after": 14,
                "unlabeled_before": 121,
                "remaining_unlabeled": 120,
            },
            {
                "iteration": 1,
                "pool_indices": list(range(1, 76)),
                "candidate_indices": [1],
                "candidate_labels": [0],
                "accepted_indices": [1],
                "accepted_labels": [0],
                "labeled_before": 14,
                "labeled_after": 15,
                "unlabeled_before": 120,
                "remaining_unlabeled": 119,
            },
        ],
    }


def _co_diagnostics() -> dict[str, object]:
    pool = list(range(75))
    next_index = 75
    trace = []
    removed_all: set[int] = set()
    for round_index in range(30):
        selected_indices = pool[:4]
        labels = [1, 0, 0, 0]
        selected = [
            {
                "pool_position": position,
                "unlabeled_index": value,
                "label": label,
                "confidence": 0.9,
            }
            for position, (value, label) in enumerate(zip(selected_indices, labels, strict=True))
        ]
        replenished = list(range(next_index, next_index + 8))
        next_index += 8
        pool_after = [*pool[4:], *replenished]
        removed_all.update(selected_indices)
        trace.append(
            {
                "round": round_index + 1,
                "round_status": "completed",
                "overlap_policy": "ordered_multiset_view1_then_view2",
                "pool_indices_before": pool,
                "pool_size_before": len(pool),
                "selected_by_view1": selected,
                "selected_by_view2": selected,
                "overlap_indices": selected_indices,
                "conflicting_overlap_indices": [],
                "multiset_additions": [
                    {
                        "proposal_order": position,
                        "source_view": "view1" if position < 4 else "view2",
                        "unlabeled_index": selected_indices[position % 4],
                        "label": labels[position % 4],
                    }
                    for position in range(8)
                ],
                "removed_indices": selected_indices,
                "requested_replenishment_count": 8,
                "replenished_indices": replenished,
                "pool_indices_after": pool_after,
                "pool_size_after": len(pool_after),
                "pool_growth": 4,
                "reservoir_remaining": 700 - 8 * (round_index + 1),
                "training_size_view1_before": 12 + 8 * round_index,
                "training_size_view1_after": 20 + 8 * round_index,
                "training_size_view2_before": 12 + 8 * round_index,
                "training_size_view2_after": 20 + 8 * round_index,
            }
        )
        pool = pool_after
    return {
        "protocol": "fixed_pool_binary",
        "seed": 123,
        "p": 1,
        "n": 3,
        "u": 75,
        "k": 30,
        "negative_label": 0,
        "positive_label": 1,
        "initial_pool_indices": list(range(75)),
        "n_iter": 30,
        "shared_labeled_multiset": True,
        "overlap_policy": "ordered_multiset_view1_then_view2",
        "selection_score_space": "log_probability",
        "combination_score_space": "summed_log_probability",
        "probability_underflow_safe": True,
        "same_label_overlap_count": 120,
        "conflicting_overlap_count": 0,
        "unique_pseudo_labeled_examples": len(removed_all),
        "pseudo_labels_added_to_shared_l": 240,
        "pseudo_labels_received_by_view1": 240,
        "pseudo_labels_received_by_view2": 240,
        "final_labeled_size": 252,
        "remaining_unlabeled_count": 776 - len(removed_all),
        "remaining_unlabeled_indices": [],
        "round_trace": trace,
    }


def _co_v2_diagnostics(*, seed: int) -> dict[str, object]:
    diagnostics = _co_diagnostics()
    diagnostics.update(
        {
            "protocol": "fixed_pool_binary_feature_selection",
            "seed": historical_acceptance.derive_seed(seed, "method"),
            "selection_score_space": "craven_1998_normalized_nb",
            "dynamic_feature_selection": "mutual_information_presence",
            "feature_selection_max_features": 2000,
            "final_feature_count_view1": 2000,
            "final_feature_count_view2": 1200,
            "final_features_sha256_view1": "1" * 64,
            "final_features_sha256_view2": "2" * 64,
            "final_maximum_mutual_information_view1": 0.5,
            "final_maximum_mutual_information_view2": 0.25,
            "selection_diagnostics_scope": "training_and_pseudo_labels_only",
            "test_metrics_used_for_protocol_selection": False,
        }
    )
    for round_values in diagnostics["round_trace"]:
        round_values.update(
            {
                "feature_selection": "mutual_information_presence",
                "selected_feature_count_view1": 2000,
                "selected_feature_count_view2": 1200,
                "selected_features_sha256_view1": "3" * 64,
                "selected_features_sha256_view2": "4" * 64,
                "maximum_mutual_information_view1": 0.5,
                "maximum_mutual_information_view2": 0.25,
            }
        )
    return diagnostics


def _co_nigam_diagnostics(*, seed: int) -> dict[str, object]:
    pool = list(range(75))
    next_index = 75
    trace: list[dict[str, object]] = []
    proposal_total1 = 0
    proposal_total2 = 0
    overlap_total = 0
    conflict_total = 0
    training_size = 12
    round_index = 0
    while pool:
        quota = min(4, len(pool))
        selected_indices1 = pool[:quota]
        labels1 = [1, *([0] * (quota - 1))]
        if round_index == 0 and len(pool) >= 7:
            # One conflicting overlap: view 2 calls a view-1 negative positive.
            selected_indices2 = [pool[1], pool[4], pool[5], pool[6]]
        elif round_index == 1 and len(pool) >= 7:
            # One same-label overlap: both views call pool[1] negative.
            selected_indices2 = [pool[4], pool[1], pool[5], pool[6]]
        elif len(pool) >= 8:
            selected_indices2 = pool[4:8]
        else:
            # Exercise a final undersized same-pool round.
            selected_indices2 = pool[:quota]
        labels2 = [1, *([0] * (quota - 1))]

        def _selected(
            indices: list[int],
            labels: list[int],
            *,
            confidence: float,
            current_pool: list[int],
        ) -> list[dict[str, object]]:
            return [
                {
                    "pool_position": current_pool.index(value),
                    "unlabeled_index": value,
                    "label": label,
                    "confidence": confidence,
                }
                for value, label in zip(indices, labels, strict=True)
            ]

        selected1 = _selected(
            selected_indices1,
            labels1,
            confidence=0.9,
            current_pool=pool,
        )
        selected2 = _selected(
            selected_indices2,
            labels2,
            confidence=0.8,
            current_pool=pool,
        )
        selected_indices = [*selected_indices1, *selected_indices2]
        selected_labels = [*labels1, *labels2]
        labels_by_index1 = dict(zip(selected_indices1, labels1, strict=True))
        labels_by_index2 = dict(zip(selected_indices2, labels2, strict=True))
        overlap_set = set(selected_indices1).intersection(selected_indices2)
        overlap = [value for value in pool if value in overlap_set]
        conflicts = [
            value for value in overlap if labels_by_index1[value] != labels_by_index2[value]
        ]
        removed_set = set(selected_indices)
        removed = [value for value in pool if value in removed_set]
        replenish_count = min(8, 776 - next_index)
        replenished = list(range(next_index, next_index + replenish_count))
        next_index += replenish_count
        pool_after = [*[value for value in pool if value not in removed_set], *replenished]
        addition_count = len(selected_indices)
        removed_count = len(removed)
        same_label_count = len(overlap) - len(conflicts)
        trace.append(
            {
                "round": round_index + 1,
                "round_status": "completed",
                "overlap_policy": "ordered_multiset_view1_then_view2",
                "pool_indices_before": pool,
                "pool_size_before": len(pool),
                "selected_by_view1": selected1,
                "selected_by_view2": selected2,
                "overlap_indices": overlap,
                "conflicting_overlap_indices": conflicts,
                "multiset_additions": [
                    {
                        "proposal_order": position,
                        "source_view": "view1" if position < len(selected1) else "view2",
                        "unlabeled_index": value,
                        "label": selected_labels[position],
                    }
                    for position, value in enumerate(selected_indices)
                ],
                "removed_indices": removed,
                "proposal_count_view1": len(selected1),
                "proposal_count_view2": len(selected2),
                "multiset_addition_count": addition_count,
                "unique_removed_count": removed_count,
                "duplicate_multiset_addition_count": addition_count - removed_count,
                "same_label_overlap_count": same_label_count,
                "conflicting_overlap_count": len(conflicts),
                "requested_replenishment_count": 8,
                "replenished_indices": replenished,
                "pool_indices_after": pool_after,
                "pool_size_after": len(pool_after),
                "pool_growth": replenish_count - removed_count,
                "reservoir_remaining": 776 - next_index,
                "training_size_view1_before": training_size,
                "training_size_view1_after": training_size + addition_count,
                "training_size_view2_before": training_size,
                "training_size_view2_after": training_size + addition_count,
            }
        )
        proposal_total1 += len(selected1)
        proposal_total2 += len(selected2)
        overlap_total += len(overlap)
        conflict_total += len(conflicts)
        training_size += addition_count
        pool = pool_after
        round_index += 1
    return {
        "protocol": "shared_pool_exhaustive_multiset",
        "seed": historical_acceptance.derive_seed(seed, "method"),
        "p": 1,
        "n": 3,
        "u": 75,
        "k": 0,
        "negative_label": 0,
        "positive_label": 1,
        "initial_pool_indices": list(range(75)),
        "n_iter": len(trace),
        "shared_labeled_multiset": True,
        "overlap_policy": "ordered_multiset_view1_then_view2",
        "views_select_from_same_pre_round_pool": True,
        "selection_score_space": "posterior_probability",
        "combination_score_space": "summed_log_probability",
        "probability_underflow_safe": True,
        "pseudo_label_proposals_view1": proposal_total1,
        "pseudo_label_proposals_view2": proposal_total2,
        "overlap_count": overlap_total,
        "duplicate_multiset_additions": overlap_total,
        "same_label_overlap_count": overlap_total - conflict_total,
        "conflicting_overlap_count": conflict_total,
        "unique_pseudo_labeled_examples": 776,
        "pseudo_labels_added_to_shared_l": proposal_total1 + proposal_total2,
        "pseudo_labels_received_by_view1": proposal_total1 + proposal_total2,
        "pseudo_labels_received_by_view2": proposal_total1 + proposal_total2,
        "final_labeled_size": training_size,
        "remaining_unlabeled_count": 0,
        "remaining_unlabeled_indices": [],
        "round_trace": trace,
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


def _write_replay(
    run_json_path: Path,
    *,
    dataset_fingerprint: str,
    split_fingerprint: str,
    seed: int,
    protocol_name: str,
) -> dict[str, str]:
    replay_root = run_json_path.parent / "sampling_split"
    replay_root.mkdir(parents=True, exist_ok=True)
    if protocol_name in {"co-training-v2", "co-training-nigam"}:
        if protocol_name == "co-training-v2":
            arrays = historical_acceptance._co_v2_expected_indices(seed=seed)
            sampling_plan = historical_acceptance._co_v2_sampling_plan()
        else:
            arrays = historical_acceptance._co_nigam_expected_indices(seed=seed)
            sampling_plan = historical_acceptance._co_nigam_sampling_plan()
        np.savez_compressed(
            replay_root / "arrays.npz",
            **{f"idx__{name}": values for name, values in arrays.items()},
        )
        sampling_stats = historical_acceptance._expected_sampling_contract(
            historical_acceptance._PROTOCOLS[protocol_name], seed=seed
        )["stats"]
        split_payload = {
            "schema_version": 1,
            "created_at": "2026-08-05T00:00:00+00:00",
            "dataset_fingerprint": dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "plan": sampling_plan,
            "refs": {name: "train" for name in arrays},
            "stats": sampling_stats,
            "mode": "inductive",
        }
        (replay_root / "split.json").write_text(
            json.dumps(split_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        contents = {
            name: (replay_root / name).read_bytes() for name in ("arrays.npz", "split.json")
        }
    else:
        contents = {
            "arrays.npz": f"synthetic arrays {seed}".encode(),
            "split.json": json.dumps({"seed": seed}).encode(),
        }
    files = {}
    for name, content in contents.items():
        (replay_root / name).write_bytes(content)
        files[name] = {"sha256": hashlib.sha256(content).hexdigest()}
    manifest = {
        "dataset_fingerprint": dataset_fingerprint,
        "files": files,
        "format": "modssc.sampling.storage.v1",
        "schema_version": 1,
        "split_fingerprint": split_fingerprint,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (replay_root / "MANIFEST.json").write_bytes(manifest_bytes)
    return {
        "format": "modssc.sampling.storage.v1",
        "manifest": "MANIFEST.json",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "path": "sampling_split",
    }


def _write_run(
    path: Path,
    *,
    seed: int,
    accuracy: float,
    status: str = "success",
    protocol_name: str = "co-training",
    split_fingerprint: str | None = None,
    view_accuracies: dict[str, float] | None = None,
) -> Path:
    protocol = historical_acceptance._PROTOCOLS[protocol_name]
    if split_fingerprint is None:
        if protocol_name == "co-training-v2":
            split_fingerprint = historical_acceptance._co_v2_split_fingerprint(seed=seed)
        elif protocol_name == "co-training-nigam":
            split_fingerprint = historical_acceptance._co_nigam_split_fingerprint(seed=seed)
        else:
            split_fingerprint = hashlib.sha256(f"{protocol_name}:{seed}".encode()).hexdigest()
    metrics = {"test": {"accuracy": accuracy}}
    for view_name, target_error in protocol.secondary_target_errors:
        view_accuracy = (view_accuracies or {}).get(view_name, 1.0 - target_error)
        metrics[f"test_{view_name}"] = {"accuracy": view_accuracy}
    path.parent.mkdir(parents=True, exist_ok=True)
    replay = _write_replay(
        path,
        dataset_fingerprint=protocol.dataset_fingerprint,
        split_fingerprint=split_fingerprint,
        seed=seed,
        protocol_name=protocol_name,
    )
    sampling_contract = historical_acceptance._expected_sampling_contract(protocol, seed=seed)
    if protocol_name == "self-training-v2":
        diagnostics = _self_v2_diagnostics()
        diagnostics["seed"] = historical_acceptance.derive_seed(seed, "method")
    elif protocol_name == "self-training":
        diagnostics = _self_diagnostics()
        diagnostics["seed"] = historical_acceptance.derive_seed(seed, "method")
    elif protocol_name == "co-training-v2":
        diagnostics = _co_v2_diagnostics(seed=seed)
    elif protocol_name == "co-training-nigam":
        diagnostics = _co_nigam_diagnostics(seed=seed)
    else:
        diagnostics = _co_diagnostics()
    config_hash = hashlib.sha256(f"{protocol_name}:{seed}:config".encode()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "run": {"seed": seed, "status": status},
                "hashes": {
                    "config_hash": config_hash,
                    "effective_config_hash": config_hash,
                },
                "protocol": {
                    "kind": "inductive",
                    "report_splits": ["test"],
                    "split_for_model_selection": None,
                    "use_test_split": False,
                },
                "config": historical_acceptance._critical_config(protocol, seed=seed),
                "metrics": metrics,
                "artifacts": {
                    "method": {
                        "id": protocol.method_id,
                        "profile": protocol.protocol_id,
                        "diagnostics": diagnostics,
                    },
                    "dataset": {
                        "id": protocol.dataset_id,
                        "fingerprint": protocol.dataset_fingerprint,
                        "content_sha256": protocol.dataset_content_sha256,
                    },
                    "sampling": {
                        **sampling_contract,
                        "split_fingerprint": split_fingerprint,
                        "replay": replay,
                    },
                },
                "versions": {
                    "git_dirty": False,
                    "git_sha": _GIT_SHA,
                    "git_diff_sha256": _GIT_DIFF_SHA256,
                    "python": "3.12.13",
                    "numpy": "2.3.2",
                    "scikit_learn": "1.8.0",
                },
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return path


def _runs(
    root: Path,
    *,
    count: int,
    error: float = 0.05,
    protocol_name: str = "co-training",
) -> list[Path]:
    protocol = historical_acceptance._PROTOCOLS[protocol_name]
    return [
        _write_run(
            root / f"seed-{seed}" / "run.json",
            seed=seed,
            accuracy=1.0 - error,
            protocol_name=protocol_name,
        )
        for seed in protocol.expected_seeds[:count]
    ]


def _replace_nested(path: Path, keys: tuple[str, ...], value: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reseal_replay_files(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    replay_root = path.parent / "sampling_split"
    manifest_path = replay_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("arrays.npz", "split.json"):
        manifest["files"][name]["sha256"] = hashlib.sha256(
            (replay_root / name).read_bytes()
        ).hexdigest()
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    payload["artifacts"]["sampling"]["replay"]["manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_co_training_complete_numeric_match(tmp_path: Path) -> None:
    report = evaluate_historical_runs(
        protocol_name="co-training",
        run_json_paths=_runs(tmp_path, count=5),
    )

    assert report["completeness"]["complete"] is True
    assert report["statistics"]["mean_error"] == pytest.approx(0.05)
    assert report["statistics"]["sample_std_error"] == pytest.approx(0.0)
    assert report["statistics"]["std_ddof"] == 1
    assert report["statistics"]["ci95_low"] == pytest.approx(0.05)
    assert report["statistics"]["ci95_high"] == pytest.approx(0.05)
    assert report["numeric_status"] == "numeric_matched"
    assert report["protocol_diagnostics_passed"] is True
    assert report["result_status"] == "replicated_paper_approx"
    assert report["scientific_status"] == "paper_approx"
    assert list(report["secondary_diagnostics"]) == ["fulltext", "inlinks"]
    assert report["secondary_diagnostics"]["fulltext"] == {
        "metric": "test_fulltext.error:one_minus_accuracy",
        "n": 5,
        "mean_error": pytest.approx(0.062),
        "sample_std_error": pytest.approx(0.0),
        "std_ddof": 1,
        "ci95_low": pytest.approx(0.062),
        "ci95_high": pytest.approx(0.062),
        "student_t_critical": pytest.approx(2.7764451051977987),
        "student_t_critical_source": report["statistics"]["student_t_critical_source"],
        "target_error": 0.062,
        "absolute_difference": pytest.approx(0.0, abs=1e-12),
        "target_in_ci95": True,
        "margin_absolute": 0.02,
        "within_margin": True,
    }
    assert report["secondary_diagnostics"]["inlinks"]["target_error"] == 0.116
    assert report["secondary_diagnostics"]["inlinks"]["target_in_ci95"] is True
    assert [item["seed"] for item in report["runs"]] == [1, 2, 3, 4, 5]
    assert report["runs"][0]["test_fulltext_error"] == pytest.approx(0.062)
    assert report["runs"][0]["test_inlinks_error"] == pytest.approx(0.116)
    assert len(report["runs"][0]["run_json"]["sha256"]) == 64
    sealed = dict(report["sealed_provenance"])
    seal_sha256 = sealed.pop("seal_sha256")
    expected_seal = hashlib.sha256(
        json.dumps(sealed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert seal_sha256 == expected_seal
    assert sealed["method"] == {
        "id": "co_training",
        "profile": "paper:blum-mitchell-1998-webkb-course-table2",
    }
    assert sealed["dataset"] == {
        "id": "webkb_course_cotraining",
        "fingerprint": "5a1d45139e2a1ccb17abf374fb6ec17dc7d0bb3f9ff7caf08935d7731bb80683",
        "content_sha256": "894e2f310924fd66239632029db7738b8e1fcd330ffb86cb201cf6937ed9a264",
    }
    assert sealed["code"] == {
        "git_dirty": False,
        "git_sha": _GIT_SHA,
        "git_diff_sha256": _GIT_DIFF_SHA256,
    }
    assert len({item["split_fingerprint"] for item in sealed["runs"]}) == 5


def test_evaluate_co_training_v2_complete_numeric_match_and_historical_controls(
    tmp_path: Path,
) -> None:
    report = evaluate_historical_runs(
        protocol_name="co-training-v2",
        run_json_paths=_runs(tmp_path, count=5, protocol_name="co-training-v2"),
    )

    assert report["protocol"] == {
        "name": "co-training-v2",
        "profile": "paper:blum-mitchell-1998-webkb-course-confirmation-v2",
        "method_id": "co_training",
        "dataset_id": "webkb_course_cotraining",
        "dataset_fingerprint": ("5a1d45139e2a1ccb17abf374fb6ec17dc7d0bb3f9ff7caf08935d7731bb80683"),
        "dataset_content_sha256": (
            "894e2f310924fd66239632029db7738b8e1fcd330ffb86cb201cf6937ed9a264"
        ),
        "expected_seeds": [6, 7, 8, 9, 10],
        "target_error": 0.05,
        "margin_absolute": 0.02,
    }
    assert report["completeness"] == {
        "expected": 5,
        "successful": 5,
        "duplicate_seeds": [],
        "missing_seeds": [],
        "unexpected_seeds": [],
        "complete": True,
    }
    assert report["statistics"]["mean_error"] == pytest.approx(0.05)
    assert report["statistics"]["ci95_low"] == pytest.approx(0.05)
    assert report["statistics"]["ci95_high"] == pytest.approx(0.05)
    assert report["statistics"]["target_in_ci95"] is True
    assert report["statistics"]["within_margin"] is True
    assert report["numeric_status"] == "numeric_matched"
    assert report["protocol_diagnostics_passed"] is True
    assert report["result_status"] == "replicated_paper_approx"
    assert report["secondary_diagnostics"]["fulltext"]["target_error"] == 0.062
    assert report["secondary_diagnostics"]["fulltext"]["target_in_ci95"] is True
    assert report["secondary_diagnostics"]["inlinks"]["target_error"] == 0.116
    assert report["secondary_diagnostics"]["inlinks"]["target_in_ci95"] is True
    assert [record["seed"] for record in report["runs"]] == [6, 7, 8, 9, 10]


def test_evaluate_co_training_nigam_complete_numeric_match_and_controls(
    tmp_path: Path,
) -> None:
    report = evaluate_historical_runs(
        protocol_name="co-training-nigam",
        run_json_paths=_runs(
            tmp_path,
            count=10,
            error=0.054,
            protocol_name="co-training-nigam",
        ),
    )

    assert report["protocol"] == {
        "name": "co-training-nigam",
        "profile": "paper:nigam-ghani2000-webkb-table2",
        "method_id": "co_training",
        "dataset_id": "webkb_course_cotraining",
        "dataset_fingerprint": ("5a1d45139e2a1ccb17abf374fb6ec17dc7d0bb3f9ff7caf08935d7731bb80683"),
        "dataset_content_sha256": (
            "894e2f310924fd66239632029db7738b8e1fcd330ffb86cb201cf6937ed9a264"
        ),
        "expected_seeds": list(range(21, 31)),
        "target_error": 0.054,
        "margin_absolute": 0.02,
    }
    assert report["statistics"]["mean_error"] == pytest.approx(0.054)
    assert report["statistics"]["n"] == 10
    assert report["statistics"]["target_in_ci95"] is True
    assert report["numeric_status"] == "numeric_matched"
    assert report["protocol_diagnostics_passed"] is True
    assert report["result_status"] == "replicated_paper_approx"
    assert report["scientific_status"] == "paper_approx"
    assert report["critical_unknowns"] == [
        "the ten historical split seeds are unavailable",
        "historical HTML and anchor-text tokenization is under-specified",
        "the historical cross-view collision policy is unpublished; the ordered "
        "multiset policy is a reconstruction",
    ]
    assert list(report["secondary_diagnostics"]) == ["nb12", "nb788"]
    assert report["secondary_diagnostics"]["nb12"]["target_error"] == 0.130
    assert report["secondary_diagnostics"]["nb12"]["target_in_ci95"] is True
    assert report["secondary_diagnostics"]["nb788"]["target_error"] == 0.033
    assert report["secondary_diagnostics"]["nb788"]["target_in_ci95"] is True
    assert [record["seed"] for record in report["runs"]] == list(range(21, 31))
    assert report["runs"][0]["test_nb12_error"] == pytest.approx(0.130)
    assert report["runs"][0]["test_nb788_error"] == pytest.approx(0.033)


def test_co_training_nigam_contract_locks_sampling_method_and_tokenizer() -> None:
    protocol = historical_acceptance._PROTOCOLS["co-training-nigam"]
    config = historical_acceptance._critical_config(protocol, seed=21)

    assert protocol.expected_seeds == tuple(range(21, 31))
    assert config["sampling"]["plan"]["split"]["stratify"] is False
    assert config["sampling"]["plan"]["labeling"] == {
        "mode": "count",
        "value": 12,
        "strategy": "random",
        "min_per_class": 0,
        "per_class": False,
        "fixed_indices": None,
        "class_counts": {"0": 9, "1": 3},
        "selection_order": "permutation",
    }
    assert config["method"]["params"] == {
        "classifier_id": "multinomial_nb",
        "classifier_backend": "sklearn",
        "classifier_params": {"alpha": 1.0, "fit_prior": True},
        "view_keys": ["fulltext", "inlinks"],
        "protocol": "shared_pool_exhaustive_multiset",
        "p": 1,
        "n": 3,
        "u": 75,
        "k": 0,
        "positive_label": 1,
        "negative_label": 0,
        "confidence_threshold": None,
        "dynamic_feature_selection": "none",
        "feature_selection_max_features": None,
        "selection_score": "posterior_probability",
    }
    assert config["views"]["plan"]["views"][0]["preprocess"]["steps"][1]["params"] == {
        "dense": True,
        "strip_html": True,
        "min_df": 1,
    }
    assert historical_acceptance._expected_sampling_contract(protocol, seed=21)["stats"][
        "train_labeled"
    ] == {"n": 12, "classes": {"0": 9, "1": 3}}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("n_iter", 97),
        ("unique_pseudo_labeled_examples", 775),
        ("pseudo_labels_added_to_shared_l", 779),
        ("final_labeled_size", 791),
        ("remaining_unlabeled_count", 1),
        ("views_select_from_same_pre_round_pool", False),
        ("overlap_policy", "sequential_unique_view1_then_view2"),
        ("addition_policy", "sequential_unique_view1_then_view2"),
        ("word_likelihood_smoothing", "none"),
        ("class_prior_smoothing", "empirical"),
        ("test_metrics_used_for_protocol_selection", True),
    ],
)
def test_co_training_nigam_rejects_inconsistent_exhaustion_diagnostics(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    diagnostics = _co_nigam_diagnostics(seed=21)
    diagnostics[key] = value

    with pytest.raises(HistoricalAcceptanceError, match=key):
        historical_acceptance._validate_co_diagnostics(
            diagnostics,
            path=tmp_path / "run.json",
            protocol=historical_acceptance._PROTOCOLS["co-training-nigam"],
            expected_seed=historical_acceptance.derive_seed(21, "method"),
        )


def test_co_training_nigam_accepts_variable_same_pool_multiset_trace(tmp_path: Path) -> None:
    diagnostics = _co_nigam_diagnostics(seed=21)
    assert diagnostics["n_iter"] == 98
    assert diagnostics["pseudo_label_proposals_view1"] == 390
    assert diagnostics["pseudo_label_proposals_view2"] == 390
    assert diagnostics["overlap_count"] == 4
    assert diagnostics["same_label_overlap_count"] == 3
    assert diagnostics["conflicting_overlap_count"] == 1
    assert diagnostics["pseudo_labels_added_to_shared_l"] == 780
    assert diagnostics["final_labeled_size"] == 792
    assert diagnostics["round_trace"][-1]["pool_size_before"] == 2
    historical_acceptance._validate_co_diagnostics(
        diagnostics,
        path=tmp_path / "run.json",
        protocol=historical_acceptance._PROTOCOLS["co-training-nigam"],
        expected_seed=historical_acceptance.derive_seed(21, "method"),
    )


def test_co_training_nigam_rejects_inconsistent_same_pool_overlap_trace(
    tmp_path: Path,
) -> None:
    diagnostics = _co_nigam_diagnostics(seed=21)
    diagnostics["round_trace"][0]["overlap_indices"] = []

    with pytest.raises(HistoricalAcceptanceError, match="same-pool overlap/removal trace"):
        historical_acceptance._validate_co_diagnostics(
            diagnostics,
            path=tmp_path / "run.json",
            protocol=historical_acceptance._PROTOCOLS["co-training-nigam"],
            expected_seed=historical_acceptance.derive_seed(21, "method"),
        )


def test_co_training_v2_contract_is_distinct_and_v1_remains_unchanged() -> None:
    v1 = historical_acceptance._critical_config(
        historical_acceptance._PROTOCOLS["co-training"], seed=1
    )
    v2 = historical_acceptance._critical_config(
        historical_acceptance._PROTOCOLS["co-training-v2"], seed=6
    )

    assert v1["sampling"]["plan"]["split"]["stratify"] is True
    assert v1["method"]["params"]["protocol"] == "fixed_pool_binary"
    assert "dynamic_feature_selection" not in v1["method"]["params"]
    assert v1["views"]["plan"]["views"][0]["preprocess"]["steps"][1]["params"] == {
        "dense": True,
        "strip_html": True,
    }

    assert v2["run"]["seed"] == 6
    assert v2["sampling"]["plan"]["split"]["stratify"] is False
    assert v2["method"]["profile"] == ("paper:blum-mitchell-1998-webkb-course-confirmation-v2")
    assert v2["method"]["params"]["protocol"] == "fixed_pool_binary_feature_selection"
    assert v2["method"]["params"]["dynamic_feature_selection"] == ("mutual_information_presence")
    assert v2["method"]["params"]["feature_selection_max_features"] == 2000
    assert v2["method"]["params"]["selection_score"] == "craven_1998_normalized_nb"
    assert v2["views"]["plan"]["views"][0]["preprocess"]["steps"][1]["params"] == {
        "dense": True,
        "strip_html": True,
        "min_df": 1,
    }


def test_co_training_v2_sampling_accepts_seed_specific_class_counts(tmp_path: Path) -> None:
    paths = _runs(tmp_path, count=5, protocol_name="co-training-v2")
    for position, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifacts"]["sampling"]["stats"]["test"]["classes"] = {
            "0": 200 + position,
            "1": 63 - position,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = evaluate_historical_runs(
        protocol_name="co-training-v2",
        run_json_paths=paths,
    )
    assert report["completeness"]["complete"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("deterministic_fingerprint", "not deterministic for seed"),
        ("replay_indices", "indices do not replay from seed"),
    ],
)
def test_co_training_v2_rejects_non_deterministic_replay(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _runs(tmp_path, count=5, protocol_name="co-training-v2")
    path = paths[0]
    replay_root = path.parent / "sampling_split"
    if mutation == "deterministic_fingerprint":
        wrong = historical_acceptance._co_v2_split_fingerprint(seed=7)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifacts"]["sampling"]["split_fingerprint"] = wrong
        split_path = replay_root / "split.json"
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        split_payload["split_fingerprint"] = wrong
        split_path.write_text(json.dumps(split_payload), encoding="utf-8")
        manifest_path = replay_root / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["split_fingerprint"] = wrong
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        arrays_path = replay_root / "arrays.npz"
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        arrays["idx__train_labeled"] = arrays["idx__train_labeled"].copy()
        arrays["idx__train_labeled"][0] = arrays["idx__train_labeled"][0] + 1
        np.savez_compressed(arrays_path, **arrays)
    _reseal_replay_files(path)

    with pytest.raises(HistoricalAcceptanceError, match=message):
        evaluate_historical_runs(protocol_name="co-training-v2", run_json_paths=paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("score", "selection_score_space"),
        ("test_selection", "test_metrics_used_for_protocol_selection"),
        ("feature_count", "final_feature_count_view1 must be in"),
        ("feature_digest", "final_features_sha256_view2"),
        ("feature_maximum", "final_maximum_mutual_information_view1"),
        ("round_selector", "feature_selection"),
        ("round_feature_count", "selected_feature_count_view2 must be in"),
    ],
)
def test_co_training_v2_rejects_incompatible_selection_diagnostics(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    diagnostics = _co_v2_diagnostics(seed=6)
    if mutation == "score":
        diagnostics["selection_score_space"] = "log_probability"
    elif mutation == "test_selection":
        diagnostics["test_metrics_used_for_protocol_selection"] = True
    elif mutation == "feature_count":
        diagnostics["final_feature_count_view1"] = 2001
    elif mutation == "feature_digest":
        diagnostics["final_features_sha256_view2"] = "invalid"
    elif mutation == "feature_maximum":
        diagnostics["final_maximum_mutual_information_view1"] = float("nan")
    elif mutation == "round_selector":
        diagnostics["round_trace"][0]["feature_selection"] = "none"
    else:
        diagnostics["round_trace"][0]["selected_feature_count_view2"] = 0

    with pytest.raises(HistoricalAcceptanceError, match=message):
        historical_acceptance._validate_co_diagnostics(
            diagnostics,
            path=tmp_path / "run.json",
            protocol=historical_acceptance._PROTOCOLS["co-training-v2"],
            expected_seed=historical_acceptance.derive_seed(6, "method"),
        )


def test_evaluate_self_training_uses_ddof_one_and_rejects_numeric_mismatch(
    tmp_path: Path,
) -> None:
    paths = _runs(tmp_path, count=50, error=0.20, protocol_name="self-training")
    _write_run(paths[0], seed=1, accuracy=0.79, protocol_name="self-training")
    report = evaluate_historical_runs(protocol_name="self-training", run_json_paths=paths)

    errors = [0.21] + [0.20] * 49
    mean = sum(errors) / 50
    expected_std = math.sqrt(sum((value - mean) ** 2 for value in errors) / 49)
    assert report["statistics"]["sample_std_error"] == pytest.approx(expected_std)
    assert report["statistics"]["target_in_ci95"] is False
    assert report["statistics"]["within_margin"] is False
    assert report["numeric_status"] == "numeric_not_matched"
    assert report["result_status"] == "failed_replication"
    assert report["scientific_status"] == "paper_approx"
    assert "secondary_diagnostics" not in report
    assert set(report["runs"][0]) == {
        "seed",
        "test_accuracy",
        "test_error",
        "split_fingerprint",
        "run_json",
    }


def test_self_training_v2_protocol_locks_fresh_seeds_and_dynamic_distance_contract() -> None:
    protocol = historical_acceptance._PROTOCOLS["self-training-v2"]
    assert protocol.expected_seeds == tuple(range(51, 101))
    config = historical_acceptance._critical_config(protocol, seed=51)

    assert config["run"]["seed"] == 51
    assert [step["id"] for step in config["preprocess"]["plan"]["steps"]] == [
        "labels.encode",
        "core.ensure_2d",
        "core.to_numpy",
    ]
    assert config["method"]["profile"] == protocol.protocol_id
    assert config["method"]["params"] == {
        "classifier_id": "knn",
        "classifier_backend": "numpy",
        "classifier_params": {"k": 1, "metric": "euclidean", "weights": "uniform"},
        "max_iter": 40,
        "confidence_threshold": None,
        "max_new_labels": None,
        "min_new_labels": 1,
        "use_group_propagation": False,
        "selection_strategy": "li_zhou_2005_1nn_distance",
        "paper_pool_size_unspecified": 75,
        "paper_candidates_per_class_unspecified": 1,
        "paper_distance_confidence_unspecified": "nearest_neighbor_distance",
        "paper_feature_scaling_unspecified": "dynamic_labeled_minmax",
    }


def test_self_training_v2_diagnostics_authenticate_the_persistent_pool(tmp_path: Path) -> None:
    protocol = historical_acceptance._PROTOCOLS["self-training-v2"]
    diagnostics = _self_v2_diagnostics()
    historical_acceptance._validate_self_diagnostics(
        diagnostics,
        path=tmp_path / "run.json",
        protocol=protocol,
    )

    diagnostics["round_trace"][1]["pool_indices"] = [1, *range(3, 77)]
    with pytest.raises(HistoricalAcceptanceError, match="persistent pool was not retained"):
        historical_acceptance._validate_self_diagnostics(
            diagnostics,
            path=tmp_path / "run.json",
            protocol=protocol,
        )

    recycled = _self_v2_diagnostics()
    recycled["round_trace"][1]["pool_indices"][-1] = 0
    with pytest.raises(HistoricalAcceptanceError, match="pool trace is inconsistent"):
        historical_acceptance._validate_self_diagnostics(
            recycled,
            path=tmp_path / "run.json",
            protocol=protocol,
        )

    out_of_range = _self_v2_diagnostics()
    out_of_range["round_trace"][0]["pool_indices"][-1] = 121
    with pytest.raises(HistoricalAcceptanceError, match="pool trace is inconsistent"):
        historical_acceptance._validate_self_diagnostics(
            out_of_range,
            path=tmp_path / "run.json",
            protocol=protocol,
        )

    with pytest.raises(HistoricalAcceptanceError, match="seed must equal 52"):
        historical_acceptance._validate_self_diagnostics(
            _self_v2_diagnostics(),
            path=tmp_path / "run.json",
            protocol=protocol,
            expected_seed=52,
        )


def test_self_training_v2_end_to_end_acceptance_is_capped_at_paper_approx(
    tmp_path: Path,
) -> None:
    report = evaluate_historical_runs(
        protocol_name="self-training-v2",
        run_json_paths=_runs(
            tmp_path,
            count=50,
            error=0.079,
            protocol_name="self-training-v2",
        ),
    )

    assert [record["seed"] for record in report["runs"]] == list(range(51, 101))
    assert report["numeric_status"] == "numeric_matched"
    assert report["protocol_diagnostics_passed"] is True
    assert report["result_status"] == "replicated_paper_approx"
    assert report["scientific_status"] == "paper_approx"


def test_numeric_match_requires_target_inside_confidence_interval(tmp_path: Path) -> None:
    report = evaluate_historical_runs(
        protocol_name="co-training",
        run_json_paths=_runs(tmp_path, count=5, error=0.06),
    )

    assert report["statistics"]["within_margin"] is True
    assert report["statistics"]["target_in_ci95"] is False
    assert report["numeric_status"] == "numeric_not_matched"


def test_co_training_aggregates_each_view_with_sample_student_interval(
    tmp_path: Path,
) -> None:
    fulltext_errors = [0.03, 0.05, 0.07, 0.09, 0.11]
    inlinks_errors = [0.08, 0.10, 0.12, 0.14, 0.16]
    paths = [
        _write_run(
            tmp_path / f"seed-{seed}" / "run.json",
            seed=seed,
            accuracy=0.95,
            view_accuracies={
                "fulltext": 1.0 - fulltext_errors[seed - 1],
                "inlinks": 1.0 - inlinks_errors[seed - 1],
            },
        )
        for seed in range(1, 6)
    ]

    report = evaluate_historical_runs(protocol_name="co-training", run_json_paths=paths)

    for view_name, errors, target_error in (
        ("fulltext", fulltext_errors, 0.062),
        ("inlinks", inlinks_errors, 0.116),
    ):
        diagnostic = report["secondary_diagnostics"][view_name]
        mean = sum(errors) / len(errors)
        sample_std = math.sqrt(sum((error - mean) ** 2 for error in errors) / (len(errors) - 1))
        half_width = diagnostic["student_t_critical"] * sample_std / math.sqrt(len(errors))
        assert diagnostic["n"] == 5
        assert diagnostic["mean_error"] == pytest.approx(mean)
        assert diagnostic["sample_std_error"] == pytest.approx(sample_std)
        assert diagnostic["std_ddof"] == 1
        assert diagnostic["ci95_low"] == pytest.approx(mean - half_width)
        assert diagnostic["ci95_high"] == pytest.approx(mean + half_width)
        assert diagnostic["target_error"] == target_error
        assert diagnostic["absolute_difference"] == pytest.approx(abs(mean - target_error))
        assert diagnostic["target_in_ci95"] is True


def test_co_training_view_diagnostics_never_change_primary_numeric_status(
    tmp_path: Path,
) -> None:
    secondary_mismatch = [
        _write_run(
            tmp_path / "secondary-mismatch" / f"seed-{seed}" / "run.json",
            seed=seed,
            accuracy=0.95,
            view_accuracies={"fulltext": 0.5, "inlinks": 0.5},
        )
        for seed in range(1, 6)
    ]
    primary_match = evaluate_historical_runs(
        protocol_name="co-training", run_json_paths=secondary_mismatch
    )

    assert primary_match["numeric_status"] == "numeric_matched"
    assert primary_match["protocol_diagnostics_passed"] is False
    assert primary_match["result_status"] == "failed_replication"
    assert all(
        diagnostic["target_in_ci95"] is False
        for diagnostic in primary_match["secondary_diagnostics"].values()
    )

    primary_mismatch = [
        _write_run(
            tmp_path / "primary-mismatch" / f"seed-{seed}" / "run.json",
            seed=seed,
            accuracy=0.75,
        )
        for seed in range(1, 6)
    ]
    secondary_match = evaluate_historical_runs(
        protocol_name="co-training", run_json_paths=primary_mismatch
    )

    assert secondary_match["numeric_status"] == "numeric_not_matched"
    assert secondary_match["protocol_diagnostics_passed"] is True
    assert secondary_match["result_status"] == "failed_replication"
    assert all(
        diagnostic["target_in_ci95"] is True
        for diagnostic in secondary_match["secondary_diagnostics"].values()
    )


@pytest.mark.parametrize(
    ("keys", "value", "message"),
    [
        (("metrics", "test_fulltext"), None, "metrics.test_fulltext must be a mapping"),
        (
            ("metrics", "test_fulltext", "accuracy"),
            True,
            "metrics.test_fulltext.accuracy",
        ),
        (
            ("metrics", "test_fulltext", "accuracy"),
            float("nan"),
            "metrics.test_fulltext.accuracy",
        ),
        (
            ("metrics", "test_fulltext", "accuracy"),
            -0.01,
            "metrics.test_fulltext.accuracy",
        ),
        (("metrics", "test_inlinks"), None, "metrics.test_inlinks must be a mapping"),
        (
            ("metrics", "test_inlinks", "accuracy"),
            float("inf"),
            "metrics.test_inlinks.accuracy",
        ),
        (
            ("metrics", "test_inlinks", "accuracy"),
            1.01,
            "metrics.test_inlinks.accuracy",
        ),
    ],
)
def test_co_training_requires_finite_bounded_view_accuracies(
    tmp_path: Path,
    keys: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    paths = _runs(tmp_path, count=5)
    _replace_nested(paths[2], keys, value)

    with pytest.raises(HistoricalAcceptanceError, match=message):
        evaluate_historical_runs(protocol_name="co-training", run_json_paths=paths)


def test_co_training_accepts_closed_accuracy_bounds_for_views(tmp_path: Path) -> None:
    paths = [
        _write_run(
            tmp_path / f"seed-{seed}" / "run.json",
            seed=seed,
            accuracy=0.95,
            view_accuracies={"fulltext": 0.0, "inlinks": 1.0},
        )
        for seed in range(1, 6)
    ]

    report = evaluate_historical_runs(protocol_name="co-training", run_json_paths=paths)

    assert report["secondary_diagnostics"]["fulltext"]["mean_error"] == 1.0
    assert report["secondary_diagnostics"]["inlinks"]["mean_error"] == 0.0


@pytest.mark.parametrize(
    ("protocol", "count"),
    [("co-training", 5), ("co-training-nigam", 10), ("self-training", 50)],
)
def test_student_t_fallback_has_prefixed_protocol_quantiles(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    count: int,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "scipy.stats":
            raise ImportError("blocked for fallback test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    value, source = _student_t_critical_95(count)

    assert source == "prefixed_student_t"
    assert value == pytest.approx(
        {5: 2.7764451051977987, 10: 2.2621571628540993, 50: 2.009575234489209}[count]
    )
    assert protocol in historical_acceptance._PROTOCOLS


def test_student_t_uses_scipy_when_available() -> None:
    pytest.importorskip("scipy.stats")
    value, source = _student_t_critical_95(5)
    assert source == "scipy.stats.t"
    assert value == pytest.approx(2.7764451051977987)


def test_student_t_fallback_rejects_unknown_sample_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "scipy.stats":
            raise ImportError("blocked for fallback test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(HistoricalAcceptanceError, match="no dependency-free"):
        _student_t_critical_95(6)


def test_discovery_accepts_exactly_one_source_and_finds_nested_runs(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "nested" / "run.json", seed=1, accuracy=0.9)
    assert discover_run_jsons(sweep_root=tmp_path, run_json_paths=None) == [run]
    assert discover_run_jsons(sweep_root=None, run_json_paths=[run]) == [run]

    with pytest.raises(HistoricalAcceptanceError, match="exactly one"):
        discover_run_jsons(sweep_root=tmp_path, run_json_paths=[run])
    with pytest.raises(HistoricalAcceptanceError, match="exactly one"):
        discover_run_jsons(sweep_root=None, run_json_paths=None)
    with pytest.raises(HistoricalAcceptanceError, match="not a directory"):
        discover_run_jsons(sweep_root=tmp_path / "absent", run_json_paths=None)
    (tmp_path / "empty").mkdir()
    with pytest.raises(HistoricalAcceptanceError, match="no run.json"):
        discover_run_jsons(sweep_root=tmp_path / "empty", run_json_paths=None)


def test_evaluator_rejects_duplicate_missing_unexpected_and_unknown_protocol(
    tmp_path: Path,
) -> None:
    paths = _runs(tmp_path / "valid", count=5)
    duplicate = _write_run(tmp_path / "duplicate" / "run.json", seed=1, accuracy=0.95)
    with pytest.raises(HistoricalAcceptanceError, match="duplicate seeds: 1"):
        evaluate_historical_runs(protocol_name="co-training", run_json_paths=[*paths, duplicate])

    unexpected = _write_run(tmp_path / "unexpected" / "run.json", seed=7, accuracy=0.95)
    with pytest.raises(HistoricalAcceptanceError, match=r"missing=\[5\], unexpected=\[7\]"):
        evaluate_historical_runs(
            protocol_name="co-training", run_json_paths=[*paths[:4], unexpected]
        )

    with pytest.raises(HistoricalAcceptanceError, match="unknown protocol"):
        evaluate_historical_runs(protocol_name="absent", run_json_paths=[])


@pytest.mark.parametrize(
    ("keys", "value", "message"),
    [
        (("artifacts",), None, "artifacts must be a mapping"),
        (("artifacts", "method"), None, "artifacts.method must be a mapping"),
        (("artifacts", "method", "id"), "pseudo_label", "artifacts.method.id"),
        (("artifacts", "method", "profile"), "standardized", "artifacts.method.profile"),
        (("artifacts", "dataset"), None, "artifacts.dataset must be a mapping"),
        (("artifacts", "dataset", "id"), "wine", "artifacts.dataset.id"),
        (("artifacts", "dataset", "fingerprint"), "0" * 64, "dataset.fingerprint"),
        (("artifacts", "dataset", "content_sha256"), "0" * 64, "content_sha256"),
        (("versions",), None, "versions must be a mapping"),
        (("versions", "git_dirty"), True, "versions.git_dirty must be false"),
        (("versions", "git_dirty"), None, "versions.git_dirty must be false"),
        (("versions", "git_sha"), "g" * 40, "versions.git_sha"),
        (("versions", "git_sha"), "a" * 39, "versions.git_sha"),
        (("versions", "git_diff_sha256"), "g" * 64, "versions.git_diff_sha256"),
        (("versions", "python"), "", "versions.python"),
        (("versions", "numpy"), None, "versions.numpy"),
        (("versions", "scikit_learn"), "   ", "versions.scikit_learn"),
        (("artifacts", "sampling"), None, "artifacts.sampling must be a mapping"),
        (
            ("artifacts", "sampling", "split_fingerprint"),
            "not-a-digest",
            "artifacts.sampling.split_fingerprint",
        ),
    ],
)
def test_evaluator_rejects_unlocked_or_unauthenticated_runs(
    tmp_path: Path,
    keys: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    path = _write_run(tmp_path / "run.json", seed=1, accuracy=0.95)
    _replace_nested(path, keys, value)

    with pytest.raises(HistoricalAcceptanceError, match=message):
        evaluate_historical_runs(protocol_name="co-training", run_json_paths=[path])


@pytest.mark.parametrize(
    ("keys", "value", "message"),
    [
        (("versions", "git_sha"), "c" * 40, "approved commit"),
        (
            ("versions", "git_diff_sha256"),
            "d" * 64,
            "versions.git_diff_sha256 differs",
        ),
        (("versions", "python"), "3.13.0", "environment differs"),
        (("versions", "numpy"), "9.0.0", "environment differs"),
        (("versions", "scikit_learn"), "9.0.0", "environment differs"),
    ],
)
def test_evaluator_requires_one_code_and_environment_across_runs(
    tmp_path: Path,
    keys: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    paths = _runs(tmp_path, count=5)
    _replace_nested(paths[1], keys, value)

    with pytest.raises(HistoricalAcceptanceError, match=message):
        evaluate_historical_runs(protocol_name="co-training", run_json_paths=paths)


def test_evaluator_rejects_reused_split_fingerprint(tmp_path: Path) -> None:
    paths = _runs(tmp_path, count=5)
    first = json.loads(paths[0].read_text(encoding="utf-8"))
    split_fingerprint = first["artifacts"]["sampling"]["split_fingerprint"]
    _replace_nested(paths[1], ("artifacts", "sampling", "split_fingerprint"), split_fingerprint)
    second = json.loads(paths[1].read_text(encoding="utf-8"))
    manifest_path = paths[1].parent / "sampling_split" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_fingerprint"] = split_fingerprint
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    second["artifacts"]["sampling"]["replay"]["manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    paths[1].write_text(json.dumps(second), encoding="utf-8")

    with pytest.raises(HistoricalAcceptanceError, match="duplicate split fingerprints"):
        evaluate_historical_runs(protocol_name="co-training", run_json_paths=paths)


def test_hex_provenance_is_case_insensitive_and_sealed_lowercase(tmp_path: Path) -> None:
    paths = _runs(tmp_path, count=5)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["versions"]["git_sha"] = payload["versions"]["git_sha"].upper()
        payload["versions"]["git_diff_sha256"] = payload["versions"]["git_diff_sha256"].upper()
        payload["artifacts"]["sampling"]["split_fingerprint"] = payload["artifacts"]["sampling"][
            "split_fingerprint"
        ].upper()
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = evaluate_historical_runs(protocol_name="co-training", run_json_paths=paths)

    assert report["sealed_provenance"]["code"]["git_sha"] == _GIT_SHA
    assert report["sealed_provenance"]["code"]["git_diff_sha256"] == _GIT_DIFF_SHA256
    assert all(
        item["split_fingerprint"] == item["split_fingerprint"].lower()
        for item in report["sealed_provenance"]["runs"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_config_key", "config.method.params.k is required"),
        ("different_config_value", "config.method.params.k must equal 30"),
        ("different_effective_hash", "effective config hash differs"),
    ],
)
def test_evaluator_rejects_modified_critical_contract(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path = _write_run(tmp_path / "run.json", seed=1, accuracy=0.95)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing_config_key":
        del payload["config"]["method"]["params"]["k"]
    elif mutation == "different_config_value":
        payload["config"]["method"]["params"]["k"] = 31
    else:
        payload["hashes"]["effective_config_hash"] = "c" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HistoricalAcceptanceError, match=message):
        evaluate_historical_runs(protocol_name="co-training", run_json_paths=[path])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_manifest", "cannot authenticate replay manifest"),
        ("manifest_sha", "manifest SHA-256 does not match"),
        ("file_set", "must contain arrays.npz and split.json"),
        ("missing_file", "cannot authenticate replay file arrays.npz"),
        ("file_sha", "replay file arrays.npz SHA-256 mismatch"),
    ],
)
def test_replay_authentication_rejects_every_unsealed_layer(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path = _write_run(tmp_path / mutation / "run.json", seed=1, accuracy=0.95)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sampling = payload["artifacts"]["sampling"]
    replay_root = path.parent / "sampling_split"
    manifest_path = replay_root / "MANIFEST.json"
    if mutation == "missing_manifest":
        manifest_path.unlink()
    elif mutation == "manifest_sha":
        sampling["replay"]["manifest_sha256"] = "0" * 64
    elif mutation == "file_set":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["files"]["split.json"]
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
        manifest_path.write_bytes(manifest_bytes)
        sampling["replay"]["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    elif mutation == "missing_file":
        (replay_root / "arrays.npz").unlink()
    else:
        (replay_root / "arrays.npz").write_bytes(b"tampered")

    protocol = historical_acceptance._PROTOCOLS["co-training"]
    with pytest.raises(HistoricalAcceptanceError, match=message):
        historical_acceptance._validate_replay(
            run_json_path=path,
            sampling=sampling,
            protocol=protocol,
            split_fingerprint=sampling["split_fingerprint"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("integer_type", "final_labeled_size must be an integer"),
        ("list_type", "round_trace must be a list"),
        ("iterations", "n_iter must be in"),
        ("final_sizes", "final L/U sizes are inconsistent"),
        ("trace_length", "round_trace length is inconsistent"),
        ("quota", "round quotas are inconsistent"),
        ("acceptance", "paper candidates must all be accepted"),
        ("pool", "pool trace is inconsistent"),
        ("labeled_trajectory", "labeled trajectory is inconsistent"),
        ("unlabeled_trajectory", "unlabeled trajectory is inconsistent"),
        ("aggregate", "aggregate trace is inconsistent"),
    ],
)
def test_self_training_diagnostics_reject_inconsistent_trajectories(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    diagnostics = _self_diagnostics()
    trace = diagnostics["round_trace"]
    round_zero = trace[0]
    if mutation == "integer_type":
        diagnostics["final_labeled_size"] = True
    elif mutation == "list_type":
        diagnostics["round_trace"] = None
    elif mutation == "iterations":
        diagnostics["n_iter"] = 41
    elif mutation == "final_sizes":
        diagnostics["final_labeled_size"] = 15
    elif mutation == "trace_length":
        diagnostics["n_iter"] = 2
    elif mutation == "quota":
        round_zero["accepted_labels"] = []
    elif mutation == "acceptance":
        round_zero["accepted_indices"] = [1]
    elif mutation == "pool":
        round_zero["pool_indices"] = []
    elif mutation == "labeled_trajectory":
        round_zero["labeled_after"] = 15
    elif mutation == "unlabeled_trajectory":
        round_zero["remaining_unlabeled"] = 119
    else:
        diagnostics["n_iter"] = 0
        diagnostics["round_trace"] = []

    with pytest.raises(HistoricalAcceptanceError, match=message):
        historical_acceptance._validate_self_diagnostics(diagnostics, path=tmp_path / "run.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("initial_pool", "initial pool must contain 75"),
        ("remaining", "remaining U size is inconsistent"),
        ("trace_length", "round_trace must contain 30"),
        ("quota", "round quotas must be 4\\+4"),
        ("labels", "positive/negative quota is inconsistent"),
        ("removed", "removed set is inconsistent"),
        ("replenished", "must replenish eight"),
        ("pool_sizes", "pool sizes are inconsistent"),
        ("replenishment", "replenishment trace is inconsistent"),
        ("aggregate", "unique promotions are inconsistent"),
    ],
)
def test_co_training_diagnostics_reject_inconsistent_trajectories(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    diagnostics = _co_diagnostics()
    trace = diagnostics["round_trace"]
    round_zero = trace[0]
    if mutation == "initial_pool":
        diagnostics["initial_pool_indices"] = []
    elif mutation == "remaining":
        diagnostics["remaining_unlabeled_count"] = 0
    elif mutation == "trace_length":
        diagnostics["round_trace"] = trace[:-1]
    elif mutation == "quota":
        round_zero["selected_by_view1"] = round_zero["selected_by_view1"][:-1]
    elif mutation == "labels":
        round_zero["selected_by_view1"][0]["label"] = 0
    elif mutation == "removed":
        round_zero["removed_indices"] = [0, 0, 1, 2]
    elif mutation == "replenished":
        round_zero["replenished_indices"] = round_zero["replenished_indices"][:-1]
    elif mutation == "pool_sizes":
        round_zero["pool_size_before"] = 74
    elif mutation == "replenishment":
        round_zero["pool_growth"] = 0
    else:
        diagnostics["unique_pseudo_labeled_examples"] += 1
        diagnostics["remaining_unlabeled_count"] -= 1

    with pytest.raises(HistoricalAcceptanceError, match=message):
        historical_acceptance._validate_co_diagnostics(diagnostics, path=tmp_path / "run.json")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"run": {"seed": 1, "status": "failed"}, "metrics": {}}, "run.status"),
        (
            {
                "run": {"seed": 1, "status": "success"},
                "metrics": {},
                "error": "contradictory",
            },
            "must not contain an error",
        ),
        ({"run": {"seed": True, "status": "success"}, "metrics": {}}, "run.seed"),
        ({"run": {"seed": 1, "status": "success"}, "metrics": None}, "metrics"),
        (
            {"run": {"seed": 1, "status": "success"}, "metrics": {"test": None}},
            "metrics.test",
        ),
        (
            {
                "run": {"seed": 1, "status": "success"},
                "metrics": {"test": {"accuracy": True}},
            },
            "metrics.test.accuracy",
        ),
        (
            {
                "run": {"seed": 1, "status": "success"},
                "metrics": {"test": {"accuracy": float("nan")}},
            },
            "metrics.test.accuracy",
        ),
        (
            {
                "run": {"seed": 1, "status": "success"},
                "metrics": {"test": {"accuracy": 1.1}},
            },
            "metrics.test.accuracy",
        ),
    ],
)
def test_evaluator_rejects_incomplete_run_payloads(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HistoricalAcceptanceError, match=message):
        evaluate_historical_runs(protocol_name="co-training", run_json_paths=[path])


@pytest.mark.parametrize("content", ["[]", "not-json"])
def test_evaluator_rejects_invalid_json(tmp_path: Path, content: str) -> None:
    path = tmp_path / "run.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(HistoricalAcceptanceError):
        evaluate_historical_runs(protocol_name="co-training", run_json_paths=[path])


def test_write_report_is_atomic_and_writes_one_tsv_row(tmp_path: Path) -> None:
    report = evaluate_historical_runs(
        protocol_name="co-training", run_json_paths=_runs(tmp_path / "runs", count=5)
    )
    output_json = tmp_path / "reports" / "acceptance.json"
    output_tsv = tmp_path / "reports" / "acceptance.tsv"
    write_report(report=report, output_json=output_json, output_tsv=output_tsv)

    assert json.loads(output_json.read_text(encoding="utf-8"))["numeric_status"] == (
        "numeric_matched"
    )
    with output_tsv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["scientific_status"] == "paper_approx"
    assert rows[0]["result_status"] == "replicated_paper_approx"
    assert not list(output_tsv.parent.glob("*.tmp"))

    with pytest.raises(HistoricalAcceptanceError, match="different paths"):
        write_report(report=report, output_json=output_json, output_tsv=output_json)


def test_atomic_text_cleans_temporary_file_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(historical_acceptance.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic"):
        _atomic_write_text(tmp_path / "report.tsv", "header\n")
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_success_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sweep = tmp_path / "sweep"
    _runs(sweep, count=5)
    output_json = tmp_path / "report.json"
    output_tsv = tmp_path / "report.tsv"
    assert (
        main(
            [
                "--protocol",
                "co-training",
                "--expected-git-sha",
                _GIT_SHA,
                "--sweep-root",
                str(sweep),
                "--output-json",
                str(output_json),
                "--output-tsv",
                str(output_tsv),
            ]
        )
        == 0
    )
    assert "paper_approx (replicated_paper_approx; numeric_matched)" in capsys.readouterr().out

    mismatch_sweep = tmp_path / "mismatch"
    _runs(mismatch_sweep, count=5, error=0.25)
    assert (
        main(
            [
                "--protocol",
                "co-training",
                "--expected-git-sha",
                _GIT_SHA,
                "--sweep-root",
                str(mismatch_sweep),
                "--output-json",
                str(output_json),
                "--output-tsv",
                str(output_tsv),
            ]
        )
        == 3
    )
    assert "failed_replication" in capsys.readouterr().out

    assert (
        main(
            [
                "--protocol",
                "co-training",
                "--expected-git-sha",
                _GIT_SHA,
                "--run-json",
                str(sweep / "seed-1" / "run.json"),
                "--output-json",
                str(output_json),
                "--output-tsv",
                str(output_tsv),
            ]
        )
        == 2
    )
    assert "seed set mismatch" in capsys.readouterr().err
