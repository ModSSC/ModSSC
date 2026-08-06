from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from bench.campaign.acceptance.historical import (
    HistoricalAcceptanceError,
    HistoricalProtocol,
    _validate_replay,
)

_DIAGNOSTIC_PROFILE = "paper:blum-mitchell-1998-webkb-course-v2:diagnostic-dev"
_CONFIRMATION_PROFILE = "paper:blum-mitchell-1998-webkb-course-confirmation-v2"
_DATASET_ID = "webkb_course_cotraining"
_DATASET_FINGERPRINT = "5a1d45139e2a1ccb17abf374fb6ec17dc7d0bb3f9ff7caf08935d7731bb80683"
_DATASET_CONTENT_SHA256 = "894e2f310924fd66239632029db7738b8e1fcd330ffb86cb201cf6937ed9a264"
_DIAGNOSTIC_SEEDS = tuple(range(1, 6))
_CONFIRMATION_SEEDS = tuple(range(6, 11))
_ALLOWED_METRIC_SPLITS = {
    "train_labeled",
    "train_labeled_fulltext",
    "train_labeled_inlinks",
}
_TEST_TOKEN = re.compile(r"(?:^|_)test(?:_|$)", flags=re.IGNORECASE)

_METHOD_PARAMS = {
    "classifier_id": "multinomial_nb",
    "classifier_backend": "sklearn",
    "classifier_params": {"alpha": 1.0, "fit_prior": True},
    "view_keys": ["fulltext", "inlinks"],
    "protocol": "fixed_pool_binary_feature_selection",
    "p": 1,
    "n": 3,
    "u": 75,
    "k": 30,
    "positive_label": 1,
    "negative_label": 0,
    "confidence_threshold": None,
    "dynamic_feature_selection": "mutual_information_presence",
    "feature_selection_max_features": 2000,
    "selection_score": "craven_1998_normalized_nb",
}

_DIAGNOSTIC_PROTOCOL = HistoricalProtocol(
    protocol_id=_DIAGNOSTIC_PROFILE,
    method_id="co_training",
    dataset_id=_DATASET_ID,
    dataset_fingerprint=_DATASET_FINGERPRINT,
    dataset_content_sha256=_DATASET_CONTENT_SHA256,
    expected_seeds=_DIAGNOSTIC_SEEDS,
    target_error=0.05,
    margin_absolute=0.02,
    critical_unknowns=(),
)


def _mapping(value: Any, *, field: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalAcceptanceError(f"{path}: {field} must be a mapping")
    return value


def _sequence(value: Any, *, field: str, path: Path) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise HistoricalAcceptanceError(f"{path}: {field} must be a sequence")
    return value


def _locked_subset(actual: Any, expected: Any, *, field: str, path: Path) -> None:
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


def _hex_digest(value: Any, *, length: int, field: str, path: Path) -> str:
    if not (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdefABCDEF" for character in value)
    ):
        raise HistoricalAcceptanceError(f"{path}: {field} must be a {length}-character hex digest")
    return value.lower()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise HistoricalAcceptanceError(f"cannot read file: {path}") from exc


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HistoricalAcceptanceError(f"cannot read YAML card: {path}") from exc
    return _mapping(value, field="YAML root", path=path)


def _card_contract(
    *,
    profile: str,
    seed: int,
    seeds: tuple[int, ...],
    report_split: str,
    stratify: bool,
) -> dict[str, Any]:
    return {
        "run": {
            "seed": seed,
            "seeds": list(seeds),
            "seeded_sections": ["sampling", "preprocess", "views"],
            "fail_fast": True,
            "benchmark_mode": False,
        },
        "dataset": {
            "id": _DATASET_ID,
            "download": False,
            "options": {},
        },
        "sampling": {
            "seed": seed,
            "plan": {
                "split": {
                    "kind": "holdout",
                    "test_fraction": 0.25023786869647957,
                    "val_fraction": 0.0,
                    "stratify": stratify,
                    "shuffle": True,
                },
                "labeling": {
                    "mode": "count",
                    "value": 12,
                    "strategy": "proportional",
                    "min_per_class": 1,
                    "per_class": False,
                    "fixed_indices": None,
                },
                "imbalance": {"kind": "none"},
            },
        },
        "preprocess": {"seed": seed, "fit_on": "train"},
        "views": {"seed": seed},
        "method": {
            "kind": "inductive",
            "id": "co_training",
            "profile": profile,
            "device": {"device": "cpu", "dtype": "float32"},
            "params": _METHOD_PARAMS,
        },
        "evaluation": {
            "split_for_model_selection": None,
            "report_splits": [report_split],
            "metrics": ["accuracy", "macro_f1"],
        },
    }


def validate_cards(*, diagnostic_card: Path, confirmation_card: Path) -> dict[str, Any]:
    diagnostic = _load_yaml_mapping(diagnostic_card)
    confirmation = _load_yaml_mapping(confirmation_card)
    _locked_subset(
        diagnostic,
        _card_contract(
            profile=_DIAGNOSTIC_PROFILE,
            seed=1,
            seeds=_DIAGNOSTIC_SEEDS,
            report_split="train_labeled",
            stratify=True,
        ),
        field="diagnostic card",
        path=diagnostic_card,
    )
    _locked_subset(
        confirmation,
        _card_contract(
            profile=_CONFIRMATION_PROFILE,
            seed=6,
            seeds=_CONFIRMATION_SEEDS,
            report_split="test",
            stratify=False,
        ),
        field="confirmation card",
        path=confirmation_card,
    )

    for field in ("dataset", "preprocess", "views"):
        if diagnostic.get(field) != confirmation.get(field):
            if field in {"preprocess", "views"}:
                diagnostic_value = dict(
                    _mapping(diagnostic[field], field=field, path=diagnostic_card)
                )
                confirmation_value = dict(
                    _mapping(confirmation[field], field=field, path=confirmation_card)
                )
                diagnostic_value.pop("seed", None)
                confirmation_value.pop("seed", None)
                if diagnostic_value == confirmation_value:
                    continue
            raise HistoricalAcceptanceError(
                f"{confirmation_card}: {field} differs from the sealed diagnostic card"
            )
    diagnostic_method = _mapping(diagnostic.get("method"), field="method", path=diagnostic_card)
    confirmation_method = _mapping(
        confirmation.get("method"), field="method", path=confirmation_card
    )
    for field in ("kind", "id", "device", "params"):
        if diagnostic_method.get(field) != confirmation_method.get(field):
            raise HistoricalAcceptanceError(
                f"{confirmation_card}: method.{field} differs from the diagnostic card"
            )

    return {
        "diagnostic": {
            "path": str(diagnostic_card.resolve()),
            "sha256": _file_sha256(diagnostic_card),
            "profile": _DIAGNOSTIC_PROFILE,
            "seeds": list(_DIAGNOSTIC_SEEDS),
        },
        "confirmation": {
            "path": str(confirmation_card.resolve()),
            "sha256": _file_sha256(confirmation_card),
            "profile": _CONFIRMATION_PROFILE,
            "seeds": list(_CONFIRMATION_SEEDS),
        },
    }


def _metric_bundle(value: Any, *, field: str, path: Path) -> None:
    metrics = _mapping(value, field=field, path=path)
    if set(metrics) != {"accuracy", "macro_f1"}:
        raise HistoricalAcceptanceError(
            f"{path}: {field} must contain exactly accuracy and macro_f1"
        )
    for name, metric in metrics.items():
        if (
            isinstance(metric, bool)
            or not isinstance(metric, int | float)
            or not math.isfinite(float(metric))
            or not 0.0 <= float(metric) <= 1.0
        ):
            raise HistoricalAcceptanceError(f"{path}: {field}.{name} must be finite and in [0, 1]")


def _reject_test_diagnostic_fields(value: Any, *, field: str, path: Path) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            nested = f"{field}.{key_text}"
            if _TEST_TOKEN.search(key_text):
                if key_text == "test_metrics_used_for_protocol_selection" and item is False:
                    continue
                raise HistoricalAcceptanceError(
                    f"{path}: forbidden test-derived diagnostic field {nested}"
                )
            _reject_test_diagnostic_fields(item, field=nested, path=path)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            _reject_test_diagnostic_fields(item, field=f"{field}[{index}]", path=path)


def _integer(
    values: Mapping[str, Any],
    key: str,
    *,
    field: str,
    path: Path,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HistoricalAcceptanceError(f"{path}: {field}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise HistoricalAcceptanceError(f"{path}: {field}.{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise HistoricalAcceptanceError(f"{path}: {field}.{key} must be <= {maximum}")
    return value


def _finite_number(values: Mapping[str, Any], key: str, *, field: str, path: Path) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise HistoricalAcceptanceError(f"{path}: {field}.{key} must be finite")
    return float(value)


def _validate_diagnostics(value: Any, *, path: Path) -> dict[str, Any]:
    field = "artifacts.method.diagnostics"
    diagnostics = _mapping(value, field=field, path=path)
    _reject_test_diagnostic_fields(diagnostics, field=field, path=path)
    _locked_subset(
        diagnostics,
        {
            "protocol": "fixed_pool_binary_feature_selection",
            "p": 1,
            "n": 3,
            "u": 75,
            "k": 30,
            "n_iter": 30,
            "positive_label": 1,
            "negative_label": 0,
            "shared_labeled_multiset": True,
            "overlap_policy": "ordered_multiset_view1_then_view2",
            "combination_score_space": "summed_log_probability",
            "probability_underflow_safe": True,
            "dynamic_feature_selection": "mutual_information_presence",
            "feature_selection_max_features": 2000,
            "selection_score_space": "craven_1998_normalized_nb",
            "selection_diagnostics_scope": "training_and_pseudo_labels_only",
            "test_metrics_used_for_protocol_selection": False,
        },
        field=field,
        path=path,
    )
    summary: dict[str, Any] = {}
    for view in (1, 2):
        count_key = f"final_feature_count_view{view}"
        digest_key = f"final_features_sha256_view{view}"
        maximum_key = f"final_maximum_mutual_information_view{view}"
        summary[count_key] = _integer(
            diagnostics,
            count_key,
            field=field,
            path=path,
            minimum=1,
            maximum=2000,
        )
        summary[digest_key] = _hex_digest(
            diagnostics.get(digest_key),
            length=64,
            field=f"{field}.{digest_key}",
            path=path,
        )
        summary[maximum_key] = _finite_number(
            diagnostics,
            maximum_key,
            field=field,
            path=path,
        )

    trace = _sequence(diagnostics.get("round_trace"), field=f"{field}.round_trace", path=path)
    if len(trace) != 30:
        raise HistoricalAcceptanceError(f"{path}: {field}.round_trace must contain 30 rounds")
    for index, raw_round in enumerate(trace, start=1):
        round_field = f"{field}.round_trace[{index - 1}]"
        round_values = _mapping(raw_round, field=round_field, path=path)
        _locked_subset(
            round_values,
            {
                "round": index,
                "feature_selection": "mutual_information_presence",
            },
            field=round_field,
            path=path,
        )
        for view in (1, 2):
            count_key = f"selected_feature_count_view{view}"
            digest_key = f"selected_features_sha256_view{view}"
            maximum_key = f"maximum_mutual_information_view{view}"
            _integer(
                round_values,
                count_key,
                field=round_field,
                path=path,
                minimum=1,
                maximum=2000,
            )
            _hex_digest(
                round_values.get(digest_key),
                length=64,
                field=f"{round_field}.{digest_key}",
                path=path,
            )
            _finite_number(round_values, maximum_key, field=round_field, path=path)
    summary["rounds"] = len(trace)
    summary["pseudo_labels_added_to_shared_l"] = _integer(
        diagnostics,
        "pseudo_labels_added_to_shared_l",
        field=field,
        path=path,
        minimum=1,
    )
    summary["conflicting_overlap_count"] = _integer(
        diagnostics,
        "conflicting_overlap_count",
        field=field,
        path=path,
        minimum=0,
    )
    return summary


def _diagnostic_run_contract(*, seed: int) -> dict[str, Any]:
    return {
        "run": {
            "seed": seed,
            "seeded_sections": ["sampling", "preprocess", "views"],
            "benchmark_mode": False,
        },
        "dataset": {"id": _DATASET_ID, "download": False, "options": {}},
        "sampling": {
            "seed": seed,
            "plan": {
                "split": {
                    "kind": "holdout",
                    "test_fraction": 0.25023786869647957,
                    "val_fraction": 0.0,
                    "stratify": True,
                    "shuffle": True,
                },
                "labeling": {
                    "mode": "count",
                    "value": 12,
                    "strategy": "proportional",
                    "min_per_class": 1,
                    "per_class": False,
                    "fixed_indices": None,
                },
            },
        },
        "preprocess": {"seed": seed, "fit_on": "train"},
        "views": {"seed": seed},
        "method": {
            "kind": "inductive",
            "method_id": "co_training",
            "profile": _DIAGNOSTIC_PROFILE,
            "device": {"device": "cpu", "dtype": "float32"},
            "params": _METHOD_PARAMS,
        },
        "evaluation": {
            "split_for_model_selection": None,
            "report_splits": ["train_labeled"],
            "metrics": ["accuracy", "macro_f1"],
        },
    }


def _load_diagnostic_run(
    path: Path,
    *,
    expected_git_sha: str,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalAcceptanceError(f"cannot read run.json: {path}") from exc
    root = _mapping(payload, field="run.json", path=path)
    run = _mapping(root.get("run"), field="run", path=path)
    if run.get("status") != "success" or root.get("error") is not None:
        raise HistoricalAcceptanceError(f"{path}: diagnostic run must be successful")
    seed = run.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise HistoricalAcceptanceError(f"{path}: run.seed must be an integer")

    _locked_subset(
        root.get("protocol"),
        {
            "kind": "inductive",
            "report_splits": ["train_labeled"],
            "split_for_model_selection": None,
            "use_test_split": False,
        },
        field="protocol",
        path=path,
    )
    config = _mapping(root.get("config"), field="config", path=path)
    _locked_subset(
        config,
        _diagnostic_run_contract(seed=seed),
        field="config",
        path=path,
    )

    metrics = _mapping(root.get("metrics"), field="metrics", path=path)
    if set(metrics) != _ALLOWED_METRIC_SPLITS:
        raise HistoricalAcceptanceError(
            f"{path}: diagnostic metrics must contain exactly "
            f"{sorted(_ALLOWED_METRIC_SPLITS)!r}, got {sorted(metrics)!r}"
        )
    for split, bundle in metrics.items():
        _metric_bundle(bundle, field=f"metrics.{split}", path=path)

    versions = _mapping(root.get("versions"), field="versions", path=path)
    if versions.get("git_dirty") is not False:
        raise HistoricalAcceptanceError(f"{path}: versions.git_dirty must be false")
    observed_git_sha = _hex_digest(
        versions.get("git_sha"), length=40, field="versions.git_sha", path=path
    )
    if observed_git_sha != expected_git_sha:
        raise HistoricalAcceptanceError(
            f"{path}: versions.git_sha differs from the approved commit"
        )
    git_diff_sha256 = _hex_digest(
        versions.get("git_diff_sha256"),
        length=64,
        field="versions.git_diff_sha256",
        path=path,
    )
    environment: dict[str, str] = {}
    for key in ("python", "python_implementation", "numpy", "scikit_learn", "modssc", "platform"):
        value = versions.get(key)
        if not isinstance(value, str) or not value.strip():
            raise HistoricalAcceptanceError(f"{path}: versions.{key} must be non-empty")
        environment[key] = value

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

    artifacts = _mapping(root.get("artifacts"), field="artifacts", path=path)
    dataset = _mapping(artifacts.get("dataset"), field="artifacts.dataset", path=path)
    _locked_subset(
        dataset,
        {
            "id": _DATASET_ID,
            "fingerprint": _DATASET_FINGERPRINT,
            "content_sha256": _DATASET_CONTENT_SHA256,
        },
        field="artifacts.dataset",
        path=path,
    )
    method = _mapping(artifacts.get("method"), field="artifacts.method", path=path)
    _locked_subset(
        method,
        {"id": "co_training", "kind": "inductive", "profile": _DIAGNOSTIC_PROFILE},
        field="artifacts.method",
        path=path,
    )
    diagnostics = _validate_diagnostics(method.get("diagnostics"), path=path)

    sampling = _mapping(artifacts.get("sampling"), field="artifacts.sampling", path=path)
    _locked_subset(
        sampling,
        {
            "seed": seed,
            "stats": {
                "train": {"n": 788, "classes": {"0": 616, "1": 172}},
                "val": {"n": 0, "classes": {}},
                "test": {"n": 263, "classes": {"0": 205, "1": 58}},
                "train_labeled": {"n": 12, "classes": {"0": 9, "1": 3}},
                "train_unlabeled": {"n": 776},
            },
        },
        field="artifacts.sampling",
        path=path,
    )
    split_fingerprint = _hex_digest(
        sampling.get("split_fingerprint"),
        length=64,
        field="artifacts.sampling.split_fingerprint",
        path=path,
    )
    replay_manifest_sha256 = _validate_replay(
        run_json_path=path,
        sampling=sampling,
        protocol=_DIAGNOSTIC_PROTOCOL,
        split_fingerprint=split_fingerprint,
    )
    provenance = {
        "git_diff_sha256": git_diff_sha256,
        "environment": environment,
        "config_hash": config_hash,
        "split_fingerprint": split_fingerprint,
        "replay_manifest_sha256": replay_manifest_sha256,
        "run_contract_sha256": _canonical_sha256(
            {
                "protocol": root["protocol"],
                "config": _diagnostic_run_contract(seed=seed),
                "diagnostics": method["diagnostics"],
            }
        ),
    }
    source = {"path": str(path.resolve()), "sha256": _file_sha256(path)}
    return seed, diagnostics, {"source": source, "provenance": provenance}


def evaluate_diagnostic_runs(
    *,
    run_json_paths: Sequence[Path],
    expected_git_sha: str,
    diagnostic_card: Path,
    confirmation_card: Path,
) -> dict[str, Any]:
    expected_git_sha = _hex_digest(
        expected_git_sha,
        length=40,
        field="expected_git_sha",
        path=Path("diagnostic arguments"),
    )
    cards = validate_cards(
        diagnostic_card=diagnostic_card,
        confirmation_card=confirmation_card,
    )
    by_seed: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    duplicates: set[int] = set()
    for path in run_json_paths:
        seed, diagnostics, evidence = _load_diagnostic_run(
            Path(path), expected_git_sha=expected_git_sha
        )
        if seed in by_seed:
            duplicates.add(seed)
        else:
            by_seed[seed] = (diagnostics, evidence)
    if duplicates:
        raise HistoricalAcceptanceError(f"duplicate seeds: {sorted(duplicates)}")
    if set(by_seed) != set(_DIAGNOSTIC_SEEDS):
        raise HistoricalAcceptanceError(
            "diagnostic seed set mismatch; "
            f"missing={sorted(set(_DIAGNOSTIC_SEEDS) - set(by_seed))}, "
            f"unexpected={sorted(set(by_seed) - set(_DIAGNOSTIC_SEEDS))}"
        )

    reference = by_seed[_DIAGNOSTIC_SEEDS[0]][1]["provenance"]
    split_fingerprints: set[str] = set()
    for seed in _DIAGNOSTIC_SEEDS:
        provenance = by_seed[seed][1]["provenance"]
        if provenance["git_diff_sha256"] != reference["git_diff_sha256"]:
            raise HistoricalAcceptanceError("versions.git_diff_sha256 differs between runs")
        if provenance["environment"] != reference["environment"]:
            raise HistoricalAcceptanceError("runtime environment differs between runs")
        fingerprint = provenance["split_fingerprint"]
        if fingerprint in split_fingerprints:
            raise HistoricalAcceptanceError("duplicate split fingerprints")
        split_fingerprints.add(fingerprint)

    run_seals = []
    run_sources = []
    for seed in _DIAGNOSTIC_SEEDS:
        diagnostics, evidence = by_seed[seed]
        provenance = evidence["provenance"]
        run_seals.append(
            {
                "seed": seed,
                "run_json_sha256": evidence["source"]["sha256"],
                "config_hash": provenance["config_hash"],
                "split_fingerprint": provenance["split_fingerprint"],
                "replay_manifest_sha256": provenance["replay_manifest_sha256"],
                "run_contract_sha256": provenance["run_contract_sha256"],
                "diagnostic_summary": diagnostics,
            }
        )
        run_sources.append({"seed": seed, "run_json": evidence["source"]})

    sealed_provenance = {
        "schema_version": 1,
        "kind": "co_training_v2_test_metric_blind_diagnostic_gate",
        "diagnostic_card": {
            key: value for key, value in cards["diagnostic"].items() if key != "path"
        },
        "confirmation_card": {
            key: value for key, value in cards["confirmation"].items() if key != "path"
        },
        "code": {
            "git_sha": expected_git_sha,
            "git_dirty": False,
            "git_diff_sha256": reference["git_diff_sha256"],
        },
        "environment": reference["environment"],
        "runs": run_seals,
        "gate": {
            "status": "passed",
            "confirmation_authorized": True,
            "allowed_metric_splits": sorted(_ALLOWED_METRIC_SPLITS),
            "test_metrics_present": False,
            "selection_basis": "training_and_pseudo_labels_only",
        },
    }
    sealed_provenance["seal_sha256"] = _canonical_sha256(sealed_provenance)
    return {
        "schema_version": 1,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "scientific_scope": {
            "test_metric_blind": True,
            "strict_epistemic_blind": False,
            "reason": (
                "Seeds 1-5 replay partitions whose v1 test results were already observed; "
                "this gate authenticates only the absence of test metrics from the v2 diagnostic."
            ),
        },
        "cards": cards,
        "sealed_provenance": sealed_provenance,
        "runs": run_sources,
    }


def discover_run_jsons(*, sweep_root: Path) -> list[Path]:
    if not sweep_root.is_dir():
        raise HistoricalAcceptanceError(f"sweep root is not a directory: {sweep_root}")
    paths = sorted(sweep_root.rglob("run.json"))
    if not paths:
        raise HistoricalAcceptanceError("no run.json was found")
    return paths


def write_immutable_report(*, report: Mapping[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_json.parent,
            prefix=f".{output_json.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, output_json)
        except FileExistsError as exc:
            raise HistoricalAcceptanceError(
                f"refusing to overwrite immutable report: {output_json}"
            ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Fail-closed, test-metric-blind gate for the Co-Training WebKB v2 diagnostic.")
    )
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--diagnostic-card", type=Path, required=True)
    parser.add_argument("--confirmation-card", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = evaluate_diagnostic_runs(
            run_json_paths=discover_run_jsons(sweep_root=args.sweep_root),
            expected_git_sha=args.expected_git_sha,
            diagnostic_card=args.diagnostic_card,
            confirmation_card=args.confirmation_card,
        )
        write_immutable_report(report=report, output_json=args.output_json)
    except HistoricalAcceptanceError as exc:
        print(f"historical diagnostic gate failed: {exc}", file=sys.stderr)
        return 2
    print(f"diagnostic gate passed; confirmation card sealed: {args.output_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
