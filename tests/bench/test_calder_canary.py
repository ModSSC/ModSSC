from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from bench.campaign.protocols.calder.official import (
    OFFICIAL_KNN_SHA256,
    OFFICIAL_PERMUTATIONS_SHA256,
    PERMUTATIONS_ARTIFACT_SHA256,
)
from tools.replication_audit.calder import canary as calder_canary
from tools.replication_audit.calder.artifacts import (
    CALDER_CONFIGS,
    EFFECTIVE_CONFIG_KIND,
    seal_calder_artifact_lock,
)
from tools.replication_audit.calder.campaigns import CANARY_CAMPAIGN_ID, PRODUCTION_CAMPAIGN_ID
from tools.replication_audit.calder.canary import (
    CalderCanaryError,
    validate_calder_canary,
    verify_calder_canary_acceptance,
    verify_embedded_calder_release_evidence,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_LABELED_INDICES_SHA256,
    SOURCE_REPLAY_ORACLE_RELATIVE,
    SOURCE_REPLAY_ORACLE_SHA256,
    SOURCE_REPLAY_PREDICTION_SHA256,
    SOURCE_REPLAY_SCORE_SHA256,
    SOURCE_REPLAY_SPLIT_FINGERPRINT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _discrete_accuracy(accuracy_percent: float, *, budget: int) -> float:
    unlabeled_count = 70_000 - budget * 10
    compatible = calder_canary._archive_compatible_correct_counts(
        accuracy_percent,
        unlabeled_count=unlabeled_count,
    )
    return compatible[len(compatible) // 2] / unlabeled_count


def _fixture(tmp_path: Path, monkeypatch) -> dict[str, object]:
    repo = tmp_path / "repo"
    oracle_target = repo / SOURCE_REPLAY_ORACLE_RELATIVE
    oracle_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / SOURCE_REPLAY_ORACLE_RELATIVE, oracle_target)
    safe_permutations_relative = Path(
        "bench/assets/calder2020/protocol_inputs/splits/"
        "mnist-table1-permutations.ragged-int64-v1.npz"
    )
    safe_permutations_target = repo / safe_permutations_relative
    safe_permutations_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / safe_permutations_relative, safe_permutations_target)
    generated = repo / "bench" / "generated" / "calder"
    official = tmp_path / "protocol-inputs"
    results = official / "references"
    results.mkdir(parents=True)
    expected_percent = {
        ("laplace_learning", 1): 11.46,
        ("laplace_learning", 5): 69.00,
        ("poisson_learning", 1): 91.51,
        ("poisson_learning", 5): 95.96,
    }
    archive_paths = {}
    for method_id, filename in {
        "laplace_learning": "mnist-vae-k10-laplace-accuracy.csv",
        "poisson_learning": "mnist-vae-k10-poisson-accuracy.csv",
    }.items():
        path = results / filename
        path.write_text(
            "".join(
                f"10,{expected_percent[(method_id, 1)]:.2f}\n"
                f"50,{expected_percent[(method_id, 5)]:.2f}\n"
                for _ in range(100)
            ),
            encoding="utf-8",
        )
        archive_paths[method_id] = path
    monkeypatch.setattr(
        calder_canary,
        "OFFICIAL_RESULTS_SHA256",
        {method_id: _sha256(path) for method_id, path in archive_paths.items()},
    )
    monkeypatch.setattr(calder_canary, "verify_calder_artifact_lock", lambda _lock: None)
    monkeypatch.setattr(calder_canary, "verify_calder_official_assets", lambda _root: {})

    graph_fingerprint = "graph:" + "8" * 64
    preprocess_fingerprint = "preprocess:" + "9" * 64
    config_records = []
    config_by_identity: dict[tuple[str, int], tuple[str, str]] = {}
    for relative in CALDER_CONFIGS:
        source = yaml.safe_load(
            (REPO_ROOT / "bench" / "configs" / "reproductions" / relative).read_text(
                encoding="utf-8"
            )
        )
        source["graph"]["expected_fingerprint"] = graph_fingerprint
        source["graph"]["expected_preprocess_fingerprint"] = preprocess_fingerprint
        target = generated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
        budget_match = re.search(r"table1-([1-5])-label", relative.name)
        assert budget_match is not None
        method_id = relative.parts[0]
        budget = int(budget_match.group(1))
        repo_path = target.relative_to(repo).as_posix()
        digest = _sha256(target)
        config_records.append(
            {
                "path": relative.as_posix(),
                "repo_path": repo_path,
                "sha256": digest,
            }
        )
        config_by_identity[(method_id, budget)] = (repo_path, digest)

    lock = seal_calder_artifact_lock(
        {
            "pins": {
                "official_commit": "official-commit",
                "official_knn_sha256": OFFICIAL_KNN_SHA256,
                "official_permutations_sha256": OFFICIAL_PERMUTATIONS_SHA256,
                "permutations_artifact_sha256": PERMUTATIONS_ARTIFACT_SHA256,
                "graph_fingerprint": graph_fingerprint,
                "preprocess_fingerprint": preprocess_fingerprint,
            },
            "dataset": {
                "prepared_fingerprint": "4" * 64,
                "content_evidence": {"content_sha256": "5" * 64},
            },
            "official_evidence": {
                "commit": "official-commit",
                "knn_sha256": OFFICIAL_KNN_SHA256,
                "permutations_sha256": OFFICIAL_PERMUTATIONS_SHA256,
                "permutations_artifact_sha256": PERMUTATIONS_ARTIFACT_SHA256,
                "labels_sha256": "7" * 64,
            },
            "artifacts": {"protocol_inputs": {"root": str(official)}},
        }
    )
    lock_path = tmp_path / "artifact-lock.json"
    _write_json(lock_path, lock)

    effective_manifest = seal_calder_artifact_lock(
        {
            "schema_version": 1,
            "kind": EFFECTIVE_CONFIG_KIND,
            "artifact_lock_sha256": lock["lock_sha256"],
            "configs": config_records,
        }
    )
    effective_manifest_path = generated / "MANIFEST.json"
    _write_json(effective_manifest_path, effective_manifest)

    tasks = []
    run_payloads = {}
    states = []
    for index, (method_id, budget) in enumerate(sorted(expected_percent)):
        short = method_id.removesuffix("_learning")
        config_path, config_sha256 = config_by_identity[(method_id, budget)]
        task_id = f"task-{index}"
        tasks.append(
            SimpleNamespace(
                task_id=task_id,
                method_id=method_id,
                protocol_id=(f"calder-2020-mnist-table1-{short}-{budget}-label-per-class"),
                config_path=config_path,
                source_config_sha256=config_sha256,
                campaign_id=CANARY_CAMPAIGN_ID,
                track="paper",
                seed=0,
                required_seed_count=1,
                dataset_id="mnist",
                assigned_site="local-cpu",
                resource_profile="cpu_graph",
                fidelity_status="not_claimable",
                label_budget=f"per_class:{budget}",
                expected_git_sha="a" * 40,
                expected_git_diff_sha256="2" * 64,
                environment_lock_sha256="3" * 64,
                expected_dataset_fingerprint="4" * 64,
                expected_dataset_content_sha256="5" * 64,
            )
        )
        run_dir = tmp_path / "results" / task_id
        run_path = run_dir / "run.json"
        run_path.parent.mkdir(parents=True)
        run_path.write_text("{}\n", encoding="utf-8")
        digest = f"{index + 6:x}" * 64
        digest = digest[:64]
        states.append(
            {
                "task_id": task_id,
                "status": "success",
                "result_dirs": [str(run_dir)],
                "run_json_paths": [str(run_path)],
                "run_json_sha256": [digest],
            }
        )
        if method_id == "laplace_learning":
            diagnostics = {
                "solver": "calder2020_conjugate_gradient",
                "converged": True,
                "iterations": 12,
                "absolute_residual": 1e-6,
            }
        else:
            diagnostics = {
                "solver": "paper_iteration",
                "decision_rule": "paper_class_prior_correction",
                "converged": True,
                "iterations": 50,
                "mixing_residual": 1e-6,
            }
        run_payloads[task_id] = {
            "metrics": {
                "unlabeled": {
                    "accuracy": _discrete_accuracy(
                        expected_percent[(method_id, budget)],
                        budget=budget,
                    )
                }
            },
            "artifacts": {
                "method": {"diagnostics": diagnostics},
                "sampling": {
                    "split_fingerprint": (
                        SOURCE_REPLAY_SPLIT_FINGERPRINT
                        if (method_id, budget) == ("laplace_learning", 5)
                        else "1" * 64
                    ),
                    "replay": {
                        "format": "modssc.sampling.storage.v1",
                        "manifest": "MANIFEST.json",
                        "manifest_sha256": "2" * 64,
                        "path": "sampling_split",
                    },
                    "stats": {"train_unlabeled": {"n": 70_000 - budget * 10}},
                },
            },
        }
    meta = {
        "campaign_id": CANARY_CAMPAIGN_ID,
        "task_count": 4,
        "manifest_sha256": "6" * 64,
    }
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    _write_json(tmp_path / "manifest.meta.json", meta)
    monkeypatch.setattr(
        calder_canary,
        "load_manifest",
        lambda *_args, **_kwargs: (meta, tasks),
    )
    by_id = {task.task_id: task for task in tasks}

    def validate_result(path: Path, task):
        state = next(row for row in states if row["task_id"] == task.task_id)
        assert path == Path(state["result_dirs"][0])
        return (
            Path(state["run_json_paths"][0]),
            run_payloads[task.task_id],
            state["run_json_sha256"][0],
        )

    monkeypatch.setattr(calder_canary, "validate_result_directory", validate_result)
    reconcile = {
        "schema_version": 1,
        "campaign_id": CANARY_CAMPAIGN_ID,
        "manifest_sha256": meta["manifest_sha256"],
        "status": "complete",
        "task_count": 4,
        "tasks": states,
    }
    reconcile_path = tmp_path / "reconcile.json"
    _write_json(reconcile_path, reconcile)

    cells = []
    for method_id in ("laplace_learning", "poisson_learning"):
        short = method_id.removesuffix("_learning")
        for budget in range(1, 6):
            config_path, config_sha256 = config_by_identity[(method_id, budget)]
            cells.append(
                {
                    "protocol_id": (f"calder-2020-mnist-table1-{short}-{budget}-label-per-class"),
                    "config": config_path,
                    "effective_config_sha256": config_sha256,
                    "seeds": "from_config",
                    "resource_profile": "cpu_graph",
                    "site": "local-cpu",
                    "fidelity_status": "paper_matched",
                    "expected_dataset_fingerprint": "4" * 64,
                    "expected_dataset_content_sha256": "5" * 64,
                }
            )
    production = {
        "schema_version": 1,
        "campaign_id": PRODUCTION_CAMPAIGN_ID,
        "track": "paper",
        "default_site": "local-cpu",
        "calder_artifacts": {
            "artifact_lock_sha256": lock["lock_sha256"],
            "source_replay_oracle": {
                "path": SOURCE_REPLAY_ORACLE_RELATIVE.as_posix(),
                "sha256": SOURCE_REPLAY_ORACLE_SHA256,
            },
            "effective_manifest": {
                "path": effective_manifest_path.relative_to(repo).as_posix(),
                "sha256": _sha256(effective_manifest_path),
                "lock_sha256": effective_manifest["lock_sha256"],
            },
            "effective_configs": {path: digest for path, digest in config_by_identity.values()},
            "official": {
                "commit": "official-commit",
                "labels_sha256": "7" * 64,
                "permutations_sha256": OFFICIAL_PERMUTATIONS_SHA256,
                "knn_sha256": OFFICIAL_KNN_SHA256,
            },
            "dataset": {
                "prepared_fingerprint": "4" * 64,
                "content_sha256": "5" * 64,
            },
            "graph": {
                "fingerprint": graph_fingerprint,
                "preprocess_fingerprint": preprocess_fingerprint,
            },
        },
        "code": {
            "git_sha": "a" * 40,
            "git_diff_sha256": "2" * 64,
            "environment_lock_sha256": "3" * 64,
            "require_clean": True,
        },
        "expect": {
            "config_count": 10,
            "task_count": 1000,
            "tasks_per_method": {
                "laplace_learning": 500,
                "poisson_learning": 500,
            },
            "tasks_by_profile": {"cpu_graph": 1000},
            "tasks_by_site": {"local-cpu": 1000},
        },
        "cells": cells,
    }
    production_path = tmp_path / "production.yaml"
    production_path.write_text(yaml.safe_dump(production, sort_keys=False), encoding="utf-8")
    test_config_sha256 = config_by_identity[("laplace_learning", 5)][1]
    monkeypatch.setattr(
        calder_canary,
        "_SOURCE_REPLAY_CONFIG_SHA256",
        test_config_sha256,
    )
    test_oracle = {
        "bindings": {
            "environment_lock_sha256": "3" * 64,
            "dataset_fingerprint": "4" * 64,
            "dataset_content_sha256": "5" * 64,
            "graph_fingerprint": graph_fingerprint,
            "preprocess_fingerprint": preprocess_fingerprint,
            "official_commit": "official-commit",
            "labels_sha256": "7" * 64,
            "official_permutations_sha256": OFFICIAL_PERMUTATIONS_SHA256,
            "official_graph_sha256": OFFICIAL_KNN_SHA256,
            "effective_config_sha256": test_config_sha256,
            "labeled_indices_sorted_sha256": SOURCE_REPLAY_LABELED_INDICES_SHA256,
            "split_fingerprint": SOURCE_REPLAY_SPLIT_FINGERPRINT,
        },
        "archive": {
            "accuracy_percent_text": "69.00",
            "compatible_correct_count": [48_263, 48_268],
            "node_delta": 1,
        },
        "replay": {
            "unlabeled_count": 69_950,
            "correct_count": 48_269,
            "accuracy": 48_269 / 69_950,
            "prediction_count": 70_000,
            "prediction_shape": [70_000],
            "prediction_byte_count": 560_000,
            "prediction_encoding": "numpy-int64-little-endian-c-order",
            "prediction_sha256": {
                "official_source": SOURCE_REPLAY_PREDICTION_SHA256,
                "modssc": SOURCE_REPLAY_PREDICTION_SHA256,
            },
            "score_shape": [70_000, 10],
            "score_byte_count": 5_600_000,
            "score_encoding": "numpy-float64-little-endian-c-order",
            "score_sha256": {
                "official_source": SOURCE_REPLAY_SCORE_SHA256,
                "modssc": SOURCE_REPLAY_SCORE_SHA256,
            },
            "differing_predictions": 0,
            "max_absolute_score_delta": 0.0,
            "iterations": {"official_source": 148, "modssc": 148},
            "residual": {
                "official_system": 9.842962973879334e-06,
                "modssc_recursive": 9.842962973957787e-06,
            },
        },
    }
    monkeypatch.setattr(
        calder_canary,
        "_load_source_replay_oracle",
        lambda _root: test_oracle,
    )
    return {
        "repo": repo,
        "lock_path": lock_path,
        "manifest_path": manifest_path,
        "reconcile_path": reconcile_path,
        "production_path": production_path,
        "output_path": tmp_path / "acceptance.json",
        "run_payloads": run_payloads,
        "reconcile": reconcile,
        "meta": meta,
        "tasks": tasks,
        "by_id": by_id,
        "config_by_identity": config_by_identity,
        "effective_manifest_path": effective_manifest_path,
        "test_oracle": test_oracle,
    }


def _validate(inputs: dict[str, object]):
    return validate_calder_canary(
        repo_root=inputs["repo"],
        artifact_lock_path=inputs["lock_path"],
        manifest_path=inputs["manifest_path"],
        reconcile_path=inputs["reconcile_path"],
        production_spec_path=inputs["production_path"],
        output_path=inputs["output_path"],
    )


def test_checked_in_source_replay_oracle_is_valid() -> None:
    oracle = calder_canary._load_source_replay_oracle(REPO_ROOT)

    assert oracle["kind"] == "modssc.calder2020-laplace-source-replay-oracle"
    assert oracle["replay"]["correct_count"] == 48_269
    assert oracle["replay"]["prediction_sha256"]["modssc"] == (SOURCE_REPLAY_PREDICTION_SHA256)
    assert oracle["replay"]["score_sha256"]["modssc"] == (SOURCE_REPLAY_SCORE_SHA256)
    historical_source = oracle["bindings"]["modssc_source"]
    assert historical_source["module"] == ("modssc.transductive.methods.classic.laplace_learning")
    assert len(historical_source["sha256"]) == 64


def test_source_replay_oracle_requires_its_historical_execution_commit(monkeypatch) -> None:
    monkeypatch.setattr(calder_canary, "_SOURCE_REPLAY_MODSSC_GIT_SHA", "f" * 40)

    with pytest.raises(CalderCanaryError, match="execution commit is absent"):
        calder_canary._load_source_replay_oracle(REPO_ROOT)


def test_canary_validation_passes_and_is_replayable(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path, monkeypatch)

    report = _validate(inputs)
    replay = _validate(inputs)
    acceptance = verify_calder_canary_acceptance(
        inputs["output_path"],
        repo_root=inputs["repo"],
        production_spec_path=inputs["production_path"],
    )

    assert replay == report
    assert report.status == "passed"
    assert report.passed_count == 4
    assert acceptance["status"] == "passed"
    assert acceptance["comparison_basis"].startswith("locked_permutation_0")
    assert len(acceptance["comparisons"]) == 4
    assert all(row["matching_archived_rows"] for row in acceptance["comparisons"])
    assert all(row["diagnostics_ok"] for row in acceptance["comparisons"])


def _prepare_laplace_source_replay_case(
    inputs: dict[str, object],
    *,
    correct_count: int,
) -> SimpleNamespace:
    task = next(
        task
        for task in inputs["tasks"]
        if (task.method_id, calder_canary._budget_from_task(task)) == ("laplace_learning", 5)
    )
    payload = inputs["run_payloads"][task.task_id]
    payload["metrics"]["unlabeled"]["accuracy"] = correct_count / 69_950
    payload["artifacts"]["method"]["diagnostics"] = {
        "solver": "calder2020_conjugate_gradient",
        "converged": True,
        "iterations": 148,
        "absolute_residual": 9.842962973957787e-06,
        "prediction_evidence": {
            "encoding": "numpy-int64-little-endian-c-order",
            "shape": [70_000],
            "count": 70_000,
            "byte_count": 560_000,
            "sha256": SOURCE_REPLAY_PREDICTION_SHA256,
        },
        "score_evidence": {
            "encoding": "numpy-float64-little-endian-c-order",
            "shape": [70_000, 10],
            "byte_count": 5_600_000,
            "sha256": SOURCE_REPLAY_SCORE_SHA256,
        },
    }
    state = next(row for row in inputs["reconcile"]["tasks"] if row["task_id"] == task.task_id)
    run_path = Path(state["run_json_paths"][0])
    sampling_dir = run_path.parent / "sampling_split"
    sampling_dir.mkdir()
    safe_permutations = (
        Path(inputs["repo"]) / "bench/assets/calder2020/protocol_inputs/splits/"
        "mnist-table1-permutations.ragged-int64-v1.npz"
    )
    with np.load(safe_permutations, allow_pickle=False) as archive:
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        values = np.asarray(archive["values"], dtype=np.int64)
        labeled_indices = np.ascontiguousarray(
            np.sort(values[offsets[4] : offsets[5]]),
            dtype="<i8",
        )
    np.savez(sampling_dir / "arrays.npz", idx__train_labeled=labeled_indices)
    manifest_path = sampling_dir / "MANIFEST.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    payload["artifacts"]["sampling"]["replay"]["manifest_sha256"] = _sha256(manifest_path)
    return task


def test_laplace_archive_boundary_requires_exact_source_replay_oracle(
    tmp_path, monkeypatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    task = _prepare_laplace_source_replay_case(inputs, correct_count=48_269)

    report = _validate(inputs)
    acceptance = verify_calder_canary_acceptance(
        inputs["output_path"],
        repo_root=inputs["repo"],
        production_spec_path=inputs["production_path"],
    )
    comparison = next(row for row in acceptance["comparisons"] if row["task_id"] == task.task_id)

    assert report.status == "passed"
    assert comparison["correct_count"] == 48_269
    assert comparison["matching_archived_rows"][0]["archive_compatible_correct_count"] == [
        48_263,
        48_268,
    ]
    assert comparison["matching_archived_rows"][0]["node_delta"] == 1
    assert comparison["numeric_environment_exception"]["oracle_sha256"] == (
        SOURCE_REPLAY_ORACLE_SHA256
    )
    assert comparison["numeric_environment_exception"]["modssc_score_sha256"] == (
        SOURCE_REPLAY_SCORE_SHA256
    )

    inputs = _fixture(tmp_path / "wrong-score", monkeypatch)
    task = _prepare_laplace_source_replay_case(inputs, correct_count=48_269)
    inputs["run_payloads"][task.task_id]["artifacts"]["method"]["diagnostics"]["score_evidence"][
        "sha256"
    ] = "0" * 64
    report = _validate(inputs)
    assert report.status == "failed"
    assert report.passed_count == 3

    inputs = _fixture(tmp_path / "two-nodes", monkeypatch)
    _prepare_laplace_source_replay_case(inputs, correct_count=48_270)
    report = _validate(inputs)
    assert report.status == "failed"
    assert report.passed_count == 3


def test_canary_numeric_or_diagnostic_mismatch_fails_closed(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    task = inputs["tasks"][0]
    inputs["run_payloads"][task.task_id]["metrics"]["unlabeled"]["accuracy"] = 0.5

    report = _validate(inputs)

    assert report.status == "failed"
    assert report.passed_count == 3
    with pytest.raises(CalderCanaryError, match="does not authorize"):
        verify_calder_canary_acceptance(
            inputs["output_path"],
            repo_root=inputs["repo"],
            production_spec_path=inputs["production_path"],
        )

    inputs = _fixture(tmp_path / "diagnostics", monkeypatch)
    task = next(task for task in inputs["tasks"] if task.method_id == "poisson_learning")
    inputs["run_payloads"][task.task_id]["artifacts"]["method"]["diagnostics"]["decision_rule"] = (
        "different"
    )
    report = _validate(inputs)
    assert report.status == "failed"
    comparison = next(
        row
        for row in json.loads(Path(report.output_path).read_text())["comparisons"]
        if row["task_id"] == task.task_id
    )
    assert comparison["diagnostic_failures"] == ["decision_rule"]


def test_canary_rejects_incomplete_reconcile_and_changed_production(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    reconcile = dict(inputs["reconcile"])
    reconcile["status"] = "incomplete"
    _write_json(inputs["reconcile_path"], reconcile)
    with pytest.raises(CalderCanaryError, match="not a complete matching report"):
        _validate(inputs)

    inputs = _fixture(tmp_path / "changed", monkeypatch)
    report = _validate(inputs)
    Path(inputs["production_path"]).write_text(
        Path(inputs["production_path"]).read_text() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CalderCanaryError, match="production spec differs"):
        verify_calder_canary_acceptance(
            Path(report.output_path),
            repo_root=inputs["repo"],
            production_spec_path=inputs["production_path"],
        )


def test_canary_rejects_task_and_archive_drift(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    inputs["tasks"][0].seed = 1
    with pytest.raises(CalderCanaryError, match="task contract differs"):
        _validate(inputs)

    inputs = _fixture(tmp_path / "archive", monkeypatch)
    archive = tmp_path / "archive/protocol-inputs/references/mnist-vae-k10-laplace-accuracy.csv"
    archive.write_text("10,11.46\n50,69.00\n", encoding="utf-8")
    monkeypatch.setattr(
        calder_canary,
        "OFFICIAL_RESULTS_SHA256",
        {
            **calder_canary.OFFICIAL_RESULTS_SHA256,
            "laplace_learning": _sha256(archive),
        },
    )
    with pytest.raises(CalderCanaryError, match="must contain 100 rows"):
        _validate(inputs)


def test_acceptance_rehashes_effective_configs_and_rejects_swapped_cells(
    tmp_path, monkeypatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    report = _validate(inputs)
    config_path, _digest = inputs["config_by_identity"][("laplace_learning", 1)]
    resolved = inputs["repo"] / config_path
    resolved.write_text(
        resolved.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CalderCanaryError, match="configuration SHA-256 differs"):
        verify_calder_canary_acceptance(
            Path(report.output_path),
            repo_root=inputs["repo"],
            production_spec_path=inputs["production_path"],
        )

    inputs = _fixture(tmp_path / "swapped", monkeypatch)
    production = yaml.safe_load(Path(inputs["production_path"]).read_text(encoding="utf-8"))
    first, second = production["cells"][:2]
    first["config"], second["config"] = second["config"], first["config"]
    first["effective_config_sha256"], second["effective_config_sha256"] = (
        second["effective_config_sha256"],
        first["effective_config_sha256"],
    )
    Path(inputs["production_path"]).write_text(
        yaml.safe_dump(production, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(CalderCanaryError, match="cell contract differs"):
        _validate(inputs)


def test_acceptance_rejects_duplicate_forged_comparisons(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    report = _validate(inputs)
    acceptance = json.loads(Path(report.output_path).read_text(encoding="utf-8"))
    acceptance["comparisons"] = [dict(acceptance["comparisons"][0]) for _ in range(4)]
    unsigned = dict(acceptance)
    unsigned.pop("acceptance_sha256")
    acceptance["acceptance_sha256"] = calder_canary._canonical_sha256(unsigned)
    forged = tmp_path / "forged.json"
    _write_json(forged, acceptance)

    with pytest.raises(CalderCanaryError, match="duplicate comparisons"):
        verify_calder_canary_acceptance(
            forged,
            repo_root=inputs["repo"],
            production_spec_path=inputs["production_path"],
        )


def test_acceptance_replays_sources_and_rejects_coherently_resealed_result(
    tmp_path, monkeypatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    report = _validate(inputs)
    acceptance = json.loads(Path(report.output_path).read_text(encoding="utf-8"))
    comparison = next(
        row
        for row in acceptance["comparisons"]
        if (row["method_id"], row["budget_per_class"]) == ("laplace_learning", 5)
    )
    compatible = calder_canary._archive_compatible_correct_counts(
        50.0,
        unlabeled_count=69_950,
    )
    comparison.update(
        {
            "accuracy": 0.5,
            "accuracy_percent": 50.0,
            "correct_count": 34_975,
            "numeric_environment_exception": None,
            "matching_archived_rows": [
                {
                    "line_number": 999_999,
                    "label_count": 50,
                    "accuracy_percent": 50.0,
                    "archive_compatible_correct_count": [
                        min(compatible),
                        max(compatible),
                    ],
                    "node_delta": 0,
                }
            ],
        }
    )
    unsigned = dict(acceptance)
    unsigned.pop("acceptance_sha256")
    acceptance["acceptance_sha256"] = calder_canary._canonical_sha256(unsigned)
    forged = tmp_path / "coherently-resealed.json"
    _write_json(forged, acceptance)

    with pytest.raises(CalderCanaryError, match="differs from its replayed"):
        verify_calder_canary_acceptance(
            forged,
            repo_root=inputs["repo"],
            production_spec_path=inputs["production_path"],
        )


def test_embedded_release_and_evaluate_paper_reject_modified_metadata(
    tmp_path, monkeypatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    report = _validate(inputs)
    acceptance_path = Path(report.output_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    campaign = tmp_path / "production-campaign"
    campaign.mkdir()
    manifest_path = campaign / "manifest.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    embedded = campaign / "release-evidence.json"
    shutil.copyfile(acceptance_path, embedded)
    meta = {
        "campaign_id": PRODUCTION_CAMPAIGN_ID,
        "spec_sha256": acceptance["production_spec_sha256"],
        "release_evidence": {
            "kind": acceptance["kind"],
            "path": "release-evidence.json",
            "file_sha256": _sha256(embedded),
            "acceptance_sha256": acceptance["acceptance_sha256"],
            "canary_manifest_sha256": acceptance["manifest_sha256"],
            "production_evidence": acceptance["production_evidence"],
        },
    }
    tasks = []
    for (method_id, budget), (config_path, digest) in sorted(inputs["config_by_identity"].items()):
        short = method_id.removesuffix("_learning")
        for seed in range(100):
            tasks.append(
                SimpleNamespace(
                    campaign_id=PRODUCTION_CAMPAIGN_ID,
                    track="paper",
                    assigned_site="local-cpu",
                    resource_profile="cpu_graph",
                    config_path=config_path,
                    source_config_sha256=digest,
                    seed=seed,
                    expected_git_sha="a" * 40,
                    expected_git_diff_sha256="2" * 64,
                    environment_lock_sha256="3" * 64,
                    expected_dataset_fingerprint="4" * 64,
                    expected_dataset_content_sha256="5" * 64,
                    protocol_id=(f"calder-2020-mnist-table1-{short}-{budget}-label-per-class"),
                )
            )

    verified = verify_embedded_calder_release_evidence(
        manifest_path,
        manifest_meta=meta,
        tasks=tasks,
    )
    assert verified["acceptance_sha256"] == acceptance["acceptance_sha256"]

    changed = json.loads(json.dumps(meta))
    changed["release_evidence"]["acceptance_sha256"] = "0" * 64
    with pytest.raises(CalderCanaryError, match="metadata differs"):
        verify_embedded_calder_release_evidence(
            manifest_path,
            manifest_meta=changed,
            tasks=tasks,
        )

    # The historical release gate is repository-only.  Public paper
    # acceptance intentionally has no dependency on this archived contract.


def test_canary_cli_exit_codes(tmp_path, monkeypatch, capsys) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    validate_args = [
        "validate",
        "--repo-root",
        str(inputs["repo"]),
        "--artifact-lock",
        str(inputs["lock_path"]),
        "--manifest",
        str(inputs["manifest_path"]),
        "--reconcile",
        str(inputs["reconcile_path"]),
        "--production-spec",
        str(inputs["production_path"]),
        "--output",
        str(inputs["output_path"]),
    ]
    assert calder_canary.main(validate_args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"

    verify_args = [
        "verify-production",
        "--acceptance",
        str(inputs["output_path"]),
        "--repo-root",
        str(inputs["repo"]),
        "--production-spec",
        str(inputs["production_path"]),
    ]
    assert calder_canary.main(verify_args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"

    monkeypatch.setattr(
        calder_canary,
        "verify_calder_canary_acceptance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CalderCanaryError("blocked")),
    )
    with pytest.raises(SystemExit) as exc_info:
        calder_canary.main(verify_args)
    assert exc_info.value.code == 2
    assert "calder-canary: blocked" in capsys.readouterr().err
