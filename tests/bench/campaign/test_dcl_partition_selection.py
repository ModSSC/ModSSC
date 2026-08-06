from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import bench.campaign.dcl_partition_selection as selection_module
from bench.campaign.cli import main
from bench.campaign.dcl_partition_selection import select_dcl_vote_partitions
from bench.campaign.errors import CampaignError
from bench.campaign.manifest import (
    finalize_task_row,
    load_manifest,
    sha256_file,
    write_manifest,
)
from bench.campaign.models import CampaignTask
from bench.utils.hashing import hash_any
from bench.utils.io import atomic_write_json

from .helpers import build_test_campaign

_PROTOCOL_ID = "zhou-goldman-2004-vote-table3-partition-screening"
_PROFILE = "paper:zhou-goldman-2004-vote-table3"


@dataclass(frozen=True)
class _Fixture:
    manifest: Path
    meta: Path
    reconcile: Path
    output: Path
    result_root: Path
    tasks: tuple[CampaignTask, ...]


def _retarget_tasks(tmp_path: Path, *, seeds: list[int]) -> tuple[Path, Path, list[CampaignTask]]:
    _, _, base_campaign = build_test_campaign(tmp_path / "base")
    base_meta, base_tasks = load_manifest(base_campaign / "manifest.jsonl")
    source = base_tasks[0].to_dict()
    for field in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
        source.pop(field)

    tasks: list[CampaignTask] = []
    for task_index, seed in enumerate(seeds):
        split_fingerprint = hash_any(
            {"kind": "dcl-vote-test-partition", "task_index": task_index, "seed": seed}
        )
        payload = {
            **source,
            "campaign_id": "dcl-vote-selection-test",
            "track": "paper",
            "protocol_id": _PROTOCOL_ID,
            "method_profile": _PROFILE,
            "label_budget": "60",
            "required_seed_count": len(seeds),
            "seed": seed,
            "data_seed": seed,
            "split_seed": seed,
            "sampling_component_seeds": {
                "partition": seed,
                "split": seed,
                "labeling": seed,
                "imbalance": seed,
            },
            "model_seed": seed,
            "method_id": "democratic_co_learning",
            "method_kind": "inductive",
            "dataset_id": "vote",
            "modality": "tabular",
            "regime": None,
            "resource_profile": "cpu_test",
            "assigned_site": "local",
            "expected_dataset_fingerprint": "dataset-fp",
            "expected_dataset_content_sha256": None,
            "dataset_request_sha256": hash_any({"dataset": "vote"}),
            "split_request_sha256": hash_any(
                {"kind": "dcl-vote-test-partition", "task_index": task_index}
            ),
            "expected_split_fingerprint": split_fingerprint,
            "fidelity_status": "not_claimable",
        }
        tasks.append(finalize_task_row(payload, task_index=task_index))

    campaign_dir = tmp_path / "campaign"
    manifest, meta, _ = write_manifest(
        tasks,
        output_dir=campaign_dir,
        campaign_id="dcl-vote-selection-test",
        spec_sha256=str(base_meta["spec_sha256"]),
        expected_git_sha=str(base_meta["expected_git_sha"]),
        expected_git_diff_sha256=base_meta.get("expected_git_diff_sha256"),
        environment_lock_sha256=str(base_meta["environment_lock_sha256"]),
    )
    return manifest, meta, tasks


def _write_result(
    result_root: Path,
    task: CampaignTask,
    *,
    diagnostics: dict[str, Any],
    arrays_seed: int | None = None,
) -> dict[str, Any]:
    result_dir = result_root / task.output_relpath
    run_dir = result_dir / "run"
    replay_dir = run_dir / "sampling_split"
    replay_dir.mkdir(parents=True)
    split_fingerprint = str(task.expected_split_fingerprint)
    atomic_write_json(
        replay_dir / "split.json",
        {
            "dataset_fingerprint": task.expected_dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "stats": {"test": {"class_count": 999999}},
        },
    )
    replay_arrays_seed = task.seed if arrays_seed is None else arrays_seed
    (replay_dir / "arrays.npz").write_bytes(f"immutable-split-arrays-{replay_arrays_seed}".encode())
    files = {
        name: {"sha256": sha256_file(replay_dir / name)} for name in ("split.json", "arrays.npz")
    }
    atomic_write_json(
        replay_dir / "MANIFEST.json",
        {
            "schema_version": 1,
            "format": "modssc.sampling.storage.v1",
            "dataset_fingerprint": task.expected_dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "files": files,
        },
    )
    replay_manifest_digest = sha256_file(replay_dir / "MANIFEST.json")
    accuracy = 1.0 if diagnostics.get("pseudo_labels_added_total") == 0 else 0.0
    run_payload = {
        "run": {"status": "success", "seed": task.seed},
        "task_info": {
            "method_id": task.method_id,
            "dataset_id": task.dataset_id,
            "method_kind": task.method_kind,
            "class_counts_test": {"leak": 10**9},
        },
        "versions": {
            "git_sha": task.expected_git_sha,
            "git_dirty": False,
            "git_diff_sha256": task.expected_git_diff_sha256,
        },
        "artifacts": {
            "dataset": {"fingerprint": task.expected_dataset_fingerprint},
            "sampling": {
                "split_fingerprint": split_fingerprint,
                "replay": {
                    "format": "modssc.sampling.storage.v1",
                    "path": "sampling_split",
                    "manifest": "MANIFEST.json",
                    "manifest_sha256": replay_manifest_digest,
                },
                "stats": {"test": {"leak": 10**9}},
            },
            "method": {
                "profile": task.method_profile,
                "diagnostics": diagnostics,
            },
        },
        "metrics": {"test": {"accuracy": accuracy}},
    }
    atomic_write_json(run_dir / "run.json", run_payload)
    (result_dir / "effective.yaml").write_text("immutable: true\n", encoding="utf-8")
    run_digest = sha256_file(run_dir / "run.json")
    atomic_write_json(
        result_dir / "task.json",
        {
            "schema_version": 1,
            "task": task.to_dict(),
            "site_id": task.assigned_site,
            "environment_lock_sha256": task.environment_lock_sha256,
        },
    )
    atomic_write_json(
        result_dir / "SUCCESS.json",
        {
            "schema_version": 1,
            "task_id": task.task_id,
            "row_sha256": task.row_sha256,
            "status": "success",
            "run_json_sha256": run_digest,
            "effective_config_path": "effective.yaml",
            "effective_config_sha256": sha256_file(result_dir / "effective.yaml"),
            "dataset_content_sha256": task.expected_dataset_content_sha256,
        },
    )
    return {
        "task_index": task.task_index,
        "task_id": task.task_id,
        "method_id": task.method_id,
        "dataset_id": task.dataset_id,
        "resource_profile": task.resource_profile,
        "assigned_site": task.assigned_site,
        "status": "success",
        "result_dirs": [str(result_dir.resolve())],
        "run_json_paths": [str((run_dir / "run.json").resolve())],
        "run_json_sha256": [run_digest],
    }


def _build_fixture(
    tmp_path: Path,
    *,
    statuses: dict[int, str] | None = None,
    diagnostics_by_seed: dict[int, dict[str, Any]] | None = None,
    pseudo_by_seed: dict[int, Any] | None = None,
    duplicate_arrays: dict[int, int] | None = None,
    seeds: list[int] | None = None,
) -> _Fixture:
    candidate_seeds = list(range(1, 23)) if seeds is None else seeds
    manifest, meta_path, tasks = _retarget_tasks(tmp_path, seeds=candidate_seeds)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    result_root = tmp_path / "results"
    states: list[dict[str, Any]] = []
    for task in tasks:
        status = (statuses or {}).get(task.seed, "success")
        if status != "success":
            states.append(
                {
                    "task_index": task.task_index,
                    "task_id": task.task_id,
                    "method_id": task.method_id,
                    "dataset_id": task.dataset_id,
                    "resource_profile": task.resource_profile,
                    "assigned_site": task.assigned_site,
                    "status": status,
                    "result_dirs": [],
                    "run_json_paths": [],
                    "run_json_sha256": [],
                }
            )
            continue
        default_pseudo = 0 if task.seed == 1 else 1
        diagnostics = {
            "pseudo_labels_added_total": (pseudo_by_seed or {}).get(task.seed, default_pseudo),
            "converged": True,
            "n_iter": 3,
        }
        diagnostics.update((diagnostics_by_seed or {}).get(task.seed, {}))
        states.append(
            _write_result(
                result_root,
                task,
                diagnostics=diagnostics,
                arrays_seed=(duplicate_arrays or {}).get(task.seed),
            )
        )
    reconcile = tmp_path / "reconcile.json"
    atomic_write_json(
        reconcile,
        {
            "schema_version": 1,
            "campaign_id": meta["campaign_id"],
            "manifest_sha256": meta["manifest_sha256"],
            "task_count": len(tasks),
            "status": "incomplete",
            # Reverse completion order deliberately: selection must use manifest seeds.
            "tasks": list(reversed(states)),
            "aggregation": {
                "forbidden_test_summary": {"accuracy": 0.999999},
            },
        },
    )
    return _Fixture(
        manifest=manifest,
        meta=meta_path,
        reconcile=reconcile,
        output=tmp_path / "selection.json",
        result_root=result_root,
        tasks=tuple(tasks),
    )


def test_selects_seed_order_from_diagnostics_and_records_rejections_and_sha(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)

    result = select_dcl_vote_partitions(
        fixture.manifest,
        meta_path=fixture.meta,
        reconcile_path=fixture.reconcile,
        output_path=fixture.output,
    )
    payload = json.loads(fixture.output.read_text(encoding="utf-8"))

    assert result.selected_count == 20
    assert result.rejected_count == 1
    assert result.cutoff_seed == 21
    assert payload["cutoff_seed"] == 21
    assert [row["seed"] for row in payload["evaluated_candidates"]] == list(range(1, 22))
    assert [row["seed"] for row in payload["selected"]] == list(range(2, 22))
    assert [row["seed"] for row in payload["rejected"]] == [1]
    assert payload["rejected"][0]["decision"] == "rejected_no_pseudo_labels"
    assert payload["selection_rule"]["test_information_used"] is False
    assert payload["source"]["reconcile_sha256"] == sha256_file(fixture.reconcile)
    assert result.output_sha256 == sha256_file(fixture.output)

    for row in payload["evaluated_candidates"]:
        task = next(task for task in fixture.tasks if task.task_id == row["task_id"])
        replay = fixture.result_root / task.output_relpath / "run" / "sampling_split"
        assert row["task_row_sha256"] == task.row_sha256
        assert row["split_fingerprint"] == task.expected_split_fingerprint
        assert row["run_json_sha256"] == sha256_file(
            fixture.result_root / task.output_relpath / "run" / "run.json"
        )
        assert row["split_manifest_sha256"] == sha256_file(replay / "MANIFEST.json")
        assert row["split_json_sha256"] == sha256_file(replay / "split.json")
        assert row["split_arrays_sha256"] == sha256_file(replay / "arrays.npz")

    serialized = fixture.output.read_text(encoding="utf-8")
    assert '"metrics"' not in serialized
    assert "class_counts_test" not in serialized
    assert '"stats"' not in serialized
    assert "0.999999" not in serialized


def test_cli_writes_same_deterministic_lock_and_refuses_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _build_fixture(tmp_path)
    direct_output = tmp_path / "direct.json"
    select_dcl_vote_partitions(
        fixture.manifest,
        meta_path=fixture.meta,
        reconcile_path=fixture.reconcile,
        output_path=direct_output,
    )

    exit_code = main(
        [
            "select-dcl-vote-partitions",
            "--manifest",
            str(fixture.manifest),
            "--meta",
            str(fixture.meta),
            "--reconcile",
            str(fixture.reconcile),
            "--output",
            str(fixture.output),
            "--protocol-id",
            _PROTOCOL_ID,
        ]
    )
    cli_result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert cli_result["selected_count"] == 20
    assert fixture.output.read_bytes() == direct_output.read_bytes()
    original = fixture.output.read_bytes()
    with pytest.raises(CampaignError, match="E_DCL_SELECTION_OUTPUT_EXISTS"):
        select_dcl_vote_partitions(
            fixture.manifest,
            meta_path=fixture.meta,
            reconcile_path=fixture.reconcile,
            output_path=fixture.output,
        )
    assert fixture.output.read_bytes() == original


def test_does_not_skip_an_unresolved_seed_before_cutoff(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, statuses={2: "failed", 22: "missing"})

    with pytest.raises(CampaignError, match="E_DCL_SELECTION_PREFIX_INCOMPLETE"):
        select_dcl_vote_partitions(
            fixture.manifest,
            meta_path=fixture.meta,
            reconcile_path=fixture.reconcile,
            output_path=fixture.output,
        )

    assert not fixture.output.exists()


def test_allows_unresolved_candidates_strictly_after_cutoff(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, statuses={22: "missing"})

    result = select_dcl_vote_partitions(
        fixture.manifest,
        meta_path=fixture.meta,
        reconcile_path=fixture.reconcile,
        output_path=fixture.output,
    )

    assert result.cutoff_seed == 21
    assert result.selected_count == 20


@pytest.mark.parametrize(
    ("diagnostic_override", "error_code"),
    [
        ({"pseudo_labels_added_total": True}, "E_DCL_SELECTION_MISMATCH"),
        ({"pseudo_labels_added_total": -1}, "E_DCL_SELECTION_MISMATCH"),
        ({"pseudo_labels_added_total": 1.0}, "E_DCL_SELECTION_MISMATCH"),
        ({"pseudo_labels_added_total": None}, "E_DCL_SELECTION_MISMATCH"),
        ({"converged": False}, "E_DCL_SELECTION_UNRESOLVED"),
        ({"n_iter": 20}, "E_DCL_SELECTION_UNRESOLVED"),
        ({"n_iter": True}, "E_DCL_SELECTION_MISMATCH"),
    ],
)
def test_rejects_invalid_or_unresolved_method_diagnostics(
    tmp_path: Path,
    diagnostic_override: dict[str, Any],
    error_code: str,
) -> None:
    fixture = _build_fixture(
        tmp_path,
        diagnostics_by_seed={1: diagnostic_override},
    )

    with pytest.raises(CampaignError, match=error_code):
        select_dcl_vote_partitions(
            fixture.manifest,
            meta_path=fixture.meta,
            reconcile_path=fixture.reconcile,
            output_path=fixture.output,
        )


def test_refuses_fewer_than_twenty_eligible_partitions(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path,
        pseudo_by_seed={seed: 0 for seed in range(1, 4)},
    )

    with pytest.raises(CampaignError, match="E_DCL_SELECTION_INSUFFICIENT"):
        select_dcl_vote_partitions(
            fixture.manifest,
            meta_path=fixture.meta,
            reconcile_path=fixture.reconcile,
            output_path=fixture.output,
        )


def test_refuses_duplicate_split_arrays_and_reconciled_task_rows(tmp_path: Path) -> None:
    duplicate_arrays_fixture = _build_fixture(
        tmp_path / "arrays",
        duplicate_arrays={2: 1},
    )
    with pytest.raises(CampaignError, match="E_DCL_SELECTION_DUPLICATE"):
        select_dcl_vote_partitions(
            duplicate_arrays_fixture.manifest,
            meta_path=duplicate_arrays_fixture.meta,
            reconcile_path=duplicate_arrays_fixture.reconcile,
            output_path=duplicate_arrays_fixture.output,
        )

    duplicate_state_fixture = _build_fixture(tmp_path / "states")
    report = json.loads(duplicate_state_fixture.reconcile.read_text(encoding="utf-8"))
    report["tasks"][-1] = dict(report["tasks"][-2])
    atomic_write_json(duplicate_state_fixture.reconcile, report)
    with pytest.raises(CampaignError, match="E_DCL_SELECTION_DUPLICATE"):
        select_dcl_vote_partitions(
            duplicate_state_fixture.manifest,
            meta_path=duplicate_state_fixture.meta,
            reconcile_path=duplicate_state_fixture.reconcile,
            output_path=duplicate_state_fixture.output,
        )


def test_refuses_tampered_split_artifact_after_reconciliation(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    first = fixture.tasks[0]
    arrays_path = (
        fixture.result_root / first.output_relpath / "run" / "sampling_split" / "arrays.npz"
    )
    arrays_path.write_bytes(b"tampered-after-reconciliation")

    with pytest.raises(CampaignError, match="E_DCL_SELECTION_MISMATCH"):
        select_dcl_vote_partitions(
            fixture.manifest,
            meta_path=fixture.meta,
            reconcile_path=fixture.reconcile,
            output_path=fixture.output,
        )


def test_selection_source_has_no_forbidden_test_information_access() -> None:
    source = inspect.getsource(selection_module)

    assert '.get("metrics")' not in source
    assert "['metrics']" not in source
    assert '["metrics"]' not in source
    assert "class_counts_test" not in source
    assert '.get("stats")' not in source
    assert "['stats']" not in source
    assert '["stats"]' not in source


def test_cli_help_exposes_conditioned_partition_selection(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["select-dcl-vote-partitions", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--manifest" in output
    assert "--reconcile" in output
    assert "--output" in output
    assert "--protocol-id" in output
