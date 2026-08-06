from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bench.campaign import generate as generate_module
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.main import _ALLOWED_SPLITS
from tools.hpc.slurm_renderer import render_slurm_sites

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "tools" / "hpc" / "specs" / "dcl-vote-partition-screening.example.yaml"
REGIONAL_PROFILE = REPO_ROOT / "tools/hpc/config/profiles/regional.example.yaml"
PRODUCTION_CONFIG = (
    REPO_ROOT / "bench" / "configs" / "reproductions" / "democratic_co_learning" / "vote.yaml"
)
SCREENING_CONFIG = (
    REPO_ROOT
    / "bench"
    / "configs"
    / "reproductions"
    / "democratic_co_learning"
    / "vote_partition_screening.yaml"
)


def _clean_placeholder_runtime(**_kwargs: object) -> dict[str, object]:
    return {
        "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
    }


def test_dcl_vote_screening_manifest_is_deterministic_cpu_only_and_not_claimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )

    first = generate_campaign(
        SPEC_PATH,
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "first",
        _allow_template_placeholders=True,
    )
    second = generate_campaign(
        SPEC_PATH,
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "second",
        _allow_template_placeholders=True,
    )
    meta, tasks = load_manifest(Path(first.manifest_path))
    render_slurm_sites(
        site_paths=[REGIONAL_PROFILE],
        campaign_dir=Path(first.output_dir),
        allow_template_placeholders=True,
    )
    _, repeated_tasks = load_manifest(Path(second.manifest_path))

    assert first.task_count == 100
    assert meta["campaign_id"] == "dcl-vote-partition-screening-v1"
    assert meta["counts_by_method"] == {"democratic_co_learning": 100}
    assert meta["counts_by_profile"] == {"cpu_tabular": 100}
    assert meta["counts_by_site"] == {"regional": 100}
    assert len({task.config_path for task in tasks}) == 1
    assert {task.seed for task in tasks} == set(range(1, 101))
    assert {task.required_seed_count for task in tasks} == {100}
    assert {task.method_id for task in tasks} == {"democratic_co_learning"}
    assert {task.method_profile for task in tasks} == {"paper:zhou-goldman-2004-vote-table3"}
    assert {task.dataset_id for task in tasks} == {"vote"}
    assert {task.protocol_id for task in tasks} == {
        "zhou-goldman-2004-vote-table3-partition-screening"
    }
    assert {task.assigned_site for task in tasks} == {"regional"}
    assert {task.resource_profile for task in tasks} == {"cpu_tabular"}
    assert {task.fidelity_status for task in tasks} == {"not_claimable"}
    assert len({task.dataset_request_sha256 for task in tasks}) == 1
    assert len({task.expected_split_fingerprint for task in tasks}) == 100
    assert all(task.data_seed == task.seed for task in tasks)
    assert all(task.sampling_component_seeds is not None for task in tasks)
    for component in ("partition", "split", "labeling", "imbalance"):
        assert (
            len(
                {
                    task.sampling_component_seeds[component]
                    for task in tasks
                    if task.sampling_component_seeds is not None
                }
            )
            == 100
        )
    assert [task.task_id for task in tasks] == [task.task_id for task in repeated_tasks]

    wrapper = tmp_path / "first" / "submit" / "regional" / "cpu_tabular.slurm"
    assert wrapper.is_file()
    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-99%32" in wrapper_text
    assert "--gres" not in wrapper_text
    resources = json.loads(
        (tmp_path / "first" / "profiles" / "resources.json").read_text(encoding="utf-8")
    )
    assert len(resources["resources"]) == 1
    cpu_profile = resources["resources"][0]
    assert cpu_profile["site_id"] == "regional"
    assert cpu_profile["profile_id"] == "cpu_tabular"
    assert cpu_profile["architecture"] == "CPU"
    assert cpu_profile["accelerators_per_task"] == 0


def test_dcl_vote_screening_spec_stays_an_explicit_template() -> None:
    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))

    assert SPEC_PATH.name.endswith(".example.yaml")
    assert raw["code"] == {
        "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
        "require_clean": True,
        "environment_lock_sha256": "REPLACE_WITH_ENVIRONMENT_LOCK_SHA256",
    }
    assert raw["expect"] == {
        "config_count": 1,
        "task_count": 100,
        "tasks_per_method": {"democratic_co_learning": 100},
        "tasks_by_profile": {"cpu_tabular": 100},
        "tasks_by_site": {"regional": 100},
    }
    assert len(raw["cells"]) == 1
    cell = raw["cells"][0]
    assert cell["seeds"] == list(range(1, 101))
    assert cell["fidelity_status"] == "not_claimable"
    assert cell["site"] == "regional"
    assert cell["resource_profile"] == "cpu_tabular"
    assert cell["config"].endswith("democratic_co_learning/vote_partition_screening.yaml")


def test_dcl_vote_screening_card_is_training_equivalent_but_test_blind() -> None:
    production = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    screening = yaml.safe_load(SCREENING_CONFIG.read_text(encoding="utf-8"))

    for section in ("dataset", "sampling", "preprocess", "method"):
        assert screening[section] == production[section]
    assert screening["evaluation"]["split_for_model_selection"] is None
    assert screening["evaluation"]["report_splits"] == ["train_labeled"]
    assert "test" not in screening["evaluation"]["report_splits"]
    assert screening["evaluation"]["metrics"] == production["evaluation"]["metrics"]
    assert production["evaluation"]["report_splits"] == ["test"]
    assert "train_labeled" in _ALLOWED_SPLITS
