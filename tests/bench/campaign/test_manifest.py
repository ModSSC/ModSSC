from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from bench.campaign import generate as generate_module
from bench.campaign.errors import CampaignError
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import (
    derive_row_sha256,
    derive_task_id,
    finalize_task_row,
    load_manifest,
    validate_task,
    write_manifest,
)
from bench.campaign.models import CampaignTask
from bench.utils.hashing import derive_seed, hash_any
from modssc.sampling.plan import SamplingPlan

from .helpers import build_test_campaign


def _test_partition_selection() -> dict[str, object]:
    return {
        "kind": "modssc.dcl-vote-conditioned-partition-selection",
        "selection_path": "bench/campaigns/locks/dcl/selected-partitions.json",
        "selection_sha256": "1" * 64,
        "selection_rank": 1,
        "source_task_id": "2" * 64,
        "source_task_row_sha256": "3" * 64,
        "replay_path": "bench/campaigns/locks/dcl/splits/seed-001",
        "split_fingerprint": "4" * 64,
        "split_manifest_sha256": "5" * 64,
        "split_json_sha256": "6" * 64,
        "split_arrays_sha256": "7" * 64,
    }


def test_generation_is_deterministic_and_expands_one_seed_per_task(tmp_path) -> None:
    repo, _, campaign_one = build_test_campaign(tmp_path / "one")
    spec = repo / "campaign.yaml"
    campaign_two = tmp_path / "two-output"
    generate_campaign(spec, repo_root=repo, output_dir=campaign_two)

    assert (campaign_one / "manifest.jsonl").read_bytes() == (
        campaign_two / "manifest.jsonl"
    ).read_bytes()
    assert (campaign_one / "manifest.meta.json").read_bytes() == (
        campaign_two / "manifest.meta.json"
    ).read_bytes()
    meta, tasks = load_manifest(campaign_one / "manifest.jsonl")
    assert meta["task_count"] == 2
    assert [task.seed for task in tasks] == [1, 2]
    assert len({task.task_id for task in tasks}) == 2
    assert all(task.method_id == "pseudo_label" for task in tasks)
    assert (campaign_one / "profiles" / "cpu_test.indices").read_text() == "0\n1\n"
    assert all(task.expected_dataset_fingerprint == "dataset-fp" for task in tasks)
    assert all(task.expected_split_fingerprint is not None for task in tasks)
    assert all(task.dataset_lock_sha256 is not None for task in tasks)
    assert {task.required_seed_count for task in tasks} == {2}


def test_partition_selection_uses_schema_v4_and_changes_task_identity(tmp_path) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    payload = tasks[0].to_dict()
    for key in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
        payload.pop(key)
    payload["partition_selection"] = _test_partition_selection()

    locked = finalize_task_row(payload, task_index=0)
    changed_payload = dict(payload)
    changed_payload["partition_selection"] = {
        **_test_partition_selection(),
        "split_arrays_sha256": "8" * 64,
    }
    changed = finalize_task_row(changed_payload, task_index=0)

    assert locked.schema_version == 4
    assert locked.partition_selection == _test_partition_selection()
    assert CampaignTask.from_dict(locked.to_dict()) == locked
    assert changed.task_id != locked.task_id


def test_task_identity_rejects_unknown_schema_version() -> None:
    with pytest.raises(CampaignError, match="schema_version 1, 2, 3, or 4"):
        derive_task_id({"schema_version": 5})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda selection: selection.pop("split_arrays_sha256"),
        lambda selection: selection.update(selection_rank=0),
        lambda selection: selection.update(replay_path="../outside"),
        lambda selection: selection.update(split_fingerprint="not-a-sha"),
    ],
)
def test_schema_v3_rejects_noncanonical_partition_selection(
    tmp_path,
    mutate,
) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    payload = tasks[0].to_dict()
    for key in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
        payload.pop(key)
    selection = _test_partition_selection()
    mutate(selection)
    payload["partition_selection"] = selection

    with pytest.raises(CampaignError, match="partition_selection"):
        finalize_task_row(payload, task_index=0)


def test_legacy_v1_task_line_keeps_historical_identity_and_round_trips(tmp_path) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path / "source")
    _, generated = load_manifest(campaign_dir / "manifest.jsonl")
    legacy = generated[0].to_dict()
    legacy["schema_version"] = 1
    legacy.pop("sampling_component_seeds")
    legacy.pop("partition_selection")
    for field in (
        "claim_scope_id",
        "campaign_stage",
        "claim_eligible",
        "gate_policy_id",
        "gate_policy_sha256",
    ):
        legacy.pop(field)
    # In v1 this field recorded the master sampling seed, not the component RNG seed.
    legacy["split_seed"] = legacy["seed"]
    legacy["task_id"] = derive_task_id(legacy)
    legacy["output_relpath"] = f"tasks/{legacy['task_id'][:2]}/{legacy['task_id']}"
    legacy["row_sha256"] = derive_row_sha256(legacy)

    loaded = CampaignTask.from_dict(legacy)
    validate_task(loaded)

    assert loaded.schema_version == 1
    assert loaded.sampling_component_seeds is None
    assert loaded.to_dict() == legacy
    output = tmp_path / "legacy"
    manifest_path, _, _ = write_manifest(
        [loaded],
        output_dir=output,
        campaign_id=loaded.campaign_id,
        spec_sha256="legacy-spec",
        expected_git_sha=loaded.expected_git_sha,
        expected_git_diff_sha256=loaded.expected_git_diff_sha256,
        environment_lock_sha256=loaded.environment_lock_sha256,
    )
    _, reloaded = load_manifest(manifest_path)
    assert reloaded == [loaded]


def test_generation_exposes_effective_component_seeds_and_matches_split_fingerprint(
    tmp_path,
) -> None:
    repo, config_path, _ = build_test_campaign(tmp_path / "base")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["sampling"]["plan"]["component_seeds"] = {"split": 2005}
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    output = tmp_path / "component-seeds"
    generate_campaign(repo / "campaign.yaml", repo_root=repo, output_dir=output)
    _, tasks = load_manifest(output / "manifest.jsonl")
    normalized_plan = SamplingPlan.from_dict(raw["sampling"]["plan"])

    assert {task.split_seed for task in tasks} == {2005}
    assert {task.sampling_component_seeds["split"] for task in tasks} == {2005}
    assert {task.sampling_component_seeds["labeling"] for task in tasks} == {
        derive_seed(1, "labeling"),
        derive_seed(2, "labeling"),
    }
    for task in tasks:
        assert task.expected_split_fingerprint == hash_any(
            {
                "schema_version": 1,
                "dataset_fingerprint": "dataset-fp",
                "plan": normalized_plan.as_dict(),
                "seed": task.seed,
            }
        )


def test_standardized_generation_requires_a_dataset_lock(tmp_path) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    spec = repo / "campaign.yaml"
    raw = spec.read_text(encoding="utf-8").replace("  dataset_lock_file: dataset-lock.yaml\n", "")
    spec.write_text(raw, encoding="utf-8")

    with pytest.raises(CampaignError, match="DATASET_UNPINNED"):
        generate_campaign(spec, repo_root=repo, output_dir=tmp_path / "missing-lock")


def test_dataset_content_digest_is_part_of_every_task_identity(tmp_path) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    lock_path = repo / "dataset-lock.yaml"
    lock = {
        "schema_version": 2,
        "datasets": {
            "adult": {"fingerprint": "dataset-fp", "content_sha256": "content-one"},
            "cifar10": {"fingerprint": "dataset-fp", "content_sha256": "content-one"},
            "toy": {"fingerprint": "dataset-fp", "content_sha256": "content-one"},
        },
    }
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    first_dir = tmp_path / "first"
    generate_campaign(repo / "campaign.yaml", repo_root=repo, output_dir=first_dir)
    _, first_tasks = load_manifest(first_dir / "manifest.jsonl")
    assert all(task.expected_dataset_content_sha256 == "content-one" for task in first_tasks)

    for entry in lock["datasets"].values():
        entry["content_sha256"] = "content-two"
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    second_dir = tmp_path / "second"
    generate_campaign(repo / "campaign.yaml", repo_root=repo, output_dir=second_dir)
    _, second_tasks = load_manifest(second_dir / "manifest.jsonl")

    assert [task.task_id for task in first_tasks] != [task.task_id for task in second_tasks]


def test_task_identity_includes_config_resource_site_modality_and_regime(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    original = tasks[0]
    base = original.to_dict()
    for key in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
        base.pop(key)

    variants = {
        finalize_task_row({**base, "config_path": "different.yaml"}, task_index=0).task_id,
        finalize_task_row({**base, "resource_profile": "other"}, task_index=0).task_id,
        finalize_task_row({**base, "assigned_site": "other"}, task_index=0).task_id,
        finalize_task_row({**base, "modality": "vision"}, task_index=0).task_id,
        finalize_task_row({**base, "regime": "R6"}, task_index=0).task_id,
        finalize_task_row({**base, "required_seed_count": 3}, task_index=0).task_id,
        finalize_task_row(
            {
                **base,
                "split_seed": 2005,
                "sampling_component_seeds": {
                    **base["sampling_component_seeds"],
                    "split": 2005,
                },
            },
            task_index=0,
        ).task_id,
    }
    assert original.task_id not in variants
    assert len(variants) == 7


def test_task_validation_rejects_nonpositive_required_seed_count(tmp_path) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    payload = tasks[0].to_dict()
    for key in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
        payload.pop(key)
    payload["required_seed_count"] = 0

    with pytest.raises(CampaignError, match="required_seed_count"):
        finalize_task_row(payload, task_index=0)


def test_task_validation_rejects_inconsistent_effective_split_seed(tmp_path) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    payload = tasks[0].to_dict()
    for key in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
        payload.pop(key)
    payload["split_seed"] = payload["split_seed"] + 1

    with pytest.raises(CampaignError, match="sampling_component_seeds.split"):
        finalize_task_row(payload, task_index=0)


def test_manifest_writer_refuses_mixed_commits_and_environments(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    meta, tasks = load_manifest(campaign / "manifest.jsonl")
    base = tasks[1].to_dict()
    for key in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
        base.pop(key)
    mixed = finalize_task_row(
        {**base, "expected_git_sha": "other-sha", "environment_lock_sha256": "other-env"},
        task_index=1,
    )

    with pytest.raises(CampaignError, match="MANIFEST_MIXED"):
        write_manifest(
            [tasks[0], mixed],
            output_dir=tmp_path / "mixed",
            campaign_id=tasks[0].campaign_id,
            spec_sha256=str(meta["spec_sha256"]),
            expected_git_sha=tasks[0].expected_git_sha,
            expected_git_diff_sha256=tasks[0].expected_git_diff_sha256,
            environment_lock_sha256=tasks[0].environment_lock_sha256,
        )


def test_manifest_detects_tampering(tmp_path) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    manifest = campaign_dir / "manifest.jsonl"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["seed"] = 999
    rows[0] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="MANIFEST_HASH_MISMATCH"):
        load_manifest(manifest)


def test_task_validation_requires_canonical_output_path(tmp_path) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    altered = replace(tasks[0], output_relpath="../outside", row_sha256="pending")
    altered = replace(altered, row_sha256=derive_row_sha256(altered.to_dict()))

    with pytest.raises(CampaignError, match="OUTPUT_PATH_INVALID"):
        validate_task(altered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("campaign_id", "../campaign"),
        ("resource_profile", "cpu;touch-pwned"),
        ("assigned_site", "site\nexport BAD=1"),
    ],
)
def test_task_validation_rejects_unsafe_identifiers(tmp_path, field: str, value: str) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    payload = tasks[0].to_dict()
    for key in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
        payload.pop(key)
    payload[field] = value

    with pytest.raises(CampaignError, match="MANIFEST_SCHEMA"):
        finalize_task_row(payload, task_index=0)


def test_generation_refuses_count_drift(tmp_path) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    spec = repo / "campaign.yaml"
    raw = spec.read_text(encoding="utf-8").replace("task_count: 2", "task_count: 3")
    spec.write_text(raw, encoding="utf-8")

    with pytest.raises(CampaignError, match="EXPECTATION_FAILED"):
        generate_campaign(spec, repo_root=repo, output_dir=tmp_path / "bad")


@pytest.mark.parametrize("campaign_id", ["../escape", "campaign;touch-pwned", "bad\nvalue"])
def test_generation_rejects_unsafe_campaign_id(tmp_path, campaign_id: str) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    spec = repo / "campaign.yaml"
    raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
    raw["campaign_id"] = campaign_id
    spec.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    output = tmp_path / "unsafe-campaign"

    with pytest.raises(CampaignError, match="SPEC_INVALID"):
        generate_campaign(spec, repo_root=repo, output_dir=output)

    assert not output.exists()


def test_generation_refuses_existing_destination_without_mutating_it(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    manifest_before = (campaign / "manifest.jsonl").read_bytes()
    marker = campaign / "do-not-overwrite"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(CampaignError, match="DESTINATION_EXISTS"):
        generate_campaign(repo / "campaign.yaml", repo_root=repo, output_dir=campaign)

    assert (campaign / "manifest.jsonl").read_bytes() == manifest_before
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_generation_lock_rejects_a_concurrent_writer(tmp_path) -> None:
    destination = tmp_path / "campaign"

    with (
        generate_module._AtomicCampaignDirectory(destination),
        pytest.raises(CampaignError, match="DESTINATION_BUSY"),
        generate_module._AtomicCampaignDirectory(destination),
    ):
        pytest.fail("a second writer acquired the same destination")

    assert destination.is_dir()


def test_generation_cleans_staging_if_destination_appears(tmp_path) -> None:
    destination = tmp_path / "campaign"

    with (
        pytest.raises(CampaignError, match="DESTINATION_EXISTS"),
        generate_module._AtomicCampaignDirectory(destination),
    ):
        destination.mkdir()

    assert destination.is_dir()
    assert not list(tmp_path.glob(".campaign.staging-*"))


def test_generation_checks_exact_profile_and_site_counts(tmp_path) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    spec = repo / "campaign.yaml"
    raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
    raw["expect"]["tasks_by_profile"] = {"cpu_test": 1}
    raw["expect"]["tasks_by_site"] = {"local": 2}
    spec.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    output = tmp_path / "count-drift"

    with pytest.raises(CampaignError, match="tasks_by_profile does not match"):
        generate_campaign(spec, repo_root=repo, output_dir=output)

    assert not output.exists()


def test_focused_ten_method_example_builds_expected_5550_tasks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    spec = repo_root / "tools" / "hpc" / "specs" / "article10-standardized.example.yaml"
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        lambda **kwargs: {
            "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
            "git_dirty": False,
            "git_diff_sha256": "0" * 64,
        },
    )

    with pytest.raises(CampaignError, match="TEMPLATE_PLACEHOLDER"):
        generate_campaign(spec, repo_root=repo_root, output_dir=tmp_path / "unresolved")

    generated = generate_campaign(
        spec,
        repo_root=repo_root,
        output_dir=tmp_path / "focused",
        _allow_template_placeholders=True,
    )
    meta, tasks = load_manifest(Path(generated.manifest_path))

    assert generated.task_count == 5550
    assert meta["counts_by_method"] == {
        "democratic_co_learning": 555,
        "fixmatch": 555,
        "flexmatch": 555,
        "free_match": 555,
        "grand": 555,
        "laplace_learning": 555,
        "poisson_learning": 555,
        "pseudo_label": 555,
        "softmatch": 555,
        "tri_training": 555,
    }
    assert meta["counts_by_profile"] == {
        "a100_gpu": 3600,
        "cpu_graph": 555,
        "v100_gpu": 1395,
    }
    assert meta["counts_by_site"] == {"slurm-gpu": 4995, "regional": 555}
    assert len(tasks) == 5550
    assert len({task.config_path for task in tasks}) == 1110
    assert len({task.task_id for task in tasks}) == 5550
    assert {task.required_seed_count for task in tasks} == {5}
    assert {task.method_id for task in tasks if task.resource_profile == "cpu_graph"} == {
        "poisson_learning"
    }
