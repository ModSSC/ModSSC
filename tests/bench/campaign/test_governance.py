from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from bench.campaign import governance
from bench.campaign.build_manifest import (
    build_manifest,
    collect_environment_identity,
    environment_identity_sha256,
)
from bench.campaign.catalog import TECHNICAL_METHOD_CATALOG
from bench.campaign.errors import CampaignError
from bench.campaign.generate import generate_campaign
from bench.campaign.governance import (
    _check_frozen_dependencies,
    _default_dataset_checker,
    _model_ids,
    load_resource_catalog,
)
from bench.campaign.manifest import (
    finalize_task_row,
    load_manifest,
    sha256_file,
    write_manifest,
)
from bench.utils.io import atomic_write_json
from modssc.data_loader import download_dataset
from modssc.data_loader.types import LoadedDataset, Split
from modssc.graph.artifacts import GraphArtifact
from modssc.graph.cache import GraphCache
from tools.hpc import cli
from tools.hpc.preflight import (
    HPCPreflightReport,
    load_allocation_snapshot,
    load_site_resources,
    run_preflight,
)

from .helpers import build_test_campaign, write_yaml


def _allocation(
    path: Path,
    *,
    total: float = 10.0,
    consumed: float = 0.0,
    updated_at: datetime | str | None = None,
) -> Path:
    write_yaml(
        path,
        {
            "schema_version": 1,
            "updated_at": (datetime.now(UTC).isoformat() if updated_at is None else updated_at),
            "reserve_fraction": 0.15,
            "architectures": {
                "A100": {
                    "total_hours": total,
                    "consumed_hours": consumed,
                    "other_committed_hours": 0,
                }
            },
        },
    )
    return path


def _gpu_site(path: Path) -> Path:
    write_yaml(
        path,
        {
            "schema_version": 1,
            "site_id": "local",
            "scheduler": "slurm",
            "profiles": {
                "cpu_test": {
                    "architecture": "A100",
                    "accelerators_per_task": 1,
                    "concurrency": 64,
                    "initial_concurrency": 64,
                    "promoted_concurrency": 128,
                    "max_walltime": "02:00:00",
                    "directives": {
                        "nodes": 1,
                        "ntasks": 1,
                        "gres": "gpu:1",
                        "constraint": "a100",
                        "time": "02:00:00",
                    },
                }
            },
        },
    )
    return path


def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "MODSSC_HF_LOCAL_FILES_ONLY",
    ):
        monkeypatch.setenv(name, "1")


def _mixed_architecture_campaign(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo, cpu_config, _unused_campaign = build_test_campaign(tmp_path / "base")
    cpu_raw = yaml.safe_load(cpu_config.read_text(encoding="utf-8"))
    cpu_raw["method"]["profile"] = "paper:test-historical-numpy"
    cpu_raw["method"]["params"] = {
        "learners": [
            {
                "classifier_id": "decision_tree",
                "classifier_backend": "numpy",
                "classifier_params": {
                    "min_num_obj": 2,
                    "unpruned": True,
                    "binary_splits": False,
                },
            }
        ]
    }
    write_yaml(cpu_config, cpu_raw)

    gpu_config = (
        repo
        / "bench"
        / "configs"
        / "best"
        / "R1"
        / "inductive"
        / "pseudo_label"
        / "vision"
        / "cifar10.yaml"
    )
    gpu_raw = json.loads(json.dumps(cpu_raw))
    gpu_raw["dataset"]["id"] = "cifar10"
    gpu_raw["method"]["device"]["device"] = "cuda"
    gpu_raw["method"]["params"] = {}
    gpu_raw["preprocess"]["plan"]["steps"][0]["params"] = {"model_id": "stub:gpu"}
    write_yaml(gpu_config, gpu_raw)

    spec_path = repo / "campaign.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["expect"] = {"config_count": 2, "task_count": 4, "tasks_per_method": 4}
    spec["profile_rules"] = [
        {
            "profile": "cpu_tabular",
            "site": "regional",
            "modalities": ["tabular"],
        },
        {
            "profile": "a100_gpu",
            "site": "slurm-gpu",
            "modalities": ["vision"],
        },
    ]
    write_yaml(spec_path, spec)
    campaign = tmp_path / "campaign"
    generate_campaign(spec_path, repo_root=repo, output_dir=campaign)

    slurm_gpu = tmp_path / "site.yaml"
    write_yaml(
        slurm_gpu,
        {
            "schema_version": 1,
            "site_id": "slurm-gpu",
            "scheduler": "slurm",
            "profiles": {
                "a100_gpu": {
                    "architecture": "A100",
                    "accelerators_per_task": 1,
                    "concurrency": 2,
                    "max_walltime": "02:00:00",
                    "directives": {
                        "nodes": 1,
                        "ntasks": 1,
                        "gres": "gpu:1",
                        "time": "02:00:00",
                    },
                }
            },
        },
    )
    regional = tmp_path / "regional.yaml"
    write_yaml(
        regional,
        {
            "schema_version": 1,
            "site_id": "regional",
            "scheduler": "slurm",
            "profiles": {
                "cpu_tabular": {
                    "architecture": "CPU",
                    "accelerators_per_task": 0,
                    "concurrency": 2,
                    "max_walltime": "02:00:00",
                    "directives": {
                        "nodes": 1,
                        "ntasks": 1,
                        "cpus-per-task": 2,
                        "time": "02:00:00",
                    },
                }
            },
        },
    )
    return repo, campaign, slurm_gpu, regional


@pytest.mark.parametrize(
    ("architecture", "dataset_id", "models", "historical_calls", "site", "profile"),
    [
        ("A100", "cifar10", ["stub:gpu"], 0, "slurm-gpu", "a100_gpu"),
        ("CPU", "toy", [], 1, "regional", "cpu_tabular"),
    ],
)
def test_preflight_scopes_assets_by_resource_architecture_but_keeps_global_budget(
    tmp_path,
    monkeypatch,
    architecture: str,
    dataset_id: str,
    models: list[str],
    historical_calls: int,
    site: str,
    profile: str,
) -> None:
    repo, campaign, slurm_gpu, regional = _mixed_architecture_campaign(tmp_path)
    allocation = _allocation(tmp_path / "allocation.yaml")
    dataset_cache = tmp_path / "datasets"
    dataset_cache.mkdir()
    _offline_env(monkeypatch)
    checked_datasets: list[str] = []
    checked_models: list[str] = []
    checked_frozen_configs: list[set[str]] = []
    checked_historical: list[str] = []

    def dataset_checker(_raw, task):
        checked_datasets.append(task.dataset_id)
        return "dataset-fp"

    def frozen_checker(config_tasks, _configs):
        checked_frozen_configs.append(set(config_tasks))
        return [], {"vae": [], "aet": [], "graphs": []}

    def historical_classifier_checker(classifier_id, _classifier_params):
        checked_historical.append(classifier_id)
        return {
            "classifier_id": classifier_id,
            "backend": "numpy",
            "implementation": "test:Classifier",
        }

    monkeypatch.setattr(governance, "_check_frozen_dependencies", frozen_checker)
    version = {
        "python": "3.12.13",
        "torch": "2.10.1",
        "scikit_learn": "1.8.0",
        "cuda_available": "true" if architecture == "A100" else "false",
        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
    }
    result = run_preflight(
        campaign / "manifest.jsonl",
        allocation_path=allocation,
        site_paths=[slurm_gpu, regional],
        repo_root=repo,
        output_path=tmp_path / f"preflight-{architecture}.json",
        environment_lock_sha256="unlocked",
        dataset_cache_dir=dataset_cache,
        model_cache_root=None,
        require_architecture=architecture,
        version_provider=lambda: version,
        method_importer=lambda _kind, _method: None,
        dataset_checker=dataset_checker,
        model_checker=lambda model_id: checked_models.append(model_id),
        historical_classifier_checker=historical_classifier_checker,
    )

    assert result.status == "pass"
    assert result.task_count == 4
    assert result.planned_gpu_hours == {"A100": pytest.approx(4.0)}
    assert set(checked_datasets) == {dataset_id}
    assert checked_models == models
    assert len(checked_historical) == historical_calls
    assert len(checked_frozen_configs) == 1
    assert checked_frozen_configs[0] == {
        task.config_path
        for task in load_manifest(campaign / "manifest.jsonl")[1]
        if task.assigned_site == site and task.resource_profile == profile
    }

    report = json.loads((tmp_path / f"preflight-{architecture}.json").read_text(encoding="utf-8"))
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    covered_ids = {
        task.task_id
        for task in tasks
        if task.assigned_site == site and task.resource_profile == profile
    }
    assert report["task_count"] == 4
    assert report["covered_task_count"] == 2
    assert set(report["task_coverage"]["task_ids"]) == covered_ids
    assert report["task_coverage"]["architecture"] == architecture
    assert report["planned_gpu_hours"] == {"A100": pytest.approx(4.0)}
    cache_check = next(check for check in report["checks"] if check["name"] == "cache_roots")
    assert cache_check["model_required"] is False


def test_preflight_without_architecture_covers_the_complete_mixed_manifest(
    tmp_path, monkeypatch
) -> None:
    repo, campaign, slurm_gpu, regional = _mixed_architecture_campaign(tmp_path)
    allocation = _allocation(tmp_path / "allocation.yaml")
    dataset_cache = tmp_path / "datasets"
    dataset_cache.mkdir()
    _offline_env(monkeypatch)
    checked_datasets: list[str] = []
    checked_models: list[str] = []

    result = run_preflight(
        campaign / "manifest.jsonl",
        allocation_path=allocation,
        site_paths=[slurm_gpu, regional],
        repo_root=repo,
        output_path=tmp_path / "preflight-all.json",
        environment_lock_sha256="unlocked",
        dataset_cache_dir=dataset_cache,
        version_provider=lambda: {
            "python": "3.12.13",
            "torch": "2.10.1",
            "scikit_learn": "1.8.0",
        },
        method_importer=lambda _kind, _method: None,
        dataset_checker=lambda _raw, task: checked_datasets.append(task.dataset_id) or "dataset-fp",
        model_checker=lambda model_id: checked_models.append(model_id),
        historical_classifier_checker=lambda classifier_id, _params: {
            "classifier_id": classifier_id,
            "backend": "numpy",
            "implementation": "test:Classifier",
        },
    )

    assert result.status == "pass"
    assert set(checked_datasets) == {"toy", "cifar10"}
    assert checked_models == ["stub:gpu"]
    report = json.loads((tmp_path / "preflight-all.json").read_text(encoding="utf-8"))
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    assert report["required_architecture"] is None
    assert report["covered_task_count"] == len(tasks) == 4
    assert report["task_coverage"]["scope"] == "all"
    assert set(report["task_coverage"]["task_ids"]) == {task.task_id for task in tasks}


def test_preflight_passes_with_exact_assets_and_budget(tmp_path, monkeypatch) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    site = _gpu_site(tmp_path / "gpu-site.yaml")
    allocation = _allocation(tmp_path / "allocation.yaml")
    estimates = tmp_path / "estimates.yaml"
    write_yaml(
        estimates,
        {"schema_version": 1, "profiles": {"local.cpu_test": {"p95_seconds": 3600}}},
    )
    dataset_cache = tmp_path / "datasets"
    model_cache = tmp_path / "models"
    dataset_cache.mkdir()
    model_cache.mkdir()
    _offline_env(monkeypatch)
    imported: list[tuple[str, str]] = []

    result = run_preflight(
        campaign / "manifest.jsonl",
        allocation_path=allocation,
        site_paths=[site],
        repo_root=repo,
        output_path=tmp_path / "preflight.json",
        runtime_estimates_path=estimates,
        environment_lock_sha256="unlocked",
        dataset_cache_dir=dataset_cache,
        model_cache_root=model_cache,
        version_provider=lambda: {
            "python": "3.12.13",
            "torch": "2.10.1",
            "scikit_learn": "1.8.0",
            "cuda_available": "true",
            "cuda_device_name": "NVIDIA A100-SXM4-80GB",
        },
        method_importer=lambda kind, method: imported.append((kind, method)),
        dataset_checker=lambda _raw, _task: "dataset-fp",
        model_checker=lambda _model: object(),
        require_architecture="A100",
    )

    assert result.status == "pass"
    assert result.error_count == 0
    assert result.planned_gpu_hours == {"A100": pytest.approx(2.0)}
    assert {method for _, method in imported} == set(TECHNICAL_METHOD_CATALOG)
    report = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert report["required_architecture"] == "A100"
    assert all(check["status"] == "pass" for check in report["checks"])
    assert report["checks"][-1]["profiles"][0]["estimate_basis"] == "calibrated_p95"


def test_cpu_graphlearning_preflight_does_not_require_a_torch_version(
    tmp_path, monkeypatch
) -> None:
    repo, _, source_campaign = build_test_campaign(tmp_path / "base")
    source_meta, source_tasks = load_manifest(source_campaign / "manifest.jsonl")
    graph_tasks = []
    for task in source_tasks:
        payload = task.to_dict()
        for field in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
            payload.pop(field)
        payload["method_id"] = "laplace_learning" if task.task_index == 0 else "poisson_learning"
        payload["method_kind"] = "transductive"
        graph_tasks.append(finalize_task_row(payload, task_index=task.task_index))
    campaign = tmp_path / "graph-campaign"
    write_manifest(
        graph_tasks,
        output_dir=campaign,
        campaign_id=str(source_meta["campaign_id"]),
        spec_sha256=str(source_meta["spec_sha256"]),
        expected_git_sha=str(source_meta["expected_git_sha"]),
        expected_git_diff_sha256=source_meta.get("expected_git_diff_sha256"),
        environment_lock_sha256=str(source_meta["environment_lock_sha256"]),
    )
    site = tmp_path / "local-cpu.yaml"
    write_yaml(
        site,
        {
            "schema_version": 1,
            "site_id": "local",
            "scheduler": "local",
            "profiles": {
                "cpu_test": {
                    "architecture": "CPU",
                    "accelerators_per_task": 0,
                    "concurrency": 2,
                    "walltime": "00:10:00",
                    "max_walltime": "00:10:00",
                }
            },
        },
    )
    allocation = _allocation(tmp_path / "allocation.yaml")
    dataset_cache = tmp_path / "datasets"
    dataset_cache.mkdir()
    _offline_env(monkeypatch)

    result = run_preflight(
        campaign / "manifest.jsonl",
        allocation_path=allocation,
        site_paths=[site],
        repo_root=repo,
        output_path=tmp_path / "preflight.json",
        environment_lock_sha256="unlocked",
        dataset_cache_dir=dataset_cache,
        require_architecture="CPU",
        version_provider=lambda: {
            "python": "3.12.13",
            "scikit_learn": "1.8.0",
        },
        method_importer=lambda _kind, _method: None,
        dataset_checker=lambda _raw, _task: "dataset-fp",
    )

    assert result.status == "pass"
    report = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    version_check = next(check for check in report["checks"] if check["name"] == "runtime_versions")
    assert version_check["expected"] == {
        "python": "3.12.13",
        "scikit_learn": "1.8",
    }
    assert "torch" not in version_check["actual"]


def test_non_graphlearning_preflight_still_requires_torch_2_10(tmp_path, monkeypatch) -> None:
    repo, _, campaign = build_test_campaign(tmp_path / "base")
    site = tmp_path / "local-cpu.yaml"
    write_yaml(
        site,
        {
            "schema_version": 1,
            "site_id": "local",
            "scheduler": "local",
            "profiles": {
                "cpu_test": {
                    "architecture": "CPU",
                    "accelerators_per_task": 0,
                    "concurrency": 2,
                    "max_walltime": "00:10:00",
                }
            },
        },
    )
    allocation = _allocation(tmp_path / "allocation.yaml")
    dataset_cache = tmp_path / "datasets"
    dataset_cache.mkdir()
    _offline_env(monkeypatch)

    result = run_preflight(
        campaign / "manifest.jsonl",
        allocation_path=allocation,
        site_paths=[site],
        repo_root=repo,
        output_path=tmp_path / "preflight.json",
        environment_lock_sha256="unlocked",
        dataset_cache_dir=dataset_cache,
        require_architecture="CPU",
        version_provider=lambda: {
            "python": "3.12.13",
            "scikit_learn": "1.8.0",
        },
        method_importer=lambda _kind, _method: None,
        dataset_checker=lambda _raw, _task: "dataset-fp",
    )

    assert result.status == "blocked"
    report = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    version_check = next(check for check in report["checks"] if check["name"] == "runtime_versions")
    assert version_check["expected"]["torch"] == "2.10"
    assert "torch: expected prefix=2.10, got None" in version_check["errors"]


def test_preflight_bootstraps_without_a_calibrated_p95(tmp_path, monkeypatch) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    site = _gpu_site(tmp_path / "gpu-site.yaml")
    allocation = _allocation(tmp_path / "allocation.yaml")
    dataset_cache = tmp_path / "datasets"
    model_cache = tmp_path / "models"
    dataset_cache.mkdir()
    model_cache.mkdir()
    _offline_env(monkeypatch)
    monkeypatch.setenv("MODSSC_EXECUTION_JOB_ID", "67890")
    monkeypatch.setenv("MODSSC_EXECUTION_JOB_NAME", "modssc-preflight")
    monkeypatch.setenv("MODSSC_EXECUTION_CLUSTER", "slurm-gpu")

    result = run_preflight(
        campaign / "manifest.jsonl",
        allocation_path=allocation,
        site_paths=[site],
        repo_root=repo,
        output_path=tmp_path / "preflight.json",
        environment_lock_sha256="unlocked",
        dataset_cache_dir=dataset_cache,
        model_cache_root=model_cache,
        version_provider=lambda: {
            "python": "3.12.13",
            "torch": "2.10.1",
            "scikit_learn": "1.8.0",
            "cuda_available": "true",
            "cuda_device_name": "NVIDIA A100-SXM4-80GB",
        },
        method_importer=lambda _kind, _method: None,
        dataset_checker=lambda _raw, _task: "dataset-fp",
        model_checker=lambda _model: object(),
        require_architecture="A100",
    )

    assert result.status == "pass"
    assert result.planned_gpu_hours == {"A100": pytest.approx(4.0)}
    report = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert report["scheduler"] == {
        "job_id": "67890",
        "job_name": "modssc-preflight",
        "cluster_name": "slurm-gpu",
    }
    profile = report["checks"][-1]["profiles"][0]
    assert profile["estimate_basis"] == "configured_walltime_upper_bound"
    assert profile["p95_seconds"] is None
    assert profile["runtime_estimate_seconds"] == 7200.0
    assert profile["requested_walltime_seconds"] == 7200.0
    assert profile["max_walltime_seconds"] == 7200


@pytest.mark.parametrize(
    ("offset", "expected_status", "error_text"),
    [
        (timedelta(hours=-23), "pass", None),
        (timedelta(hours=-24), "blocked", "snapshot is stale"),
        (timedelta(hours=-25), "blocked", "snapshot is stale"),
        (timedelta(seconds=1), "blocked", "in the future"),
    ],
)
def test_preflight_enforces_allocation_freshness(
    tmp_path,
    monkeypatch,
    offset: timedelta,
    expected_status: str,
    error_text: str | None,
) -> None:
    now = datetime(2030, 1, 2, 12, tzinfo=UTC)
    repo, _, campaign = build_test_campaign(tmp_path)
    allocation = _allocation(tmp_path / "allocation.yaml", updated_at=now + offset)
    site = _gpu_site(tmp_path / "site.yaml")
    dataset_cache = tmp_path / "datasets"
    model_cache = tmp_path / "models"
    dataset_cache.mkdir()
    model_cache.mkdir()
    estimates = tmp_path / "estimates.yaml"
    write_yaml(
        estimates,
        {"schema_version": 1, "profiles": {"local.cpu_test": {"p95_seconds": 3600}}},
    )
    _offline_env(monkeypatch)

    result = run_preflight(
        campaign / "manifest.jsonl",
        allocation_path=allocation,
        site_paths=[site],
        repo_root=repo,
        output_path=tmp_path / "preflight.json",
        runtime_estimates_path=estimates,
        environment_lock_sha256="unlocked",
        dataset_cache_dir=dataset_cache,
        model_cache_root=model_cache,
        max_allocation_age_hours=24.0,
        now_provider=lambda: now,
        version_provider=lambda: {
            "python": "3.12.13",
            "torch": "2.10.1",
            "scikit_learn": "1.8.0",
        },
        method_importer=lambda _kind, _method: None,
        dataset_checker=lambda _raw, _task: "dataset-fp",
        model_checker=lambda _model: object(),
    )

    assert result.status == expected_status
    report = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert report["created_at"] == now.isoformat()
    assert report["max_allocation_age_hours"] == 24.0
    freshness = next(check for check in report["checks"] if check["name"] == "allocation_freshness")
    if error_text is None:
        assert freshness["status"] == "pass"
        assert datetime.fromisoformat(report["expires_at"]) == now + timedelta(hours=1)
    else:
        assert error_text in "\n".join(freshness["errors"])


def test_preflight_checks_distinct_prepared_pools_for_one_dataset_request(
    tmp_path, monkeypatch
) -> None:
    repo, config_path, _campaign = build_test_campaign(tmp_path / "base")
    second_config = (
        repo
        / "bench"
        / "configs"
        / "best"
        / "R2"
        / "inductive"
        / "pseudo_label"
        / "tabular"
        / "toy.yaml"
    )
    second_raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    second_raw["sampling"]["plan"]["policy"] = {"merge_official_splits": True}
    write_yaml(second_config, second_raw)
    spec_path = repo / "campaign.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["expect"] = {"config_count": 2, "task_count": 4, "tasks_per_method": 4}
    write_yaml(spec_path, spec)
    campaign = tmp_path / "campaign"
    generate_campaign(spec_path, repo_root=repo, output_dir=campaign)

    site = _gpu_site(tmp_path / "site.yaml")
    allocation = _allocation(tmp_path / "allocation.yaml")
    dataset_cache = tmp_path / "datasets"
    model_cache = tmp_path / "models"
    dataset_cache.mkdir()
    model_cache.mkdir()
    _offline_env(monkeypatch)
    checked: list[str] = []

    def checker(raw, task):
        checked.append(task.split_request_sha256)
        policy = raw["sampling"]["plan"].get("policy", {})
        return "merged-fingerprint" if policy.get("merge_official_splits") else "dataset-fp"

    result = run_preflight(
        campaign / "manifest.jsonl",
        allocation_path=allocation,
        site_paths=[site],
        repo_root=repo,
        output_path=tmp_path / "preflight.json",
        environment_lock_sha256="unlocked",
        dataset_cache_dir=dataset_cache,
        model_cache_root=model_cache,
        version_provider=lambda: {
            "python": "3.12.13",
            "torch": "2.10.1",
            "scikit_learn": "1.8.0",
        },
        method_importer=lambda _kind, _method: None,
        dataset_checker=checker,
        model_checker=lambda _model: object(),
    )

    assert result.status == "blocked"
    assert len(checked) == 4
    report = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    datasets = next(check for check in report["checks"] if check["name"] == "datasets")
    assert datasets["request_count"] == 1
    assert datasets["prepared_request_count"] == 4
    assert "dataset fingerprint differs" in "\n".join(datasets["errors"])


def test_preflight_full_rehash_blocks_mutated_cached_array(tmp_path, monkeypatch) -> None:
    repo, _config_path, _campaign = build_test_campaign(tmp_path / "base")
    cache_dir = tmp_path / "base" / "cache"
    dataset = download_dataset("toy", cache_dir=cache_dir)
    fingerprint = str(dataset.meta["dataset_fingerprint"])
    content_sha256 = str(dataset.meta["dataset_content_sha256"])
    lock = {
        "schema_version": 2,
        "datasets": {
            dataset_id: {
                "fingerprint": fingerprint,
                "content_sha256": content_sha256,
            }
            for dataset_id in ("adult", "cifar10", "toy")
        },
    }
    write_yaml(repo / "dataset-lock.yaml", lock)
    campaign = tmp_path / "campaign"
    generate_campaign(repo / "campaign.yaml", repo_root=repo, output_dir=campaign)

    site = _gpu_site(tmp_path / "site.yaml")
    allocation = _allocation(tmp_path / "allocation.yaml")
    estimates = tmp_path / "content-estimates.yaml"
    write_yaml(
        estimates,
        {"schema_version": 1, "profiles": {"local.cpu_test": {"p95_seconds": 3600}}},
    )
    model_cache = tmp_path / "models"
    model_cache.mkdir()
    _offline_env(monkeypatch)
    common = {
        "allocation_path": allocation,
        "site_paths": [site],
        "repo_root": repo,
        "runtime_estimates_path": estimates,
        "environment_lock_sha256": "unlocked",
        "dataset_cache_dir": cache_dir,
        "model_cache_root": model_cache,
        "version_provider": lambda: {
            "python": "3.12.13",
            "torch": "2.10.1",
            "scikit_learn": "1.8.0",
        },
        "method_importer": lambda _kind, _method: None,
        "model_checker": lambda _model: object(),
    }
    passed = run_preflight(
        campaign / "manifest.jsonl",
        output_path=tmp_path / "passed.json",
        **common,
    )
    assert passed.status == "pass"

    cache_fingerprint = str(dataset.meta["dataset_cache_fingerprint"])
    array_path = cache_dir / "processed" / cache_fingerprint / "train_X.npy"
    values = np.load(array_path)
    values.flat[0] = values.flat[0] + 1
    np.save(array_path, values)
    blocked = run_preflight(
        campaign / "manifest.jsonl",
        output_path=tmp_path / "blocked.json",
        **common,
    )

    assert blocked.status == "blocked"
    report = json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8"))
    errors = "\n".join(error for check in report["checks"] for error in check["errors"])
    assert "dataset content file digest differs" in errors.lower()


def test_preflight_blocks_reserve_versions_assets_and_offline_mode(tmp_path, monkeypatch) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    site = _gpu_site(tmp_path / "gpu-site.yaml")
    allocation = _allocation(tmp_path / "allocation.yaml", total=2.0)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    monkeypatch.delenv("MODSSC_HF_LOCAL_FILES_ONLY", raising=False)

    result = run_preflight(
        campaign / "manifest.jsonl",
        allocation_path=allocation,
        site_paths=[site],
        repo_root=repo,
        output_path=tmp_path / "blocked.json",
        environment_lock_sha256="wrong",
        dataset_cache_dir=tmp_path / "missing-datasets",
        model_cache_root=tmp_path / "missing-models",
        version_provider=lambda: {"python": "3.11", "torch": "1", "scikit_learn": "1"},
        method_importer=lambda _kind, method: (_ for _ in ()).throw(RuntimeError(method)),
        dataset_checker=lambda _raw, _task: (_ for _ in ()).throw(RuntimeError("missing")),
    )

    assert result.status == "blocked"
    report = json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8"))
    assert report["error_count"] >= 20
    assert "after the 15% reserve" in "\n".join(
        error for check in report["checks"] for error in check["errors"]
    )


def test_scientific_preflight_rehashes_the_active_environment_manifest(
    tmp_path, monkeypatch
) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    identity = collect_environment_identity()
    digest = environment_identity_sha256(identity)
    spec_path = repo / "campaign.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["campaign_id"] = "article10-test"
    spec["code"] = {
        "git_sha": None,
        "require_clean": True,
        "git_diff_sha256": None,
        "environment_lock_sha256": digest,
    }
    spec["scientific_scope"] = {
        "claim_scope_id": "article10",
        "stage": "production",
        "claim_eligible": True,
    }
    write_yaml(spec_path, spec)
    gate_path = repo / "bench" / "campaigns" / "scientific-gates.yaml"
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    gate["track_statuses"]["standardized"] = "passed"
    write_yaml(gate_path, gate)
    write_yaml(
        repo / "dataset-lock.yaml",
        {
            "schema_version": 2,
            "datasets": {
                dataset_id: {
                    "fingerprint": "dataset-fp",
                    "content_sha256": "c" * 64,
                }
                for dataset_id in ("adult", "cifar10", "toy")
            },
        },
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "scientific test"], cwd=repo, check=True)
    campaign = tmp_path / "scientific"
    generate_campaign(spec_path, repo_root=repo, output_dir=campaign)
    site = _gpu_site(tmp_path / "site.yaml")
    allocation = _allocation(tmp_path / "allocation.yaml")
    estimates = tmp_path / "estimates.yaml"
    write_yaml(
        estimates,
        {"schema_version": 1, "profiles": {"local.cpu_test": {"p95_seconds": 3600}}},
    )
    dataset_cache = tmp_path / "datasets"
    model_cache = tmp_path / "models"
    dataset_cache.mkdir()
    model_cache.mkdir()
    _offline_env(monkeypatch)

    def versions() -> dict[str, str]:
        return {
            "python": "3.12.13",
            "torch": "2.10.1",
            "scikit_learn": "1.8.0",
            "cuda_available": "true",
            "cuda_device_name": "NVIDIA A100",
        }

    common = {
        "allocation_path": allocation,
        "site_paths": [site],
        "repo_root": repo,
        "runtime_estimates_path": estimates,
        "dataset_cache_dir": dataset_cache,
        "model_cache_root": model_cache,
        "version_provider": versions,
        "method_importer": lambda _kind, _method: None,
        "dataset_checker": lambda _raw, _task: {
            "fingerprint": "dataset-fp",
            "content_sha256": "c" * 64,
            "content_manifest_sha256": "d" * 64,
            "cache_state_sha256": "e" * 64,
            "cache_fingerprint": "dataset-fp",
        },
        "model_checker": lambda _model: object(),
        "require_architecture": "A100",
    }

    blocked = run_preflight(
        campaign / "manifest.jsonl",
        output_path=tmp_path / "blocked.json",
        environment_lock_sha256=digest,
        **common,
    )
    assert blocked.status == "blocked"

    environment_manifest = tmp_path / "environment.json"
    atomic_write_json(environment_manifest, build_manifest(repo))
    passed = run_preflight(
        campaign / "manifest.jsonl",
        output_path=tmp_path / "passed.json",
        environment_manifest_path=environment_manifest,
        **common,
    )
    assert passed.status == "pass"
    report = json.loads((tmp_path / "passed.json").read_text(encoding="utf-8"))
    assert report["build_manifest_sha256"] == sha256_file(environment_manifest)
    assert (
        next(check for check in report["checks"] if check["name"] == "build_manifest")["status"]
        == "pass"
    )

    payload = json.loads(environment_manifest.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    atomic_write_json(environment_manifest, payload)
    tampered = run_preflight(
        campaign / "manifest.jsonl",
        output_path=tmp_path / "tampered.json",
        environment_manifest_path=environment_manifest,
        **common,
    )
    assert tampered.status == "blocked"
    tampered_report = json.loads((tmp_path / "tampered.json").read_text(encoding="utf-8"))
    build_check = next(
        check for check in tampered_report["checks"] if check["name"] == "build_manifest"
    )
    assert "tracked_tree_sha256" in "\n".join(build_check["errors"])


def test_allocation_and_resource_schema_validation(tmp_path) -> None:
    allocation = _allocation(tmp_path / "allocation.yaml")
    loaded = load_allocation_snapshot(allocation)
    assert loaded["architectures"]["A100"]["total_hours"] == 10.0

    invalid = tmp_path / "invalid.yaml"
    write_yaml(invalid, {"schema_version": 1, "updated_at": "x", "reserve_fraction": 0.1})
    with pytest.raises(CampaignError, match="at least 0.15"):
        load_allocation_snapshot(invalid)

    for value, message in (
        ("2026-07-23T12:00:00", "explicit timezone"),
        ("2026-07-23", "explicit timezone"),
        ("not-a-timestamp", "valid ISO-8601"),
    ):
        invalid_timestamp = tmp_path / f"invalid-{len(value)}-{value[:2]}.yaml"
        write_yaml(
            invalid_timestamp,
            {
                "schema_version": 1,
                "updated_at": value,
                "reserve_fraction": 0.15,
                "architectures": {"A100": {"total_hours": 1, "consumed_hours": 0}},
            },
        )
        with pytest.raises(CampaignError, match=message):
            load_allocation_snapshot(invalid_timestamp)

    site = _gpu_site(tmp_path / "site.yaml")
    resources = load_site_resources([site])
    assert resources[("local", "cpu_test")]["promoted_concurrency"] == 128

    catalog = tmp_path / "resources.json"
    catalog.write_text(
        json.dumps({"schema_version": 1, "resources": list(resources.values())}),
        encoding="utf-8",
    )
    assert load_resource_catalog(catalog) == resources


@pytest.mark.parametrize("maximum", [True, "24", float("inf"), 0.0])
def test_preflight_rejects_invalid_allocation_maximum(tmp_path, maximum) -> None:
    with pytest.raises(CampaignError, match="finite positive"):
        run_preflight(
            tmp_path / "manifest.jsonl",
            allocation_path=tmp_path / "allocation.yaml",
            site_paths=[],
            repo_root=tmp_path,
            output_path=tmp_path / "preflight.json",
            max_allocation_age_hours=maximum,
        )


def test_preflight_rejects_a_naive_clock(tmp_path) -> None:
    with pytest.raises(CampaignError, match="clock must include a timezone"):
        run_preflight(
            tmp_path / "manifest.jsonl",
            allocation_path=tmp_path / "allocation.yaml",
            site_paths=[],
            repo_root=tmp_path,
            output_path=tmp_path / "preflight.json",
            now_provider=lambda: datetime(2030, 1, 1),
        )


def test_model_id_discovery_is_recursive() -> None:
    assert _model_ids(
        {
            "steps": [
                {"params": {"model_id": "st:all-MiniLM-L6-v2"}},
                {"params": {"model_id_audio": "wav2vec2:base"}},
            ],
            "classifier_id": "not-a-preprocess-model",
        }
    ) == {"st:all-MiniLM-L6-v2", "wav2vec2:base"}


def test_default_dataset_checker_fingerprints_the_effective_merged_pool(
    tmp_path, monkeypatch
) -> None:
    repo, config_path, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["sampling"]["plan"]["policy"] = {"merge_official_splits": True}
    source = LoadedDataset(
        train=Split(X=np.array([[1.0], [2.0]]), y=np.array([0, 1])),
        test=Split(X=np.array([[3.0]]), y=np.array([1])),
        meta={
            "dataset_fingerprint": "raw-provider-fingerprint",
            "dataset_content_sha256": "content-sha",
        },
    )
    monkeypatch.setattr(governance, "load_dataset", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        governance,
        "verify_dataset_content",
        lambda *_args, **_kwargs: {
            "content_sha256": "content-sha",
            "content_manifest_sha256": "manifest-sha",
            "cache_state_sha256": "state-sha",
            "cache_fingerprint": "raw-provider-fingerprint",
        },
    )

    checked = _default_dataset_checker(raw, tasks[0])
    prepared = governance.sampling_orch.prepare_dataset(source, plan_dict=raw["sampling"]["plan"])

    assert checked["fingerprint"] == prepared.meta["dataset_fingerprint"]
    assert checked["fingerprint"] != source.meta["dataset_fingerprint"]
    assert checked["content_sha256"] == "content-sha"
    assert prepared.train.y.shape == (3,)
    assert prepared.test is None


def test_frozen_paper_dependencies_verify_vae_aet_and_graph(tmp_path, monkeypatch) -> None:
    _repo, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = tasks[0]

    preprocess_root = tmp_path / "preprocess"
    monkeypatch.setenv("MODSSC_PREPROCESS_CACHE_DIR", str(preprocess_root))
    vae_dir = preprocess_root / "vae_models" / "shared-abc"
    vae_dir.mkdir(parents=True)
    model = vae_dir / "model.pt"
    state = vae_dir / "state.npz"
    model.write_bytes(b"model")
    state.write_bytes(b"state")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    state_sha = hashlib.sha256(state.read_bytes()).hexdigest()
    (vae_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": "vae-fingerprint",
                "params": {"preset": "poisson_mnist"},
                "cache": {"cache_key": "shared"},
                "file_sha256": {"model.pt": model_sha, "state.npz": state_sha},
            }
        ),
        encoding="utf-8",
    )

    aet_dir = preprocess_root / "pretrained_features" / "aet"
    aet_dir.mkdir(parents=True)
    aet_features = aet_dir / "cifar_aet.npz"
    aet_labels = aet_dir / "cifar_labels.npz"
    aet_features.write_bytes(b"features")
    aet_labels.write_bytes(b"labels")
    aet_features_sha = hashlib.sha256(aet_features.read_bytes()).hexdigest()
    aet_labels_sha = hashlib.sha256(aet_labels.read_bytes()).hexdigest()

    graph_root = tmp_path / "graph"
    graph_fingerprint = "graph-fingerprint"
    graph_spec = {"scheme": "knn", "metric": "euclidean", "k": 1}
    graph = GraphArtifact(
        n_nodes=2,
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_weight=np.ones(2, dtype=np.float32),
        directed=False,
        meta={
            "fingerprint": graph_fingerprint,
            "preprocess_fingerprint": "preprocess-fingerprint",
        },
    )
    GraphCache(root=graph_root).save(
        fingerprint=graph_fingerprint,
        graph=graph,
        manifest={
            "fingerprint": graph_fingerprint,
            "dataset_fingerprint": task.expected_dataset_fingerprint,
            "preprocess_fingerprint": "preprocess-fingerprint",
            "spec": graph_spec,
            "seed": 1,
        },
    )
    raw = {
        "preprocess": {
            "plan": {
                "steps": [
                    {
                        "id": "core.vae",
                        "params": {
                            "preset": "poisson_mnist",
                            "cache_key": "shared",
                            "expected_model_fingerprint": "vae-fingerprint",
                            "require_cache_hit": True,
                        },
                    },
                    {
                        "id": "vision.aet",
                        "params": {
                            "source": "precomputed",
                            "expected_features_sha256": aet_features_sha,
                            "expected_labels_sha256": aet_labels_sha,
                        },
                    },
                ]
            }
        },
        "graph": {
            "require_cache_hit": True,
            "cache_dir": str(graph_root),
            "seed": 1,
            "expected_fingerprint": graph_fingerprint,
            "expected_preprocess_fingerprint": "preprocess-fingerprint",
            "spec": graph_spec,
        },
    }
    config_tasks = {task.config_path: task}
    configs = {task.config_path: raw}
    raw["preprocess"]["plan"]["steps"].append(
        {
            "id": "vision.aet",
            "params": dict(raw["preprocess"]["plan"]["steps"][1]["params"]),
        }
    )

    errors, artifacts = _check_frozen_dependencies(config_tasks, configs)
    assert errors == []
    assert artifacts["vae"][0]["fingerprint"] == "vae-fingerprint"
    assert artifacts["aet"][0]["features_sha256"] == aet_features_sha
    assert artifacts["aet"][0]["labels_sha256"] == aet_labels_sha
    assert artifacts["graphs"][0]["fingerprint"] == graph_fingerprint

    vae_params = raw["preprocess"]["plan"]["steps"][0]["params"]
    aet_params = raw["preprocess"]["plan"]["steps"][1]["params"]
    aet_params.pop("expected_features_sha256")
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "frozen AET requires expected_features_sha256" in "\n".join(errors)
    aet_params["expected_features_sha256"] = aet_features_sha

    aet_params.pop("expected_labels_sha256")
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "frozen AET requires expected_labels_sha256" in "\n".join(errors)
    aet_params["expected_labels_sha256"] = aet_labels_sha

    aet_params["expected_features_sha256"] = "not-a-digest"
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "invalid frozen AET parameters" in "\n".join(errors)
    aet_params["expected_features_sha256"] = aet_features_sha

    aet_params["features_path"] = str(tmp_path / "missing-aet.npz")
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "expected one verified frozen AET" in "\n".join(errors)
    aet_params.pop("features_path")

    aet_params["expected_labels_sha256"] = "0" * 64
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "expected one verified frozen AET" in "\n".join(errors)
    aet_params["expected_labels_sha256"] = aet_labels_sha

    vae_params.pop("expected_model_fingerprint")
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "requires expected_model_fingerprint" in "\n".join(errors)
    vae_params["expected_model_fingerprint"] = "vae-fingerprint"

    raw["graph"].pop("expected_fingerprint")
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "requires expected_fingerprint" in "\n".join(errors)
    raw["graph"]["expected_fingerprint"] = graph_fingerprint

    raw["graph"].pop("expected_preprocess_fingerprint")
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "requires expected_preprocess_fingerprint" in "\n".join(errors)
    raw["graph"]["expected_preprocess_fingerprint"] = "preprocess-fingerprint"

    vae_params["expected_model_fingerprint"] = "vae-other"
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "expected one verified frozen VAE" in "\n".join(errors)
    vae_params["expected_model_fingerprint"] = "vae-fingerprint"

    raw["graph"]["expected_preprocess_fingerprint"] = "preprocess-other"
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "expected one verified frozen graph" in "\n".join(errors)
    raw["graph"]["expected_preprocess_fingerprint"] = "preprocess-fingerprint"

    state.write_bytes(b"tampered")
    errors, _ = _check_frozen_dependencies(config_tasks, configs)
    assert "expected one verified frozen VAE" in "\n".join(errors)


def test_preflight_cli_returns_nonzero_for_a_blocked_gate(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def blocked(*_args, **kwargs) -> HPCPreflightReport:
        captured.update(kwargs)
        return HPCPreflightReport(
            campaign_id="campaign",
            status="blocked",
            task_count=1,
            report_path=str(tmp_path / "preflight.json"),
            error_count=1,
            planned_gpu_hours={"A100": 1.0},
        )

    monkeypatch.setattr(cli, "run_preflight", blocked)
    assert (
        cli.main(
            [
                "preflight",
                "--manifest",
                "manifest.jsonl",
                "--allocation",
                "allocation.yaml",
                "--site",
                "site.yaml",
                "--output",
                str(tmp_path / "preflight.json"),
                "--max-allocation-age-hours",
                "6",
            ]
        )
        == 2
    )
    assert captured["max_allocation_age_hours"] == 6.0
