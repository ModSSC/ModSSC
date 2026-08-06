from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.campaign import cli
from bench.campaign.dataset_lock import create_dataset_lock
from bench.campaign.errors import CampaignError
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from modssc.data_loader import download_dataset

from .helpers import build_test_campaign, minimal_config, write_yaml


def test_lock_datasets_bootstraps_a_schema_v2_lock_and_campaign(tmp_path) -> None:
    repo, _config, _old_campaign = build_test_campaign(tmp_path)
    cache_dir = tmp_path / "cache"
    dataset = download_dataset("toy", cache_dir=cache_dir)
    lock_path = repo / "dataset-lock.yaml"

    result = create_dataset_lock(
        repo / "campaign.yaml",
        repo_root=repo,
        output_path=lock_path,
        dataset_cache_dir=cache_dir,
        overwrite=True,
    )

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert lock["schema_version"] == 2
    assert result.dataset_count == 1
    assert result.prepared_request_count == 1
    assert lock["datasets"]["toy"] == {
        "fingerprint": dataset.meta["dataset_fingerprint"],
        "content_sha256": dataset.meta["dataset_content_sha256"],
    }
    campaign = tmp_path / "locked-campaign"
    generate_campaign(repo / "campaign.yaml", repo_root=repo, output_dir=campaign)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    assert all(
        task.expected_dataset_content_sha256 == dataset.meta["dataset_content_sha256"]
        for task in tasks
    )


def test_lock_datasets_refuses_multiple_prepared_identities_for_one_id(tmp_path) -> None:
    repo, config_path, _campaign = build_test_campaign(tmp_path)
    cache_dir = tmp_path / "cache"
    download_dataset("toy", cache_dir=cache_dir)
    second = (
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
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["sampling"]["plan"]["policy"] = {"merge_official_splits": True}
    write_yaml(second, raw)

    with pytest.raises(CampaignError, match="DATASET_LOCK_DIVERGENCE"):
        create_dataset_lock(
            repo / "campaign.yaml",
            repo_root=repo,
            output_path=repo / "dataset-lock.yaml",
            dataset_cache_dir=cache_dir,
            overwrite=True,
        )


def test_lock_datasets_cli_is_offline_only(tmp_path, capsys) -> None:
    repo, config_path, _campaign = build_test_campaign(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["dataset"]["download"] = True
    write_yaml(config_path, raw)

    code = cli.main(
        [
            "lock-datasets",
            "--spec",
            str(repo / "campaign.yaml"),
            "--repo-root",
            str(repo),
            "--output",
            str(repo / "new-dataset-lock.yaml"),
            "--dataset-cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert code == 2
    assert "offline-only" in capsys.readouterr().err
    assert not Path(repo / "new-dataset-lock.yaml").exists()


def test_lock_datasets_refuses_implicit_overwrite(tmp_path) -> None:
    repo, _config_path, _campaign = build_test_campaign(tmp_path)

    with pytest.raises(CampaignError, match="DATASET_LOCK_EXISTS"):
        create_dataset_lock(
            repo / "campaign.yaml",
            repo_root=repo,
            output_path=repo / "dataset-lock.yaml",
            dataset_cache_dir=tmp_path / "cache",
        )


def test_lock_datasets_observes_paper_protocols_with_distinct_merged_pool(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    cache_dir = tmp_path / "cache"
    download_dataset("toy", cache_dir=cache_dir)
    base = minimal_config(output_dir=tmp_path / "runs", cache_dir=cache_dir)
    base["run"]["seeds"] = [7]
    base["method"]["profile"] = "paper:toy-official"
    official = repo / "official.yaml"
    write_yaml(official, base)
    merged_raw = yaml.safe_load(official.read_text(encoding="utf-8"))
    merged_raw["method"]["profile"] = "paper:toy-merged"
    merged_raw["sampling"]["plan"]["policy"] = {"merge_official_splits": True}
    merged = repo / "merged.yaml"
    write_yaml(merged, merged_raw)
    spec = repo / "paper-campaign.yaml"
    write_yaml(
        spec,
        {
            "schema_version": 1,
            "campaign_id": "paper-observation-test",
            "track": "paper",
            "code": {
                "git_sha": "not-used-for-observation",
                "require_clean": False,
                "environment_lock_sha256": "not-used-for-observation",
            },
            "cells": [
                {
                    "protocol_id": "toy-official",
                    "config": "official.yaml",
                    "seeds": "from_config",
                },
                {
                    "protocol_id": "toy-merged",
                    "config": "merged.yaml",
                    "seeds": "from_config",
                },
            ],
        },
    )

    result = create_dataset_lock(
        spec,
        repo_root=repo,
        output_path=repo / "paper-dataset-observations.yaml",
        dataset_cache_dir=cache_dir,
    )

    observations = result.protocols
    assert result.protocol_count == 2
    assert result.prepared_request_count == 2
    assert observations["toy-official"]["dataset_id"] == "toy"
    assert observations["toy-merged"]["dataset_id"] == "toy"
    assert (
        observations["toy-official"]["content_sha256"]
        == observations["toy-merged"]["content_sha256"]
    )
    assert observations["toy-official"]["fingerprint"] != observations["toy-merged"]["fingerprint"]
    assert (
        observations["toy-official"]["dataset_request_sha256s"]
        == observations["toy-merged"]["dataset_request_sha256s"]
    )
    assert (
        observations["toy-official"]["split_request_sha256s"]
        != observations["toy-merged"]["split_request_sha256s"]
    )
    payload = yaml.safe_load((repo / "paper-dataset-observations.yaml").read_text(encoding="utf-8"))
    assert payload["kind"] == "modssc.paper-dataset-observations"
