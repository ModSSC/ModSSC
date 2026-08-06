from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from bench.campaign.build_manifest import collect_environment_identity, environment_identity_sha256
from bench.campaign.errors import CampaignError
from bench.campaign.executor import _verify_preflight_report
from bench.campaign.manifest import load_manifest, sha256_file
from bench.campaign.model_artifacts import model_artifact_lock_sha256
from bench.campaign.preflight_coverage import build_task_coverage
from bench.utils.io import atomic_write_json

from .helpers import build_test_campaign, preflight_governance


def test_scientific_preflight_binds_model_lock_and_detects_postflight_change(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    meta, tasks = load_manifest(campaign / "manifest.jsonl")
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"weights")
    stat = artifact.stat()
    model_lock = {
        "schema_version": 1,
        "models": [
            {
                "model_id": "external:test",
                "provider": "test",
                "artifact_free": False,
                "revision": "commit",
                "files": [
                    {
                        "path": "weights.bin",
                        "size": stat.st_size,
                        "sha256": "a" * 64,
                    }
                ],
            }
        ],
    }
    identity = collect_environment_identity(model_artifact_lock=model_lock)
    environment_digest = environment_identity_sha256(identity)
    environment_manifest = tmp_path / "environment.json"
    atomic_write_json(
        environment_manifest,
        {
            "schema_version": 2,
            "environment_lock": identity,
            "environment_lock_sha256": environment_digest,
            "model_artifacts_sha256": model_artifact_lock_sha256(model_lock),
        },
    )
    task = replace(
        tasks[0],
        campaign_id="article10-canary-r3-wave1-v1",
        resource_profile="a100_dev",
        environment_lock_sha256=environment_digest,
    )
    report = tmp_path / "preflight.json"
    atomic_write_json(
        report,
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "max_authorization_age_hours": 24.0,
            "status": "pass",
            "campaign_id": task.campaign_id,
            "manifest_sha256": meta["manifest_sha256"],
            "required_architecture": "A100",
            "task_coverage": build_task_coverage([task.task_id], architecture="A100"),
            "environment_lock_sha256": environment_digest,
            "environment_manifest_sha256": sha256_file(environment_manifest),
            "model_artifacts_sha256": model_artifact_lock_sha256(model_lock),
            **preflight_governance([task]),
            "model_artifact_attestations": [
                {
                    "model_id": "external:test",
                    "path": str(artifact),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "sha256": "a" * 64,
                }
            ],
        },
    )

    _verify_preflight_report(
        task,
        meta,
        report,
        environment_manifest_path=environment_manifest,
    )

    artifact.write_bytes(b"changed")
    with pytest.raises(CampaignError, match="changed after preflight"):
        _verify_preflight_report(
            task,
            meta,
            report,
            environment_manifest_path=environment_manifest,
        )
