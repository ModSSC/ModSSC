from __future__ import annotations

import copy
from typing import Any

import pytest

from bench.limits import apply_limits
from bench.schema import LimitsConfig
from bench.utils.hashing import hash_any
from bench.utils.identity import (
    build_resume_identity,
    effective_config_sha256,
    protocol_identity_payload,
    protocol_sha256,
    software_identity_payload,
    software_sha256,
)


def _effective_config() -> dict[str, Any]:
    return {
        "run": {
            "name": "mac-run",
            "seed": 2718,
            "seeds": [2718, 3141],
            "output_dir": "/Users/researcher/modssc/runs",
            "log_level": "detailed",
            "fail_fast": True,
            "resume_policy": "auto",
            "checkpoint_dir": "/Users/researcher/modssc/checkpoints",
            "artifact_root": "/Users/researcher/.cache/modssc/artifacts",
            "input_artifacts": [
                {
                    "path": "models/encoder",
                    "kind": "tree",
                    "sha256": "c" * 64,
                }
            ],
            "benchmark_mode": True,
        },
        "limits": {
            "profile": "macbook",
            "max_method_batch_size": 32,
        },
        "dataset": {
            "id": "cifar10",
            "download": True,
            "cache_dir": "/Users/researcher/.cache/modssc/datasets",
            "options": {"max_train_samples": 40_000},
        },
        "sampling": {"seed": 2718, "plan": {"labeling": {"value": 250}}},
        "preprocess": {
            "seed": 2718,
            "cache": True,
            "cache_dir": "/Users/researcher/.cache/modssc/preprocess",
            "plan": {"steps": [{"id": "core.to_numpy"}]},
        },
        "views": {
            "seed": 2718,
            "cache": True,
            "cache_dir": "/Users/researcher/.cache/modssc/views",
            "plan": {"views": []},
        },
        "graph": {
            "enabled": True,
            "cache": True,
            "cache_dir": "/Users/researcher/.cache/modssc/graph",
            "spec": {"k": 10, "chunk_size": 512},
        },
        "method": {
            "kind": "inductive",
            "id": "fixmatch",
            "params": {"batch_size": 32, "threshold": 0.95},
        },
    }


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": "3.11.13",
        "python_implementation": "CPython",
        "platform": "macOS-15.6-arm64-arm-64bit",
        "executable": "/Users/researcher/modssc/.venv/bin/python",
        "modssc": "0.1.0",
        "distribution_sha256": "a" * 64,
        "numpy": "2.3.1",
        "scikit_learn": "1.7.1",
        "torch": "2.8.0",
        "torch_geometric": "2.6.1",
        "git_sha": "0123456789abcdef" * 2 + "01234567",
        "git_dirty": True,
        "git_diff_sha256": "b" * 64,
        "cuda": None,
        "cudnn": None,
    }


def test_protocol_identity_is_portable_across_operational_paths() -> None:
    mac = _effective_config()
    jean_zay = copy.deepcopy(mac)
    jean_zay["run"].update(
        {
            "name": "slurm-487122",
            "output_dir": "/lustre/fswork/projects/rech/example/runs",
            "log_level": "basic",
            "fail_fast": False,
            "resume_policy": "required",
            "checkpoint_dir": "/lustre/fsn1/projects/rech/example/checkpoints",
            "artifact_root": "/lustre/fsn1/projects/rech/example/artifacts",
        }
    )
    jean_zay["limits"] = {"profile": "a100"}
    jean_zay["dataset"].update(
        {
            "download": False,
            "cache_dir": "/lustre/fsn1/projects/rech/example/datasets",
        }
    )
    for section in ("preprocess", "views", "graph"):
        jean_zay[section]["cache"] = False
        jean_zay[section]["cache_dir"] = f"/lustre/{section}"

    payload = protocol_identity_payload(mac)
    assert "limits" not in payload
    assert payload["run"] == {
        "seed": 2718,
        "seeds": [2718, 3141],
        "input_artifacts": [
            {
                "path": "models/encoder",
                "kind": "tree",
                "sha256": "c" * 64,
            }
        ],
        "benchmark_mode": True,
    }
    assert payload["dataset"]["options"]["max_train_samples"] == 40_000
    assert payload["method"]["params"]["batch_size"] == 32
    assert protocol_sha256(mac) == protocol_sha256(jean_zay)
    assert effective_config_sha256(mac) == hash_any(mac)


def test_native_protocol_identity_rejects_non_mapping_configs() -> None:
    with pytest.raises(TypeError, match="config must be a mapping"):
        effective_config_sha256([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config must be a mapping"):
        protocol_identity_payload([])  # type: ignore[arg-type]


def test_protocol_identity_invalidates_scientific_changes_and_keeps_limit_effects() -> None:
    raw = _effective_config()
    raw["method"]["params"]["batch_size"] = 256

    limited_32, _, _ = apply_limits(
        copy.deepcopy(raw),
        limits=LimitsConfig(max_method_batch_size=32),
    )
    limited_64, _, _ = apply_limits(
        copy.deepcopy(raw),
        limits=LimitsConfig(max_method_batch_size=64),
    )

    assert "limits" not in protocol_identity_payload(limited_32)
    assert protocol_identity_payload(limited_32)["method"]["params"]["batch_size"] == 32
    assert protocol_sha256(limited_32) != protocol_sha256(limited_64)

    changed_threshold = copy.deepcopy(limited_32)
    changed_threshold["method"]["params"]["threshold"] = 0.8
    assert protocol_sha256(limited_32) != protocol_sha256(changed_threshold)

    changed_seed = copy.deepcopy(limited_32)
    changed_seed["run"]["seed"] += 1
    assert protocol_sha256(limited_32) != protocol_sha256(changed_seed)

    changed_artifact = copy.deepcopy(limited_32)
    changed_artifact["run"]["input_artifacts"][0]["sha256"] = "d" * 64
    assert protocol_sha256(limited_32) != protocol_sha256(changed_artifact)


def test_artifact_root_is_operational_but_every_contract_field_is_identity_bearing() -> None:
    reference = _effective_config()
    moved_root = copy.deepcopy(reference)
    moved_root["run"]["artifact_root"] = "/another/machine/cache"
    assert protocol_sha256(reference) == protocol_sha256(moved_root)

    for field, value in (
        ("path", "models/other-encoder"),
        ("kind", "file"),
        ("sha256", "d" * 64),
    ):
        changed = copy.deepcopy(reference)
        changed["run"]["input_artifacts"][0][field] = value
        assert protocol_sha256(reference) != protocol_sha256(changed)


def test_software_identity_is_host_independent_but_code_sensitive() -> None:
    mac = _runtime_versions()
    jean_zay = copy.deepcopy(mac)
    jean_zay.update(
        {
            "platform": "Linux-6.1.0-x86_64-with-glibc2.36",
            "executable": "/gpfswork/rech/example/venv/bin/python",
            "cuda": "12.6",
            "cudnn": 90501,
        }
    )

    payload = software_identity_payload(mac)
    assert "platform" not in payload
    assert "executable" not in payload
    assert "cuda" not in payload
    assert "cudnn" not in payload
    assert software_sha256(mac) == software_sha256(jean_zay)

    changed_dependency = copy.deepcopy(mac)
    changed_dependency["torch"] = "2.9.0"
    assert software_sha256(mac) != software_sha256(changed_dependency)

    changed_distribution = copy.deepcopy(mac)
    changed_distribution["distribution_sha256"] = "d" * 64
    assert software_sha256(mac) != software_sha256(changed_distribution)

    changed_code = copy.deepcopy(mac)
    changed_code["git_diff_sha256"] = "c" * 64
    assert software_sha256(mac) != software_sha256(changed_code)


def test_checkpoint_identity_survives_mac_to_jean_zay_only_for_same_protocol_and_code() -> None:
    mac_config = _effective_config()
    cluster_config = copy.deepcopy(mac_config)
    cluster_config["run"]["output_dir"] = "/lustre/runs"
    cluster_config["run"]["checkpoint_dir"] = "/lustre/checkpoints"
    cluster_config["dataset"]["cache_dir"] = "/lustre/datasets"

    mac_versions = _runtime_versions()
    cluster_versions = copy.deepcopy(mac_versions)
    cluster_versions["platform"] = "Linux-x86_64"
    cluster_versions["executable"] = "/lustre/venv/bin/python"
    cluster_versions["cuda"] = "12.6"

    mac_identity = build_resume_identity(
        mac_config,
        seed=2718,
        runtime_versions=mac_versions,
    )
    cluster_identity = build_resume_identity(
        cluster_config,
        seed=2718,
        runtime_versions=cluster_versions,
    )
    assert mac_identity.sha256 == cluster_identity.sha256

    changed_code = copy.deepcopy(cluster_versions)
    changed_code["git_sha"] = "f" * 40
    code_identity = build_resume_identity(
        cluster_config,
        seed=2718,
        runtime_versions=changed_code,
    )
    assert code_identity.sha256 != mac_identity.sha256

    changed_protocol = copy.deepcopy(cluster_config)
    changed_protocol["sampling"]["plan"]["labeling"]["value"] = 4_000
    protocol_identity = build_resume_identity(
        changed_protocol,
        seed=2718,
        runtime_versions=cluster_versions,
    )
    assert protocol_identity.sha256 != mac_identity.sha256
