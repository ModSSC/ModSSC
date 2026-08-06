from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

from bench.campaign.build_manifest import environment_identity_sha256
from bench.campaign.model_artifacts import model_artifact_lock_sha256
from tools.replication_audit.calder import campaigns
from tools.replication_audit.calder.artifacts import (
    CALDER_CONFIGS,
    EFFECTIVE_CONFIG_KIND,
    CalderArtifactError,
    load_calder_config_family,
    materialized_calder_graph_spec,
    seal_calder_artifact_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch) -> dict[str, object]:
    repo = tmp_path / "repo"
    oracle_target = repo / campaigns.SOURCE_REPLAY_ORACLE_RELATIVE
    oracle_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        REPO_ROOT / campaigns.SOURCE_REPLAY_ORACLE_RELATIVE,
        oracle_target,
    )
    reproduction_root = repo / "bench" / "configs" / "reproductions"
    for relative in CALDER_CONFIGS:
        target = reproduction_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            REPO_ROOT / "bench" / "configs" / "reproductions" / relative,
            target,
        )
    family = load_calder_config_family(repo)
    git_sha = "a" * 40
    pins = {
        "preprocess_fingerprint": "preprocess:" + "1" * 64,
        "graph_fingerprint": "graph:" + "2" * 64,
    }
    lock = seal_calder_artifact_lock(
        {
            "schema_version": 2,
            "kind": "modssc.calder2020-mnist-table1-artifacts",
            "builder": {
                "git_sha": git_sha,
                "config_files": list(family.files),
            },
            "pins": {
                **pins,
                "official_commit": "b" * 40,
                "official_knn_sha256": "3" * 64,
                "official_permutations_sha256": "4" * 64,
            },
            "dataset": {
                "id": "mnist",
                "source_fingerprint": "5" * 64,
                "prepared_fingerprint": "6" * 64,
                "content_evidence": {"content_sha256": "7" * 64},
            },
            "official_evidence": {
                "commit": "b" * 40,
                "knn_sha256": "3" * 64,
                "permutations_sha256": "4" * 64,
                "labels_sha256": "0" * 64,
            },
        }
    )
    lock_path = tmp_path / "calder-lock.json"
    _json_dump(lock_path, lock)

    generated = repo / "bench" / "generated" / "calder-effective"
    records = []
    for relative in CALDER_CONFIGS:
        source = yaml.safe_load((reproduction_root / relative).read_text(encoding="utf-8"))
        source["graph"]["spec"] = materialized_calder_graph_spec(source["graph"]["spec"])
        source["graph"]["expected_fingerprint"] = pins["graph_fingerprint"]
        source["graph"]["expected_preprocess_fingerprint"] = pins["preprocess_fingerprint"]
        source["graph"]["require_cache_hit"] = True
        target = generated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
        records.append(
            {
                "path": relative.as_posix(),
                "repo_path": target.relative_to(repo).as_posix(),
                "sha256": _sha256_file(target),
            }
        )
    effective_manifest = seal_calder_artifact_lock(
        {
            "schema_version": 1,
            "kind": EFFECTIVE_CONFIG_KIND,
            "artifact_lock_sha256": lock["lock_sha256"],
            "artifact_builder_git_sha": git_sha,
            "source_configs": list(family.files),
            "pins": pins,
            "configs": records,
        }
    )
    effective_manifest_path = generated / "MANIFEST.json"
    _json_dump(effective_manifest_path, effective_manifest)

    model_lock = {"schema_version": 1, "models": []}
    environment = {
        "schema_version": 2,
        "python": "3.12.13",
        "implementation": "CPython",
        "machine": "arm64",
        "distributions": [],
        "model_artifacts": model_lock,
        "model_artifacts_sha256": model_artifact_lock_sha256(model_lock),
    }
    build_manifest = {
        "schema_version": 2,
        "git": {
            "sha": git_sha,
            "dirty": False,
            "diff_sha256": "8" * 64,
        },
        "runtime": {"scheduler_cluster_name": None},
        "environment_lock": environment,
        "environment_lock_sha256": environment_identity_sha256(environment),
    }
    build_manifest_path = tmp_path / "build.json"
    _json_dump(build_manifest_path, build_manifest)

    monkeypatch.setattr(campaigns, "verify_calder_artifact_lock", lambda _lock: None)
    monkeypatch.setattr(
        campaigns,
        "validate_build_manifest",
        lambda *_args, **_kwargs: {
            "git_sha": git_sha,
            "tracked_tree_sha256": "9" * 64,
        },
    )
    return {
        "repo": repo,
        "lock": lock,
        "lock_path": lock_path,
        "effective_manifest": effective_manifest,
        "effective_manifest_path": effective_manifest_path,
        "build_manifest": build_manifest,
        "build_manifest_path": build_manifest_path,
        "output": tmp_path / "campaign-specs",
    }


def _generate(inputs: dict[str, object]) -> campaigns.CalderCampaignSpecs:
    return campaigns.generate_calder_campaign_specs(
        repo_root=inputs["repo"],
        artifact_lock_path=inputs["lock_path"],
        effective_manifest_path=inputs["effective_manifest_path"],
        build_manifest_path=inputs["build_manifest_path"],
        output_dir=inputs["output"],
    )


def test_generates_exact_local_canary_and_production_specs_idempotently(
    tmp_path, monkeypatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)

    result = _generate(inputs)
    second = _generate(inputs)

    assert second == result
    assert result.canary_task_count == 4
    assert result.production_task_count == 1000
    canary = yaml.safe_load(Path(result.canary_path).read_text(encoding="utf-8"))
    production = yaml.safe_load(Path(result.production_path).read_text(encoding="utf-8"))
    assert canary["campaign_id"] == campaigns.CANARY_CAMPAIGN_ID
    assert canary["expect"] == {
        "config_count": 4,
        "task_count": 4,
        "tasks_per_method": {"laplace_learning": 2, "poisson_learning": 2},
        "tasks_by_profile": {"cpu_graph": 4},
        "tasks_by_site": {"local-cpu": 4},
    }
    assert [(cell["protocol_id"], cell["seeds"]) for cell in canary["cells"]] == [
        ("calder-2020-mnist-table1-laplace-1-label-per-class", [0]),
        ("calder-2020-mnist-table1-laplace-5-label-per-class", [0]),
        ("calder-2020-mnist-table1-poisson-1-label-per-class", [0]),
        ("calder-2020-mnist-table1-poisson-5-label-per-class", [0]),
    ]
    assert production["campaign_id"] == campaigns.PRODUCTION_CAMPAIGN_ID
    assert production["expect"]["task_count"] == 1000
    assert production["expect"]["config_count"] == 10
    assert all(cell["seeds"] == "from_config" for cell in production["cells"])
    assert all(cell["site"] == "local-cpu" for cell in production["cells"])
    assert all(cell["resource_profile"] == "cpu_graph" for cell in production["cells"])
    assert all(cell["fidelity_status"] == "paper_matched" for cell in production["cells"])
    assert {cell["expected_dataset_fingerprint"] for cell in production["cells"]} == {"6" * 64}
    assert {cell["expected_dataset_content_sha256"] for cell in production["cells"]} == {"7" * 64}
    assert production["code"] == {
        "git_sha": "a" * 40,
        "git_diff_sha256": "8" * 64,
        "require_clean": True,
        "environment_lock_sha256": result.environment_lock_sha256,
    }
    assert production["default_site"] == "local-cpu"
    assert production["calder_artifacts"]["artifact_lock_sha256"] == inputs["lock"]["lock_sha256"]
    assert production["calder_artifacts"]["effective_manifest"] == {
        "path": Path(inputs["effective_manifest_path"]).relative_to(inputs["repo"]).as_posix(),
        "sha256": _sha256_file(inputs["effective_manifest_path"]),
        "lock_sha256": inputs["effective_manifest"]["lock_sha256"],
    }
    assert len(production["calder_artifacts"]["effective_configs"]) == 10
    assert production["calder_artifacts"]["source_replay_oracle"] == {
        "path": campaigns.SOURCE_REPLAY_ORACLE_RELATIVE.as_posix(),
        "sha256": campaigns.SOURCE_REPLAY_ORACLE_SHA256,
    }
    assert {
        cell["config"]: cell["effective_config_sha256"] for cell in production["cells"]
    } == production["calder_artifacts"]["effective_configs"]


def test_refuses_changed_source_replay_oracle(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    oracle = Path(inputs["repo"]) / campaigns.SOURCE_REPLAY_ORACLE_RELATIVE
    oracle.write_text(
        oracle.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        campaigns.CalderCampaignError,
        match="source-replay oracle SHA-256 differs",
    ):
        _generate(inputs)


def test_refuses_unverified_lock_and_invalid_build_manifest(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        campaigns,
        "verify_calder_artifact_lock",
        lambda _lock: (_ for _ in ()).throw(CalderArtifactError("bad lock")),
    )
    with pytest.raises(campaigns.CalderCampaignError, match="lock verification failed"):
        _generate(inputs)

    inputs = _fixture(tmp_path / "build", monkeypatch)
    monkeypatch.setattr(
        campaigns,
        "validate_build_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad build")),
    )
    with pytest.raises(campaigns.CalderCampaignError, match="build manifest verification failed"):
        _generate(inputs)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("commit", "different commits"),
        ("lock", "different artifact lock"),
        ("source", "source configs differ"),
        ("pins", "graph pins differ"),
        ("seal", "MANIFEST SHA-256 differs"),
        ("count", "must contain ten configs"),
        ("order", "order or membership differs"),
    ],
)
def test_refuses_incoherent_lock_and_effective_manifest(
    tmp_path, monkeypatch, case: str, message: str
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    manifest = copy.deepcopy(inputs["effective_manifest"])
    lock = copy.deepcopy(inputs["lock"])
    if case == "commit":
        lock["builder"]["git_sha"] = "c" * 40
        lock = seal_calder_artifact_lock(lock)
        _json_dump(inputs["lock_path"], lock)
        manifest["artifact_lock_sha256"] = lock["lock_sha256"]
    elif case == "lock":
        manifest["artifact_lock_sha256"] = "d" * 64
    elif case == "source":
        manifest["source_configs"] = []
    elif case == "pins":
        manifest["pins"]["graph_fingerprint"] = "different"
    elif case == "seal":
        manifest["kind"] = "different"
        _json_dump(inputs["effective_manifest_path"], manifest)
        with pytest.raises(campaigns.CalderCampaignError, match=message):
            _generate(inputs)
        return
    elif case == "count":
        manifest["configs"].pop()
    else:
        manifest["configs"].reverse()
    manifest = seal_calder_artifact_lock(manifest)
    _json_dump(inputs["effective_manifest_path"], manifest)

    with pytest.raises(campaigns.CalderCampaignError, match=message):
        _generate(inputs)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("sha", "SHA-256 differs"),
        ("content", "differs outside"),
        ("placeholder", "placeholder"),
        ("protocol", "protocol identity differs"),
        ("outside", "below"),
    ],
)
def test_refuses_tampered_or_outside_effective_configs(
    tmp_path, monkeypatch, case: str, message: str
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    manifest = copy.deepcopy(inputs["effective_manifest"])
    record = manifest["configs"][0]
    path = inputs["repo"] / record["repo_path"]
    if case == "sha":
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif case == "content":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["evaluation"]["metrics"] = ["macro_f1"]
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        record["sha256"] = _sha256_file(path)
    elif case == "placeholder":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["run"]["output_dir"] = "REPLACE_WITH_OUTPUT"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        record["sha256"] = _sha256_file(path)
        source = inputs["repo"] / "bench/configs/reproductions" / CALDER_CONFIGS[0]
        source_raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        source_raw["run"]["output_dir"] = "REPLACE_WITH_OUTPUT"
        source.write_text(yaml.safe_dump(source_raw, sort_keys=False), encoding="utf-8")
        family = load_calder_config_family(inputs["repo"])
        inputs["lock"]["builder"]["config_files"] = list(family.files)
        updated_lock = seal_calder_artifact_lock(inputs["lock"])
        _json_dump(inputs["lock_path"], updated_lock)
        manifest["artifact_lock_sha256"] = updated_lock["lock_sha256"]
        manifest["source_configs"] = list(family.files)
    elif case == "protocol":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["run"]["seeds"] = [0]
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        record["sha256"] = _sha256_file(path)
        source = inputs["repo"] / "bench/configs/reproductions" / CALDER_CONFIGS[0]
        source_raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        source_raw["run"]["seeds"] = [0]
        source.write_text(yaml.safe_dump(source_raw, sort_keys=False), encoding="utf-8")
        family = load_calder_config_family(inputs["repo"])
        inputs["lock"]["builder"]["config_files"] = list(family.files)
        updated_lock = seal_calder_artifact_lock(inputs["lock"])
        _json_dump(inputs["lock_path"], updated_lock)
        manifest["artifact_lock_sha256"] = updated_lock["lock_sha256"]
        manifest["source_configs"] = list(family.files)
    else:
        outside = tmp_path / "outside.yaml"
        shutil.copyfile(path, outside)
        record["repo_path"] = "../outside.yaml"
    manifest = seal_calder_artifact_lock(manifest)
    _json_dump(inputs["effective_manifest_path"], manifest)

    with pytest.raises(campaigns.CalderCampaignError, match=message):
        _generate(inputs)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("dirty", "clean worktree"),
        ("slurm", "local unscheduled runtime"),
        ("environment", "environment lock SHA-256 differs"),
        ("dataset", "prepared dataset fingerprint"),
        ("content", "content SHA-256"),
    ],
)
def test_refuses_unpinned_local_build_or_dataset(
    tmp_path, monkeypatch, case: str, message: str
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    build = copy.deepcopy(inputs["build_manifest"])
    lock = copy.deepcopy(inputs["lock"])
    if case == "dirty":
        build["git"]["dirty"] = True
    elif case == "slurm":
        build["runtime"]["scheduler_cluster_name"] = "scheduled-cluster"
    elif case == "environment":
        build["environment_lock_sha256"] = "0" * 64
    elif case == "dataset":
        lock["dataset"]["prepared_fingerprint"] = "not-a-sha"
    else:
        lock["dataset"]["content_evidence"]["content_sha256"] = "not-a-sha"
    _json_dump(inputs["build_manifest_path"], build)
    updated_lock = seal_calder_artifact_lock(lock)
    _json_dump(inputs["lock_path"], updated_lock)
    if case in {"dataset", "content"}:
        manifest = copy.deepcopy(inputs["effective_manifest"])
        manifest["artifact_lock_sha256"] = updated_lock["lock_sha256"]
        _json_dump(
            inputs["effective_manifest_path"],
            seal_calder_artifact_lock(manifest),
        )

    with pytest.raises(campaigns.CalderCampaignError, match=message):
        _generate(inputs)


def test_refuses_different_or_partial_existing_output(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    result = _generate(inputs)
    Path(result.canary_path).write_text("different\n", encoding="utf-8")
    with pytest.raises(campaigns.CalderCampaignError, match="replace different"):
        _generate(inputs)

    inputs = _fixture(tmp_path / "partial", monkeypatch)
    inputs["output"].mkdir(parents=True)
    (inputs["output"] / campaigns.CANARY_FILE_NAME).write_text("", encoding="utf-8")
    with pytest.raises(campaigns.CalderCampaignError, match="refusing to modify existing"):
        _generate(inputs)


def test_cli_prints_summary_and_reports_validation_error(tmp_path, monkeypatch, capsys) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    args = [
        "--repo-root",
        str(inputs["repo"]),
        "--artifact-lock",
        str(inputs["lock_path"]),
        "--effective-manifest",
        str(inputs["effective_manifest_path"]),
        "--build-manifest",
        str(inputs["build_manifest_path"]),
        "--output-dir",
        str(inputs["output"]),
    ]

    assert campaigns.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["production_task_count"] == 1000

    monkeypatch.setattr(
        campaigns,
        "generate_calder_campaign_specs",
        lambda **_kwargs: (_ for _ in ()).throw(campaigns.CalderCampaignError("stopped")),
    )
    with pytest.raises(SystemExit) as exc_info:
        campaigns.main(args)
    assert exc_info.value.code == 2
    assert "calder-campaigns: stopped" in capsys.readouterr().err
