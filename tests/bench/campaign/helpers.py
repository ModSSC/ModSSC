from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.utils.hashing import hash_any
from bench.utils.io import atomic_write_json
from modssc.sampling.plan import SamplingPlan

_GATE_REGISTRY = (
    Path(__file__).resolve().parents[3] / "bench" / "campaigns" / "scientific-gates.yaml"
)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def install_test_gate_registry(repo_root: Path, *, source: Path | None = None) -> Path:
    """Install the exact gate policy a synthetic repository will pin."""

    target = repo_root / "bench" / "campaigns" / "scientific-gates.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source or _GATE_REGISTRY, target)
    return target


def preflight_governance(tasks: list[Any]) -> dict[str, Any]:
    """Return governance fields shared by every task covered by one preflight."""

    if not tasks:
        raise ValueError("preflight task coverage must not be empty")
    fields = (
        "claim_scope_id",
        "campaign_stage",
        "claim_eligible",
        "gate_policy_id",
        "gate_policy_sha256",
    )
    payload = {field: getattr(tasks[0], field) for field in fields}
    if any(any(getattr(task, field) != payload[field] for field in fields) for task in tasks[1:]):
        raise ValueError("one preflight cannot mix scientific governance contracts")
    return payload


def minimal_config(*, output_dir: Path, cache_dir: Path) -> dict[str, Any]:
    return {
        "run": {
            "name": "campaign_toy",
            "seed": 1,
            "seeds": [1, 2],
            "seeded_sections": ["sampling", "preprocess"],
            "output_dir": str(output_dir),
            "fail_fast": True,
            "benchmark_mode": False,
        },
        "dataset": {
            "id": "toy",
            "download": False,
            "cache_dir": str(cache_dir),
        },
        "sampling": {
            "seed": 1,
            "plan": {
                "split": {
                    "kind": "holdout",
                    "test_fraction": 0.2,
                    "val_fraction": 0.1,
                    "stratify": True,
                    "shuffle": True,
                },
                "labeling": {
                    "mode": "per_class",
                    "value": 1,
                    "strategy": "balanced",
                    "min_per_class": 1,
                    "per_class": True,
                    "fixed_indices": None,
                },
            },
        },
        "preprocess": {
            "seed": 1,
            "fit_on": "train_labeled",
            "cache": False,
            "plan": {
                "output_key": "features.X",
                "steps": [{"id": "core.to_numpy"}],
            },
        },
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "cpu", "dtype": "float32"},
            "params": {},
        },
        "evaluation": {
            "split_for_model_selection": "val",
            "report_splits": ["val", "test"],
            "metrics": ["accuracy"],
        },
    }


def build_test_campaign(
    tmp_path: Path,
    *,
    with_site: bool = False,
    array_block_size: int | None = None,
) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    install_test_gate_registry(repo)
    config_path = (
        repo
        / "bench"
        / "configs"
        / "best"
        / "R1"
        / "inductive"
        / "pseudo_label"
        / "tabular"
        / "toy.yaml"
    )
    write_yaml(
        config_path,
        minimal_config(output_dir=tmp_path / "source-output", cache_dir=tmp_path / "cache"),
    )
    spec_path = repo / "campaign.yaml"
    dataset_lock_path = repo / "dataset-lock.yaml"
    write_yaml(
        dataset_lock_path,
        {
            "schema_version": 1,
            "datasets": {
                "adult": "dataset-fp",
                "cifar10": "dataset-fp",
                "toy": "dataset-fp",
            },
        },
    )
    write_yaml(
        spec_path,
        {
            "schema_version": 1,
            "campaign_id": "test-campaign",
            "track": "standardized",
            "selection": {
                "config_root": "bench/configs/best",
                "methods": ["pseudo_label"],
                "seeds": "from_config",
                "dataset_lock_file": "dataset-lock.yaml",
            },
            "code": {
                "git_sha": "test-sha",
                "require_clean": False,
                "git_diff_sha256": "0" * 64,
                "environment_lock_sha256": "unlocked",
            },
            # Unit-test campaigns exercise the execution machinery but are not
            # scientific claims.  Keeping that distinction explicit prevents
            # production-only pin and preflight rules from being weakened for
            # the sake of local fixtures.
            "scientific_scope": {
                "claim_scope_id": "article10",
                "stage": "diagnostic",
                "claim_eligible": False,
            },
            "expect": {"config_count": 1, "task_count": 2, "tasks_per_method": 2},
            "profile_rules": [
                {"profile": "cpu_test", "site": "local", "methods": ["pseudo_label"]}
            ],
        },
    )
    site_paths: list[Path] = []
    if with_site:
        site_path = repo / "site.yaml"
        write_yaml(
            site_path,
            {
                "schema_version": 1,
                "site_id": "local",
                "scheduler": "slurm",
                "environment_lock_sha256": "unlocked",
                "setup": [],
                "profiles": {
                    "cpu_test": {
                        "concurrency": 2,
                        "directives": {
                            "nodes": 1,
                            "ntasks": 1,
                            "cpus-per-task": 2,
                            "time": "00:10:00",
                        },
                    }
                },
            },
        )
        if array_block_size is not None:
            site_payload = yaml.safe_load(site_path.read_text(encoding="utf-8"))
            site_payload["profiles"]["cpu_test"]["array_block_size"] = int(array_block_size)
            write_yaml(site_path, site_payload)
        site_paths.append(site_path)
    campaign_dir = tmp_path / "generated"
    generate_campaign(
        spec_path,
        repo_root=repo,
        output_dir=campaign_dir,
    )
    if site_paths:
        from tools.hpc.resources import plan_resource_sites
        from tools.hpc.slurm_renderer import render_slurm_sites

        _, generated_tasks = load_manifest(campaign_dir / "manifest.jsonl")
        plan_resource_sites(
            site_paths=site_paths,
            tasks=generated_tasks,
            campaign_dir=campaign_dir,
        )
        render_slurm_sites(site_paths=site_paths, campaign_dir=campaign_dir)
    return repo, config_path, campaign_dir


def run_payload(
    *,
    cfg: Any,
    git_sha: str = "test-sha",
    replay: dict[str, str] | None = None,
    dataset_fingerprint: str = "dataset-fp",
    split_fingerprint: str = "split-fp",
) -> dict[str, Any]:
    return {
        "run": {
            "name": cfg.run.name,
            "seed": cfg.run.seed,
            "run_id": "fake-run-id",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "status": "success",
            "benchmark_mode": bool(cfg.run.benchmark_mode),
            "config_path": "effective.yaml",
            "error_code": None,
        },
        "hashes": {"config_hash": "a", "effective_config_hash": "b"},
        "resolution": {
            "device": {"requested": "cpu", "resolved": "cpu"},
            "backend": {"requested": {}, "resolved": {}},
            "dtype": {"requested": {}, "resolved": {}},
            "normalization": {"requested": {}, "resolved": {}},
            "splits": {"requested": ["val", "test"], "resolved": {}},
            "limits": {"requested": None, "resolved": None, "changes": []},
        },
        "protocol": {
            "kind": cfg.method.kind,
            "use_test_split": True,
            "report_splits": ["val", "test"],
            "split_for_model_selection": "val",
        },
        "versions": {
            "python": "3.12",
            "modssc": "0",
            "numpy": "0",
            "git_sha": git_sha,
            "git_dirty": False,
            "git_diff_sha256": "0" * 64,
        },
        "run_info": {"run_time_seconds": 1.0},
        "task_info": {
            "dataset_id": cfg.dataset.id,
            "method_id": cfg.method.method_id,
            "method_kind": cfg.method.kind,
        },
        "graph_info": None,
        "config": {},
        "artifacts": {
            "dataset": {"fingerprint": dataset_fingerprint},
            "sampling": {
                "split_fingerprint": split_fingerprint,
                "replay": replay
                or {"format": "modssc.sampling.storage.v1", "path": "sampling_split"},
            },
            "method": {"profile": cfg.method.profile},
        },
        "metrics": {"test": {"accuracy": 0.5}},
        "hpo": None,
        "fallback_events": [],
        "error": None,
    }


@dataclass
class FakeRunResult:
    code: int
    run_dir: Path
    run_json_path: Path


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, config_path: Path, *, raw: dict[str, Any], cfg: Any) -> FakeRunResult:
        self.calls.append({"config_path": config_path, "raw": raw, "cfg": cfg})
        run_dir = Path(cfg.run.output_dir) / "fake-run"
        run_dir.mkdir(parents=True, exist_ok=False)
        replay = run_dir / "sampling_split"
        replay.mkdir()
        dataset_fingerprint = "dataset-fp"
        split_fingerprint = hash_any(
            {
                "schema_version": 1,
                "dataset_fingerprint": dataset_fingerprint,
                "plan": SamplingPlan.from_dict(raw["sampling"]["plan"]).as_dict(),
                "seed": raw["sampling"]["seed"],
            }
        )
        atomic_write_json(
            replay / "split.json",
            {
                "dataset_fingerprint": dataset_fingerprint,
                "split_fingerprint": split_fingerprint,
            },
        )
        (replay / "arrays.npz").write_bytes(b"fake-npz")
        files = {
            name: {"sha256": hashlib.sha256((replay / name).read_bytes()).hexdigest()}
            for name in ("split.json", "arrays.npz")
        }
        atomic_write_json(
            replay / "MANIFEST.json",
            {
                "schema_version": 1,
                "format": "modssc.sampling.storage.v1",
                "dataset_fingerprint": dataset_fingerprint,
                "split_fingerprint": split_fingerprint,
                "files": files,
            },
        )
        replay_artifact = {
            "format": "modssc.sampling.storage.v1",
            "path": "sampling_split",
            "manifest": "MANIFEST.json",
            "manifest_sha256": hashlib.sha256((replay / "MANIFEST.json").read_bytes()).hexdigest(),
        }
        run_json_path = run_dir / "run.json"
        atomic_write_json(
            run_json_path,
            run_payload(
                cfg=cfg,
                replay=replay_artifact,
                dataset_fingerprint=dataset_fingerprint,
                split_fingerprint=split_fingerprint,
            ),
        )
        (run_dir / "config.yaml").write_text("fake: true\n", encoding="utf-8")
        return FakeRunResult(code=0, run_dir=run_dir, run_json_path=run_json_path)


def fake_versions(**_: Any) -> dict[str, Any]:
    return {
        "git_sha": "test-sha",
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
    }


def rewrite_success_digest(result_dir: Path) -> None:
    import hashlib

    run_json = result_dir / "run" / "run.json"
    digest = hashlib.sha256(run_json.read_bytes()).hexdigest()
    marker = json.loads((result_dir / "SUCCESS.json").read_text(encoding="utf-8"))
    marker["run_json_sha256"] = digest
    atomic_write_json(result_dir / "SUCCESS.json", marker)
