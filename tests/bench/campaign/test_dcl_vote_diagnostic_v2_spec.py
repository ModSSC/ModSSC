from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from bench.campaign import generate as generate_module
from bench.campaign.dcl_partition_lock import (
    DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS,
    DCL_DIAGNOSTIC_CONTROL_PROTOCOLS,
    DCL_DIAGNOSTIC_METHOD_PROFILE,
)
from bench.campaign.errors import CampaignError
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.campaign.paper_acceptance import evaluate_paper_campaign
from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_ROOT = REPO_ROOT / "tools" / "hpc" / "specs"

SPECS = {
    "dcl-vote-controls-v2.example.yaml": (80, DCL_DIAGNOSTIC_CONTROL_PROTOCOLS),
    "dcl-vote-confidence-primary-v2.example.yaml": (
        40,
        {
            protocol_id: values
            for protocol_id, values in DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS.items()
            if values in {("training_accuracy", "wald"), ("kfold_oof", "wald")}
        },
    ),
    "dcl-vote-confidence-conditional-v2.example.yaml": (
        40,
        {
            protocol_id: values
            for protocol_id, values in DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS.items()
            if values in {("kfold_oof", "wilson"), ("kfold_oof", "clopper_pearson")}
        },
    ),
}


def _fake_clean_versions(**_: object) -> dict[str, object]:
    return {
        "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
    }


@pytest.mark.parametrize(("spec_name", "expected"), SPECS.items())
def test_dcl_vote_v2_diagnostic_specs_replay_all_v1_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec_name: str,
    expected: tuple[int, dict[str, object]],
) -> None:
    expected_count, expected_protocols = expected
    monkeypatch.setattr(generate_module, "collect_runtime_versions", _fake_clean_versions)

    generated = generate_campaign(
        SPEC_ROOT / spec_name,
        repo_root=REPO_ROOT,
        output_dir=tmp_path / spec_name,
        _allow_template_placeholders=True,
    )
    meta, tasks = load_manifest(Path(generated.manifest_path))

    assert generated.task_count == expected_count
    assert meta["counts_by_method"] == {"democratic_co_learning": expected_count}
    assert meta["counts_by_profile"] == {"v100_gpu": expected_count}
    assert meta["counts_by_site"] == {"slurm-gpu": expected_count}
    assert {task.protocol_id for task in tasks} == set(expected_protocols)
    assert {task.method_profile for task in tasks} == {DCL_DIAGNOSTIC_METHOD_PROFILE}
    assert {task.fidelity_status for task in tasks} == {"not_claimable"}
    assert {task.dataset_id for task in tasks} == {"vote"}

    for config_path in {task.config_path for task in tasks}:
        config_tasks = [task for task in tasks if task.config_path == config_path]
        assert {task.seed for task in config_tasks} == set(range(1, 21))
        assert {task.partition_selection["selection_rank"] for task in config_tasks} == set(
            range(1, 21)
        )
        assert all(
            task.expected_split_fingerprint == task.partition_selection["split_fingerprint"]
            for task in config_tasks
        )


def test_dcl_vote_v2_diagnostic_configs_separate_test_controls_from_confidence() -> None:
    config_root = REPO_ROOT / "bench" / "configs" / "diagnostics" / "democratic_co_learning"
    configs = {
        path.name: ExperimentConfig.from_dict(load_yaml(path))
        for path in sorted(config_root.glob("*_v2.yaml"))
    }

    assert len(configs) == 8
    controls = [
        config for config in configs.values() if config.method.params["control_mode"] != "dcl"
    ]
    confidence = [
        config for config in configs.values() if config.method.params["control_mode"] == "dcl"
    ]
    assert len(controls) == 4
    assert len(confidence) == 4
    assert {tuple(config.evaluation.report_splits) for config in controls} == {("test",)}
    assert {tuple(config.evaluation.report_splits) for config in confidence} == {("train_labeled",)}
    assert all(config.evaluation.split_for_model_selection is None for config in configs.values())
    assert all(
        config.method.profile == DCL_DIAGNOSTIC_METHOD_PROFILE for config in configs.values()
    )
    assert all(config.method.params["diagnostic_trace"] is True for config in configs.values())
    assert {
        (
            config.method.params["confidence_estimator"],
            config.method.params["confidence_interval"],
        )
        for config in confidence
    } == {
        ("training_accuracy", "wald"),
        ("kfold_oof", "wald"),
        ("kfold_oof", "wilson"),
        ("kfold_oof", "clopper_pearson"),
    }


def test_dcl_vote_v2_diagnostic_manifest_cannot_enter_paper_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generate_module, "collect_runtime_versions", _fake_clean_versions)
    generated = generate_campaign(
        SPEC_ROOT / "dcl-vote-confidence-primary-v2.example.yaml",
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "diagnostic",
        _allow_template_placeholders=True,
    )

    with pytest.raises(CampaignError, match="cannot enter paper acceptance"):
        evaluate_paper_campaign(
            Path(generated.manifest_path),
            reconcile_path=tmp_path / "must-not-be-read.json",
            acceptance_path=REPO_ROOT / "bench" / "campaigns" / "article10-paper-acceptance.yaml",
            gate_registry_path=REPO_ROOT / "bench" / "campaigns" / "scientific-gates.yaml",
            output_dir=tmp_path / "acceptance",
        )


@pytest.mark.parametrize(
    ("spec_name", "config_name", "parameter", "value", "message"),
    [
        (
            "dcl-vote-controls-v2.example.yaml",
            "vote_control_naive_bayes_v2.yaml",
            "confidence_seed",
            1,
            "control protocol and immutable parameters differ",
        ),
        (
            "dcl-vote-controls-v2.example.yaml",
            "vote_control_combining_only_v2.yaml",
            "confidence_interval",
            "wilson",
            "control protocol and immutable parameters differ",
        ),
        (
            "dcl-vote-confidence-primary-v2.example.yaml",
            "vote_confidence_10fold_wald_v2.yaml",
            "confidence_level",
            0.9,
            "scientific core differs at method parameters",
        ),
        (
            "dcl-vote-confidence-primary-v2.example.yaml",
            "vote_confidence_resub_wald_v2.yaml",
            "control_mode",
            "learner_0",
            "confidence protocol and method parameters differ",
        ),
    ],
)
def test_dcl_vote_v2_protocol_ids_reject_silent_parameter_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec_name: str,
    config_name: str,
    parameter: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(generate_module, "collect_runtime_versions", _fake_clean_versions)
    original_load_yaml = generate_module.load_yaml

    def tampered_load_yaml(path: str | Path) -> dict[str, Any]:
        payload = original_load_yaml(path)
        if Path(path).name == config_name:
            payload = deepcopy(payload)
            payload["method"]["params"][parameter] = value
        return payload

    monkeypatch.setattr(generate_module, "load_yaml", tampered_load_yaml)

    with pytest.raises(CampaignError, match=message):
        generate_campaign(
            SPEC_ROOT / spec_name,
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "rejected",
            _allow_template_placeholders=True,
        )


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        ("dataset", "dataset"),
        ("sampling", "sampling"),
        ("preprocess", "preprocess"),
        ("max_iter", "method parameters"),
        ("min_confidence", "method parameters"),
        ("learner_order", "method parameters"),
        ("classifier_backend", "method parameters"),
        ("evaluation", "evaluation"),
    ],
)
def test_dcl_vote_v2_rejects_scientific_core_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    field: str,
) -> None:
    monkeypatch.setattr(generate_module, "collect_runtime_versions", _fake_clean_versions)
    original_load_yaml = generate_module.load_yaml

    def tampered_load_yaml(path: str | Path) -> dict[str, Any]:
        payload = original_load_yaml(path)
        if Path(path).name != "vote_control_naive_bayes_v2.yaml":
            return payload
        payload = deepcopy(payload)
        if mutation == "dataset":
            payload["dataset"]["download"] = True
        elif mutation == "sampling":
            payload["sampling"]["plan"]["split"]["test_fraction"] = 0.4
        elif mutation == "preprocess":
            payload["preprocess"]["plan"]["steps"].reverse()
        elif mutation == "max_iter":
            payload["method"]["params"]["max_iter"] = 19
        elif mutation == "min_confidence":
            payload["method"]["params"]["min_confidence"] = 0.49
        elif mutation == "learner_order":
            payload["method"]["params"]["classifier_specs"].reverse()
        elif mutation == "classifier_backend":
            payload["method"]["params"]["classifier_specs"][0]["classifier_backend"] = "sklearn"
        elif mutation == "evaluation":
            payload["evaluation"]["metrics"] = ["accuracy"]
        else:  # pragma: no cover - protected by the parametrization above.
            raise AssertionError(mutation)
        return payload

    monkeypatch.setattr(generate_module, "load_yaml", tampered_load_yaml)

    with pytest.raises(CampaignError, match=rf"scientific core differs at {field}"):
        generate_campaign(
            SPEC_ROOT / "dcl-vote-controls-v2.example.yaml",
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "rejected-core",
            _allow_template_placeholders=True,
        )
