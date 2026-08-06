from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from bench.campaign.cli import main
from bench.campaign.errors import CampaignError
from bench.campaign.executor import execute_task
from bench.campaign.generate import generate_campaign
from bench.campaign.scientific_gates import (
    ARTICLE10_METHODS,
    evaluate_gate,
    load_gate_registry,
)

from .helpers import FakeRunner, build_test_campaign, fake_versions, write_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / "bench" / "campaigns" / "scientific-gates.yaml"


def _passed_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "registry_id": "test-passed",
        "methods": {
            method_id: {
                "algorithmic_conformity": "passed",
                "conformity_basis": "pinned_official_implementation",
                "evidence": [f"tests/{method_id}.txt"],
                "reviewed_by": "test-reviewer",
                "reviewed_at": "2026-07-23T10:00:00+02:00",
            }
            for method_id in ARTICLE10_METHODS
        },
        "dependencies": {
            "flexmatch": ["fixmatch"],
            "free_match": ["fixmatch"],
            "softmatch": ["fixmatch"],
        },
        "protected_campaign_prefixes": ["article10-"],
        "exempt_campaign_ids": [
            "article10-canary-r3-wave1-v1",
            "article10-canary-r3-wave2-v1",
        ],
    }


def test_registry_opens_paper_blocks_standardized_scope_and_exempts_canaries(
    tmp_path: Path,
) -> None:
    registry = load_gate_registry(REGISTRY_PATH)

    democratic = evaluate_gate(
        registry,
        campaign_id="article10-paper-minimal-v1",
        track="paper",
        method_id="democratic_co_learning",
    )
    grand = evaluate_gate(
        registry,
        campaign_id="article10-paper-minimal-v1",
        track="paper",
        method_id="grand",
    )
    laplace = evaluate_gate(
        registry,
        campaign_id="article10-paper-minimal-v1",
        track="paper",
        method_id="laplace_learning",
    )
    poisson = evaluate_gate(
        registry,
        campaign_id="article10-paper-minimal-v1",
        track="paper",
        method_id="poisson_learning",
    )
    flex = evaluate_gate(
        registry,
        campaign_id="article10-paper-v1",
        track="paper",
        method_id="flexmatch",
    )
    standardized = evaluate_gate(
        registry,
        campaign_id="article10-standardized-v1",
        track="standardized",
        method_id="pseudo_label",
    )
    fixmatch_canary = evaluate_gate(
        registry,
        campaign_id="article10-canary-r3-wave1-v1",
        track="standardized",
        method_id="fixmatch",
        campaign_stage="canary",
        claim_eligible=False,
    )
    replacement_fixmatch_canary = evaluate_gate(
        registry,
        campaign_id="article10-canary-r3-wave1-v2",
        track="standardized",
        method_id="fixmatch",
        campaign_stage="canary",
        claim_eligible=False,
    )
    flexmatch_canary = evaluate_gate(
        registry,
        campaign_id="article10-canary-r3-wave2-v1",
        track="standardized",
        method_id="flexmatch",
        campaign_stage="canary",
        claim_eligible=False,
    )

    assert democratic.allowed is True
    assert democratic.blockers == ()
    assert laplace.allowed is True
    assert laplace.blockers == ()
    assert poisson.allowed is True
    assert poisson.blockers == ()
    assert (
        registry.method_conformity_bases["democratic_co_learning"] == "independent_equation_oracle"
    )
    assert grand.allowed is True
    assert grand.blockers == ()
    assert registry.method_conformity_bases["grand"] == "pinned_official_implementation"
    assert flex.allowed is True
    assert flex.blockers == ()
    assert registry.track_statuses == {"paper": "passed", "standardized": "pending"}
    assert standardized.allowed is False
    assert standardized.blockers == ("track_status:standardized=pending",)
    assert fixmatch_canary.allowed is True
    assert fixmatch_canary.to_dict()["blockers"] == []
    assert replacement_fixmatch_canary.allowed is True
    assert replacement_fixmatch_canary.to_dict()["blockers"] == []
    assert flexmatch_canary.allowed is True
    assert flexmatch_canary.blockers == ()

    pending_payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    for method_id in ("fixmatch", "flexmatch", "free_match", "softmatch"):
        pending_payload["methods"][method_id]["algorithmic_conformity"] = "pending"
    pending_path = tmp_path / "pending-match.yaml"
    write_yaml(pending_path, pending_payload)
    pending_registry = load_gate_registry(pending_path)

    blocked_flex = evaluate_gate(
        pending_registry,
        campaign_id="article10-paper-v1",
        track="paper",
        method_id="flexmatch",
    )
    assert blocked_flex.blockers == (
        "method_conformity:flexmatch=pending",
        "dependency_conformity:fixmatch=pending",
    )
    blocked_canary = evaluate_gate(
        pending_registry,
        campaign_id="article10-canary-r3-wave2-v1",
        track="standardized",
        method_id="flexmatch",
        campaign_stage="canary",
        claim_eligible=False,
    )
    assert blocked_canary.blockers == ("dependency_conformity:fixmatch=pending",)


def test_canary_adaptive_match_wave_opens_after_fixmatch_passes(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["methods"]["fixmatch"]["algorithmic_conformity"] = "pending"
    pending_path = tmp_path / "fixmatch-pending.yaml"
    write_yaml(pending_path, payload)
    pending_registry = load_gate_registry(pending_path)
    for method_id in ("flexmatch", "free_match", "softmatch"):
        decision = evaluate_gate(
            pending_registry,
            campaign_id="article10-canary-r3-wave2-v1",
            track="standardized",
            method_id=method_id,
            campaign_stage="canary",
            claim_eligible=False,
        )
        assert decision.blockers == ("dependency_conformity:fixmatch=pending",)

    payload["methods"]["fixmatch"] = {
        "algorithmic_conformity": "passed",
        "conformity_basis": "pinned_official_implementation",
        "evidence": ["canary/fixmatch-parity.json"],
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-07-23T10:00:00+02:00",
    }
    path = tmp_path / "fixmatch-passed.yaml"
    write_yaml(path, payload)
    registry = load_gate_registry(path)

    for method_id in ("flexmatch", "free_match", "softmatch"):
        decision = evaluate_gate(
            registry,
            campaign_id="article10-canary-r3-wave2-v1",
            track="standardized",
            method_id=method_id,
            campaign_stage="canary",
            claim_eligible=False,
        )
        assert decision.allowed is True
        assert decision.blockers == ()


def test_nonclaimable_canary_keeps_algorithmic_dependencies(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["protected_campaign_prefixes"] = ["other-"]
    payload["methods"]["fixmatch"]["algorithmic_conformity"] = "pending"
    path = tmp_path / "named-exemption.yaml"
    write_yaml(path, payload)

    decision = evaluate_gate(
        load_gate_registry(path),
        campaign_id="article10-canary-r3-wave2-v1",
        track="standardized",
        method_id="softmatch",
        campaign_stage="canary",
        claim_eligible=False,
    )
    assert decision.allowed is False
    assert decision.blockers == ("dependency_conformity:fixmatch=pending",)


def test_all_passed_registry_opens_paper_dependencies_and_standardized(tmp_path: Path) -> None:
    path = tmp_path / "passed.yaml"
    write_yaml(path, _passed_registry())
    registry = load_gate_registry(path)

    assert registry.track_statuses == {"paper": "passed", "standardized": "passed"}
    assert evaluate_gate(
        registry,
        campaign_id="paper",
        track="paper",
        method_id="softmatch",
    ).allowed
    assert evaluate_gate(
        registry,
        campaign_id="standardized",
        track="standardized",
        method_id="grand",
    ).allowed


def test_gate_registry_rejects_unreviewed_pass(tmp_path: Path) -> None:
    payload = _passed_registry()
    payload["methods"]["fixmatch"]["evidence"] = []  # type: ignore[index]
    path = tmp_path / "invalid.yaml"
    write_yaml(path, payload)

    with pytest.raises(CampaignError, match="passed method fixmatch needs evidence"):
        load_gate_registry(path)


def test_gate_registry_accepts_independent_equation_oracle_basis(tmp_path: Path) -> None:
    payload = _passed_registry()
    payload["methods"]["democratic_co_learning"]["conformity_basis"] = "independent_equation_oracle"
    path = tmp_path / "independent-oracle.yaml"
    write_yaml(path, payload)

    registry = load_gate_registry(path)

    assert (
        registry.method_conformity_bases["democratic_co_learning"] == "independent_equation_oracle"
    )


def test_gate_registry_rejects_pass_without_declared_conformity_basis(
    tmp_path: Path,
) -> None:
    payload = _passed_registry()
    del payload["methods"]["democratic_co_learning"]["conformity_basis"]
    path = tmp_path / "missing-conformity-basis.yaml"
    write_yaml(path, payload)

    with pytest.raises(
        CampaignError,
        match="passed method democratic_co_learning needs conformity_basis",
    ):
        load_gate_registry(path)


def test_gate_registry_rejects_unknown_conformity_basis(tmp_path: Path) -> None:
    payload = _passed_registry()
    payload["methods"]["democratic_co_learning"]["conformity_basis"] = "unreviewed_reimplementation"
    path = tmp_path / "unknown-conformity-basis.yaml"
    write_yaml(path, payload)

    with pytest.raises(
        CampaignError,
        match=r"methods\.democratic_co_learning\.conformity_basis must be one of",
    ):
        load_gate_registry(path)


def test_gate_registry_validates_explicit_track_statuses(tmp_path: Path) -> None:
    payload = _passed_registry()
    payload["track_statuses"] = {"paper": "passed", "standardized": "pending"}
    path = tmp_path / "track-statuses.yaml"
    write_yaml(path, payload)

    registry = load_gate_registry(path)

    assert registry.track_status("paper") == "passed"
    assert registry.track_status("standardized") == "pending"
    assert registry.track_status("unknown") == "missing"


@pytest.mark.parametrize(
    ("track_statuses", "message"),
    [
        ([], "track_statuses must be a mapping"),
        ({"paper": "passed"}, "must contain exactly paper and standardized"),
        (
            {"paper": "passed", "standardized": "pending", "extra": "passed"},
            "must contain exactly paper and standardized",
        ),
        (
            {"paper": "open", "standardized": "pending"},
            r"track_statuses\.paper must be pending, passed, or failed",
        ),
    ],
)
def test_gate_registry_rejects_invalid_track_statuses(
    tmp_path: Path,
    track_statuses: Any,
    message: str,
) -> None:
    payload = _passed_registry()
    payload["track_statuses"] = track_statuses
    path = tmp_path / "invalid-track-statuses.yaml"
    write_yaml(path, payload)

    with pytest.raises(CampaignError, match=message):
        load_gate_registry(path)


def test_executor_auto_discovers_and_enforces_repository_gate(tmp_path: Path) -> None:
    repo, _, _ = build_test_campaign(tmp_path)
    spec_path = repo / "campaign.yaml"
    registry_path = repo / "bench" / "campaigns" / "scientific-gates.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["methods"]["fixmatch"]["algorithmic_conformity"] = "pending"
    registry["dependencies"]["pseudo_label"] = ["fixmatch"]
    write_yaml(registry_path, registry)
    campaign = tmp_path / "blocked-generated"
    generate_campaign(spec_path, repo_root=repo, output_dir=campaign)
    runner = FakeRunner()

    with pytest.raises(CampaignError, match="E_SCIENTIFIC_GATE_BLOCKED"):
        execute_task(
            campaign / "manifest.jsonl",
            repo_root=repo,
            result_root=tmp_path / "results",
            work_root=tmp_path / "work",
            site_id="local",
            index=0,
            runner=runner,
            version_collector=fake_versions,
        )
    assert runner.calls == []
    attempt_paths = list((tmp_path / "results" / "attempts").glob("*/*/*/attempt.json"))
    assert len(attempt_paths) == 1
    attempt = yaml.safe_load(attempt_paths[0].read_text(encoding="utf-8"))
    assert attempt["failure_phase"] == "precondition"
    assert attempt["failure_class"] == "deterministic"
    assert attempt["retryable"] is False


def test_gate_status_cli_uses_nonzero_for_blocked_and_zero_for_open(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "gate-status",
                "--registry",
                str(REGISTRY_PATH),
                "--campaign-id",
                "article10-paper-v1",
                "--track",
                "paper",
                "--method",
                "fixmatch",
            ]
        )
        == 0
    )
    assert '"allowed": true' in capsys.readouterr().out

    assert (
        main(
            [
                "gate-status",
                "--registry",
                str(REGISTRY_PATH),
                "--campaign-id",
                "article10-standardized-v1",
                "--track",
                "standardized",
            ]
        )
        == 1
    )
    blocked_output = capsys.readouterr().out
    assert '"allowed": false' in blocked_output
    assert "track_status:standardized=pending" in blocked_output

    passed = tmp_path / "passed.yaml"
    passed.write_text(yaml.safe_dump(_passed_registry()), encoding="utf-8")
    assert (
        main(
            [
                "gate-status",
                "--registry",
                str(passed),
                "--campaign-id",
                "article10-standardized-v1",
                "--track",
                "standardized",
            ]
        )
        == 0
    )
    assert '"allowed": true' in capsys.readouterr().out
