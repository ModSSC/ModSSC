from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from tools.replication_audit.calder.canary import ACCEPTANCE_KIND

from .helpers import build_test_campaign, write_yaml


def test_public_generator_treats_historical_calder_campaign_as_scheduler_neutral(
    tmp_path: Path,
) -> None:
    repo, _config_path, _unused = build_test_campaign(tmp_path / "base")
    spec_path = repo / "campaign.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["campaign_id"] = "article10-calder-paper-local-v1"
    write_yaml(spec_path, spec)

    output = tmp_path / "generated"
    generated = generate_campaign(
        spec_path,
        repo_root=repo,
        output_dir=output,
    )

    meta, tasks = load_manifest(Path(generated.manifest_path))
    assert len(tasks) == 2
    assert "release_evidence" not in meta


def test_historical_calder_release_validator_is_repo_only() -> None:
    assert "release_evidence_path" not in inspect.signature(generate_campaign).parameters
    assert ACCEPTANCE_KIND == "modssc.calder2020-mnist-table1-canary-acceptance"
