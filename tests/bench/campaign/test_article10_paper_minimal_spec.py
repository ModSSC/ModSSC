from __future__ import annotations

from pathlib import Path

import pytest

from bench.campaign import generate as generate_module
from bench.campaign.errors import CampaignError
from bench.campaign.generate import generate_campaign

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "tools" / "hpc" / "specs" / "article10-paper-minimal.example.yaml"


def test_public_minimal_paper_spec_fails_closed_without_private_dcl_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        lambda **kwargs: {
            "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
            "git_dirty": False,
            "git_diff_sha256": "0" * 64,
        },
    )

    with pytest.raises(
        CampaignError,
        match="E_CAMPAIGN_PARTITION_SELECTION_PRIVATE_REQUIRED",
    ):
        generate_campaign(
            SPEC_PATH,
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "article10-paper-minimal",
            _allow_template_placeholders=True,
        )
