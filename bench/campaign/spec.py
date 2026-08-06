from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .catalog import scientific_scope
from .errors import CampaignError
from .identifiers import validate_safe_identifier


def load_spec(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", f"cannot read spec: {path}") from exc
    if not isinstance(raw, dict):
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "campaign spec must be a mapping")
    if raw.get("schema_version") != 1:
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "spec schema_version must equal 1")
    validate_safe_identifier(
        raw.get("campaign_id"),
        field="campaign_id",
        code="E_CAMPAIGN_SPEC_INVALID",
    )
    track = raw.get("track")
    if track not in {"paper", "standardized"}:
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "track must be 'paper' or 'standardized'")
    scientific_scope(raw)
    return raw


def string_list(value: Any, *, field: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", f"{field} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", f"{field} must contain non-empty strings")
    return [str(item) for item in value]
