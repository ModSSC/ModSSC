"""Scientific campaign scopes, kept separate from execution-site routing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import CampaignError

ARTICLE10_CLAIM_SCOPE_ID = "article10"
HISTORICAL_REPLACEMENTS_CLAIM_SCOPE_ID = "historical-replacements"

ARTICLE10_METHODS = (
    "pseudo_label",
    "tri_training",
    "democratic_co_learning",
    "fixmatch",
    "flexmatch",
    "free_match",
    "softmatch",
    "laplace_learning",
    "poisson_learning",
    "grand",
)

TECHNICAL_METHOD_CATALOG = frozenset((*ARTICLE10_METHODS, "co_training"))
CLAIM_SCOPES = {
    ARTICLE10_CLAIM_SCOPE_ID: frozenset(ARTICLE10_METHODS),
    HISTORICAL_REPLACEMENTS_CLAIM_SCOPE_ID: frozenset({"co_training"}),
}
CAMPAIGN_STAGES = frozenset({"diagnostic", "canary", "production"})


def scientific_scope(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit governance contract for a campaign specification.

    Schema-v1 specifications predate this block.  Their conservative migration
    defaults remain claim-eligible Article10 production; new diagnostic and
    replacement campaigns must opt in explicitly.
    """

    raw = spec.get("scientific_scope", {})
    if not isinstance(raw, Mapping):
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "scientific_scope must be a mapping")
    allowed = {
        "claim_scope_id",
        "stage",
        "claim_eligible",
        "gate_policy_id",
        "gate_policy_sha256",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID",
            f"scientific_scope contains unsupported fields: {unknown}",
        )
    claim_scope_id = raw.get("claim_scope_id", ARTICLE10_CLAIM_SCOPE_ID)
    stage = raw.get("stage", "production")
    claim_eligible = raw.get("claim_eligible", True)
    gate_policy_id = raw.get("gate_policy_id", "modssc-scientific-gates-v2")
    gate_policy_sha256 = raw.get("gate_policy_sha256", "from_registry")
    if claim_scope_id not in CLAIM_SCOPES:
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", f"unknown claim scope: {claim_scope_id!r}")
    if stage not in CAMPAIGN_STAGES:
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", f"unknown campaign stage: {stage!r}")
    if not isinstance(claim_eligible, bool):
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID", "scientific_scope.claim_eligible must be boolean"
        )
    if not isinstance(gate_policy_id, str) or not gate_policy_id.strip():
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID",
            "scientific_scope.gate_policy_id must be a non-empty string",
        )
    if gate_policy_sha256 != "from_registry" and (
        not isinstance(gate_policy_sha256, str)
        or len(gate_policy_sha256) != 64
        or any(character not in "0123456789abcdef" for character in gate_policy_sha256)
    ):
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID",
            "scientific_scope.gate_policy_sha256 must be from_registry or a lowercase SHA-256",
        )
    if stage != "production" and claim_eligible:
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID",
            "diagnostic and canary campaigns cannot be claim-eligible",
        )
    return {
        "claim_scope_id": str(claim_scope_id),
        "campaign_stage": str(stage),
        "claim_eligible": claim_eligible,
        "gate_policy_id": gate_policy_id,
        "gate_policy_sha256": gate_policy_sha256,
    }


def validate_method_scope(*, method_id: str, claim_scope_id: str) -> None:
    if method_id not in TECHNICAL_METHOD_CATALOG:
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID", f"method is absent from the technical catalog: {method_id}"
        )
    if method_id not in CLAIM_SCOPES[claim_scope_id]:
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID",
            f"method {method_id} does not belong to claim scope {claim_scope_id}",
        )


__all__ = [
    "ARTICLE10_CLAIM_SCOPE_ID",
    "ARTICLE10_METHODS",
    "CAMPAIGN_STAGES",
    "CLAIM_SCOPES",
    "HISTORICAL_REPLACEMENTS_CLAIM_SCOPE_ID",
    "TECHNICAL_METHOD_CATALOG",
    "scientific_scope",
    "validate_method_scope",
]
