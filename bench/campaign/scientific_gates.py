from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from .catalog import ARTICLE10_METHODS, CLAIM_SCOPES, TECHNICAL_METHOD_CATALOG
from .errors import CampaignError
from .models import CampaignTask

_VALID_STATUSES = {"pending", "passed", "failed"}
_TRACKS = ("paper", "standardized")
_VALID_CONFORMITY_BASES = {
    "pinned_official_implementation",
    "independent_equation_oracle",
}


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    campaign_id: str
    track: str
    method_id: str
    claim_scope_id: str
    campaign_stage: str
    claim_eligible: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "campaign_id": self.campaign_id,
            "track": self.track,
            "method_id": self.method_id,
            "claim_scope_id": self.claim_scope_id,
            "campaign_stage": self.campaign_stage,
            "claim_eligible": self.claim_eligible,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class ScientificGateRegistry:
    registry_id: str
    method_statuses: dict[str, str]
    method_conformity_bases: dict[str, str | None]
    track_statuses: dict[str, str]
    dependencies: dict[str, tuple[str, ...]]
    claim_scopes: dict[str, frozenset[str]]
    protected_campaign_prefixes: tuple[str, ...]
    exempt_campaign_ids: frozenset[str]

    def status(self, method_id: str) -> str:
        return self.method_statuses.get(method_id, "missing")

    def track_status(self, track: str) -> str:
        return self.track_statuses.get(track, "missing")


def _non_empty_strings(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CampaignError("E_SCIENTIFIC_GATE_SCHEMA", f"{field} must be a list[str]")
    return [str(item) for item in value]


def load_gate_registry(path: Path) -> ScientificGateRegistry:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignError(
            "E_SCIENTIFIC_GATE_SCHEMA", f"cannot read scientific gate registry: {path}"
        ) from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") not in {1, 2}:
        raise CampaignError(
            "E_SCIENTIFIC_GATE_SCHEMA",
            "scientific gate registry must use schema_version 1 or 2",
        )
    schema_version = int(raw["schema_version"])
    registry_id = raw.get("registry_id")
    if not isinstance(registry_id, str) or not registry_id.strip():
        raise CampaignError("E_SCIENTIFIC_GATE_SCHEMA", "registry_id must be non-empty")
    raw_methods = raw.get("methods")
    if not isinstance(raw_methods, Mapping):
        raise CampaignError("E_SCIENTIFIC_GATE_SCHEMA", "methods must be a mapping")
    expected_methods = (
        set(ARTICLE10_METHODS) if schema_version == 1 else set(TECHNICAL_METHOD_CATALOG)
    )
    missing = sorted(expected_methods - set(raw_methods))
    extra = sorted(set(raw_methods) - expected_methods)
    if missing or extra:
        raise CampaignError(
            "E_SCIENTIFIC_GATE_SCHEMA",
            f"registry methods differ from the technical catalog; missing={missing}, extra={extra}",
        )

    statuses: dict[str, str] = {}
    conformity_bases: dict[str, str | None] = {}
    for method_id in sorted(expected_methods):
        entry = raw_methods[method_id]
        if not isinstance(entry, Mapping):
            raise CampaignError(
                "E_SCIENTIFIC_GATE_SCHEMA", f"methods.{method_id} must be a mapping"
            )
        status = entry.get("algorithmic_conformity")
        if status not in _VALID_STATUSES:
            raise CampaignError(
                "E_SCIENTIFIC_GATE_SCHEMA",
                f"methods.{method_id}.algorithmic_conformity must be pending, passed, or failed",
            )
        evidence = _non_empty_strings(
            entry.get("evidence", []), field=f"methods.{method_id}.evidence"
        )
        conformity_basis = entry.get("conformity_basis")
        if conformity_basis is not None and conformity_basis not in _VALID_CONFORMITY_BASES:
            raise CampaignError(
                "E_SCIENTIFIC_GATE_SCHEMA",
                (
                    f"methods.{method_id}.conformity_basis must be one of "
                    f"{sorted(_VALID_CONFORMITY_BASES)}"
                ),
            )
        if status == "passed":
            reviewed_by = entry.get("reviewed_by")
            reviewed_at = entry.get("reviewed_at")
            if not evidence or not isinstance(reviewed_by, str) or not reviewed_by.strip():
                raise CampaignError(
                    "E_SCIENTIFIC_GATE_SCHEMA",
                    f"passed method {method_id} needs evidence and reviewed_by",
                )
            if conformity_basis not in _VALID_CONFORMITY_BASES:
                raise CampaignError(
                    "E_SCIENTIFIC_GATE_SCHEMA",
                    f"passed method {method_id} needs conformity_basis",
                )
            if not isinstance(reviewed_at, str) or not reviewed_at.strip():
                raise CampaignError(
                    "E_SCIENTIFIC_GATE_SCHEMA", f"passed method {method_id} needs reviewed_at"
                )
            try:
                datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise CampaignError(
                    "E_SCIENTIFIC_GATE_SCHEMA",
                    f"methods.{method_id}.reviewed_at must be ISO-8601",
                ) from exc
        statuses[method_id] = str(status)
        conformity_bases[method_id] = (
            str(conformity_basis) if conformity_basis is not None else None
        )

    raw_track_statuses = raw.get("track_statuses")
    if raw_track_statuses is None:
        # Schema-v1 registries created before track-scoped gates remain valid
        # and retain their historical method-only behavior.
        track_statuses = {track: "passed" for track in _TRACKS}
    else:
        if not isinstance(raw_track_statuses, Mapping):
            raise CampaignError("E_SCIENTIFIC_GATE_SCHEMA", "track_statuses must be a mapping")
        missing_tracks = sorted(set(_TRACKS) - set(raw_track_statuses))
        extra_tracks = sorted(set(raw_track_statuses) - set(_TRACKS))
        if missing_tracks or extra_tracks:
            raise CampaignError(
                "E_SCIENTIFIC_GATE_SCHEMA",
                (
                    "track_statuses must contain exactly paper and standardized; "
                    f"missing={missing_tracks}, extra={extra_tracks}"
                ),
            )
        track_statuses = {}
        for track in _TRACKS:
            status = raw_track_statuses[track]
            if status not in _VALID_STATUSES:
                raise CampaignError(
                    "E_SCIENTIFIC_GATE_SCHEMA",
                    f"track_statuses.{track} must be pending, passed, or failed",
                )
            track_statuses[track] = str(status)

    raw_dependencies = raw.get("dependencies", {})
    if not isinstance(raw_dependencies, Mapping):
        raise CampaignError("E_SCIENTIFIC_GATE_SCHEMA", "dependencies must be a mapping")
    dependencies: dict[str, tuple[str, ...]] = {}
    for method_id, values in raw_dependencies.items():
        if method_id not in expected_methods:
            raise CampaignError(
                "E_SCIENTIFIC_GATE_SCHEMA", f"unknown dependency method: {method_id}"
            )
        items = _non_empty_strings(values, field=f"dependencies.{method_id}")
        unknown = sorted(set(items) - expected_methods)
        if unknown:
            raise CampaignError(
                "E_SCIENTIFIC_GATE_SCHEMA",
                f"dependencies.{method_id} contains unknown methods: {unknown}",
            )
        dependencies[str(method_id)] = tuple(items)

    prefixes = tuple(
        _non_empty_strings(
            raw.get("protected_campaign_prefixes", []), field="protected_campaign_prefixes"
        )
    )
    if schema_version == 1 and not prefixes:
        raise CampaignError(
            "E_SCIENTIFIC_GATE_SCHEMA", "protected_campaign_prefixes must not be empty"
        )
    raw_exemptions = raw.get("exempt_campaign_ids", [])
    exemptions = frozenset(_non_empty_strings(raw_exemptions, field="exempt_campaign_ids"))
    raw_claim_scopes = raw.get("claim_scopes")
    if schema_version == 1:
        claim_scopes = {"article10": frozenset(ARTICLE10_METHODS)}
    else:
        if not isinstance(raw_claim_scopes, Mapping):
            raise CampaignError("E_SCIENTIFIC_GATE_SCHEMA", "schema v2 requires claim_scopes")
        claim_scopes = {}
        for scope_id, value in raw_claim_scopes.items():
            methods = frozenset(_non_empty_strings(value, field=f"claim_scopes.{scope_id}"))
            if scope_id not in CLAIM_SCOPES or methods != CLAIM_SCOPES[scope_id]:
                raise CampaignError(
                    "E_SCIENTIFIC_GATE_SCHEMA",
                    f"claim_scopes.{scope_id} differs from the technical catalog",
                )
            claim_scopes[str(scope_id)] = methods
        if claim_scopes != CLAIM_SCOPES:
            raise CampaignError(
                "E_SCIENTIFIC_GATE_SCHEMA",
                "claim_scopes must contain article10 and historical-replacements",
            )
    return ScientificGateRegistry(
        registry_id=registry_id,
        method_statuses=statuses,
        method_conformity_bases=conformity_bases,
        track_statuses=track_statuses,
        dependencies=dependencies,
        claim_scopes=claim_scopes,
        protected_campaign_prefixes=prefixes,
        exempt_campaign_ids=exemptions,
    )


def evaluate_gate(
    registry: ScientificGateRegistry,
    *,
    campaign_id: str,
    track: str,
    method_id: str,
    claim_scope_id: str = "article10",
    campaign_stage: str = "production",
    claim_eligible: bool = True,
) -> GateDecision:
    blockers: list[str] = []
    scope_methods = registry.claim_scopes.get(claim_scope_id)
    if scope_methods is None:
        blockers.append(f"claim_scope_not_registered:{claim_scope_id}")
    elif method_id not in scope_methods:
        blockers.append(f"method_outside_claim_scope:{claim_scope_id}:{method_id}")
    if method_id not in registry.method_statuses:
        blockers.append(f"method_not_registered:{method_id}")
    elif claim_eligible and registry.status(method_id) != "passed":
        blockers.append(f"method_conformity:{method_id}={registry.status(method_id)}")

    # Dependencies describe an ordering constraint, not the method's own
    # conformity gate.  They therefore remain active for evidence-producing
    # canaries and every other named exemption.
    for dependency in registry.dependencies.get(method_id, ()):
        if registry.status(dependency) != "passed":
            blockers.append(f"dependency_conformity:{dependency}={registry.status(dependency)}")

    if campaign_stage not in {"diagnostic", "canary", "production"}:
        blockers.append(f"unsupported_stage:{campaign_stage}")
    if campaign_stage != "production" and claim_eligible:
        blockers.append(f"non_production_claim:{campaign_stage}")

    if track in _TRACKS and claim_eligible and registry.track_status(track) != "passed":
        blockers.append(f"track_status:{track}={registry.track_status(track)}")

    if track == "standardized":
        if claim_eligible:
            if claim_scope_id != "article10":
                blockers.append(f"standardized_scope:{claim_scope_id}")
            else:
                for required in registry.claim_scopes["article10"]:
                    if registry.status(required) != "passed":
                        blocker = f"standardized_requires:{required}={registry.status(required)}"
                        if blocker not in blockers:
                            blockers.append(blocker)
    elif track != "paper":
        blockers.append(f"unsupported_track:{track}")
    return GateDecision(
        not blockers,
        campaign_id,
        track,
        method_id,
        claim_scope_id,
        campaign_stage,
        claim_eligible,
        tuple(blockers),
    )


def guard_task(task: CampaignTask, registry_path: Path) -> GateDecision:
    registry = load_gate_registry(registry_path)
    actual_sha256 = sha256(registry_path.read_bytes()).hexdigest()
    if task.gate_policy_id != registry.registry_id or task.gate_policy_sha256 != actual_sha256:
        raise CampaignError(
            "E_SCIENTIFIC_GATE_SCHEMA",
            f"task {task.task_id} scientific gate policy identity differs",
        )
    decision = evaluate_gate(
        registry,
        campaign_id=task.campaign_id,
        track=task.track,
        method_id=task.method_id,
        claim_scope_id=task.claim_scope_id,
        campaign_stage=task.campaign_stage,
        claim_eligible=task.claim_eligible,
    )
    if not decision.allowed:
        raise CampaignError(
            "E_SCIENTIFIC_GATE_BLOCKED",
            f"task {task.task_id} is blocked by {', '.join(decision.blockers)}",
        )
    return decision


def discover_gate_registry(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise CampaignError(
                "E_SCIENTIFIC_GATE_SCHEMA", f"scientific gate registry not found: {path}"
            )
        return path
    candidate = repo_root.resolve() / "bench" / "campaigns" / "scientific-gates.yaml"
    if not candidate.is_file():
        raise CampaignError(
            "E_SCIENTIFIC_GATE_SCHEMA",
            f"scientific gate registry not found: {candidate}",
        )
    return candidate
