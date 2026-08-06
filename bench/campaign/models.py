from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from bench.partition_selection_schema import (
    DCL_PARTITION_SELECTION_KIND,
    PARTITION_SELECTION_DIGEST_FIELDS,
    PARTITION_SELECTION_TASK_FIELDS,
)

from .errors import CampaignError

_SAMPLING_COMPONENTS = ("partition", "split", "labeling", "imbalance")


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CampaignError("E_CAMPAIGN_MANIFEST_SCHEMA", f"{key} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_SCHEMA", f"{key} must be null or a non-empty string"
        )
    return value


def _sampling_component_seeds(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("sampling_component_seeds")
    if not isinstance(raw, Mapping) or set(raw) != set(_SAMPLING_COMPONENTS):
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_SCHEMA",
            "sampling_component_seeds must contain partition, split, labeling, and imbalance",
        )
    resolved: dict[str, int] = {}
    for component in _SAMPLING_COMPONENTS:
        value = raw.get(component)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA",
                f"sampling_component_seeds.{component} must be a non-negative integer",
            )
        resolved[component] = int(value)
    return resolved


def _partition_selection(payload: dict[str, Any], *, required: bool) -> dict[str, Any] | None:
    raw = payload.get("partition_selection")
    if raw is None:
        if required:
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA",
                "schema v3 tasks require partition_selection",
            )
        return None
    if not required:
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_SCHEMA",
            "partition_selection requires task schema_version 3",
        )
    if not isinstance(raw, Mapping) or set(raw) != PARTITION_SELECTION_TASK_FIELDS:
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_SCHEMA",
            "partition_selection must contain exactly the frozen selection and replay fields",
        )
    if raw.get("kind") != DCL_PARTITION_SELECTION_KIND:
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_SCHEMA",
            f"partition_selection.kind must equal {DCL_PARTITION_SELECTION_KIND!r}",
        )
    rank = raw.get("selection_rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_SCHEMA",
            "partition_selection.selection_rank must be a positive integer",
        )
    for key in PARTITION_SELECTION_TASK_FIELDS - {"selection_rank"}:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA",
                f"partition_selection.{key} must be a non-empty string",
            )
    for key in PARTITION_SELECTION_DIGEST_FIELDS:
        value = str(raw[key])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA",
                f"partition_selection.{key} must be a lowercase SHA-256 digest",
            )
    for key in ("selection_path", "replay_path"):
        path = str(raw[key])
        if path.startswith("/") or ".." in path.split("/"):
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA",
                f"partition_selection.{key} must be a contained repository-relative path",
            )
    return dict(raw)


@dataclass(frozen=True)
class CampaignTask:
    schema_version: int
    task_index: int
    task_id: str
    campaign_id: str
    track: str
    protocol_id: str | None
    config_path: str
    source_config_sha256: str
    method_profile: str
    label_budget: str
    required_seed_count: int
    seed: int
    data_seed: int
    split_seed: int
    sampling_component_seeds: dict[str, int] | None
    model_seed: int
    seeded_sections: tuple[str, ...] | None
    method_id: str
    method_kind: str
    dataset_id: str
    modality: str | None
    regime: str | None
    resource_profile: str
    assigned_site: str
    expected_git_sha: str
    expected_git_diff_sha256: str | None
    environment_lock_sha256: str
    dataset_lock_sha256: str | None
    expected_dataset_fingerprint: str | None
    expected_dataset_content_sha256: str | None
    dataset_request_sha256: str
    split_request_sha256: str
    expected_split_fingerprint: str | None
    partition_selection: dict[str, Any] | None
    fidelity_status: str | None
    output_relpath: str
    row_sha256: str
    # Scientific governance is deliberately independent from the human-facing
    # campaign name.  These fields were added in manifest schema v4; defaults
    # keep historical v1-v3 manifests readable without silently changing their
    # recorded identity.
    claim_scope_id: str = "legacy-unscoped"
    campaign_stage: str = "legacy"
    claim_eligible: bool = False
    gate_policy_id: str = "legacy-unpinned"
    gate_policy_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.schema_version == 1:
            payload.pop("sampling_component_seeds", None)
        if self.schema_version in {1, 2}:
            payload.pop("partition_selection", None)
        if self.schema_version in {1, 2, 3}:
            for field in (
                "claim_scope_id",
                "campaign_stage",
                "claim_eligible",
                "gate_policy_id",
                "gate_policy_sha256",
            ):
                payload.pop(field, None)
        if self.seeded_sections is not None:
            payload["seeded_sections"] = list(self.seeded_sections)
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CampaignTask:
        if not isinstance(raw, dict):
            raise CampaignError("E_CAMPAIGN_MANIFEST_SCHEMA", "task row must be a mapping")
        schema_version = raw.get("schema_version")
        if schema_version not in {1, 2, 3, 4}:
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA", "task schema_version must equal 1, 2, 3, or 4"
            )
        task_index = raw.get("task_index")
        seed = raw.get("seed")
        required_seed_count = raw.get("required_seed_count")
        if isinstance(task_index, bool) or not isinstance(task_index, int) or task_index < 0:
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA", "task_index must be a non-negative integer"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CampaignError("E_CAMPAIGN_MANIFEST_SCHEMA", "seed must be an integer")
        if (
            isinstance(required_seed_count, bool)
            or not isinstance(required_seed_count, int)
            or required_seed_count <= 0
        ):
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA",
                "required_seed_count must be a positive integer",
            )
        derived_seeds: dict[str, int] = {}
        for key in ("data_seed", "split_seed", "model_seed"):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise CampaignError("E_CAMPAIGN_MANIFEST_SCHEMA", f"{key} must be an integer")
            derived_seeds[key] = int(value)
        sampling_component_seeds = None if schema_version == 1 else _sampling_component_seeds(raw)
        partition_selection = _partition_selection(
            raw,
            required=schema_version == 3
            or (schema_version == 4 and raw.get("partition_selection") is not None),
        )
        if (
            sampling_component_seeds is not None
            and derived_seeds["split_seed"] != sampling_component_seeds["split"]
        ):
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA",
                "split_seed must equal sampling_component_seeds.split",
            )
        sections_raw = raw.get("seeded_sections")
        if sections_raw is None:
            sections = None
        elif isinstance(sections_raw, list) and all(
            isinstance(item, str) and item for item in sections_raw
        ):
            sections = tuple(sections_raw)
        else:
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_SCHEMA",
                "seeded_sections must be null or a list of non-empty strings",
            )

        if schema_version < 4:
            # Governance did not form part of the signed identity of schemas
            # v1-v3.  Ignore stray forward fields rather than allowing them to
            # alter an otherwise valid legacy row.
            claim_scope_id = "legacy-unscoped"
            campaign_stage = "legacy"
            claim_eligible = False
            gate_policy_id = "legacy-unpinned"
            gate_policy_sha256 = None
        else:
            claim_scope_id = raw.get("claim_scope_id")
            campaign_stage = raw.get("campaign_stage")
            claim_eligible = raw.get("claim_eligible")
            gate_policy_id = raw.get("gate_policy_id")
            gate_policy_sha256 = raw.get("gate_policy_sha256")
        if schema_version == 4:
            for field, value in (
                ("claim_scope_id", claim_scope_id),
                ("campaign_stage", campaign_stage),
                ("gate_policy_id", gate_policy_id),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise CampaignError(
                        "E_CAMPAIGN_MANIFEST_SCHEMA",
                        f"{field} must be a non-empty string",
                    )
            if campaign_stage not in {"diagnostic", "canary", "production"}:
                raise CampaignError(
                    "E_CAMPAIGN_MANIFEST_SCHEMA",
                    "campaign_stage must be diagnostic, canary, or production",
                )
            if not isinstance(claim_eligible, bool):
                raise CampaignError(
                    "E_CAMPAIGN_MANIFEST_SCHEMA", "claim_eligible must be a boolean"
                )
            if (
                not isinstance(gate_policy_sha256, str)
                or len(gate_policy_sha256) != 64
                or any(character not in "0123456789abcdef" for character in gate_policy_sha256)
            ):
                raise CampaignError(
                    "E_CAMPAIGN_MANIFEST_SCHEMA",
                    "gate_policy_sha256 must be a lowercase SHA-256 digest",
                )

        return cls(
            schema_version=int(schema_version),
            task_index=task_index,
            task_id=_required_string(raw, "task_id"),
            campaign_id=_required_string(raw, "campaign_id"),
            track=_required_string(raw, "track"),
            protocol_id=_optional_string(raw, "protocol_id"),
            config_path=_required_string(raw, "config_path"),
            source_config_sha256=_required_string(raw, "source_config_sha256"),
            method_profile=_required_string(raw, "method_profile"),
            label_budget=_required_string(raw, "label_budget"),
            required_seed_count=required_seed_count,
            seed=seed,
            data_seed=derived_seeds["data_seed"],
            split_seed=derived_seeds["split_seed"],
            sampling_component_seeds=sampling_component_seeds,
            model_seed=derived_seeds["model_seed"],
            seeded_sections=sections,
            method_id=_required_string(raw, "method_id"),
            method_kind=_required_string(raw, "method_kind"),
            dataset_id=_required_string(raw, "dataset_id"),
            modality=_optional_string(raw, "modality"),
            regime=_optional_string(raw, "regime"),
            resource_profile=_required_string(raw, "resource_profile"),
            assigned_site=_required_string(raw, "assigned_site"),
            expected_git_sha=_required_string(raw, "expected_git_sha"),
            expected_git_diff_sha256=_optional_string(raw, "expected_git_diff_sha256"),
            environment_lock_sha256=_required_string(raw, "environment_lock_sha256"),
            dataset_lock_sha256=_optional_string(raw, "dataset_lock_sha256"),
            expected_dataset_fingerprint=_optional_string(raw, "expected_dataset_fingerprint"),
            expected_dataset_content_sha256=_optional_string(
                raw, "expected_dataset_content_sha256"
            ),
            dataset_request_sha256=_required_string(raw, "dataset_request_sha256"),
            split_request_sha256=_required_string(raw, "split_request_sha256"),
            expected_split_fingerprint=_optional_string(raw, "expected_split_fingerprint"),
            partition_selection=partition_selection,
            fidelity_status=_optional_string(raw, "fidelity_status"),
            output_relpath=_required_string(raw, "output_relpath"),
            row_sha256=_required_string(raw, "row_sha256"),
            claim_scope_id=str(claim_scope_id),
            campaign_stage=str(campaign_stage),
            claim_eligible=bool(claim_eligible),
            gate_policy_id=str(gate_policy_id),
            gate_policy_sha256=(
                str(gate_policy_sha256) if gate_policy_sha256 is not None else None
            ),
        )


@dataclass(frozen=True)
class GeneratedCampaign:
    campaign_id: str
    output_dir: str
    manifest_path: str
    meta_path: str
    task_count: int
    manifest_sha256: str
    counts_by_profile: dict[str, int]


@dataclass(frozen=True)
class TaskExecutionResult:
    task_id: str
    status: str
    result_dir: str | None
    attempt_dir: str | None
    skipped: bool = False


@dataclass(frozen=True)
class ReconcileReport:
    campaign_id: str
    status: str
    task_count: int
    counts: dict[str, int]
    report_path: str
    retry_count: int
    retry_campaign_path: str | None
    continuation_count: int = 0
    continuation_campaign_path: str | None = None
