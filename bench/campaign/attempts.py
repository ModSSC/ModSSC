"""Authenticated, immutable campaign-attempt records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from bench.utils.hashing import hash_any

from .errors import CampaignError
from .identifiers import validate_safe_identifier
from .models import CampaignTask

ATTEMPT_SCHEMA_VERSION = 2
_STATUSES = frozenset({"failed", "continuation"})
_FAILURE_POLICIES = {
    "deterministic": (False, False),
    "infrastructure": (True, False),
    "resource_oom": (False, True),
    "resource_timeout": (False, True),
}
AUTHORIZATION_EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AttemptRecord:
    payload: dict[str, Any]
    record_sha256: str
    finished_at_utc: datetime

    @property
    def attempt_id(self) -> str:
        return str(self.payload["attempt_id"])


def seal_attempt_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["schema_version"] = ATTEMPT_SCHEMA_VERSION
    record.pop("record_sha256", None)
    record["record_sha256"] = hash_any(record)
    return record


@dataclass(frozen=True)
class AuthorizationEvent:
    payload: dict[str, Any]
    record_sha256: str
    observed_at_utc: datetime

    @property
    def event_id(self) -> str:
        return str(self.payload["event_id"])


def seal_authorization_event(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["schema_version"] = AUTHORIZATION_EVENT_SCHEMA_VERSION
    record.pop("record_sha256", None)
    record["record_sha256"] = hash_any(record)
    return record


def _aware_datetime(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise CampaignError("E_CAMPAIGN_ATTEMPT_INVALID", f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CampaignError("E_CAMPAIGN_ATTEMPT_INVALID", f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID", f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignError("E_CAMPAIGN_ATTEMPT_INVALID", f"{field} must be a non-empty string")
    return value


def _validate_portable_reference(value: Any, *, field: str) -> str:
    reference = _string(value, field=field)
    if reference.startswith("/") or ".." in reference.split("/"):
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID", f"{field} must be a contained logical reference"
        )
    return reference


def validate_attempt_record(
    raw: Any,
    *,
    task: CampaignTask,
    directory_name: str | None = None,
) -> AttemptRecord:
    if not isinstance(raw, Mapping):
        raise CampaignError("E_CAMPAIGN_ATTEMPT_INVALID", "attempt record must be an object")
    payload = dict(raw)
    if payload.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID",
            f"attempt record schema_version must equal {ATTEMPT_SCHEMA_VERSION}",
        )
    expected = payload.get("record_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "record_sha256"}
    if not isinstance(expected, str) or expected != hash_any(unsigned):
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID", "attempt record digest is missing or invalid"
        )
    attempt_id = validate_safe_identifier(
        payload.get("attempt_id"),
        field="attempt_id",
        code="E_CAMPAIGN_ATTEMPT_INVALID",
    )
    if directory_name is not None and directory_name != attempt_id:
        raise CampaignError("E_CAMPAIGN_ATTEMPT_INVALID", "attempt directory and attempt_id differ")
    if payload.get("task_id") != task.task_id or payload.get("row_sha256") != task.row_sha256:
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID", "attempt task identity differs from the manifest"
        )
    status = payload.get("status")
    if status not in _STATUSES:
        raise CampaignError("E_CAMPAIGN_ATTEMPT_INVALID", "attempt status is invalid")
    if status == "continuation" and payload.get("event_class") != "planned_continuation":
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID", "continuation attempt event class is invalid"
        )
    if "work_dir" in payload or "continue_path" in payload:
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID",
            "attempt records must not contain machine-specific work or continuation paths",
        )
    finished_at_utc = _aware_datetime(payload.get("finished_at"), field="attempt finished_at")
    site_id = payload.get("site_id")
    validate_safe_identifier(
        site_id,
        field="site_id",
        code="E_CAMPAIGN_ATTEMPT_INVALID",
    )
    retryable = payload.get("retryable")
    resource_change_required = payload.get("resource_change_required")
    if not isinstance(retryable, bool) or not isinstance(resource_change_required, bool):
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID",
            "attempt retryable and resource_change_required must be booleans",
        )
    if status == "failed":
        failure_class = payload.get("failure_class")
        if failure_class not in _FAILURE_POLICIES:
            raise CampaignError(
                "E_CAMPAIGN_ATTEMPT_INVALID", "failed attempt failure_class is invalid"
            )
        expected_policy = _FAILURE_POLICIES[str(failure_class)]
        if (retryable, resource_change_required) != expected_policy:
            raise CampaignError(
                "E_CAMPAIGN_ATTEMPT_INVALID",
                "failed attempt retry policy differs from its failure class",
            )
        for field in ("error_type", "error", "traceback"):
            value = payload.get(field)
            if not isinstance(value, str):
                raise CampaignError(
                    "E_CAMPAIGN_ATTEMPT_INVALID", f"failed attempt {field} must be a string"
                )
        phase = payload.get("failure_phase")
        external_event_id = payload.get("external_event_id")
        if not (
            (isinstance(phase, str) and phase.strip())
            or (isinstance(external_event_id, str) and external_event_id.strip())
        ):
            raise CampaignError(
                "E_CAMPAIGN_ATTEMPT_INVALID",
                "failed attempt needs a failure phase or external event identity",
            )
    else:
        if payload.get("failure_class") is not None or retryable or resource_change_required:
            raise CampaignError(
                "E_CAMPAIGN_ATTEMPT_INVALID",
                "continuation attempt must not carry failure or retry semantics",
            )
        signal_number = payload.get("signal_number")
        if (
            isinstance(signal_number, bool)
            or not isinstance(signal_number, int)
            or signal_number <= 0
        ):
            raise CampaignError(
                "E_CAMPAIGN_ATTEMPT_INVALID", "continuation signal_number must be positive"
            )
        _digest(payload.get("checkpoint_payload_sha256"), field="checkpoint_payload_sha256")
        _digest(payload.get("checkpoint_manifest_sha256"), field="checkpoint_manifest_sha256")
        _validate_portable_reference(
            payload.get("checkpoint_reference"), field="checkpoint_reference"
        )
    return AttemptRecord(
        payload=payload,
        record_sha256=expected,
        finished_at_utc=finished_at_utc,
    )


def validate_authorization_event(
    raw: Any,
    *,
    task: CampaignTask,
    directory_name: str | None = None,
) -> AuthorizationEvent:
    if not isinstance(raw, Mapping):
        raise CampaignError("E_CAMPAIGN_ATTEMPT_INVALID", "authorization event must be an object")
    payload = dict(raw)
    if payload.get("schema_version") != AUTHORIZATION_EVENT_SCHEMA_VERSION:
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID",
            f"authorization event schema_version must equal {AUTHORIZATION_EVENT_SCHEMA_VERSION}",
        )
    expected = payload.get("record_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "record_sha256"}
    if not isinstance(expected, str) or expected != hash_any(unsigned):
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID", "authorization event digest is missing or invalid"
        )
    event_id = validate_safe_identifier(
        payload.get("event_id"),
        field="event_id",
        code="E_CAMPAIGN_ATTEMPT_INVALID",
    )
    if directory_name is not None and directory_name != event_id:
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID", "authorization event directory and event_id differ"
        )
    if payload.get("task_id") != task.task_id or payload.get("row_sha256") != task.row_sha256:
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID", "authorization event task identity differs"
        )
    if payload.get("event_class") != "authorization_expired":
        raise CampaignError("E_CAMPAIGN_ATTEMPT_INVALID", "authorization event class is invalid")
    validate_safe_identifier(
        payload.get("site_id"), field="site_id", code="E_CAMPAIGN_ATTEMPT_INVALID"
    )
    _digest(payload.get("preflight_report_sha256"), field="preflight_report_sha256")
    expires_at = _aware_datetime(payload.get("expires_at"), field="authorization expires_at")
    observed_at = _aware_datetime(payload.get("observed_at"), field="authorization observed_at")
    if observed_at < expires_at:
        raise CampaignError(
            "E_CAMPAIGN_ATTEMPT_INVALID",
            "authorization expiration event was observed before expiry",
        )
    return AuthorizationEvent(
        payload=payload,
        record_sha256=expected,
        observed_at_utc=observed_at,
    )


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "AUTHORIZATION_EVENT_SCHEMA_VERSION",
    "AttemptRecord",
    "AuthorizationEvent",
    "seal_authorization_event",
    "seal_attempt_record",
    "validate_authorization_event",
    "validate_attempt_record",
]
