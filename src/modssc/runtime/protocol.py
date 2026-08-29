"""Portable identities for effective scientific configurations.

The benchmark layer may parse a configuration from YAML, but checkpoint and
report compatibility are native runtime concerns.  This module therefore owns
the canonical protocol projection and hashing rules without knowing anything
about cards, schedulers, datasets, methods, or research articles.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from modssc.runtime.execution import RunIdentity
from modssc.runtime.software import software_sha256

_IDENTITY_SCHEMA_VERSION = 1

# These fields affect where or how a run is operated, but not the scientific
# protocol that a resumed method must continue.  Keep this list deliberately
# narrow: every other configuration value remains identity-bearing.
_OPERATIONAL_FIELDS: dict[str, frozenset[str]] = {
    "run": frozenset(
        {
            "name",
            "output_dir",
            "log_level",
            "fail_fast",
            "resume_policy",
            "checkpoint_dir",
            "artifact_root",
        }
    ),
    "dataset": frozenset({"cache_dir", "download"}),
    "preprocess": frozenset({"cache", "cache_dir"}),
    "graph": frozenset({"cache", "cache_dir"}),
    "views": frozenset({"cache", "cache_dir"}),
}


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the exact effective configuration serialized in a run report."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return _sha256(dict(config))


def protocol_identity_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical scientific subset of an effective configuration.

    Operational limits are removed after they have been applied: every value
    they changed elsewhere in the effective configuration remains part of the
    protocol identity.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    payload = copy.deepcopy(dict(config))
    payload.pop("limits", None)
    for section_name, excluded_fields in _OPERATIONAL_FIELDS.items():
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            continue
        filtered_section = dict(section)
        for field in excluded_fields:
            filtered_section.pop(field, None)
        payload[section_name] = filtered_section
    return payload


def protocol_sha256(config: Mapping[str, Any]) -> str:
    """Hash a portable, effective scientific protocol."""

    return _sha256(
        {
            "schema_version": _IDENTITY_SCHEMA_VERSION,
            "protocol": protocol_identity_payload(config),
        }
    )


def build_resume_identity(
    config: Mapping[str, Any],
    *,
    seed: int,
    runtime_versions: Mapping[str, Any],
) -> RunIdentity:
    """Build a checkpoint identity from effective protocol and software state."""

    return RunIdentity(
        config_sha256=protocol_sha256(config),
        seed=seed,
        code_sha256=software_sha256(runtime_versions),
    )


__all__ = [
    "build_resume_identity",
    "effective_config_sha256",
    "protocol_identity_payload",
    "protocol_sha256",
]
