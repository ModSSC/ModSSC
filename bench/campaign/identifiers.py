from __future__ import annotations

import re

from .errors import CampaignError

_SAFE_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def validate_safe_identifier(value: object, *, field: str, code: str) -> str:
    """Validate an identifier used in generated paths and shell wrappers."""

    if not isinstance(value, str) or ".." in value or _SAFE_IDENTIFIER_RE.fullmatch(value) is None:
        raise CampaignError(
            code,
            f"{field} must match [A-Za-z0-9][A-Za-z0-9._-]* and must not contain '..'",
        )
    return value
