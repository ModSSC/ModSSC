from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def stable_json_dumps(obj: Any) -> str:
    """Dump JSON with stable key ordering and compact separators.

    This is used to compute fingerprints that are independent of dict ordering.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(obj: Any, *, prefix: str = "") -> str:
    """Compute a stable SHA256 hex digest for any JSON-serializable object."""
    payload = stable_json_dumps(obj).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return f"{prefix}{h}" if prefix else h


def derive_seed(master_seed: int, *, step_id: str, step_index: int) -> int:
    """Derive a per-step seed from a master seed and step identity.

    Returned seed fits in uint32 for numpy's Generator.
    """
    raw = f"{master_seed}|{step_index}|{step_id}".encode()
    h = hashlib.sha256(raw).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def shallow_mapping(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Return a plain dict copy, converting nested mappings shallowly."""
    return {k: dict(v) if isinstance(v, Mapping) else v for k, v in obj.items()}
