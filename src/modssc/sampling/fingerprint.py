from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def stable_hash(obj: Mapping[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def derive_seed(master_seed: int, label: str) -> int:
    # stable across Python versions
    b = f"{int(master_seed)}:{label}".encode()
    h = hashlib.sha256(b).digest()
    # uint32 range
    return int.from_bytes(h[:4], "big", signed=False)
