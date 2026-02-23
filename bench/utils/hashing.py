from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(obj: Mapping[str, Any]) -> str:
    blob = stable_json_dumps(obj).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def hash_any(obj: Any) -> str:
    blob = stable_json_dumps(obj).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def derive_seed(master_seed: int, label: str) -> int:
    b = f"{int(master_seed)}:{label}".encode()
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:4], "big", signed=False)
