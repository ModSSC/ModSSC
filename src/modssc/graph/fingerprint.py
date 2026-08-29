from __future__ import annotations

import hashlib
import json
from dataclasses import is_dataclass
from typing import Any

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    """Convert objects to JSON-serializable structures.

    This is intentionally conservative: it supports primitives, dict/list/tuple,
    dataclasses (via asdict), and numpy scalars.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        # avoid dataclasses.asdict recursion for safety (explicit via __dict__)
        return _to_jsonable(obj.__dict__)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON-serializable")


def canonical_json(data: Any) -> str:
    """Stable JSON string used to build fingerprints."""
    payload = _to_jsonable(data)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_dict(d: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(d).encode("utf-8"))


def fingerprint_spec(spec: Any) -> str:
    """Fingerprint a spec dataclass or a dict."""
    if isinstance(spec, dict):
        return fingerprint_dict(spec)
    if is_dataclass(spec):
        return fingerprint_dict(spec.__dict__)
    raise TypeError("spec must be a dataclass or dict")


def fingerprint_array(arr: Any, *, max_bytes: int = 2_000_000) -> str:
    """Fingerprint the full logical content of a dense or sparse array.

    ``max_bytes`` is retained for API compatibility and now controls only the
    streaming chunk size. No content is sampled or omitted.
    """
    chunk_size = max(1, int(max_bytes))

    def update_array(digest: Any, value: Any) -> None:
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays cannot be fingerprinted reproducibly")
        dtype = array.dtype.str.encode("ascii")
        shape = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        digest.update(len(dtype).to_bytes(8, "big"))
        digest.update(dtype)
        digest.update(len(shape).to_bytes(8, "big"))
        digest.update(shape)
        if array.nbytes == 0:
            return
        raw = memoryview(array).cast("B")
        for offset in range(0, len(raw), chunk_size):
            digest.update(raw[offset : offset + chunk_size])

    # scipy sparse support without importing scipy at module import
    if hasattr(arr, "tocoo") and hasattr(arr, "data") and hasattr(arr, "indices"):
        coo = arr.tocoo()
        h = hashlib.sha256()
        h.update(b"modssc:sparse-array:v2")
        update_array(h, np.asarray(coo.row, dtype=np.int64))
        update_array(h, np.asarray(coo.col, dtype=np.int64))
        update_array(h, np.asarray(coo.data))
        shape = json.dumps(list(coo.shape), separators=(",", ":")).encode("ascii")
        h.update(len(shape).to_bytes(8, "big"))
        h.update(shape)
        return h.hexdigest()

    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(b"modssc:dense-array:v2")
    update_array(h, a)
    return h.hexdigest()
