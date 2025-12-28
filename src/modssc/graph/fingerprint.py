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
    """Fingerprint a dense or sparse array deterministically.

    This is intended as a fallback when no dataset-level fingerprint is available.
    For large arrays, it hashes only a prefix and a suffix of the raw bytes.

    Notes
    -----
    This is not a cryptographic commitment of the entire array when it is huge.
    For reproducible runs, you should pass dataset_fingerprint from data_loader whenever possible.
    """
    # scipy sparse support without importing scipy at module import
    if hasattr(arr, "tocoo") and hasattr(arr, "data") and hasattr(arr, "indices"):
        # likely sparse matrix
        coo = arr.tocoo()
        pieces = [
            str(getattr(coo, "shape", None)).encode("utf-8"),
            str(getattr(coo, "dtype", None)).encode("utf-8"),
            np.asarray(coo.row, dtype=np.int64).tobytes()[:max_bytes],
            np.asarray(coo.col, dtype=np.int64).tobytes()[:max_bytes],
            np.asarray(coo.data).tobytes()[:max_bytes],
        ]
        h = hashlib.sha256()
        for p in pieces:
            h.update(p)
        return h.hexdigest()

    a = np.asarray(arr)
    meta = f"shape={a.shape};dtype={a.dtype};".encode()
    raw = np.ascontiguousarray(a).view(np.uint8)
    n = int(raw.size)

    h = hashlib.sha256()
    h.update(meta)

    if n <= max_bytes:
        h.update(raw.tobytes())
        return h.hexdigest()

    # prefix + suffix to reduce cost while keeping determinism
    h.update(raw[:max_bytes].tobytes())
    h.update(raw[-max_bytes:].tobytes())
    return h.hexdigest()
