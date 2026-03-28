from __future__ import annotations

import os
import tempfile
from pathlib import Path


def resolve_relative_path(root: Path, relative: str, *, purpose: str) -> Path:
    """Resolve a manifest-relative path while keeping it inside ``root``."""
    root_resolved = root.expanduser().resolve(strict=False)
    rel_path = Path(relative)
    if rel_path.is_absolute():
        raise ValueError(f"{purpose} must be relative, got absolute path {relative!r}")

    candidate = (root_resolved / rel_path).resolve(strict=False)
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"{purpose} escapes its cache root: {relative!r}")
    return candidate


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
