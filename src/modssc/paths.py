from __future__ import annotations

from pathlib import Path

_PREFERRED_LOCAL_CACHE_ROOT = "modssc_cache"
_LEGACY_LOCAL_CACHE_ROOT = "cache"


def find_repo_root(start: str | Path | None = None) -> Path | None:
    current = Path(start) if start is not None else Path.cwd()
    current = current.expanduser().resolve()
    if current.is_file():
        current = current.parent

    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def default_local_cache_root(start: str | Path | None = None) -> Path | None:
    root = find_repo_root(start=start)
    if root is None:
        return None

    preferred = root / _PREFERRED_LOCAL_CACHE_ROOT
    if preferred.exists():
        return preferred

    return root / _LEGACY_LOCAL_CACHE_ROOT


def default_local_cache_subdir(name: str, *, start: str | Path | None = None) -> Path | None:
    root = default_local_cache_root(start=start)
    if root is None:
        return None
    return root / name
