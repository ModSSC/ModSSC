from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "modssc"
_PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
_EXTRAS = set(_PYPROJECT["project"]["optional-dependencies"].keys())
_REQUIRED_EXTRA_RE = re.compile(r'required_extra(?:["\']?\s*:\s*|=\s*)["\']([A-Za-z0-9_.-]+)["\']')


def test_required_extra_literals_exist_in_pyproject() -> None:
    missing: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for extra in _REQUIRED_EXTRA_RE.findall(text):
            if extra not in _EXTRAS:
                missing.add(extra)
    assert missing == set()
