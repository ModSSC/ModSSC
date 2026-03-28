from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
_IMPORT_RE = re.compile(r"^from modssc[\w.]* import .+$")
_REPO_LINK_RE = re.compile(
    r"https://github\.com/ModSSC/ModSSC/(?:blob|tree)/main/([A-Za-z0-9_./-]+)"
)


def test_documented_python_imports_resolve() -> None:
    failures: list[str] = []
    for path in DOCS_ROOT.rglob("*.md"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not _IMPORT_RE.match(stripped):
                continue
            try:
                exec(stripped, {}, {})
            except Exception as exc:  # pragma: no cover - failure path only
                failures.append(f"{path}:{lineno}: {exc}")
    assert failures == []


def test_repo_links_in_docs_point_to_existing_paths() -> None:
    missing: list[str] = []
    for path in DOCS_ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for match in _REPO_LINK_RE.finditer(text):
            target = REPO_ROOT / match.group(1)
            if not target.exists():
                missing.append(f"{path}: {match.group(1)}")
    assert missing == []
