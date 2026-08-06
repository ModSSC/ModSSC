from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"


def _nav_paths(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value.endswith(".md") else set()
    if isinstance(value, list):
        return set().union(*(_nav_paths(item) for item in value), set())
    if isinstance(value, dict):
        return set().union(*(_nav_paths(item) for item in value.values()), set())
    return set()


def test_all_public_documentation_pages_are_navigable() -> None:
    config = yaml.safe_load((REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    navigation = _nav_paths(config["nav"])
    pages = {
        path.relative_to(DOCS_ROOT).as_posix()
        for path in DOCS_ROOT.rglob("*.md")
        if "article_code" not in path.relative_to(DOCS_ROOT).parts
    }

    assert navigation == pages
