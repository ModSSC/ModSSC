from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"


def test_notebooks_parse_and_have_a_title_cell() -> None:
    for path in NOTEBOOKS_ROOT.glob("*.ipynb"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] >= 4
        assert isinstance(payload["cells"], list)
        assert payload["cells"]

        first = payload["cells"][0]
        assert first["cell_type"] == "markdown"
        text = "".join(first.get("source", []))
        assert text.startswith("# ")
