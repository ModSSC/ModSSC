from __future__ import annotations

import subprocess
import sys


def test_import_has_no_side_effects(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import modssc"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_import_does_not_pull_heavy_deps_by_default(tmp_path):
    code = """
import sys
import modssc
for name in ("torch", "tensorflow", "sklearn"):
    assert name not in sys.modules, f"{name} imported at import time"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
