from __future__ import annotations

import math

import pytest

from bench.utils.io import atomic_write_json, write_json


@pytest.mark.parametrize("writer", [write_json, atomic_write_json])
def test_json_writers_reject_non_standard_non_finite_numbers(writer, tmp_path) -> None:
    path = tmp_path / "result.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        writer(path, {"metric": math.nan})

    if path.exists():
        assert "NaN" not in path.read_text(encoding="utf-8")
