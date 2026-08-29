from __future__ import annotations

import pytest

from modssc.views import ViewSpec, ViewsPlan


def test_view_orchestrator_parses_native_input_columns() -> None:
    plan = ViewsPlan.from_dict(
        {
            "views": [
                {"name": "page", "input_columns": {"mode": "indices", "indices": [0]}},
                {"name": "links", "input_columns": {"mode": "indices", "indices": [1]}},
            ]
        }
    )

    assert plan.views[0].input_columns is not None
    assert plan.views[0].input_columns.indices == (0,)
    assert plan.views[1].input_columns is not None
    assert plan.views[1].input_columns.indices == (1,)


def test_view_orchestrator_rejects_unknown_native_view_key() -> None:
    with pytest.raises(ValueError, match="Unknown keys"):
        ViewSpec.from_dict({"name": "page", "source_columns": {"mode": "all"}})
