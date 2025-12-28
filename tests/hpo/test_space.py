from __future__ import annotations

import pytest

from modssc.hpo import HpoError, Space


def test_space_init_requires_leaves():
    with pytest.raises(HpoError, match="space must define at least one leaf"):
        Space([])


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (1, "space must be a mapping"),
        ({}, "space cannot be empty"),
        ({"a": 1}, "space leaves must be lists or dist specs"),
        ({"dist": "uniform", "low": 0, "high": 1}, "space root must be a mapping"),
        ({"a": []}, "space leaves must be non-empty lists"),
        ({"a": {}}, "space cannot contain empty mappings"),
        ({1: [1]}, "space keys must be non-empty strings"),
        (
            {"a": {"dist": "uniform", "low": 0, "high": 1, "extra": 2}},
            "Unknown keys in dist spec",
        ),
        ({"a": {"dist": "choice", "values": "bad"}}, "values must be a list"),
        ({"a": {"dist": "nope"}}, "Unknown dist"),
    ],
)
def test_space_rejects_invalid_specs(payload, match):
    with pytest.raises(HpoError, match=match):
        Space.from_dict(payload)  # type: ignore[arg-type]


def test_grid_order_and_count():
    space = Space.from_dict(
        {
            "b": [1, 2],
            "a": {
                "y": [10, 20],
                "x": [100, 200],
            },
        }
    )
    trials = list(space.iter_grid())
    assert len(trials) == 8
    assert trials[0].params == {"a.x": 100, "a.y": 10, "b": 1}
    assert trials[1].params == {"a.x": 100, "a.y": 10, "b": 2}
    assert trials[2].params == {"a.x": 100, "a.y": 20, "b": 1}
    assert trials[-1].params == {"a.x": 200, "a.y": 20, "b": 2}


def test_grid_rejects_non_discrete_dist():
    space = Space.from_dict({"a": {"dist": "uniform", "low": 0.0, "high": 1.0}})
    with pytest.raises(HpoError, match="grid search requires list or choice distributions"):
        list(space.iter_grid())
