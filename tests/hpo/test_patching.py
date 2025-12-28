from __future__ import annotations

from modssc.hpo import deep_merge, flatten_patch


def test_deep_merge_is_non_mutating():
    base = {"a": {"b": 1}, "c": [1, 2], "d": {"e": {"f": 1}}}
    patch = {"a": {"c": 2}, "c": [3], "d": {"e": {"g": 2}}, "x": 9}

    merged = deep_merge(base, patch)

    assert merged == {
        "a": {"b": 1, "c": 2},
        "c": [3],
        "d": {"e": {"f": 1, "g": 2}},
        "x": 9,
    }
    assert base == {"a": {"b": 1}, "c": [1, 2], "d": {"e": {"f": 1}}}
    assert patch == {"a": {"c": 2}, "c": [3], "d": {"e": {"g": 2}}, "x": 9}

    merged["a"]["b"] = 99
    merged["c"].append(4)
    assert base["a"]["b"] == 1
    assert base["c"] == [1, 2]


def test_flatten_patch_builds_dotted_keys():
    patch = {
        "method": {"params": {"max_iter": 10, "nested": {"x": 1}}},
        "empty": {},
        "seed": 7,
    }
    flat = flatten_patch(patch)
    assert flat == {
        "empty": {},
        "method.params.max_iter": 10,
        "method.params.nested.x": 1,
        "seed": 7,
    }
