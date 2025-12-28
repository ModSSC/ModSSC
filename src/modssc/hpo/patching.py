from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def flatten_patch(patch: dict[str, Any]) -> dict[str, object]:
    out: dict[str, object] = {}

    def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        for key in sorted(node.keys()):
            value = node[key]
            next_prefix = prefix + (str(key),)
            if isinstance(value, Mapping) and value:
                _walk(value, next_prefix)
            else:
                out[".".join(next_prefix)] = value

    _walk(patch, ())
    return out


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = copy.deepcopy(value)
    return merged
