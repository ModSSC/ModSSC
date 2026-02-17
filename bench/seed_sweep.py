from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_SEEDED_SECTIONS = ("sampling", "preprocess", "views", "graph", "augmentation", "search")


def sweep_run_name(base_name: str, *, seed: int, index: int, total: int) -> str:
    _ = index, total
    return f"{base_name}-seed{int(seed)}"


def apply_global_seed(
    raw: Mapping[str, Any],
    *,
    seed: int,
    run_name: str | None = None,
) -> dict[str, Any]:
    out = deepcopy(dict(raw))
    run = out.get("run")
    if not isinstance(run, dict):
        run = {}
        out["run"] = run
    run.pop("seeds", None)
    run["seed"] = int(seed)
    if run_name is not None:
        run["name"] = str(run_name)

    for section in _SEEDED_SECTIONS:
        block = out.get(section)
        if isinstance(block, dict):
            block["seed"] = int(seed)

    return out
