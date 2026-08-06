from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_DEFAULT_SEEDED_SECTIONS = (
    "sampling",
    "preprocess",
    "views",
    "graph",
    "augmentation",
    "search",
)
_ALLOWED_SEEDED_SECTIONS = (*_DEFAULT_SEEDED_SECTIONS, "dataset")


def sweep_run_name(base_name: str, *, seed: int, index: int, total: int) -> str:
    _ = index, total
    return f"{base_name}-seed{int(seed)}"


def apply_global_seed(
    raw: Mapping[str, Any],
    *,
    seed: int,
    run_name: str | None = None,
    seeded_sections: tuple[str, ...] | list[str] | None = None,
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

    sections = (
        _DEFAULT_SEEDED_SECTIONS
        if seeded_sections is None
        else tuple(str(s) for s in seeded_sections)
    )
    unknown = sorted(set(sections) - set(_ALLOWED_SEEDED_SECTIONS))
    if unknown:
        raise ValueError(f"Unknown seeded sections: {unknown!r}")

    for section in sections:
        block = out.get(section)
        if isinstance(block, dict):
            if section == "dataset":
                options = block.get("options")
                if options is None:
                    options = {}
                    block["options"] = options
                if not isinstance(options, dict):
                    raise ValueError("dataset.options must be a mapping to apply its seed")
                options["seed"] = int(seed)
            else:
                block["seed"] = int(seed)

    return out
