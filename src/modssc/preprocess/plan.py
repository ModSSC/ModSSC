from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from modssc.preprocess.fingerprint import fingerprint


@dataclass(frozen=True)
class StepConfig:
    """A single step configuration in a preprocessing plan."""

    step_id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    modalities: tuple[str, ...] = ()
    requires_fields: tuple[str, ...] = ()
    enabled: bool = True


@dataclass(frozen=True)
class PreprocessPlan:
    """A preprocessing plan.

    A plan is independent from a dataset. Conditional logic is applied during resolution.
    """

    steps: tuple[StepConfig, ...]
    output_key: str = "features.X"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_key": self.output_key,
            "steps": [
                {
                    "id": s.step_id,
                    "params": dict(s.params),
                    "modalities": list(s.modalities),
                    "requires_fields": list(s.requires_fields),
                    "enabled": bool(s.enabled),
                }
                for s in self.steps
            ],
        }

    def fingerprint(self) -> str:
        return fingerprint(self.to_dict(), prefix="plan:")


def load_plan(path: str | Path) -> PreprocessPlan:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, Mapping):
        raise ValueError("Plan file must contain a mapping at the root")

    output_key = str(data.get("output_key", "features.X"))
    steps_raw = data.get("steps", [])
    if not isinstance(steps_raw, Sequence):
        raise ValueError("'steps' must be a sequence")

    steps: list[StepConfig] = []
    for item in steps_raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each step must be a mapping")
        step_id = str(item.get("id") or item.get("step_id"))
        if not step_id:
            raise ValueError("Each step must define 'id'")
        params = item.get("params", {}) or {}
        if not isinstance(params, Mapping):
            raise ValueError(f"params for {step_id!r} must be a mapping")

        modalities = tuple(str(m) for m in (item.get("modalities") or ()))
        requires_fields = tuple(str(k) for k in (item.get("requires_fields") or ()))
        enabled = bool(item.get("enabled", True))
        steps.append(
            StepConfig(
                step_id=step_id,
                params=dict(params),
                modalities=modalities,
                requires_fields=requires_fields,
                enabled=enabled,
            )
        )

    return PreprocessPlan(steps=tuple(steps), output_key=output_key)


def dump_plan(plan: PreprocessPlan, path: str | Path) -> None:
    p = Path(path)
    p.write_text(yaml.safe_dump(plan.to_dict(), sort_keys=False))
