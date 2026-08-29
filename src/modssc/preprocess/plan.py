from __future__ import annotations

from collections.abc import Iterable, Mapping
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

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> PreprocessPlan:
        """Parse a strict serialized preprocessing plan."""

        if not isinstance(obj, Mapping):
            raise ValueError("preprocess plan must be a mapping")
        unknown = set(obj) - {"output_key", "steps"}
        if unknown:
            raise ValueError(f"Unknown keys in preprocess plan: {sorted(unknown)}")
        steps_raw = obj.get("steps", [])
        if not isinstance(steps_raw, list):
            raise ValueError("'steps' must be a sequence")

        steps: list[StepConfig] = []
        allowed = {"id", "step_id", "params", "modalities", "requires_fields", "enabled"}
        for index, item in enumerate(steps_raw):
            if not isinstance(item, Mapping):
                raise ValueError("Each step must be a mapping")
            unknown_step = set(item) - allowed
            if unknown_step:
                raise ValueError(
                    f"Unknown keys in preprocess plan steps[{index}]: {sorted(unknown_step)}"
                )
            if "id" in item and "step_id" in item and str(item["id"]) != str(item["step_id"]):
                raise ValueError(
                    f"preprocess plan steps[{index}] has conflicting 'id' and 'step_id' values"
                )
            step_id = str(item.get("id") or item.get("step_id") or "")
            if not step_id:
                raise ValueError("Each step must define 'id'")
            params = item.get("params", {}) or {}
            if not isinstance(params, Mapping):
                raise ValueError(f"params for {step_id!r} must be a mapping")
            steps.append(
                StepConfig(
                    step_id=step_id,
                    params=dict(params),
                    modalities=tuple(str(value) for value in (item.get("modalities") or ())),
                    requires_fields=tuple(
                        str(value) for value in (item.get("requires_fields") or ())
                    ),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        return cls(
            steps=tuple(steps),
            output_key=str(obj.get("output_key", "features.X")),
        )

    def enabled_step_ids(self) -> tuple[str, ...]:
        """Return registry identifiers for enabled steps in execution order."""

        return tuple(step.step_id for step in self.steps if step.enabled)

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

    return PreprocessPlan.from_dict(data)


def dump_plan(plan: PreprocessPlan, path: str | Path) -> None:
    p = Path(path)
    p.write_text(yaml.safe_dump(plan.to_dict(), sort_keys=False))


def steps_require_fit_indices(step_ids: Iterable[str]) -> bool:
    """Return whether any enabled native preprocessing step is fittable."""

    from modssc.preprocess.registry import step_info

    return any(step_info(step_id).get("kind") == "fittable" for step_id in step_ids)


def steps_with_runtime_role(step_ids: Iterable[str], *, role: str) -> tuple[str, ...]:
    """Select enabled native steps declaring a runtime reporting role."""

    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be a non-empty string")

    from modssc.preprocess.registry import default_step_registry

    registry = default_step_registry()
    return tuple(step_id for step_id in step_ids if role in registry.spec(step_id).runtime_roles)
