from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from modssc.preprocess.catalog import BUILTIN_STEPS
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.types import StepSpec


def _load_object(import_path: str) -> Any:
    """Load an object from 'module:qualname'."""
    if ":" not in import_path:
        raise ValueError(f"Invalid import path: {import_path!r}")
    module_name, qualname = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass
class StepRegistry:
    specs: dict[str, StepSpec]

    @classmethod
    def builtin(cls) -> StepRegistry:
        return cls(specs={s.step_id: s for s in BUILTIN_STEPS})

    def available(self) -> list[str]:
        return sorted(self.specs.keys())

    def spec(self, step_id: str) -> StepSpec:
        try:
            return self.specs[step_id]
        except KeyError as e:
            raise PreprocessValidationError(f"Unknown step id: {step_id!r}") from e

    def instantiate(self, step_id: str, *, params: dict[str, Any]) -> Any:
        spec = self.spec(step_id)
        cls_obj = _load_object(spec.import_path)
        return cls_obj(**params)


_DEFAULT = StepRegistry.builtin()


def default_step_registry() -> StepRegistry:
    return _DEFAULT


def available_steps() -> list[str]:
    return _DEFAULT.available()


def step_info(step_id: str) -> dict[str, Any]:
    spec = _DEFAULT.spec(step_id)
    return {
        "id": spec.step_id,
        "import_path": spec.import_path,
        "kind": spec.kind,
        "required_extra": spec.required_extra,
        "modalities": list(spec.modalities),
        "consumes": list(spec.consumes),
        "produces": list(spec.produces),
        "description": spec.description,
    }
