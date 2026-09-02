from __future__ import annotations

import importlib

import pytest

from modssc.data_augmentation import AugmentationPlan, StepConfig, parse_augmentation_plan
from modssc.data_augmentation.errors import DataAugmentationValidationError


def _assert_module_importable(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing.startswith("modssc"):
            raise
        pytest.skip(f"Optional dependency missing while importing {module_name}: {missing}")
    except Exception as exc:
        if exc.__class__.__name__ == "OptionalDependencyError" or 'pip install "modssc[' in str(
            exc
        ):
            pytest.skip(f"Optional dependency missing while importing {module_name}: {exc}")
        raise


def test_module_importable() -> None:
    _assert_module_importable("modssc.data_augmentation.plan")


def test_parse_augmentation_plan_accepts_yaml_and_python_step_ids() -> None:
    plan = parse_augmentation_plan(
        {
            "modality": "tabular",
            "description": "two native spellings",
            "steps": [
                {"id": "core.identity"},
                {"op_id": "tabular.gaussian_noise", "params": {"std": 0.25}},
            ],
        }
    )

    assert plan == AugmentationPlan(
        modality="tabular",
        description="two native spellings",
        steps=(
            StepConfig(op_id="core.identity", params={}),
            StepConfig(op_id="tabular.gaussian_noise", params={"std": 0.25}),
        ),
    )


def test_parse_augmentation_plan_resolves_or_preserves_native_plan() -> None:
    plan = AugmentationPlan(steps=(), description="native")
    resolved = parse_augmentation_plan(plan, modality="vision")

    assert resolved == AugmentationPlan(steps=(), modality="vision", description="native")
    assert parse_augmentation_plan(resolved, modality="vision") is resolved
    assert parse_augmentation_plan(resolved) is resolved


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ([], "must be a mapping"),
        ({"unknown": True}, "Unknown keys in augmentation plan"),
        ({"steps": {}}, "augmentation.steps must be a list"),
        ({"steps": ["bad"]}, "step must be a mapping"),
        ({"steps": [{"id": "core.identity", "unknown": True}]}, "Unknown keys"),
        (
            {"steps": [{"id": "core.identity", "op_id": "core.ensure_float32"}]},
            "conflicting",
        ),
        ({"steps": [{}]}, "must define 'id'"),
        ({"steps": [{"id": "core.identity", "params": []}]}, "params for op"),
    ],
)
def test_parse_augmentation_plan_rejects_malformed_declarations(value, match: str) -> None:
    with pytest.raises(DataAugmentationValidationError, match=match):
        parse_augmentation_plan(value)


def test_parse_augmentation_plan_rejects_modality_conflicts() -> None:
    with pytest.raises(DataAugmentationValidationError, match="conflicts"):
        parse_augmentation_plan({"modality": "text", "steps": []}, modality="vision")
    with pytest.raises(DataAugmentationValidationError, match="conflicts"):
        parse_augmentation_plan(
            AugmentationPlan(steps=(), modality="text"),
            modality="vision",
        )
