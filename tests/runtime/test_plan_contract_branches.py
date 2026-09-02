from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from modssc.preprocess.plan import (
    PreprocessPlan,
    StepConfig,
    steps_require_fit_indices,
)
from modssc.preprocess.scope import resolve_fit_indices
from modssc.views.errors import ViewsValidationError
from modssc.views.plan import (
    ColumnSelectSpec,
    ViewSpec,
    ViewsPlan,
    two_view_random_feature_split,
)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "must be a mapping"),
        ({"unknown": True}, "Unknown keys"),
        ({"steps": [{"id": "x", "unknown": True}]}, "Unknown keys"),
        ({"steps": [{"id": "a", "step_id": "b"}]}, "conflicting"),
    ],
)
def test_preprocess_plan_remaining_strict_failures(value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PreprocessPlan.from_dict(value)


def test_preprocess_plan_helpers_cover_enabled_fingerprint_and_fit_registry() -> None:
    plan = PreprocessPlan(
        steps=(
            StepConfig("tabular.standard_scaler"),
            StepConfig("core.to_numpy", enabled=False),
        )
    )
    assert plan.enabled_step_ids() == ("tabular.standard_scaler",)
    assert plan.fingerprint().startswith("plan:")
    assert steps_require_fit_indices(plan.enabled_step_ids())
    assert not steps_require_fit_indices(["core.to_numpy"])


class _GraphSampling:
    masks = {"train": np.array([True, False])}
    indices: dict[str, np.ndarray] = {}

    @staticmethod
    def is_graph() -> bool:
        return True


class _IndexSampling:
    masks: dict[str, np.ndarray] = {}
    indices = {"train": np.array([0], dtype=np.int64)}

    @staticmethod
    def is_graph() -> bool:
        return False


def test_fit_scope_rejects_unknown_graph_and_index_scopes() -> None:
    dataset = SimpleNamespace()
    with pytest.raises(ValueError, match="graph sampling"):
        resolve_fit_indices(
            dataset=dataset,
            sampling=_GraphSampling(),  # type: ignore[arg-type]
            fit_on="other",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Unsupported fit_on"):
        resolve_fit_indices(
            dataset=dataset,
            sampling=_IndexSampling(),  # type: ignore[arg-type]
            fit_on="other",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "must be a mapping"),
        ({"unknown": True}, "Unknown keys"),
    ],
)
def test_column_select_from_dict_strict_failures(value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ColumnSelectSpec.from_dict(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "must be a mapping"),
        ({"unknown": True}, "Unknown keys"),
        ({}, "define 'name'"),
        ({"name": "a", "meta": []}, "meta must be a mapping"),
    ],
)
def test_view_spec_from_dict_strict_failures(value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ViewSpec.from_dict(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "must be a mapping"),
        ({"unknown": True}, "Unknown keys"),
        ({"views": {}}, "views must be a list"),
    ],
)
def test_views_plan_from_dict_strict_failures(value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ViewsPlan.from_dict(value)


def test_view_plan_parsing_helpers_and_nested_preprocess_ids(tmp_path: Path) -> None:
    del tmp_path
    plan = ViewsPlan.from_dict(
        {
            "views": [
                {
                    "name": "a",
                    "preprocess": {
                        "steps": [
                            {"id": "to_numpy"},
                            {"id": "ensure_2d", "enabled": False},
                        ]
                    },
                    "input_columns": {"mode": "indices", "indices": [0]},
                    "columns": {"mode": "random", "fraction": 1.0},
                    "meta": {"role": "first"},
                },
                {
                    "name": "b",
                    "input_columns": {"mode": "complement", "complement_of": "a"},
                    "columns": {"mode": "complement", "complement_of": "a"},
                },
            ]
        }
    )
    assert plan.preprocess_step_ids() == ("to_numpy",)
    assert plan.views[0].meta == {"role": "first"}

    split = two_view_random_feature_split(fraction=0.25, seed_offset=3)
    assert [view.name for view in split.views] == ["view_a", "view_b"]
    assert split.views[0].columns == ColumnSelectSpec(mode="random", fraction=0.25, seed_offset=3)


def test_views_plan_validates_input_and_output_complement_order() -> None:
    valid_input = ViewsPlan(
        views=(
            ViewSpec(name="a"),
            ViewSpec(
                name="b",
                input_columns=ColumnSelectSpec(mode="complement", complement_of="a"),
            ),
        )
    )
    valid_input.validate()

    with pytest.raises(ViewsValidationError, match="input complement_of"):
        ViewsPlan(
            views=(
                ViewSpec(
                    name="a",
                    input_columns=ColumnSelectSpec(mode="complement", complement_of="b"),
                ),
                ViewSpec(name="b"),
            )
        ).validate()
    with pytest.raises(ViewsValidationError, match="uses complement_of"):
        ViewsPlan(
            views=(
                ViewSpec(
                    name="a",
                    columns=ColumnSelectSpec(mode="complement", complement_of="b"),
                ),
                ViewSpec(name="b"),
            )
        ).validate()
