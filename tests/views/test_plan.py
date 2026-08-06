from __future__ import annotations

import pytest

from modssc.views import ColumnSelectSpec, ViewSpec, ViewsPlan
from modssc.views.errors import ViewsValidationError


def test_viewsplan_requires_two_views() -> None:
    with pytest.raises(ViewsValidationError):
        ViewsPlan(views=(ViewSpec(name="a"),)).validate()


def test_viewsplan_unique_names() -> None:
    plan = ViewsPlan(views=(ViewSpec(name="a"), ViewSpec(name="a")))
    with pytest.raises(ViewsValidationError):
        plan.validate()


def test_complement_must_reference_previous_view() -> None:
    plan = ViewsPlan(
        views=(
            ViewSpec(name="b", columns=ColumnSelectSpec(mode="complement", complement_of="a")),
            ViewSpec(name="a", columns=ColumnSelectSpec(mode="random", fraction=0.5)),
        )
    )
    with pytest.raises(ViewsValidationError):
        plan.validate()


def test_column_select_spec_validation_errors() -> None:
    with pytest.raises(ViewsValidationError, match="Unknown ColumnSelectSpec.mode"):
        ColumnSelectSpec(mode="bad").validate()

    with pytest.raises(ViewsValidationError, match="requires `indices`"):
        ColumnSelectSpec(mode="indices").validate()

    with pytest.raises(ViewsValidationError, match="cannot contain negative"):
        ColumnSelectSpec(mode="indices", indices=(-1,)).validate()

    with pytest.raises(ViewsValidationError, match="fraction must be in"):
        ColumnSelectSpec(mode="random", fraction=0.0).validate()

    with pytest.raises(ViewsValidationError, match="requires `complement_of`"):
        ColumnSelectSpec(mode="complement").validate()


def test_view_spec_validation_errors() -> None:
    with pytest.raises(ViewsValidationError, match="name cannot be empty"):
        ViewSpec(name=" ").validate()

    with pytest.raises(ViewsValidationError, match="meta must be a dict"):
        ViewSpec(name="a", meta="bad").validate()

    ViewSpec(
        name="ok",
        input_columns=ColumnSelectSpec(mode="indices", indices=(0,)),
        columns=ColumnSelectSpec(mode="indices", indices=(0,)),
    ).validate()


def test_input_complement_must_reference_previous_view() -> None:
    plan = ViewsPlan(
        views=(
            ViewSpec(
                name="b",
                input_columns=ColumnSelectSpec(mode="complement", complement_of="a"),
            ),
            ViewSpec(name="a", input_columns=ColumnSelectSpec(mode="random", fraction=0.5)),
        )
    )
    with pytest.raises(ViewsValidationError, match="input complement_of"):
        plan.validate()


def test_view_spec_preserves_historical_positional_constructor() -> None:
    columns = ColumnSelectSpec(mode="indices", indices=(1, 2))
    metadata = {"role": "historical"}

    spec = ViewSpec("view", None, columns, metadata)

    assert spec.columns is columns
    assert spec.meta is metadata
    assert spec.input_columns is None
    with pytest.raises(TypeError):
        ViewSpec("view", None, columns, metadata, columns)  # type: ignore[misc]
