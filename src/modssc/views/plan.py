from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from modssc.preprocess.plan import PreprocessPlan

from .errors import ViewsValidationError


@dataclass(frozen=True)
class ColumnSelectSpec:
    """How to select columns from a 2D feature matrix.

    This is used to generate *feature views* (e.g. classic Co-Training),
    where each view sees a different subset of the features.

    Notes
    -----
    - `mode="complement"` assumes the referenced view has already been resolved.
    - `fraction` is only used for `mode="random"`.
    """

    mode: Literal["all", "indices", "random", "complement"] = "all"
    indices: tuple[int, ...] = ()
    fraction: float = 0.5
    complement_of: str | None = None
    seed_offset: int = 0

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> ColumnSelectSpec:
        if not isinstance(obj, Mapping):
            raise ValueError("view.columns must be a mapping")
        unknown = set(obj) - {"mode", "indices", "fraction", "complement_of", "seed_offset"}
        if unknown:
            raise ValueError(f"Unknown keys in view.columns: {sorted(unknown)}")
        return cls(
            mode=str(obj.get("mode", "all")),
            indices=tuple(int(index) for index in (obj.get("indices") or ())),
            fraction=float(obj.get("fraction", 0.5)),
            complement_of=(str(obj["complement_of"]) if obj.get("complement_of") else None),
            seed_offset=int(obj.get("seed_offset", 0)),
        )

    def validate(self) -> None:
        if self.mode not in ("all", "indices", "random", "complement"):
            raise ViewsValidationError(f"Unknown ColumnSelectSpec.mode={self.mode!r}")

        if self.mode == "indices":
            if not self.indices:
                raise ViewsValidationError("ColumnSelectSpec(mode='indices') requires `indices`")
            if any(int(i) < 0 for i in self.indices):
                raise ViewsValidationError(
                    "ColumnSelectSpec.indices cannot contain negative values"
                )

        if self.mode == "random":
            f = float(self.fraction)
            if not (0.0 < f <= 1.0):
                raise ViewsValidationError("ColumnSelectSpec.fraction must be in (0, 1] for random")

        if self.mode == "complement" and not self.complement_of:
            raise ViewsValidationError(
                "ColumnSelectSpec(mode='complement') requires `complement_of`"
            )


@dataclass(frozen=True)
class ViewSpec:
    """A single view definition."""

    name: str
    preprocess: PreprocessPlan | None = None
    columns: ColumnSelectSpec | None = None
    meta: dict[str, Any] | None = None
    input_columns: ColumnSelectSpec | None = field(default=None, kw_only=True)

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> ViewSpec:
        if not isinstance(obj, Mapping):
            raise ValueError("Each view must be a mapping")
        unknown = set(obj) - {"name", "preprocess", "input_columns", "columns", "meta"}
        if unknown:
            raise ValueError(f"Unknown keys in view: {sorted(unknown)}")
        name = str(obj.get("name", ""))
        if not name:
            raise ValueError("Each view must define 'name'")
        meta = obj.get("meta")
        if meta is not None and not isinstance(meta, Mapping):
            raise ValueError("view.meta must be a mapping when provided")
        return cls(
            name=name,
            preprocess=(
                PreprocessPlan.from_dict(obj["preprocess"])
                if obj.get("preprocess") is not None
                else None
            ),
            input_columns=(
                ColumnSelectSpec.from_dict(obj["input_columns"])
                if obj.get("input_columns") is not None
                else None
            ),
            columns=(
                ColumnSelectSpec.from_dict(obj["columns"])
                if obj.get("columns") is not None
                else None
            ),
            meta=dict(meta) if meta else None,
        )

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ViewsValidationError("ViewSpec.name cannot be empty")
        if self.input_columns is not None:
            self.input_columns.validate()
        if self.columns is not None:
            self.columns.validate()
        if self.meta is not None and not isinstance(self.meta, dict):
            raise ViewsValidationError("ViewSpec.meta must be a dict when provided")


@dataclass(frozen=True)
class ViewsPlan:
    """A plan that generates multiple views from the same dataset."""

    views: tuple[ViewSpec, ...]

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> ViewsPlan:
        if not isinstance(obj, Mapping):
            raise ValueError("views plan must be a mapping")
        unknown = set(obj) - {"views"}
        if unknown:
            raise ValueError(f"Unknown keys in views plan: {sorted(unknown)}")
        views_raw = obj.get("views", [])
        if not isinstance(views_raw, list):
            raise ValueError("views plan views must be a list")
        plan = cls(views=tuple(ViewSpec.from_dict(view) for view in views_raw))
        plan.validate()
        return plan

    def preprocess_step_ids(self) -> tuple[str, ...]:
        """Return enabled preprocessing steps nested in view order."""

        return tuple(
            step_id
            for view in self.views
            if view.preprocess is not None
            for step_id in view.preprocess.enabled_step_ids()
        )

    def validate(self) -> None:
        if len(self.views) < 2:
            raise ViewsValidationError("ViewsPlan must contain at least 2 views")
        names = [v.name for v in self.views]
        if len(set(names)) != len(names):
            raise ViewsValidationError("View names must be unique")
        for v in self.views:
            v.validate()

        # Complement dependency must point to a previous view in the tuple
        seen: set[str] = set()
        for v in self.views:
            if v.input_columns is not None and v.input_columns.mode == "complement":
                target = str(v.input_columns.complement_of)
                if target not in seen:
                    raise ViewsValidationError(
                        f"View {v.name!r} uses input complement_of={target!r} but that "
                        "view wasn't resolved yet. Put the referenced view earlier in ViewsPlan.views."
                    )
            if v.columns is not None and v.columns.mode == "complement":
                target = str(v.columns.complement_of)
                if target not in seen:
                    raise ViewsValidationError(
                        f"View {v.name!r} uses complement_of={target!r} but that view wasn't resolved yet. "
                        "Put the referenced view earlier in ViewsPlan.views."
                    )
            seen.add(v.name)


def two_view_random_feature_split(
    *,
    preprocess: PreprocessPlan | None = None,
    fraction: float = 0.5,
    seed_offset: int = 0,
    name_a: str = "view_a",
    name_b: str = "view_b",
) -> ViewsPlan:
    """Convenience helper for classic 2-view feature split.

    The first view picks a random subset of columns, the second view is its complement.
    """

    a = ViewSpec(
        name=name_a,
        preprocess=preprocess,
        columns=ColumnSelectSpec(
            mode="random", fraction=float(fraction), seed_offset=int(seed_offset)
        ),
        meta={"role": "primary"},
    )
    b = ViewSpec(
        name=name_b,
        preprocess=preprocess,
        columns=ColumnSelectSpec(mode="complement", complement_of=name_a),
        meta={"role": "complement"},
    )
    plan = ViewsPlan(views=(a, b))
    plan.validate()
    return plan
