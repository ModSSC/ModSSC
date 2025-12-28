from __future__ import annotations

from dataclasses import dataclass
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

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ViewsValidationError("ViewSpec.name cannot be empty")
        if self.columns is not None:
            self.columns.validate()
        if self.meta is not None and not isinstance(self.meta, dict):
            raise ViewsValidationError("ViewSpec.meta must be a dict when provided")


@dataclass(frozen=True)
class ViewsPlan:
    """A plan that generates multiple views from the same dataset."""

    views: tuple[ViewSpec, ...]

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
