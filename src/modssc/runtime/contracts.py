"""Directional contracts for composing methods, inputs, and components.

The legacy capability facade is intentionally flat and remains useful for
early availability checks.  This module owns the stricter execution contract:
methods declare requirements, materialized inputs and bound components expose
provisions, and the runtime records whether every requirement was verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from modssc.capabilities import MethodCapabilities

VerificationStatus = Literal["verified", "declared", "unverified"]
ContractStatus = Literal["compatible", "incompatible", "unverified"]
InputConsumption = Literal["scientific", "alignment_only", "unused"]
ComponentKind = Literal["torch_model", "classifier"]
RelationKind = Literal[
    "same_rows",
    "same_backend",
    "same_device",
    "same_dtype_kind",
    "same_input_schema",
    "concat_compatible",
    "mix_compatible",
]
ComponentRelationKind = Literal[
    "distinct_objects",
    "disjoint_parameters",
    "same_architecture",
    "same_device",
]

_VERIFICATION_STATUSES = frozenset({"verified", "declared", "unverified"})
_CONSUMPTION_KINDS = frozenset({"scientific", "alignment_only", "unused"})
_COMPONENT_KINDS = frozenset({"torch_model", "classifier"})
_RELATION_KINDS = frozenset(
    {
        "same_rows",
        "same_backend",
        "same_device",
        "same_dtype_kind",
        "same_input_schema",
        "concat_compatible",
        "mix_compatible",
    }
)
_COMPONENT_RELATION_KINDS = frozenset(
    {"distinct_objects", "disjoint_parameters", "same_architecture", "same_device"}
)


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _names(
    values: frozenset[str] | set[str] | tuple[str, ...] | list[str] | None,
    *,
    field_name: str,
    allow_none: bool = True,
) -> frozenset[str] | None:
    if values is None:
        if allow_none:
            return None
        return frozenset()
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized = frozenset(_name(value, field_name=field_name) for value in values)
    if allow_none and not normalized:
        raise ValueError(f"{field_name} cannot be empty; use None for unrestricted")
    return normalized


def _ranks(values: frozenset[int] | set[int] | tuple[int, ...] | None) -> frozenset[int] | None:
    if values is None:
        return None
    normalized: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("ranks must contain non-negative integers")
        normalized.add(value)
    if not normalized:
        raise ValueError("ranks cannot be empty; use None for unrestricted")
    return frozenset(normalized)


@dataclass(frozen=True)
class ValueDescriptor:
    """Metadata-only description of one materialized value or container."""

    representation: str
    container_backends: frozenset[str] = frozenset()
    dtypes: frozenset[str] = frozenset()
    dtype_kinds: frozenset[str] = frozenset()
    devices: frozenset[str] = frozenset()
    rank: int | None = None
    shape: tuple[int | None, ...] | None = None
    rows: int | None = None
    schema: tuple[tuple[str, ValueDescriptor], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "representation",
            _name(self.representation, field_name="representation"),
        )
        for field_name in ("container_backends", "dtypes", "dtype_kinds", "devices"):
            normalized = _names(
                getattr(self, field_name),
                field_name=field_name,
                allow_none=False,
            )
            object.__setattr__(self, field_name, normalized)
        if self.rank is not None and (
            isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0
        ):
            raise ValueError("rank must be a non-negative integer or None")
        if self.rows is not None and (
            isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows < 0
        ):
            raise ValueError("rows must be a non-negative integer or None")
        if self.shape is not None:
            normalized_shape: list[int | None] = []
            for dimension in self.shape:
                if dimension is not None and (
                    isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                ):
                    raise ValueError("shape dimensions must be non-negative integers or None")
                normalized_shape.append(dimension)
            object.__setattr__(self, "shape", tuple(normalized_shape))
        normalized_schema: list[tuple[str, ValueDescriptor]] = []
        for key, descriptor in self.schema:
            normalized_key = _name(key, field_name="schema key")
            if not isinstance(descriptor, ValueDescriptor):
                raise TypeError("schema values must be ValueDescriptor instances")
            normalized_schema.append((normalized_key, descriptor))
        normalized_schema.sort(key=lambda item: item[0])
        object.__setattr__(self, "schema", tuple(normalized_schema))

    @property
    def numeric(self) -> bool | None:
        """Return whether every known leaf is numeric, or ``None`` if unknown."""

        if not self.dtype_kinds:
            return None
        return self.dtype_kinds <= frozenset({"bool", "integer", "float", "complex"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "container_backends": sorted(self.container_backends),
            "dtypes": sorted(self.dtypes),
            "dtype_kinds": sorted(self.dtype_kinds),
            "devices": sorted(self.devices),
            "rank": self.rank,
            "shape": None if self.shape is None else list(self.shape),
            "rows": self.rows,
            "schema": {key: descriptor.to_dict() for key, descriptor in self.schema},
        }


@dataclass(frozen=True)
class InputRoleRequirement:
    """Requirements imposed on one exact method-facing input role."""

    role: str
    representations: frozenset[str] | None = None
    container_backends: frozenset[str] | None = None
    dtype_kinds: frozenset[str] | None = None
    dtypes: frozenset[str] | None = None
    ranks: frozenset[int] | None = None
    numeric: bool | None = None
    optional: bool = False
    non_empty: bool = True
    consumption: InputConsumption = "scientific"
    model_input: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _name(self.role, field_name="role"))
        for field_name in (
            "representations",
            "container_backends",
            "dtype_kinds",
            "dtypes",
        ):
            object.__setattr__(
                self,
                field_name,
                _names(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "ranks", _ranks(self.ranks))
        if self.numeric is not None and not isinstance(self.numeric, bool):
            raise TypeError("numeric must be bool or None")
        if self.consumption not in _CONSUMPTION_KINDS:
            raise ValueError(f"consumption must be one of {sorted(_CONSUMPTION_KINDS)!r}")
        if self.model_input is not None:
            object.__setattr__(
                self,
                "model_input",
                _name(self.model_input, field_name="model_input"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "representations": (
                None if self.representations is None else sorted(self.representations)
            ),
            "container_backends": (
                None if self.container_backends is None else sorted(self.container_backends)
            ),
            "dtype_kinds": None if self.dtype_kinds is None else sorted(self.dtype_kinds),
            "dtypes": None if self.dtypes is None else sorted(self.dtypes),
            "ranks": None if self.ranks is None else sorted(self.ranks),
            "numeric": self.numeric,
            "optional": self.optional,
            "non_empty": self.non_empty,
            "consumption": self.consumption,
            "model_input": self.model_input,
        }


@dataclass(frozen=True)
class RoleRelation:
    """A relation that must hold between two or more input roles."""

    kind: RelationKind
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in _RELATION_KINDS:
            raise ValueError(f"kind must be one of {sorted(_RELATION_KINDS)!r}")
        normalized = tuple(_name(role, field_name="relation role") for role in self.roles)
        if len(normalized) < 2:
            raise ValueError("a role relation requires at least two roles")
        object.__setattr__(self, "roles", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "roles": list(self.roles)}


@dataclass(frozen=True)
class ModelContract:
    """Static or verified input/output contract for a bound model component."""

    outputs: frozenset[str]
    input_representations: frozenset[str] | None = None
    input_dtype_kinds: frozenset[str] | None = None
    input_ranks: frozenset[int] | None = None
    verification: VerificationStatus = "declared"
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outputs",
            _names(self.outputs, field_name="outputs", allow_none=False),
        )
        for field_name in ("input_representations", "input_dtype_kinds"):
            object.__setattr__(
                self,
                field_name,
                _names(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "input_ranks", _ranks(self.input_ranks))
        if self.verification not in _VERIFICATION_STATUSES:
            raise ValueError(f"verification must be one of {sorted(_VERIFICATION_STATUSES)!r}")
        if self.source is not None:
            object.__setattr__(self, "source", _name(self.source, field_name="source"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": sorted(self.outputs),
            "input_representations": (
                None if self.input_representations is None else sorted(self.input_representations)
            ),
            "input_dtype_kinds": (
                None if self.input_dtype_kinds is None else sorted(self.input_dtype_kinds)
            ),
            "input_ranks": None if self.input_ranks is None else sorted(self.input_ranks),
            "verification": self.verification,
            "source": self.source,
        }


@dataclass(frozen=True)
class ComponentRequirement:
    """Requirement imposed by a method on one bound component slot."""

    slot: str
    kind: ComponentKind
    outputs: frozenset[str] = frozenset()
    output_alternatives: tuple[frozenset[str], ...] = ()
    input_roles: tuple[str, ...] = ()
    requires_optimizer: bool = False
    requires_ema: bool = False
    requires_scheduler: bool = False
    scheduler_types: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _name(self.slot, field_name="slot"))
        if self.kind not in _COMPONENT_KINDS:
            raise ValueError(f"kind must be one of {sorted(_COMPONENT_KINDS)!r}")
        object.__setattr__(
            self,
            "outputs",
            _names(self.outputs, field_name="outputs", allow_none=False),
        )
        alternatives: list[frozenset[str]] = []
        for index, alternative in enumerate(self.output_alternatives):
            normalized = _names(
                alternative,
                field_name=f"output_alternatives[{index}]",
                allow_none=False,
            )
            if not normalized:
                raise ValueError("output alternatives cannot be empty")
            alternatives.append(normalized)
        object.__setattr__(
            self,
            "output_alternatives",
            tuple(sorted(set(alternatives), key=lambda value: tuple(sorted(value)))),
        )
        object.__setattr__(
            self,
            "input_roles",
            tuple(
                sorted(
                    {_name(role, field_name="component input role") for role in self.input_roles}
                )
            ),
        )
        scheduler_types = _names(
            self.scheduler_types,
            field_name="scheduler_types",
            allow_none=False,
        )
        object.__setattr__(self, "scheduler_types", scheduler_types)
        if scheduler_types:
            object.__setattr__(self, "requires_scheduler", True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "kind": self.kind,
            "outputs": sorted(self.outputs),
            "output_alternatives": [
                sorted(alternative) for alternative in self.output_alternatives
            ],
            "input_roles": list(self.input_roles),
            "requires_optimizer": self.requires_optimizer,
            "requires_ema": self.requires_ema,
            "requires_scheduler": self.requires_scheduler,
            "scheduler_types": sorted(self.scheduler_types),
        }


@dataclass(frozen=True)
class ComponentProvision:
    """Facts exposed by one concrete bound model or classifier."""

    slot: str
    kind: ComponentKind
    contract: ModelContract | None
    object_id: int | None = None
    parameter_ids: frozenset[int] = frozenset()
    ema_object_id: int | None = None
    ema_parameter_ids: frozenset[int] = frozenset()
    has_optimizer: bool = False
    has_ema: bool = False
    has_scheduler: bool = False
    scheduler_type: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _name(self.slot, field_name="slot"))
        if self.kind not in _COMPONENT_KINDS:
            raise ValueError(f"kind must be one of {sorted(_COMPONENT_KINDS)!r}")
        if self.contract is not None and not isinstance(self.contract, ModelContract):
            raise TypeError("contract must be ModelContract or None")
        if self.device is not None:
            object.__setattr__(self, "device", _name(self.device, field_name="device"))
        if self.scheduler_type is not None:
            object.__setattr__(
                self,
                "scheduler_type",
                _name(self.scheduler_type, field_name="scheduler_type"),
            )
            object.__setattr__(self, "has_scheduler", True)
        object.__setattr__(self, "parameter_ids", frozenset(self.parameter_ids))
        object.__setattr__(self, "ema_parameter_ids", frozenset(self.ema_parameter_ids))

    @property
    def verification(self) -> VerificationStatus:
        return "unverified" if self.contract is None else self.contract.verification

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "kind": self.kind,
            "contract": None if self.contract is None else self.contract.to_dict(),
            "verification": self.verification,
            "has_optimizer": self.has_optimizer,
            "has_ema": self.has_ema,
            "has_scheduler": self.has_scheduler,
            "scheduler_type": self.scheduler_type,
            "device": self.device,
            "parameter_count": len(self.parameter_ids),
            "ema_parameter_count": len(self.ema_parameter_ids),
        }


@dataclass(frozen=True)
class ComponentRelation:
    """A relation required between bound component slots."""

    kind: ComponentRelationKind
    slots: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in _COMPONENT_RELATION_KINDS:
            raise ValueError(f"kind must be one of {sorted(_COMPONENT_RELATION_KINDS)!r}")
        normalized = tuple(_name(slot, field_name="component slot") for slot in self.slots)
        if len(normalized) < 2:
            raise ValueError("a component relation requires at least two slots")
        object.__setattr__(self, "slots", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "slots": list(self.slots)}


@dataclass(frozen=True)
class MethodExecutionContract:
    """Resolved requirements for one method specification."""

    base: MethodCapabilities
    inputs: tuple[InputRoleRequirement, ...] = ()
    relations: tuple[RoleRelation, ...] = ()
    components: tuple[ComponentRequirement, ...] = ()
    component_relations: tuple[ComponentRelation, ...] = ()
    source: str = "fallback"

    def __post_init__(self) -> None:
        if not isinstance(self.base, MethodCapabilities):
            raise TypeError("base must be MethodCapabilities")
        for field_name, expected in (
            ("inputs", InputRoleRequirement),
            ("relations", RoleRelation),
            ("components", ComponentRequirement),
            ("component_relations", ComponentRelation),
        ):
            values = tuple(getattr(self, field_name))
            if not all(isinstance(value, expected) for value in values):
                raise TypeError(f"{field_name} must contain {expected.__name__} values")
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "source", _name(self.source, field_name="source"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.base.regime,
            "source": self.source,
            "inputs": [requirement.to_dict() for requirement in self.inputs],
            "relations": [relation.to_dict() for relation in self.relations],
            "components": [requirement.to_dict() for requirement in self.components],
            "component_relations": [relation.to_dict() for relation in self.component_relations],
        }


@dataclass(frozen=True)
class ContractIssue:
    """One stable incompatibility or unverifiable requirement."""

    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _name(self.code, field_name="code"))
        object.__setattr__(self, "message", _name(self.message, field_name="message"))

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ExecutionContractReport:
    """Final directional composition report."""

    method_id: str
    issues: tuple[ContractIssue, ...] = ()
    unverified: tuple[ContractIssue, ...] = ()
    input_provisions: tuple[tuple[str, ValueDescriptor], ...] = ()
    component_provisions: tuple[ComponentProvision, ...] = ()
    contract: MethodExecutionContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _name(self.method_id, field_name="method_id"))
        for field_name in ("issues", "unverified"):
            values = tuple(getattr(self, field_name))
            if not all(isinstance(value, ContractIssue) for value in values):
                raise TypeError(f"{field_name} must contain ContractIssue values")
            unique = {(value.code, value.message): value for value in values}
            object.__setattr__(
                self,
                field_name,
                tuple(unique[key] for key in sorted(unique)),
            )
        provisions: list[tuple[str, ValueDescriptor]] = []
        for role, descriptor in self.input_provisions:
            if not isinstance(descriptor, ValueDescriptor):
                raise TypeError("input provisions must contain ValueDescriptor values")
            provisions.append((_name(role, field_name="input role"), descriptor))
        provisions.sort(key=lambda item: item[0])
        object.__setattr__(self, "input_provisions", tuple(provisions))
        components = tuple(self.component_provisions)
        if not all(isinstance(value, ComponentProvision) for value in components):
            raise TypeError("component_provisions must contain ComponentProvision values")
        object.__setattr__(
            self,
            "component_provisions",
            tuple(sorted(components, key=lambda value: value.slot)),
        )

    @property
    def status(self) -> ContractStatus:
        if self.issues:
            return "incompatible"
        if self.unverified:
            return "unverified"
        return "compatible"

    @property
    def compatible(self) -> bool:
        return self.status == "compatible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
            "unverified": [issue.to_dict() for issue in self.unverified],
            "contract": None if self.contract is None else self.contract.to_dict(),
            "inputs": {role: descriptor.to_dict() for role, descriptor in self.input_provisions},
            "components": [provision.to_dict() for provision in self.component_provisions],
        }


class ExecutionContractError(ValueError):
    """Raised when a resolved execution contract cannot be proven."""

    def __init__(self, report: ExecutionContractReport) -> None:
        self.report = report
        details = [f"[{issue.code}] {issue.message}" for issue in report.issues]
        details.extend(f"[{issue.code}] {issue.message}" for issue in report.unverified)
        super().__init__(
            f"Execution contract for method {report.method_id!r} is {report.status}: "
            + "; ".join(details)
        )


__all__ = [
    "ComponentProvision",
    "ComponentRelation",
    "ComponentRequirement",
    "ContractIssue",
    "ContractStatus",
    "ExecutionContractError",
    "ExecutionContractReport",
    "InputConsumption",
    "InputRoleRequirement",
    "MethodExecutionContract",
    "ModelContract",
    "RelationKind",
    "RoleRelation",
    "ValueDescriptor",
    "VerificationStatus",
]
