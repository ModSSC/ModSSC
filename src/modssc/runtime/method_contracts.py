"""Resolve method-owned execution requirements.

Methods may provide an exact ``execution_contract`` classmethod.  Until every
method owns such a declaration, this module translates the legacy capability
facade and the native model-binding shape into one deterministic directional
contract.  The fallback deliberately does not invent provisions for classic
classifiers constructed inside ``fit``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

from modssc.capabilities import MethodCapabilities
from modssc.runtime.contracts import (
    ComponentRelation,
    ComponentRequirement,
    InputRoleRequirement,
    MethodExecutionContract,
    RoleRelation,
)

_HOOK_SOURCE = "method.execution_contract"
_FALLBACK_SOURCE = "capabilities+model_binding:fallback"
_MODEL_OUTPUT_NAMES = frozenset({"feat", "logits"})
_BUNDLE_FIELD_COUNTS = {
    "single": 1,
    "teacher_student": 2,
    "pair": 2,
    "pretrain_finetune": 2,
}
_BINDING_KINDS = frozenset({"none", *_BUNDLE_FIELD_COUNTS, "shared_heads"})


def _method_hook(method_class: type[Any]) -> Callable[..., Any] | None:
    """Return a declared classmethod hook, rejecting ambiguous descriptors."""

    try:
        descriptor = inspect.getattr_static(method_class, "execution_contract")
    except AttributeError:
        return None
    if not isinstance(descriptor, classmethod):
        raise TypeError("method execution_contract must be declared as a classmethod")
    hook = method_class.execution_contract
    if not callable(hook):  # pragma: no cover - guaranteed by classmethod in normal Python
        raise TypeError("method execution_contract must be callable")
    return hook


def _feature_requirement(role: str, capabilities: MethodCapabilities) -> InputRoleRequirement:
    return InputRoleRequirement(
        role=role,
        representations=capabilities.representations,
    )


def _target_requirement(role: str, capabilities: MethodCapabilities) -> InputRoleRequirement:
    class_ids_only = capabilities.target_kinds == frozenset({"class_ids"})
    return InputRoleRequirement(
        role=role,
        dtype_kinds=frozenset({"integer"}) if class_ids_only else None,
        ranks=frozenset({1}),
        numeric=True if class_ids_only else None,
    )


def _mask_requirement(role: str) -> InputRoleRequirement:
    return InputRoleRequirement(
        role=role,
        dtype_kinds=frozenset({"bool"}),
        ranks=frozenset({1}),
    )


def _graph_requirements() -> tuple[InputRoleRequirement, ...]:
    return (
        InputRoleRequirement(
            role="fit.graph.edge_index",
            representations=frozenset({"dense"}),
            dtype_kinds=frozenset({"integer"}),
            ranks=frozenset({2}),
            numeric=True,
        ),
        InputRoleRequirement(
            role="fit.graph.edge_weight",
            ranks=frozenset({1}),
            numeric=True,
            optional=True,
        ),
        InputRoleRequirement(
            role="fit.graph.n_nodes",
            dtype_kinds=frozenset({"integer"}),
            ranks=frozenset({0}),
            numeric=True,
            optional=True,
            consumption="alignment_only",
        ),
    )


def _relation(
    relations: list[RoleRelation],
    kind: str,
    roles: Iterable[str],
) -> None:
    normalized = tuple(roles)
    if len(normalized) >= 2:
        relations.append(RoleRelation(kind, normalized))  # type: ignore[arg-type]


def _inductive_inputs(
    capabilities: MethodCapabilities,
) -> tuple[tuple[InputRoleRequirement, ...], tuple[RoleRelation, ...]]:
    inputs = [
        _feature_requirement("fit.X_l", capabilities),
        _target_requirement("fit.y_l", capabilities),
    ]
    row_relations = [RoleRelation("same_rows", ("fit.X_l", "fit.y_l"))]
    feature_roles = ["fit.X_l"]
    unlabeled_roles: list[str] = []

    if capabilities.requires_unlabeled:
        inputs.append(_feature_requirement("fit.X_u", capabilities))
        feature_roles.append("fit.X_u")
        unlabeled_roles.append("fit.X_u")
    if capabilities.requires_weak_augmentation:
        inputs.append(_feature_requirement("fit.X_u_w", capabilities))
        feature_roles.append("fit.X_u_w")
        unlabeled_roles.append("fit.X_u_w")
    for index in range(capabilities.min_strong_augmentations):
        role = f"fit.X_u_s.{index}"
        inputs.append(_feature_requirement(role, capabilities))
        feature_roles.append(role)
        unlabeled_roles.append(role)

    _relation(row_relations, "same_rows", unlabeled_roles)
    if capabilities.requires_graph:
        inputs.extend(_graph_requirements())

    relations = row_relations
    _relation(relations, "same_backend", feature_roles)
    _relation(relations, "same_device", feature_roles)
    return tuple(inputs), tuple(relations)


def _transductive_inputs(
    capabilities: MethodCapabilities,
) -> tuple[tuple[InputRoleRequirement, ...], tuple[RoleRelation, ...]]:
    inputs = [
        _feature_requirement("fit.X", capabilities),
        _target_requirement("fit.y", capabilities),
        _mask_requirement("fit.masks.train_mask"),
    ]
    row_roles = ["fit.X", "fit.y", "fit.masks.train_mask"]

    if capabilities.requires_unlabeled:
        inputs.append(_mask_requirement("fit.masks.unlabeled_mask"))
        row_roles.append("fit.masks.unlabeled_mask")
    if capabilities.requires_graph:
        inputs.extend(_graph_requirements())

    return tuple(inputs), (RoleRelation("same_rows", tuple(row_roles)),)


def _binding_kind(model_binding: Any | None) -> str:
    if model_binding is None:
        return "none"
    kind = getattr(model_binding, "kind", None)
    if not isinstance(kind, str) or kind not in _BINDING_KINDS:
        raise ValueError(f"unsupported model-binding kind: {kind!r}")
    return kind


def _bundle_fields(model_binding: Any, *, kind: str) -> tuple[str, ...]:
    raw_fields = getattr(model_binding, "bundle_fields", ())
    if isinstance(raw_fields, str):
        raise TypeError("model-binding bundle_fields must be a collection of names")
    try:
        fields = tuple(raw_fields)
    except TypeError as exc:
        raise TypeError("model-binding bundle_fields must be iterable") from exc
    expected = _BUNDLE_FIELD_COUNTS[kind]
    if len(fields) != expected:
        raise ValueError(f"model-binding kind {kind!r} requires exactly {expected} bundle fields")
    if not all(isinstance(field, str) and field.strip() for field in fields):
        raise ValueError("model-binding bundle fields must be non-empty strings")
    return tuple(field.strip() for field in fields)


def _shared_slots(model_binding: Any) -> tuple[str, ...]:
    shared_field = getattr(model_binding, "shared_bundle_field", None)
    heads_field = getattr(model_binding, "head_bundles_field", None)
    head_count = getattr(model_binding, "head_count", None)
    if not isinstance(shared_field, str) or not shared_field.strip():
        raise ValueError("shared-head binding requires shared_bundle_field")
    if not isinstance(heads_field, str) or not heads_field.strip():
        raise ValueError("shared-head binding requires head_bundles_field")
    if isinstance(head_count, bool) or not isinstance(head_count, int) or head_count <= 0:
        raise ValueError("shared-head binding requires a positive head_count")
    shared = shared_field.strip()
    heads = heads_field.strip()
    return (shared, *(f"{heads}[{index}]" for index in range(head_count)))


def _model_feature_roles(
    inputs: Iterable[InputRoleRequirement],
) -> tuple[str, ...]:
    """Return scientific feature roles, excluding labels and graph metadata."""

    return tuple(
        requirement.role
        for requirement in inputs
        if requirement.role == "fit.X"
        or requirement.role.startswith("fit.X_")
        or ".X_" in requirement.role
    )


def _components(
    method_class: type[Any],
    capabilities: MethodCapabilities,
    model_binding: Any | None,
    *,
    input_roles: tuple[str, ...],
) -> tuple[tuple[ComponentRequirement, ...], tuple[ComponentRelation, ...]]:
    kind = _binding_kind(model_binding)
    if kind == "none":
        return (), ()
    assert model_binding is not None  # narrowed by the binding kind

    legacy_model_outputs = capabilities.required_classifier_outputs & _MODEL_OUTPUT_NAMES
    default_ema = bool(getattr(getattr(method_class, "info", None), "default_model_ema", False))
    relations: list[ComponentRelation] = []

    if kind == "shared_heads":
        slots = _shared_slots(model_binding)
        components = [
            ComponentRequirement(
                slot=slots[0],
                kind="torch_model",
                input_roles=input_roles,
                requires_optimizer=True,
            )
        ]
        components.extend(
            ComponentRequirement(
                slot=slot,
                kind="torch_model",
                outputs=frozenset({"logits"}),
                requires_optimizer=True,
            )
            for slot in slots[1:]
        )
        relations.extend(
            (
                ComponentRelation("distinct_objects", slots),
                ComponentRelation("disjoint_parameters", slots),
            )
        )
        return tuple(components), tuple(relations)

    fields = _bundle_fields(model_binding, kind=kind)
    if kind == "pretrain_finetune":
        output_sets = (frozenset(), legacy_model_outputs)
    else:
        output_sets = tuple(legacy_model_outputs for _field in fields)
    components = tuple(
        ComponentRequirement(
            slot=field,
            kind="torch_model",
            outputs=outputs,
            input_roles=input_roles,
            requires_optimizer=True,
            requires_ema=default_ema,
        )
        for field, outputs in zip(fields, output_sets, strict=True)
    )
    if kind in {"pair", "teacher_student"}:
        relations.extend(
            (
                ComponentRelation("distinct_objects", fields),
                ComponentRelation("disjoint_parameters", fields),
            )
        )
    return components, tuple(relations)


def fallback_method_execution_contract(
    method_class: type[Any],
    capabilities: MethodCapabilities,
    model_binding: Any | None = None,
) -> MethodExecutionContract:
    """Build the pure capability/binding fallback without consulting method hooks."""

    if not isinstance(method_class, type):
        raise TypeError("method_class must be a class")
    if not isinstance(capabilities, MethodCapabilities):
        raise TypeError("capabilities must be MethodCapabilities")

    if capabilities.regime == "inductive":
        inputs, relations = _inductive_inputs(capabilities)
    else:
        inputs, relations = _transductive_inputs(capabilities)
    components, component_relations = _components(
        method_class,
        capabilities,
        model_binding,
        input_roles=_model_feature_roles(inputs),
    )
    return MethodExecutionContract(
        base=capabilities,
        inputs=inputs,
        relations=relations,
        components=components,
        component_relations=component_relations,
        source=_FALLBACK_SOURCE,
    )


def with_inductive_input_roles(
    contract: MethodExecutionContract,
    *,
    feature_roles: Iterable[str],
    optional_feature_roles: Iterable[str] = (),
    allow_empty_feature_roles: Iterable[str] = (),
    row_groups: Iterable[Iterable[str]] = (),
) -> MethodExecutionContract:
    """Replace fallback inductive roles with one method's exact feature routing."""

    if not isinstance(contract, MethodExecutionContract):
        raise TypeError("contract must be MethodExecutionContract")
    if contract.base.regime != "inductive":
        raise ValueError("with_inductive_input_roles requires an inductive contract")

    normalized_roles = tuple(str(role).strip() for role in feature_roles)
    if any(not role for role in normalized_roles):
        raise ValueError("feature roles must be non-empty strings")
    if len(set(normalized_roles)) != len(normalized_roles):
        raise ValueError("feature roles must be unique")
    optional = frozenset(str(role).strip() for role in optional_feature_roles)
    allow_empty = frozenset(str(role).strip() for role in allow_empty_feature_roles)
    unknown = (optional | allow_empty) - frozenset(normalized_roles)
    if unknown:
        raise ValueError(f"feature role options reference unknown roles: {sorted(unknown)!r}")

    inputs: list[InputRoleRequirement] = []
    target_added = False
    for role in normalized_roles:
        inputs.append(
            replace(
                _feature_requirement(role, contract.base),
                optional=role in optional,
                non_empty=role not in allow_empty,
            )
        )
        if role == "fit.X_l":
            inputs.append(_target_requirement("fit.y_l", contract.base))
            target_added = True
    if not target_added:
        inputs.insert(0, _target_requirement("fit.y_l", contract.base))

    declared_roles = frozenset(requirement.role for requirement in inputs)
    relations: list[RoleRelation] = []
    for raw_group in row_groups:
        group = tuple(str(role).strip() for role in raw_group)
        missing = frozenset(group) - declared_roles
        if missing:
            raise ValueError(f"row group references undeclared roles: {sorted(missing)!r}")
        _relation(relations, "same_rows", group)
    all_roles = tuple(requirement.role for requirement in inputs)
    _relation(relations, "same_backend", all_roles)
    _relation(relations, "same_device", all_roles)
    return replace(contract, inputs=tuple(inputs), relations=tuple(relations))


def _invoke_method_hook(
    hook: Callable[..., Any],
    spec: Any,
    capabilities: MethodCapabilities,
    model_binding: Any | None,
) -> Any:
    """Prefer the native three-argument API while retaining one-argument hooks."""

    signature = inspect.signature(hook)
    try:
        signature.bind(spec, capabilities, model_binding)
    except TypeError:
        try:
            signature.bind(spec)
        except TypeError as exc:
            raise TypeError(
                "method execution_contract must accept either "
                "(spec) or (spec, capabilities, model_binding)"
            ) from exc
        return hook(spec)
    return hook(spec, capabilities, model_binding)


def resolve_method_execution_contract(
    method_class: type[Any],
    spec: Any,
    capabilities: MethodCapabilities,
    model_binding: Any | None = None,
) -> MethodExecutionContract:
    """Resolve an exact hook contract or a deterministic compatibility fallback."""

    if not isinstance(method_class, type):
        raise TypeError("method_class must be a class")
    if not isinstance(capabilities, MethodCapabilities):
        raise TypeError("capabilities must be MethodCapabilities")

    hook = _method_hook(method_class)
    if hook is not None:
        resolved = _invoke_method_hook(hook, spec, capabilities, model_binding)
        if not isinstance(resolved, MethodExecutionContract):
            raise TypeError("method execution_contract must return MethodExecutionContract")
        return replace(resolved, source=_HOOK_SOURCE)

    return fallback_method_execution_contract(method_class, capabilities, model_binding)


__all__ = [
    "fallback_method_execution_contract",
    "resolve_method_execution_contract",
    "with_inductive_input_roles",
]
