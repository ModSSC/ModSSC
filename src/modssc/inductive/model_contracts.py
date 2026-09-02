"""Static composition of method requirements with bound model components.

The functions in this module deliberately do not call a model.  Native bundle
builders attach a static :class:`~modssc.runtime.contracts.ModelContract`, while
external and pre-bound bundles remain unverifiable unless their provider
attaches the same explicit contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from modssc.runtime.contracts import (
    ComponentProvision,
    ComponentRelation,
    ComponentRequirement,
    ContractIssue,
    ModelContract,
    ValueDescriptor,
)

from .model_binding import ModelBindingSpec


def _declared_bound_slots(
    spec: Any,
    binding: ModelBindingSpec,
) -> tuple[tuple[str, Any], ...]:
    if binding.kind == "none":
        return ()
    if binding.kind != "shared_heads":
        return tuple(
            (slot, getattr(spec, slot))
            for slot in binding.bundle_fields
            if hasattr(spec, slot) and getattr(spec, slot) is not None
        )

    resolved: list[tuple[str, Any]] = []
    shared_slot = binding.shared_bundle_field
    if shared_slot is not None and hasattr(spec, shared_slot):
        shared = getattr(spec, shared_slot)
        if shared is not None:
            resolved.append((shared_slot, shared))

    heads_slot = binding.head_bundles_field
    heads = getattr(spec, heads_slot, None) if heads_slot is not None else None
    if isinstance(heads, Sequence) and not isinstance(heads, (str, bytes, bytearray)):
        for index, bundle in enumerate(heads[: binding.head_count]):
            if bundle is not None:
                resolved.append((f"{heads_slot}[{index}]", bundle))
    return tuple(resolved)


def _declared_field_names(binding: ModelBindingSpec) -> frozenset[str]:
    if binding.kind != "shared_heads":
        return frozenset(binding.bundle_fields)
    return frozenset(
        field_name
        for field_name in (binding.shared_bundle_field, binding.head_bundles_field)
        if field_name is not None
    )


def _bundle_like(value: Any) -> bool:
    return value is not None and hasattr(value, "model") and hasattr(value, "optimizer")


def _generic_bound_slots(
    spec: Any,
    *,
    excluded_fields: frozenset[str],
) -> tuple[tuple[str, Any], ...]:
    """Discover explicit auxiliary bundle fields outside a binding declaration."""

    if is_dataclass(spec) and not isinstance(spec, type):
        names = tuple(field.name for field in fields(spec))
    else:
        values = getattr(spec, "__dict__", None)
        names = tuple(values) if isinstance(values, dict) else ()

    resolved: list[tuple[str, Any]] = []
    for name in sorted(set(names) - excluded_fields):
        value = getattr(spec, name, None)
        if _bundle_like(value):
            resolved.append((name, value))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            resolved.extend(
                (f"{name}[{index}]", bundle)
                for index, bundle in enumerate(value)
                if _bundle_like(bundle)
            )
    return tuple(resolved)


def _bound_slots(spec: Any, binding: ModelBindingSpec) -> tuple[tuple[str, Any], ...]:
    if spec is None:
        return ()
    declared = _declared_bound_slots(spec, binding)
    auxiliary = _generic_bound_slots(
        spec,
        excluded_fields=_declared_field_names(binding),
    )
    return (*declared, *auxiliary)


def _materialized_parameters(model: Any) -> tuple[Any, ...]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return ()
    try:
        return tuple(parameters())
    except (AttributeError, RuntimeError, TypeError):
        return ()


def _component_device(model: Any, parameters: tuple[Any, ...]) -> str | None:
    devices = {
        str(device)
        for parameter in parameters
        if (device := getattr(parameter, "device", None)) is not None
    }
    if not devices:
        buffers = getattr(model, "buffers", None)
        if callable(buffers):
            try:
                devices = {
                    str(device)
                    for buffer in buffers()
                    if (device := getattr(buffer, "device", None)) is not None
                }
            except (AttributeError, RuntimeError, TypeError):
                devices = set()
    if not devices:
        declared = getattr(model, "device", None)
        if declared is not None and str(declared).strip():
            devices = {str(declared)}
    return next(iter(devices)) if len(devices) == 1 else None


def _component_provision(slot: str, bundle: Any) -> ComponentProvision:
    model = getattr(bundle, "model", None)
    parameters = _materialized_parameters(model)
    ema_model = getattr(bundle, "ema_model", None)
    ema_parameters = _materialized_parameters(ema_model)
    scheduler = getattr(bundle, "scheduler", None)
    declared_contract = getattr(bundle, "contract", None)
    contract = declared_contract if isinstance(declared_contract, ModelContract) else None
    return ComponentProvision(
        slot=slot,
        kind="torch_model",
        contract=contract,
        object_id=None if model is None else id(model),
        parameter_ids=frozenset(id(parameter) for parameter in parameters),
        ema_object_id=None if ema_model is None else id(ema_model),
        ema_parameter_ids=frozenset(id(parameter) for parameter in ema_parameters),
        has_optimizer=getattr(bundle, "optimizer", None) is not None,
        has_ema=ema_model is not None,
        has_scheduler=scheduler is not None,
        scheduler_type=None if scheduler is None else type(scheduler).__name__,
        device=_component_device(model, parameters),
    )


def resolve_bound_component_contracts(
    spec: Any,
    binding: ModelBindingSpec,
) -> tuple[ComponentProvision, ...]:
    """Describe declared binding slots and explicit auxiliary model bundles."""

    provisions = (
        _component_provision(slot, bundle) for slot, bundle in _bound_slots(spec, binding)
    )
    return tuple(sorted(provisions, key=lambda provision: provision.slot))


def _issue(code: str, message: str) -> ContractIssue:
    return ContractIssue(code=code, message=message)


def _stable_issues(issues: Iterable[ContractIssue]) -> tuple[ContractIssue, ...]:
    unique = {(issue.code, issue.message): issue for issue in issues}
    return tuple(unique[key] for key in sorted(unique))


def _input_provision_mapping(
    provisions: Mapping[str, ValueDescriptor] | Iterable[tuple[str, ValueDescriptor]],
) -> dict[str, ValueDescriptor]:
    items = provisions.items() if isinstance(provisions, Mapping) else provisions
    normalized: dict[str, ValueDescriptor] = {}
    for role, descriptor in items:
        if not isinstance(role, str) or not role.strip():
            raise TypeError("input provision roles must be non-empty strings")
        if not isinstance(descriptor, ValueDescriptor):
            raise TypeError("input provisions must contain ValueDescriptor values")
        if role in normalized:
            raise ValueError(f"duplicate input provision role {role!r}")
        normalized[role] = descriptor
    return normalized


def _model_input_descriptor(descriptor: ValueDescriptor) -> ValueDescriptor:
    """Return metadata for the payload a native model actually receives.

    Structured graph and token containers expose several leaves with different
    dtypes.  Model rank/dtype contracts describe the primary feature payload,
    not auxiliary edge indices or attention masks.
    """

    schema = dict(descriptor.schema)
    if descriptor.representation == "graph" and "x" in schema:
        primary = schema["x"]
    elif descriptor.representation == "tokens" and "input_ids" in schema:
        primary = schema["input_ids"]
    else:
        return descriptor
    return ValueDescriptor(
        representation=descriptor.representation,
        container_backends=primary.container_backends,
        dtypes=primary.dtypes,
        dtype_kinds=primary.dtype_kinds,
        devices=primary.devices,
        rank=primary.rank,
        shape=primary.shape,
        rows=primary.rows,
        schema=descriptor.schema,
    )


def _validate_model_inputs(
    requirement: ComponentRequirement,
    contract: ModelContract,
    provisions: Mapping[str, ValueDescriptor],
    optional_roles: frozenset[str],
) -> tuple[list[ContractIssue], list[ContractIssue]]:
    issues: list[ContractIssue] = []
    unverified: list[ContractIssue] = []
    for role in requirement.input_roles:
        descriptor = provisions.get(role)
        if descriptor is None:
            if role in optional_roles:
                continue
            issues.append(
                _issue(
                    "E_COMPONENT_INPUT_ROLE_MISSING",
                    f"component slot {requirement.slot!r} requires input role {role!r}",
                )
            )
            continue
        model_input = _model_input_descriptor(descriptor)
        if (
            contract.input_representations is not None
            and model_input.representation not in contract.input_representations
        ):
            issues.append(
                _issue(
                    "E_COMPONENT_INPUT_REPRESENTATION",
                    f"component slot {requirement.slot!r} accepts input representations "
                    f"{sorted(contract.input_representations)!r}, but role {role!r} provides "
                    f"{model_input.representation!r}",
                )
            )
        if contract.input_dtype_kinds is not None:
            if not model_input.dtype_kinds:
                unverified.append(
                    _issue(
                        "E_COMPONENT_INPUT_DTYPE_UNVERIFIED",
                        f"input role {role!r} does not expose dtype metadata for component "
                        f"slot {requirement.slot!r}",
                    )
                )
            elif not model_input.dtype_kinds <= contract.input_dtype_kinds:
                issues.append(
                    _issue(
                        "E_COMPONENT_INPUT_DTYPE",
                        f"component slot {requirement.slot!r} accepts input dtype kinds "
                        f"{sorted(contract.input_dtype_kinds)!r}, but role {role!r} provides "
                        f"{sorted(model_input.dtype_kinds)!r}",
                    )
                )
        if contract.input_ranks is not None:
            if model_input.rank is None:
                unverified.append(
                    _issue(
                        "E_COMPONENT_INPUT_RANK_UNVERIFIED",
                        f"input role {role!r} does not expose rank metadata for component "
                        f"slot {requirement.slot!r}",
                    )
                )
            elif model_input.rank not in contract.input_ranks:
                issues.append(
                    _issue(
                        "E_COMPONENT_INPUT_RANK",
                        f"component slot {requirement.slot!r} accepts input ranks "
                        f"{sorted(contract.input_ranks)!r}, but role {role!r} has rank "
                        f"{model_input.rank}",
                    )
                )
    return issues, unverified


def validate_component_contracts(
    requirements: Iterable[ComponentRequirement],
    relations: Iterable[ComponentRelation],
    provisions: Iterable[ComponentProvision],
    *,
    input_provisions: Mapping[str, ValueDescriptor] | Iterable[tuple[str, ValueDescriptor]] = (),
    optional_input_roles: Iterable[str] = (),
) -> tuple[tuple[ContractIssue, ...], tuple[ContractIssue, ...]]:
    """Compose component requirements and relations without runtime probing.

    The first result contains proven incompatibilities.  The second contains
    facts that cannot be proven from explicit contracts or safe object
    metadata.  Both tuples are de-duplicated and sorted by stable code/message.
    """

    requirements = tuple(requirements)
    relations = tuple(relations)
    provisions = tuple(provisions)
    input_provision_by_role = _input_provision_mapping(input_provisions)
    raw_optional_roles = tuple(optional_input_roles)
    if not all(isinstance(role, str) and role.strip() for role in raw_optional_roles):
        raise TypeError("optional input roles must be non-empty strings")
    optional_roles = frozenset(role.strip() for role in raw_optional_roles)
    if not all(isinstance(value, ComponentRequirement) for value in requirements):
        raise TypeError("requirements must contain ComponentRequirement values")
    if not all(isinstance(value, ComponentRelation) for value in relations):
        raise TypeError("relations must contain ComponentRelation values")
    if not all(isinstance(value, ComponentProvision) for value in provisions):
        raise TypeError("provisions must contain ComponentProvision values")

    issues: list[ContractIssue] = []
    unverified: list[ContractIssue] = []
    by_slot: dict[str, ComponentProvision] = {}
    for provision in sorted(provisions, key=lambda value: value.slot):
        if provision.slot in by_slot:
            issues.append(
                _issue(
                    "E_COMPONENT_SLOT_DUPLICATE",
                    f"component slot {provision.slot!r} has multiple provisions",
                )
            )
            continue
        by_slot[provision.slot] = provision

    for requirement in requirements:
        provision = by_slot.get(requirement.slot)
        if provision is None:
            issues.append(
                _issue(
                    "E_COMPONENT_MISSING",
                    f"required component slot {requirement.slot!r} is not bound",
                )
            )
            continue
        if provision.kind != requirement.kind:
            issues.append(
                _issue(
                    "E_COMPONENT_KIND_MISMATCH",
                    f"component slot {requirement.slot!r} requires kind "
                    f"{requirement.kind!r}, got {provision.kind!r}",
                )
            )
        contract = provision.contract
        if contract is None:
            unverified.append(
                _issue(
                    "E_COMPONENT_CONTRACT_UNVERIFIED",
                    f"component slot {requirement.slot!r} has no explicit model contract",
                )
            )
        else:
            missing_outputs = sorted(requirement.outputs - contract.outputs)
            if missing_outputs:
                issues.append(
                    _issue(
                        "E_COMPONENT_OUTPUT_MISSING",
                        f"component slot {requirement.slot!r} does not declare required "
                        f"outputs {missing_outputs!r}",
                    )
                )
            if requirement.output_alternatives and not any(
                alternative <= contract.outputs for alternative in requirement.output_alternatives
            ):
                alternatives = [
                    sorted(alternative) for alternative in requirement.output_alternatives
                ]
                issues.append(
                    _issue(
                        "E_COMPONENT_OUTPUT_ALTERNATIVE_MISSING",
                        f"component slot {requirement.slot!r} does not satisfy any required "
                        f"output alternative {alternatives!r}",
                    )
                )
            if contract.verification == "unverified":
                unverified.append(
                    _issue(
                        "E_COMPONENT_CONTRACT_UNVERIFIED",
                        f"component slot {requirement.slot!r} declares an unverified "
                        "model contract",
                    )
                )
            input_issues, input_unverified = _validate_model_inputs(
                requirement,
                contract,
                input_provision_by_role,
                optional_roles,
            )
            issues.extend(input_issues)
            unverified.extend(input_unverified)
        if requirement.requires_optimizer and not provision.has_optimizer:
            issues.append(
                _issue(
                    "E_COMPONENT_OPTIMIZER_MISSING",
                    f"component slot {requirement.slot!r} requires an optimizer",
                )
            )
        if requirement.requires_scheduler and not provision.has_scheduler:
            issues.append(
                _issue(
                    "E_COMPONENT_SCHEDULER_MISSING",
                    f"component slot {requirement.slot!r} requires a scheduler",
                )
            )
        elif (
            requirement.scheduler_types
            and provision.scheduler_type not in requirement.scheduler_types
        ):
            issues.append(
                _issue(
                    "E_COMPONENT_SCHEDULER_TYPE",
                    f"component slot {requirement.slot!r} requires scheduler types "
                    f"{sorted(requirement.scheduler_types)!r}, got "
                    f"{provision.scheduler_type!r}",
                )
            )
        if requirement.requires_ema and not provision.has_ema:
            issues.append(
                _issue(
                    "E_COMPONENT_EMA_MISSING",
                    f"component slot {requirement.slot!r} requires an EMA model",
                )
            )
        elif requirement.requires_ema:
            if provision.object_id is None or provision.ema_object_id is None:
                unverified.append(
                    _issue(
                        "E_COMPONENT_EMA_IDENTITY_UNVERIFIED",
                        f"cannot verify EMA identity for component slot {requirement.slot!r}",
                    )
                )
            elif provision.object_id == provision.ema_object_id:
                issues.append(
                    _issue(
                        "E_COMPONENT_EMA_OBJECT_ALIAS",
                        f"component slot {requirement.slot!r} uses the training model "
                        "itself as EMA model",
                    )
                )
            if not provision.parameter_ids or not provision.ema_parameter_ids:
                unverified.append(
                    _issue(
                        "E_COMPONENT_EMA_PARAMETERS_UNVERIFIED",
                        f"cannot verify EMA parameter independence for component slot "
                        f"{requirement.slot!r}",
                    )
                )
            elif provision.parameter_ids & provision.ema_parameter_ids:
                issues.append(
                    _issue(
                        "E_COMPONENT_EMA_PARAMETERS_SHARED",
                        f"component slot {requirement.slot!r} shares parameters with its EMA model",
                    )
                )

    for relation in relations:
        related = [by_slot.get(slot) for slot in relation.slots]
        missing_slots = [
            slot
            for slot, provision in zip(relation.slots, related, strict=True)
            if provision is None
        ]
        if missing_slots:
            issues.append(
                _issue(
                    "E_COMPONENT_RELATION_SLOT_MISSING",
                    f"component relation {relation.kind!r} references unbound slots "
                    f"{missing_slots!r}",
                )
            )
            continue
        present = tuple(provision for provision in related if provision is not None)
        if relation.kind == "distinct_objects":
            object_ids = [provision.object_id for provision in present]
            if any(object_id is None for object_id in object_ids):
                unverified.append(
                    _issue(
                        "E_COMPONENT_RELATION_UNVERIFIED",
                        f"cannot verify distinct objects for slots {list(relation.slots)!r}",
                    )
                )
            elif len(set(object_ids)) != len(object_ids):
                issues.append(
                    _issue(
                        "E_COMPONENT_OBJECT_ALIAS",
                        f"component slots {list(relation.slots)!r} alias the same model object",
                    )
                )
        elif relation.kind == "disjoint_parameters":
            parameter_sets = [provision.parameter_ids for provision in present]
            if any(not parameter_ids for parameter_ids in parameter_sets):
                unverified.append(
                    _issue(
                        "E_COMPONENT_RELATION_UNVERIFIED",
                        f"cannot verify disjoint parameters for slots {list(relation.slots)!r}",
                    )
                )
            elif any(
                parameter_sets[left] & parameter_sets[right]
                for left in range(len(parameter_sets))
                for right in range(left + 1, len(parameter_sets))
            ):
                issues.append(
                    _issue(
                        "E_COMPONENT_PARAMETERS_SHARED",
                        f"component slots {list(relation.slots)!r} share parameters",
                    )
                )
        elif relation.kind == "same_device":
            devices = [provision.device for provision in present]
            if any(device is None for device in devices):
                unverified.append(
                    _issue(
                        "E_COMPONENT_RELATION_UNVERIFIED",
                        f"cannot verify a common device for slots {list(relation.slots)!r}",
                    )
                )
            elif len(set(devices)) != 1:
                issues.append(
                    _issue(
                        "E_COMPONENT_DEVICE_MISMATCH",
                        f"component slots {list(relation.slots)!r} use different devices "
                        f"{devices!r}",
                    )
                )
        else:
            unverified.append(
                _issue(
                    "E_COMPONENT_ARCHITECTURE_UNVERIFIED",
                    f"cannot verify a common architecture for slots {list(relation.slots)!r}",
                )
            )

    return _stable_issues(issues), _stable_issues(unverified)


__all__ = [
    "resolve_bound_component_contracts",
    "validate_component_contracts",
]
