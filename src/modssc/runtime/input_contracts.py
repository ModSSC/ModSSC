"""Metadata-only materialization and validation of method-facing inputs.

This module deliberately stays on the structural side of the execution
boundary.  Describing a value reads container metadata (shape, dtype, device,
and mapping keys), but never coerces a value with :func:`numpy.asarray`, moves a
tensor, or densifies a sparse matrix.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from modssc.capabilities import LearningRegime
from modssc.runtime.contracts import (
    ContractIssue,
    InputRoleRequirement,
    RoleRelation,
    ValueDescriptor,
)

_NUMERIC_KINDS = frozenset({"bool", "integer", "float", "complex"})
_MIXABLE_KINDS = frozenset({"float", "complex"})
_FIRST_STRONG_VIEW_KEYS = (
    "X_u_s0",
    "X_u_s_0",
    "X_u_s",
    "X_u_strong0",
)
_SECOND_STRONG_VIEW_KEYS = (
    "X_u_s1",
    "X_u_s_1",
    "X_u_strong1",
    "X_u_s2",
    "X_u_s_2",
)
_LABELED_STRONG_VIEW_KEYS = ("X_l_s", "X_l_strong", "labeled_strong")
_AUGMENTATION_VIEW_KEYS = frozenset(
    (
        *_FIRST_STRONG_VIEW_KEYS,
        *_SECOND_STRONG_VIEW_KEYS,
        *_LABELED_STRONG_VIEW_KEYS,
    )
)
_GRAPH_KEYS = frozenset(
    {
        "edge_index",
        "edge_weight",
        "edge_attr",
        "adjacency",
        "adj",
        "n_nodes",
        "num_nodes",
    }
)
_GRAPH_METADATA_KEYS = frozenset({"n_nodes", "num_nodes", "directed", "meta"})
_TOKEN_KEYS = frozenset({"input_ids", "attention_mask", "token_type_ids"})


def _module_root(value: Any) -> str:
    module = type(value).__module__.split(".", 1)[0]
    return module if module and module != "builtins" else "python"


def _shape_metadata(value: Any) -> tuple[int | None, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dimensions = tuple(shape)
    except TypeError:
        return None
    normalized: list[int | None] = []
    for dimension in dimensions:
        try:
            parsed = int(dimension)
        except (TypeError, ValueError, OverflowError):
            normalized.append(None)
            continue
        normalized.append(parsed if parsed >= 0 else None)
    return tuple(normalized)


def _dtype_kind_from_code(code: str) -> str:
    return {
        "b": "bool",
        "i": "integer",
        "u": "integer",
        "f": "float",
        "c": "complex",
        "S": "string",
        "U": "string",
        "O": "object",
        "M": "datetime",
        "m": "timedelta",
        "V": "unknown",
    }.get(code, "unknown")


def _dtype_metadata(dtype: Any) -> tuple[frozenset[str], frozenset[str]]:
    if dtype is None:
        return frozenset(), frozenset()
    name = str(dtype).removeprefix("torch.")
    kind = getattr(dtype, "kind", None)
    if isinstance(kind, str) and len(kind) == 1:
        return frozenset({name}), frozenset({_dtype_kind_from_code(kind)})
    try:
        numpy_dtype = np.dtype(name)
    except TypeError:
        lowered = name.lower()
        if lowered == "bool":
            resolved_kind = "bool"
        elif "complex" in lowered:
            resolved_kind = "complex"
        elif "float" in lowered or "bfloat" in lowered:
            resolved_kind = "float"
        elif "int" in lowered or lowered.startswith(("qint", "quint")):
            resolved_kind = "integer"
        elif "str" in lowered or "string" in lowered:
            resolved_kind = "string"
        elif "object" in lowered:
            resolved_kind = "object"
        else:
            resolved_kind = "unknown"
        return frozenset({name}), frozenset({resolved_kind})
    return (
        frozenset({str(numpy_dtype)}),
        frozenset({_dtype_kind_from_code(numpy_dtype.kind)}),
    )


def _rows_from_shape(shape: tuple[int | None, ...] | None) -> int | None:
    if not shape:
        return None
    return shape[0]


def _is_torch_value(value: Any) -> bool:
    return _module_root(value) == "torch" and hasattr(value, "shape")


def _is_scipy_sparse(value: Any) -> bool:
    module = type(value).__module__
    return module.startswith("scipy.sparse") or (
        hasattr(value, "shape")
        and hasattr(value, "nnz")
        and callable(getattr(value, "tocsr", None))
    )


def _array_descriptor(
    value: Any,
    *,
    backend: str,
    sparse: bool = False,
) -> ValueDescriptor:
    shape = _shape_metadata(value)
    dtypes, dtype_kinds = _dtype_metadata(getattr(value, "dtype", None))
    if sparse:
        representation = "sparse"
    elif backend == "numpy" and dtype_kinds & frozenset({"string", "object"}):
        representation = "objects"
    else:
        representation = "dense"
    devices: frozenset[str]
    if backend in {"numpy", "scipy"}:
        devices = frozenset({"cpu"})
    else:
        device = getattr(value, "device", None)
        devices = frozenset() if device is None else frozenset({str(device)})
    return ValueDescriptor(
        representation=representation,
        container_backends=frozenset({backend}),
        dtypes=dtypes,
        dtype_kinds=dtype_kinds,
        devices=devices,
        rank=None if shape is None else len(shape),
        shape=shape,
        rows=_rows_from_shape(shape),
    )


def _scalar_descriptor(value: Any) -> ValueDescriptor:
    if isinstance(value, np.generic):
        dtypes, dtype_kinds = _dtype_metadata(value.dtype)
        return ValueDescriptor(
            representation="scalar",
            container_backends=frozenset({"numpy"}),
            dtypes=dtypes,
            dtype_kinds=dtype_kinds,
            devices=frozenset({"cpu"}),
            rank=0,
            shape=(),
        )
    if value is None:
        dtype_name, dtype_kind, representation = "none", "unknown", "none"
    elif isinstance(value, bool):
        dtype_name, dtype_kind, representation = "bool", "bool", "scalar"
    elif isinstance(value, int):
        dtype_name, dtype_kind, representation = "int", "integer", "scalar"
    elif isinstance(value, float):
        dtype_name, dtype_kind, representation = "float", "float", "scalar"
    elif isinstance(value, complex):
        dtype_name, dtype_kind, representation = "complex", "complex", "scalar"
    elif isinstance(value, str):
        dtype_name, dtype_kind, representation = "str", "string", "text"
    elif isinstance(value, (bytes, bytearray, memoryview)):
        dtype_name, dtype_kind, representation = "bytes", "string", "bytes"
    else:
        dtype_name, dtype_kind, representation = type(value).__name__, "object", "objects"
    return ValueDescriptor(
        representation=representation,
        container_backends=frozenset({"python"}),
        dtypes=frozenset({dtype_name}),
        dtype_kinds=frozenset({dtype_kind}),
        rank=0,
        shape=(),
    )


def _common_shape(descriptors: Sequence[ValueDescriptor]) -> tuple[int | None, ...] | None:
    if not descriptors or any(descriptor.shape is None for descriptor in descriptors):
        return None
    shapes = [descriptor.shape for descriptor in descriptors]
    assert all(shape is not None for shape in shapes)
    lengths = {len(shape) for shape in shapes if shape is not None}
    if len(lengths) != 1:
        return None
    width = next(iter(lengths))
    return tuple(
        dimensions.pop() if len(dimensions := {shape[index] for shape in shapes}) == 1 else None
        for index in range(width)
    )


def _common_rows(descriptors: Iterable[ValueDescriptor]) -> int | None:
    known = [descriptor.rows for descriptor in descriptors if descriptor.rows is not None]
    if not known or len(set(known)) != 1:
        return None
    return known[0]


def _aggregate_sets(
    descriptors: Iterable[ValueDescriptor],
    field_name: str,
) -> frozenset[str]:
    return frozenset(item for descriptor in descriptors for item in getattr(descriptor, field_name))


def _merged_item_descriptor(descriptors: Sequence[ValueDescriptor]) -> ValueDescriptor:
    first = descriptors[0]
    if all(descriptor == first for descriptor in descriptors[1:]):
        return first
    representations = {descriptor.representation for descriptor in descriptors}
    ranks = {descriptor.rank for descriptor in descriptors}
    schemas = {descriptor.schema for descriptor in descriptors}
    return ValueDescriptor(
        representation=(representations.pop() if len(representations) == 1 else "mixed"),
        container_backends=_aggregate_sets(descriptors, "container_backends"),
        dtypes=_aggregate_sets(descriptors, "dtypes"),
        dtype_kinds=_aggregate_sets(descriptors, "dtype_kinds"),
        devices=_aggregate_sets(descriptors, "devices"),
        rank=ranks.pop() if len(ranks) == 1 else None,
        shape=_common_shape(descriptors),
        rows=_common_rows(descriptors),
        schema=schemas.pop() if len(schemas) == 1 else (),
    )


def _sequence_descriptor(value: Sequence[Any], stack: set[int]) -> ValueDescriptor:
    size = len(value)
    if size == 0:
        return ValueDescriptor(
            representation="sequence",
            container_backends=frozenset({"python"}),
            rank=1,
            shape=(0,),
            rows=0,
        )
    items = [_describe_value(item, stack) for item in value]
    merged = _merged_item_descriptor(items)
    shape = None if merged.shape is None else (size, *merged.shape)
    return ValueDescriptor(
        representation="sequence",
        container_backends=merged.container_backends,
        dtypes=merged.dtypes,
        dtype_kinds=merged.dtype_kinds,
        devices=merged.devices,
        rank=None if merged.rank is None else merged.rank + 1,
        shape=shape,
        rows=size,
        schema=(("item", merged),),
    )


def _mapping_representation(
    keys: frozenset[str],
    descriptors: Sequence[ValueDescriptor],
) -> str:
    if "input_ids" in keys or len(keys & _TOKEN_KEYS) >= 2:
        return "tokens"
    if keys & _GRAPH_KEYS or ({"x", "edge_index"} <= keys):
        return "graph"
    representations = {descriptor.representation for descriptor in descriptors}
    backends = _aggregate_sets(descriptors, "container_backends")
    if len(representations) > 1 or len(backends) > 1 or "mixed" in representations:
        return "mixed"
    return "structured"


def _mapping_descriptor(value: Mapping[Any, Any], stack: set[int]) -> ValueDescriptor:
    raw_items: list[tuple[str, Any]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("structured input mapping keys must be strings")
        raw_items.append((key, item))
    raw_items.sort(key=lambda item: item[0])
    described = [(key, _describe_value(item, stack)) for key, item in raw_items]
    descriptors = [descriptor for _, descriptor in described]
    keys = frozenset(key for key, _ in described)
    representation = _mapping_representation(keys, descriptors)
    aggregate_descriptors = [
        descriptor
        for key, descriptor in described
        if not (representation == "graph" and key in _GRAPH_METADATA_KEYS)
    ]
    if not aggregate_descriptors:
        aggregate_descriptors = descriptors
    common_shape = _common_shape(aggregate_descriptors)
    ranks = {descriptor.rank for descriptor in aggregate_descriptors}
    rank = ranks.pop() if len(ranks) == 1 else None

    if representation == "tokens":
        primary = dict(described).get("input_ids")
        rows = None if primary is None else primary.rows
    elif representation == "graph":
        by_name = dict(described)
        primary = by_name.get("x")
        rows = None if primary is None else primary.rows
        if rows is None:
            count = value.get("n_nodes", value.get("num_nodes"))
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                rows = count
    else:
        rows = _common_rows(aggregate_descriptors)

    return ValueDescriptor(
        representation=representation,
        container_backends=_aggregate_sets(aggregate_descriptors, "container_backends"),
        dtypes=_aggregate_sets(aggregate_descriptors, "dtypes"),
        dtype_kinds=_aggregate_sets(aggregate_descriptors, "dtype_kinds"),
        devices=_aggregate_sets(aggregate_descriptors, "devices"),
        rank=rank,
        shape=common_shape,
        rows=rows,
        schema=tuple(described),
    )


def _graph_object_descriptor(value: Any, stack: set[int]) -> ValueDescriptor:
    fields: dict[str, Any] = {}
    for name in ("x", "edge_index", "edge_weight", "n_nodes", "num_nodes"):
        try:
            item = getattr(value, name)
        except (AttributeError, RuntimeError):
            continue
        if item is not None:
            fields[name] = item
    return _mapping_descriptor(fields, stack)


def _describe_value(value: Any, stack: set[int]) -> ValueDescriptor:
    if isinstance(value, np.ndarray):
        return _array_descriptor(value, backend="numpy")
    if isinstance(value, np.generic):
        return _scalar_descriptor(value)
    if _is_scipy_sparse(value):
        return _array_descriptor(value, backend="scipy", sparse=True)
    if _is_torch_value(value):
        layout = str(getattr(value, "layout", ""))
        sparse = bool(getattr(value, "is_sparse", False)) or "sparse" in layout
        return _array_descriptor(value, backend="torch", sparse=sparse)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in stack:
            return ValueDescriptor(
                representation="objects",
                container_backends=frozenset({"python"}),
            )
        stack.add(identity)
        try:
            return _mapping_descriptor(value, stack)
        finally:
            stack.remove(identity)
    if hasattr(value, "edge_index") and (hasattr(value, "n_nodes") or hasattr(value, "num_nodes")):
        identity = id(value)
        if identity in stack:
            return ValueDescriptor(
                representation="graph",
                container_backends=frozenset(),
            )
        stack.add(identity)
        try:
            return _graph_object_descriptor(value, stack)
        finally:
            stack.remove(identity)
    if isinstance(value, range):
        return ValueDescriptor(
            representation="sequence",
            container_backends=frozenset({"python"}),
            dtypes=frozenset({"int"}),
            dtype_kinds=frozenset({"integer"}),
            rank=1,
            shape=(len(value),),
            rows=len(value),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        identity = id(value)
        if identity in stack:
            return ValueDescriptor(
                representation="sequence",
                container_backends=frozenset({"python"}),
            )
        stack.add(identity)
        try:
            return _sequence_descriptor(value, stack)
        finally:
            stack.remove(identity)
    if hasattr(value, "shape"):
        return _array_descriptor(value, backend=_module_root(value))
    return _scalar_descriptor(value)


def describe_value(value: Any) -> ValueDescriptor:
    """Describe ``value`` without materializing or transferring its contents."""

    return _describe_value(value, set())


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _put_role(
    provisions: dict[str, ValueDescriptor],
    role: str,
    value: Any,
    *,
    rows: int | None = None,
) -> None:
    if value is None:
        return
    descriptor = describe_value(value)
    if rows is not None:
        descriptor = replace(descriptor, rows=rows)
    provisions[role] = descriptor


def _view_payload_field(payload: Any, field_name: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(field_name)
    return getattr(payload, field_name, None)


def _is_scientific_view_payload(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        return "X_l" in payload or "X_u" in payload
    return hasattr(payload, "X_l") or hasattr(payload, "X_u")


def _strong_view(
    views: Mapping[str, Any] | None,
    aliases: tuple[str, ...],
) -> Any:
    if views is None:
        return None
    for alias in aliases:
        candidate = views.get(alias)
        if candidate is not None and not _is_scientific_view_payload(candidate):
            return candidate
    return None


def _materialize_graph_roles(
    provisions: dict[str, ValueDescriptor],
    graph: Any,
) -> None:
    if graph is None:
        return
    if isinstance(graph, Mapping):
        items = graph.items()
    else:
        fields: list[tuple[str, Any]] = []
        for name in ("edge_index", "edge_weight", "n_nodes", "num_nodes"):
            try:
                value = getattr(graph, name)
            except (AttributeError, RuntimeError):
                continue
            fields.append((name, value))
        items = fields
    for name, value in sorted(items, key=lambda item: str(item[0])):
        if not isinstance(name, str):
            raise TypeError("graph mapping keys must be strings")
        rows = None
        if (
            name in {"n_nodes", "num_nodes"}
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            rows = value if value >= 0 else None
        _put_role(provisions, f"fit.graph.{name}", value, rows=rows)


def _materialize_inductive(consumed_input: Any) -> dict[str, ValueDescriptor]:
    provisions: dict[str, ValueDescriptor] = {}
    for name in ("X_l", "y_l", "X_u", "X_u_w"):
        _put_role(provisions, f"fit.{name}", _field(consumed_input, name))

    views_value = _field(consumed_input, "views")
    if views_value is not None and not isinstance(views_value, Mapping):
        raise TypeError("inductive consumed_input.views must be a mapping when provided")
    views = views_value
    first_strong = _field(consumed_input, "X_u_s")
    if first_strong is None:
        first_strong = _strong_view(views, _FIRST_STRONG_VIEW_KEYS)
    second_strong = _field(consumed_input, "X_u_s_1")
    if second_strong is None:
        second_strong = _strong_view(views, _SECOND_STRONG_VIEW_KEYS)
    _put_role(provisions, "fit.X_u_s.0", first_strong)
    _put_role(provisions, "fit.X_u_s.1", second_strong)
    _put_role(
        provisions,
        "fit.X_l_s.0",
        _strong_view(views, _LABELED_STRONG_VIEW_KEYS),
    )

    if views is not None:
        for name, payload in sorted(views.items(), key=lambda item: str(item[0])):
            if not isinstance(name, str):
                raise TypeError("inductive view names must be strings")
            if name in _AUGMENTATION_VIEW_KEYS:
                continue
            for split_name in ("X_l", "X_u"):
                _put_role(
                    provisions,
                    f"fit.views.{name}.{split_name}",
                    _view_payload_field(payload, split_name),
                )

    _materialize_graph_roles(provisions, _field(consumed_input, "graph"))
    return provisions


def _materialize_transductive(consumed_input: Any) -> dict[str, ValueDescriptor]:
    method_input = _field(consumed_input, "fit", consumed_input)
    provisions: dict[str, ValueDescriptor] = {}
    for name in ("X", "y"):
        _put_role(provisions, f"fit.{name}", _field(method_input, name))
    masks = _field(method_input, "masks")
    if masks is not None and not isinstance(masks, Mapping):
        raise TypeError("transductive consumed_input.masks must be a mapping when provided")
    if masks is not None:
        for name, value in sorted(masks.items(), key=lambda item: str(item[0])):
            if not isinstance(name, str):
                raise TypeError("transductive mask names must be strings")
            _put_role(provisions, f"fit.masks.{name}", value)
    _materialize_graph_roles(provisions, _field(method_input, "graph"))
    return provisions


def materialize_input_contracts(
    *,
    regime: LearningRegime,
    consumed_input: Any,
) -> tuple[tuple[str, ValueDescriptor], ...]:
    """Describe every exact role exposed to an inductive or transductive method."""

    if regime == "inductive":
        provisions = _materialize_inductive(consumed_input)
    elif regime == "transductive":
        provisions = _materialize_transductive(consumed_input)
    else:
        raise ValueError("regime must be 'inductive' or 'transductive'")
    return tuple(sorted(provisions.items(), key=lambda item: item[0]))


def _issue(code: str, message: str) -> ContractIssue:
    return ContractIssue(code=code, message=message)


def _stable_issues(values: Iterable[ContractIssue]) -> tuple[ContractIssue, ...]:
    unique = {(value.code, value.message): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _unverified(
    code: str,
    role: str,
    field_name: str,
) -> ContractIssue:
    return _issue(
        f"{code}_UNVERIFIED",
        f"role {role!r} does not expose {field_name} metadata",
    )


def _check_allowed_set(
    *,
    role: str,
    actual: frozenset[str],
    allowed: frozenset[str] | None,
    field_name: str,
    code: str,
) -> tuple[ContractIssue | None, ContractIssue | None]:
    if allowed is None:
        return None, None
    if not actual:
        return None, _unverified(code, role, field_name)
    if actual <= allowed:
        return None, None
    return (
        _issue(
            code,
            f"role {role!r} has {field_name} {sorted(actual)!r}; "
            f"expected a subset of {sorted(allowed)!r}",
        ),
        None,
    )


def _non_empty_status(descriptor: ValueDescriptor) -> bool | None:
    if descriptor.rows is not None:
        return descriptor.rows > 0
    if descriptor.rank == 0 or descriptor.shape == ():
        return True
    return None


def _validate_requirement(
    requirement: InputRoleRequirement,
    descriptor: ValueDescriptor,
) -> tuple[list[ContractIssue], list[ContractIssue]]:
    issues: list[ContractIssue] = []
    unverified: list[ContractIssue] = []
    role = requirement.role

    if requirement.non_empty:
        non_empty = _non_empty_status(descriptor)
        if non_empty is False:
            issues.append(_issue("E_INPUT_NON_EMPTY", f"role {role!r} is empty"))
        elif non_empty is None:
            unverified.append(_unverified("E_INPUT_NON_EMPTY", role, "row-count"))

    if (
        requirement.representations is not None
        and descriptor.representation not in requirement.representations
    ):
        issues.append(
            _issue(
                "E_INPUT_REPRESENTATION",
                f"role {role!r} has representation {descriptor.representation!r}; "
                f"expected one of {sorted(requirement.representations)!r}",
            )
        )

    for actual, allowed, field_name, code in (
        (
            descriptor.container_backends,
            requirement.container_backends,
            "container backends",
            "E_INPUT_BACKEND",
        ),
        (
            descriptor.dtype_kinds,
            requirement.dtype_kinds,
            "dtype kinds",
            "E_INPUT_DTYPE_KIND",
        ),
        (descriptor.dtypes, requirement.dtypes, "dtypes", "E_INPUT_DTYPE"),
    ):
        issue, unknown = _check_allowed_set(
            role=role,
            actual=actual,
            allowed=allowed,
            field_name=field_name,
            code=code,
        )
        if issue is not None:
            issues.append(issue)
        if unknown is not None:
            unverified.append(unknown)

    if requirement.ranks is not None:
        if descriptor.rank is None:
            unverified.append(_unverified("E_INPUT_RANK", role, "rank"))
        elif descriptor.rank not in requirement.ranks:
            issues.append(
                _issue(
                    "E_INPUT_RANK",
                    f"role {role!r} has rank {descriptor.rank}; "
                    f"expected one of {sorted(requirement.ranks)!r}",
                )
            )

    if requirement.numeric is not None:
        numeric = descriptor.numeric
        if numeric is None:
            unverified.append(_unverified("E_INPUT_NUMERIC", role, "numeric dtype"))
        elif numeric is not requirement.numeric:
            issues.append(
                _issue(
                    "E_INPUT_NUMERIC",
                    f"role {role!r} numeric={numeric}; expected {requirement.numeric}",
                )
            )
    return issues, unverified


def _same_known_sets(descriptors: Sequence[ValueDescriptor], field_name: str) -> bool | None:
    values = [getattr(descriptor, field_name) for descriptor in descriptors]
    if any(not value for value in values):
        return None
    return len(set(values)) == 1


def _primary_payload_descriptor(descriptor: ValueDescriptor) -> ValueDescriptor:
    primary_key = {"graph": "x", "tokens": "input_ids"}.get(descriptor.representation)
    if primary_key is None:
        return descriptor
    return _schema_items(descriptor).get(primary_key, descriptor)


def _same_rows(descriptors: Sequence[ValueDescriptor]) -> bool | None:
    rows = [descriptor.rows for descriptor in descriptors]
    if any(row is None for row in rows):
        return None
    return len(set(rows)) == 1


def _shape_compatible(
    left: ValueDescriptor,
    right: ValueDescriptor,
    *,
    ignore_rows: bool,
) -> bool | None:
    if left.rank is not None and right.rank is not None and left.rank != right.rank:
        return False
    if left.rank is None or right.rank is None:
        return None
    if left.shape is None or right.shape is None:
        return None
    if len(left.shape) != len(right.shape):
        return False
    unknown = False
    start = 1 if ignore_rows and left.rank > 0 else 0
    for left_dimension, right_dimension in zip(
        left.shape[start:], right.shape[start:], strict=True
    ):
        if left_dimension is None or right_dimension is None:
            unknown = True
        elif left_dimension != right_dimension:
            return False
    return None if unknown else True


def _schema_items(descriptor: ValueDescriptor) -> dict[str, ValueDescriptor]:
    return dict(descriptor.schema)


def _is_structured(descriptor: ValueDescriptor) -> bool:
    return descriptor.representation in {"tokens", "graph", "structured", "mixed"}


def _same_input_schema(left: ValueDescriptor, right: ValueDescriptor) -> bool | None:
    if left.representation != right.representation:
        return False
    if _is_structured(left) or _is_structured(right):
        left_schema = _schema_items(left)
        right_schema = _schema_items(right)
        if set(left_schema) != set(right_schema):
            return False
        if not left_schema:
            return None
        return _combine_statuses(
            _same_input_schema(left_schema[key], right_schema[key]) for key in sorted(left_schema)
        )
    return _shape_compatible(left, right, ignore_rows=True)


def _metadata_sets_compatible(
    left: ValueDescriptor,
    right: ValueDescriptor,
    field_name: str,
    *,
    absent_is_compatible: bool = False,
) -> bool | None:
    left_value = getattr(left, field_name)
    right_value = getattr(right, field_name)
    if not left_value and not right_value and absent_is_compatible:
        return True
    if not left_value or not right_value:
        return None
    return left_value == right_value


def _concat_compatible(left: ValueDescriptor, right: ValueDescriptor) -> bool | None:
    if left.representation != right.representation:
        return False
    if _is_structured(left) or _is_structured(right):
        left_schema = _schema_items(left)
        right_schema = _schema_items(right)
        if set(left_schema) != set(right_schema):
            return False
        if not left_schema:
            return None
        nested = _combine_statuses(
            _concat_compatible(left_schema[key], right_schema[key]) for key in sorted(left_schema)
        )
    else:
        nested = _shape_compatible(left, right, ignore_rows=True)
    return _combine_statuses(
        (
            nested,
            _metadata_sets_compatible(left, right, "container_backends"),
            _metadata_sets_compatible(left, right, "devices", absent_is_compatible=True),
            _metadata_sets_compatible(left, right, "dtypes"),
        )
    )


def _mix_compatible(left: ValueDescriptor, right: ValueDescriptor) -> bool | None:
    concat = _concat_compatible(left, right)
    if concat is False:
        return False
    numeric = (left.numeric, right.numeric)
    if False in numeric:
        return False
    if any(value is None for value in numeric):
        numeric_status: bool | None = None
    elif not left.dtype_kinds <= _MIXABLE_KINDS or not right.dtype_kinds <= _MIXABLE_KINDS:
        return False
    else:
        numeric_status = True
    return _combine_statuses(
        (
            concat,
            numeric_status,
            _same_rows((left, right)),
            _shape_compatible(left, right, ignore_rows=False),
        )
    )


def _combine_statuses(values: Iterable[bool | None]) -> bool | None:
    unknown = False
    for value in values:
        if value is False:
            return False
        if value is None:
            unknown = True
    return None if unknown else True


def _relation_status(
    relation: RoleRelation,
    descriptors: Sequence[ValueDescriptor],
) -> bool | None:
    if relation.kind == "same_rows":
        return _same_rows(descriptors)
    if relation.kind == "same_backend":
        payloads = tuple(_primary_payload_descriptor(value) for value in descriptors)
        return _same_known_sets(payloads, "container_backends")
    if relation.kind == "same_device":
        payloads = tuple(_primary_payload_descriptor(value) for value in descriptors)
        return _same_known_sets(payloads, "devices")
    if relation.kind == "same_dtype_kind":
        return _same_known_sets(descriptors, "dtype_kinds")
    comparator = {
        "same_input_schema": _same_input_schema,
        "concat_compatible": _concat_compatible,
        "mix_compatible": _mix_compatible,
    }[relation.kind]
    first, *others = descriptors
    return _combine_statuses(comparator(first, other) for other in others)


def _provision_mapping(
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


def validate_input_contracts(
    requirements: Iterable[InputRoleRequirement],
    relations: Iterable[RoleRelation],
    provisions: Mapping[str, ValueDescriptor] | Iterable[tuple[str, ValueDescriptor]],
) -> tuple[tuple[ContractIssue, ...], tuple[ContractIssue, ...]]:
    """Validate role requirements and relations against materialized provisions.

    The first returned tuple contains proven incompatibilities.  The second
    contains requirements for which the necessary metadata is unavailable.
    Both tuples are de-duplicated and sorted by ``(code, message)`` so their
    serialization is independent of mapping and declaration order.
    """

    provision_by_role = _provision_mapping(provisions)
    requirement_values = tuple(requirements)
    relation_values = tuple(relations)
    if not all(isinstance(value, InputRoleRequirement) for value in requirement_values):
        raise TypeError("requirements must contain InputRoleRequirement values")
    if not all(isinstance(value, RoleRelation) for value in relation_values):
        raise TypeError("relations must contain RoleRelation values")
    requirement_by_role: dict[str, InputRoleRequirement] = {}
    for requirement in requirement_values:
        if requirement.role in requirement_by_role:
            raise ValueError(f"duplicate input requirement role {requirement.role!r}")
        requirement_by_role[requirement.role] = requirement

    issues: list[ContractIssue] = []
    unverified: list[ContractIssue] = []
    for requirement in sorted(requirement_values, key=lambda value: value.role):
        descriptor = provision_by_role.get(requirement.role)
        if descriptor is None:
            if not requirement.optional:
                issues.append(
                    _issue(
                        "E_INPUT_ROLE_MISSING",
                        f"required role {requirement.role!r} was not materialized",
                    )
                )
            continue
        requirement_issues, requirement_unverified = _validate_requirement(requirement, descriptor)
        issues.extend(requirement_issues)
        unverified.extend(requirement_unverified)

    for relation in sorted(relation_values, key=lambda value: (value.kind, value.roles)):
        missing = [role for role in relation.roles if role not in provision_by_role]
        if missing:
            missing_requirements = [requirement_by_role.get(role) for role in missing]
            if all(
                requirement is not None and requirement.optional
                for requirement in missing_requirements
            ):
                continue
            if any(
                requirement is not None and not requirement.optional
                for requirement in missing_requirements
            ):
                continue
            code = f"E_INPUT_RELATION_{relation.kind.upper()}_UNVERIFIED"
            unverified.append(
                _issue(
                    code,
                    f"relation {relation.kind!r} cannot be verified because roles "
                    f"{sorted(missing)!r} were not materialized",
                )
            )
            continue
        descriptors = [provision_by_role[role] for role in relation.roles]
        status = _relation_status(relation, descriptors)
        code = f"E_INPUT_RELATION_{relation.kind.upper()}"
        if status is False:
            issues.append(
                _issue(
                    code,
                    f"relation {relation.kind!r} failed for roles {list(relation.roles)!r}",
                )
            )
        elif status is None:
            unverified.append(
                _issue(
                    f"{code}_UNVERIFIED",
                    f"relation {relation.kind!r} could not be verified for roles "
                    f"{list(relation.roles)!r}",
                )
            )

    return _stable_issues(issues), _stable_issues(unverified)


__all__ = [
    "describe_value",
    "materialize_input_contracts",
    "validate_input_contracts",
]
