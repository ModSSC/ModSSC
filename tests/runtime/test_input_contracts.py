from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import modssc.runtime.input_contracts as input_contracts_module
from modssc.runtime.contracts import (
    InputRoleRequirement,
    RoleRelation,
    ValueDescriptor,
)
from modssc.runtime.input_contracts import (
    describe_value,
    materialize_input_contracts,
    validate_input_contracts,
)


class _MetadataOnlyArray:
    shape = (1_000_000, 4)
    dtype = np.dtype("float32")

    def __array__(self) -> np.ndarray:
        raise AssertionError("describing an input must not call np.asarray")


class _ShapeWithoutMetadata:
    shape = None
    dtype = None


class _BadShape:
    dtype = None

    @property
    def shape(self) -> object:
        class _NotIterable:
            def __iter__(self) -> object:
                raise TypeError("symbolic shape")

        return _NotIterable()


class _BadDimension:
    def __int__(self) -> int:
        raise ValueError("symbolic dimension")


class _SymbolicShape:
    shape = (_BadDimension(), -1)
    dtype = None


class _NamedDType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class _NamedDTypeArray:
    shape = (2,)

    def __init__(self, dtype_name: str) -> None:
        self.dtype = _NamedDType(dtype_name)


class _GraphObject:
    n_nodes = 3
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_weight = None

    @property
    def x(self) -> object:
        raise RuntimeError("features are intentionally unavailable")


def _by_role(
    provisions: tuple[tuple[str, ValueDescriptor], ...],
) -> dict[str, ValueDescriptor]:
    return dict(provisions)


def _codes(values: tuple[object, ...]) -> list[str]:
    return [value.code for value in values]  # type: ignore[attr-defined]


def test_describe_value_uses_array_metadata_without_coercion() -> None:
    descriptor = describe_value(_MetadataOnlyArray())

    assert descriptor.representation == "dense"
    assert descriptor.shape == (1_000_000, 4)
    assert descriptor.rank == 2
    assert descriptor.rows == 1_000_000
    assert descriptor.dtypes == frozenset({"float32"})
    assert descriptor.dtype_kinds == frozenset({"float"})


def test_describe_numpy_and_scipy_sparse() -> None:
    sparse = pytest.importorskip("scipy.sparse")

    dense_descriptor = describe_value(np.zeros((3, 2), dtype=np.int64))
    sparse_descriptor = describe_value(sparse.csr_matrix((3, 2), dtype=np.float32))

    assert dense_descriptor.to_dict() == {
        "representation": "dense",
        "container_backends": ["numpy"],
        "dtypes": ["int64"],
        "dtype_kinds": ["integer"],
        "devices": ["cpu"],
        "rank": 2,
        "shape": [3, 2],
        "rows": 3,
        "schema": {},
    }
    assert sparse_descriptor.representation == "sparse"
    assert sparse_descriptor.container_backends == frozenset({"scipy"})
    assert sparse_descriptor.shape == (3, 2)
    assert sparse_descriptor.dtype_kinds == frozenset({"float"})


def test_describe_torch_dense_and_sparse_without_transferring() -> None:
    torch = pytest.importorskip("torch")
    dense = torch.zeros((4, 3), dtype=torch.float64)
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([1.0, 2.0]),
        size=(2, 2),
        check_invariants=False,
    )

    dense_descriptor = describe_value(dense)
    sparse_descriptor = describe_value(sparse)

    assert dense_descriptor.representation == "dense"
    assert dense_descriptor.container_backends == frozenset({"torch"})
    assert dense_descriptor.devices == frozenset({"cpu"})
    assert dense_descriptor.dtypes == frozenset({"float64"})
    assert sparse_descriptor.representation == "sparse"
    assert sparse_descriptor.shape == (2, 2)


def test_describe_structured_token_graph_and_mixed_mappings() -> None:
    torch = pytest.importorskip("torch")
    tokens = {
        "input_ids": torch.ones((5, 8), dtype=torch.int64),
        "attention_mask": torch.ones((5, 8), dtype=torch.bool),
    }
    graph = {
        "n_nodes": 5,
        "edge_index": np.zeros((2, 3), dtype=np.int64),
        "x": np.ones((5, 4), dtype=np.float32),
    }
    mixed = {
        "numpy": np.ones((2, 2), dtype=np.float32),
        "torch": torch.ones((2, 2), dtype=torch.float32),
    }

    token_descriptor = describe_value(tokens)
    graph_descriptor = describe_value(graph)
    mixed_descriptor = describe_value(mixed)

    assert token_descriptor.representation == "tokens"
    assert token_descriptor.rows == 5
    assert token_descriptor.rank == 2
    assert [name for name, _ in token_descriptor.schema] == [
        "attention_mask",
        "input_ids",
    ]
    assert token_descriptor.dtype_kinds == frozenset({"bool", "integer"})
    assert graph_descriptor.representation == "graph"
    assert graph_descriptor.rows == 5
    assert graph_descriptor.container_backends == frozenset({"numpy"})
    assert mixed_descriptor.representation == "mixed"
    assert mixed_descriptor.container_backends == frozenset({"numpy", "torch"})
    assert list(mixed_descriptor.to_dict()["schema"]) == ["numpy", "torch"]


def test_describe_nested_sequences_without_numpy_materialization() -> None:
    descriptor = describe_value([[1, 2], [3, 4]])
    text_descriptor = describe_value(("one", "two"))

    assert descriptor.representation == "sequence"
    assert descriptor.container_backends == frozenset({"python"})
    assert descriptor.dtype_kinds == frozenset({"integer"})
    assert descriptor.rank == 2
    assert descriptor.shape == (2, 2)
    assert descriptor.rows == 2
    assert text_descriptor.representation == "sequence"
    assert text_descriptor.dtype_kinds == frozenset({"string"})
    assert text_descriptor.shape == (2,)


@pytest.mark.parametrize(
    ("value", "representation", "dtype_kind"),
    [
        (np.int16(4), "scalar", "integer"),
        (None, "none", "unknown"),
        (True, "scalar", "bool"),
        (1.5, "scalar", "float"),
        (1 + 2j, "scalar", "complex"),
        (b"bytes", "bytes", "string"),
        (object(), "objects", "object"),
    ],
)
def test_describe_scalar_kinds(value: object, representation: str, dtype_kind: str) -> None:
    descriptor = describe_value(value)

    assert descriptor.representation == representation
    assert descriptor.dtype_kinds == frozenset({dtype_kind})
    assert descriptor.rank == 0
    assert descriptor.shape == ()


def test_describe_unknown_and_symbolic_shape_metadata() -> None:
    missing = describe_value(_ShapeWithoutMetadata())
    bad_shape = describe_value(_BadShape())
    symbolic = describe_value(_SymbolicShape())
    scalar_array = describe_value(np.array(1.0, dtype=np.float32))

    assert missing.shape is None
    assert missing.rank is None
    assert missing.dtypes == frozenset()
    assert bad_shape.shape is None
    assert symbolic.shape == (None, None)
    assert symbolic.rows is None
    assert scalar_array.rows is None


def test_describe_unknown_dtype_names_without_assuming_numeric_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_dtype = np.dtype

    def reject_named_dtype(value: object) -> np.dtype:
        if isinstance(value, str):
            raise TypeError("not a NumPy dtype")
        return original_dtype(value)

    monkeypatch.setattr(input_contracts_module.np, "dtype", reject_named_dtype)
    expected = {
        "bool": "bool",
        "complex_custom": "complex",
        "bfloat_custom": "float",
        "qint_custom": "integer",
        "string_custom": "string",
        "object_custom": "object",
        "opaque": "unknown",
    }

    for dtype_name, dtype_kind in expected.items():
        descriptor = describe_value(_NamedDTypeArray(dtype_name))
        assert descriptor.dtypes == frozenset({dtype_name})
        assert descriptor.dtype_kinds == frozenset({dtype_kind})


def test_describe_numpy_object_empty_and_heterogeneous_sequences() -> None:
    objects = describe_value(np.array(["a", "b"], dtype=object))
    empty = describe_value([])
    unequal_rank = describe_value(
        [np.ones((2,), dtype=np.float32), np.ones((2, 1), dtype=np.float32)]
    )
    unequal_rows = describe_value(
        [np.ones((2,), dtype=np.float32), np.ones((3,), dtype=np.float32)]
    )
    unknown_shape = describe_value([_ShapeWithoutMetadata(), _ShapeWithoutMetadata()])

    assert objects.representation == "objects"
    assert empty.shape == (0,)
    assert empty.rows == 0
    assert unequal_rank.rank is None
    assert unequal_rank.shape is None
    assert unequal_rows.shape == (2, None)
    assert dict(unequal_rows.schema)["item"].rows is None
    assert unknown_shape.shape is None


def test_describe_generic_empty_graph_and_token_mappings() -> None:
    structured = describe_value({"a": np.ones((2, 2), dtype=np.float32)})
    empty = describe_value({})
    graph_count = describe_value({"n_nodes": 4})
    graph_unknown_count = describe_value({"n_nodes": True})
    graph_negative_count = describe_value(
        {"n_nodes": -1, "edge_index": np.empty((2, 0), dtype=np.int64)}
    )
    tokens_without_primary = describe_value(
        {
            "attention_mask": np.ones((2, 4), dtype=np.int64),
            "token_type_ids": np.ones((2, 4), dtype=np.int64),
        }
    )

    assert structured.representation == "structured"
    assert empty.representation == "structured"
    assert empty.container_backends == frozenset()
    assert graph_count.representation == "graph"
    assert graph_count.rows == 4
    assert graph_unknown_count.rows is None
    assert graph_negative_count.rows is None
    assert tokens_without_primary.representation == "tokens"
    assert tokens_without_primary.rows is None


def test_describe_graph_object_ranges_and_recursive_containers() -> None:
    graph = describe_value(_GraphObject())
    integer_range = describe_value(range(5))
    recursive_mapping: dict[str, object] = {}
    recursive_mapping["self"] = recursive_mapping
    recursive_sequence: list[object] = []
    recursive_sequence.append(recursive_sequence)
    recursive_graph = SimpleNamespace(n_nodes=1)
    recursive_graph.edge_index = recursive_graph

    mapping = describe_value(recursive_mapping)
    sequence = describe_value(recursive_sequence)
    graph_cycle = describe_value(recursive_graph)

    assert graph.representation == "graph"
    assert graph.rows == 3
    assert integer_range.shape == (5,)
    assert dict(mapping.schema)["self"].representation == "objects"
    assert dict(sequence.schema)["item"].representation == "sequence"
    assert dict(graph_cycle.schema)["edge_index"].representation == "graph"


def test_describe_rejects_non_string_structured_keys() -> None:
    with pytest.raises(TypeError, match="mapping keys"):
        describe_value({1: np.ones((1, 1), dtype=np.float32)})


def test_materialize_inductive_roles_views_augmentations_and_graph() -> None:
    consumed = SimpleNamespace(
        X_l=np.ones((2, 3), dtype=np.float32),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u=np.ones((3, 3), dtype=np.float32),
        X_u_w=np.ones((3, 3), dtype=np.float32),
        X_u_s=np.full((3, 3), 2.0, dtype=np.float32),
        X_u_s_1=None,
        views={
            "z_view": {
                "X_l": np.ones((2, 1), dtype=np.float32),
                "X_u": np.ones((3, 1), dtype=np.float32),
            },
            "a_view": {
                "X_l": {"input_ids": np.ones((2, 4), dtype=np.int64)},
                "X_u": {"input_ids": np.ones((3, 4), dtype=np.int64)},
            },
            "X_u_s1": np.full((3, 3), 3.0, dtype=np.float32),
            "labeled_strong": np.full((2, 3), 4.0, dtype=np.float32),
        },
        graph={
            "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
            "edge_weight": np.ones(2, dtype=np.float32),
            "n_nodes": 5,
        },
    )

    provisions = materialize_input_contracts(regime="inductive", consumed_input=consumed)
    described = _by_role(provisions)

    assert list(described) == sorted(described)
    assert set(described) == {
        "fit.X_l",
        "fit.X_l_s.0",
        "fit.X_u",
        "fit.X_u_s.0",
        "fit.X_u_s.1",
        "fit.X_u_w",
        "fit.graph.edge_index",
        "fit.graph.edge_weight",
        "fit.graph.n_nodes",
        "fit.views.a_view.X_l",
        "fit.views.a_view.X_u",
        "fit.views.z_view.X_l",
        "fit.views.z_view.X_u",
        "fit.y_l",
    }
    assert described["fit.X_u_s.1"].rows == 3
    assert described["fit.X_l_s.0"].rows == 2
    assert described["fit.views.a_view.X_l"].representation == "tokens"
    assert described["fit.graph.n_nodes"].rows == 5


def test_materialize_transductive_unwraps_fit_and_expands_masks() -> None:
    fit = SimpleNamespace(
        X=np.ones((4, 2), dtype=np.float32),
        y=np.array([0, -1, 1, -1], dtype=np.int64),
        masks={
            "unlabeled_mask": np.array([False, True, False, True]),
            "labeled_mask": np.array([True, False, True, False]),
        },
        graph=SimpleNamespace(
            n_nodes=4,
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_weight=None,
        ),
    )

    provisions = materialize_input_contracts(
        regime="transductive", consumed_input=SimpleNamespace(fit=fit)
    )
    described = _by_role(provisions)

    assert set(described) == {
        "fit.X",
        "fit.y",
        "fit.masks.labeled_mask",
        "fit.masks.unlabeled_mask",
        "fit.graph.edge_index",
        "fit.graph.n_nodes",
    }
    assert described["fit.masks.unlabeled_mask"].dtype_kinds == frozenset({"bool"})
    assert described["fit.graph.n_nodes"].rows == 4


def test_materialize_mapping_input_and_attribute_view_payload() -> None:
    view = SimpleNamespace(
        X_l=np.ones((1, 2), dtype=np.float32),
        X_u=np.ones((2, 2), dtype=np.float32),
    )
    consumed = {
        "X_l": np.ones((1, 2), dtype=np.float32),
        "y_l": np.array([0], dtype=np.int64),
        "X_u_s_1": np.ones((2, 2), dtype=np.float32),
        "views": {"attribute_view": view},
    }

    described = _by_role(materialize_input_contracts(regime="inductive", consumed_input=consumed))

    assert described["fit.views.attribute_view.X_l"].rows == 1
    assert described["fit.views.attribute_view.X_u"].rows == 2
    assert described["fit.X_u_s.1"].rows == 2


def test_materialize_absent_optional_structures_and_reserved_view_payload() -> None:
    scientific_alias = {
        "X_l": np.ones((1, 1), dtype=np.float32),
        "X_u": np.ones((2, 1), dtype=np.float32),
    }
    inductive = SimpleNamespace(
        X_l=np.ones((1, 1), dtype=np.float32),
        y_l=np.array([0]),
        views={"X_u_s": scientific_alias},
        graph=None,
    )
    transductive = SimpleNamespace(
        X=np.ones((1, 1), dtype=np.float32),
        y=np.array([0]),
        masks=None,
        graph=None,
    )

    inductive_roles = _by_role(
        materialize_input_contracts(regime="inductive", consumed_input=inductive)
    )
    transductive_roles = _by_role(
        materialize_input_contracts(regime="transductive", consumed_input=transductive)
    )

    assert "fit.X_u_s.0" not in inductive_roles
    assert set(transductive_roles) == {"fit.X", "fit.y"}


@pytest.mark.parametrize(
    ("regime", "consumed", "message"),
    [
        (
            "inductive",
            SimpleNamespace(views={1: {"X_l": np.ones((1, 1))}}),
            "view names",
        ),
        (
            "inductive",
            SimpleNamespace(graph={1: np.ones((1, 1))}),
            "graph mapping keys",
        ),
        (
            "transductive",
            SimpleNamespace(masks=[]),
            "masks must be a mapping",
        ),
        (
            "transductive",
            SimpleNamespace(masks={1: np.ones(1, dtype=bool)}),
            "mask names",
        ),
    ],
)
def test_materialize_rejects_non_canonical_dynamic_structures(
    regime: str, consumed: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        materialize_input_contracts(
            regime=regime,  # type: ignore[arg-type]
            consumed_input=consumed,
        )


def test_materialization_rejects_unknown_regime_and_unstructured_views() -> None:
    with pytest.raises(ValueError, match="regime"):
        materialize_input_contracts(regime="invalid", consumed_input=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="views must be a mapping"):
        materialize_input_contracts(
            regime="inductive",
            consumed_input=SimpleNamespace(views=[]),
        )


def test_validate_requirements_reports_all_issues_and_unknown_metadata() -> None:
    provisions = {
        "fit.X_l": describe_value(np.empty((0, 3), dtype=np.float32)),
        "fit.opaque": ValueDescriptor(representation="dense"),
    }
    requirements = (
        InputRoleRequirement(
            role="fit.X_l",
            representations=frozenset({"tokens"}),
            container_backends=frozenset({"torch"}),
            dtype_kinds=frozenset({"integer"}),
            dtypes=frozenset({"int64"}),
            ranks=frozenset({4}),
            numeric=False,
        ),
        InputRoleRequirement(
            role="fit.opaque",
            container_backends=frozenset({"numpy"}),
            dtype_kinds=frozenset({"float"}),
            dtypes=frozenset({"float32"}),
            ranks=frozenset({2}),
            numeric=True,
        ),
        InputRoleRequirement(role="fit.missing"),
        InputRoleRequirement(role="fit.optional", optional=True),
    )

    issues, unverified = validate_input_contracts(requirements, (), provisions)

    assert _codes(issues) == sorted(
        [
            "E_INPUT_BACKEND",
            "E_INPUT_DTYPE",
            "E_INPUT_DTYPE_KIND",
            "E_INPUT_NON_EMPTY",
            "E_INPUT_NUMERIC",
            "E_INPUT_RANK",
            "E_INPUT_REPRESENTATION",
            "E_INPUT_ROLE_MISSING",
        ]
    )
    assert _codes(unverified) == sorted(
        [
            "E_INPUT_BACKEND_UNVERIFIED",
            "E_INPUT_DTYPE_KIND_UNVERIFIED",
            "E_INPUT_DTYPE_UNVERIFIED",
            "E_INPUT_NON_EMPTY_UNVERIFIED",
            "E_INPUT_NUMERIC_UNVERIFIED",
            "E_INPUT_RANK_UNVERIFIED",
        ]
    )


def test_validate_requirements_is_deterministic_across_declaration_order() -> None:
    requirements = (
        InputRoleRequirement(role="b"),
        InputRoleRequirement(role="a", ranks=frozenset({1})),
    )
    provisions = (("a", describe_value(np.ones((2, 2), dtype=np.float32))),)

    forward = validate_input_contracts(requirements, (), provisions)
    reversed_order = validate_input_contracts(reversed(requirements), (), reversed(provisions))

    assert forward == reversed_order


def test_validate_compatible_and_scalar_requirements() -> None:
    provisions = {
        "dense": describe_value(np.ones((2, 2), dtype=np.float32)),
        "scalar": describe_value(3),
        "text": describe_value("value"),
        "unknown": ValueDescriptor(representation="dense"),
    }
    requirements = (
        InputRoleRequirement(
            role="dense",
            representations=frozenset({"dense"}),
            container_backends=frozenset({"numpy"}),
            dtype_kinds=frozenset({"float"}),
            dtypes=frozenset({"float32"}),
            ranks=frozenset({2}),
            numeric=True,
        ),
        InputRoleRequirement(role="scalar", ranks=frozenset({0}), numeric=True),
        InputRoleRequirement(role="text", numeric=False),
        InputRoleRequirement(role="unknown", non_empty=False),
    )

    assert validate_input_contracts(requirements, (), provisions) == ((), ())


@pytest.mark.parametrize(
    ("requirements", "relations", "provisions", "error", "message"),
    [
        ((), (), (("", ValueDescriptor("dense")),), TypeError, "roles"),
        ((), (), (("a", object()),), TypeError, "ValueDescriptor"),
        (
            (),
            (),
            (("a", ValueDescriptor("dense")), ("a", ValueDescriptor("dense"))),
            ValueError,
            "duplicate input provision",
        ),
        ((object(),), (), (), TypeError, "InputRoleRequirement"),
        ((), (object(),), (), TypeError, "RoleRelation"),
        (
            (InputRoleRequirement("a"), InputRoleRequirement("a")),
            (),
            (),
            ValueError,
            "duplicate input requirement",
        ),
    ],
)
def test_validate_rejects_malformed_contract_collections(
    requirements: object,
    relations: object,
    provisions: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        validate_input_contracts(
            requirements,  # type: ignore[arg-type]
            relations,  # type: ignore[arg-type]
            provisions,  # type: ignore[arg-type]
        )


def test_validate_all_supported_relations() -> None:
    provisions = {
        "a": describe_value(np.ones((2, 3), dtype=np.float32)),
        "b": describe_value(np.zeros((2, 3), dtype=np.float32)),
    }
    relations = tuple(
        RoleRelation(kind=kind, roles=("a", "b"))
        for kind in (
            "same_rows",
            "same_backend",
            "same_device",
            "same_dtype_kind",
            "same_input_schema",
            "concat_compatible",
            "mix_compatible",
        )
    )

    assert validate_input_contracts((), relations, provisions) == ((), ())


def test_backend_and_device_relations_use_structured_primary_payloads() -> None:
    primary = ValueDescriptor(
        representation="dense",
        container_backends=frozenset({"torch"}),
        devices=frozenset({"cuda:0"}),
        rank=2,
        shape=(3, 4),
        rows=3,
    )
    auxiliary = ValueDescriptor(
        representation="dense",
        container_backends=frozenset({"numpy"}),
        devices=frozenset({"cpu"}),
        rank=2,
        shape=(2, 5),
        rows=2,
    )
    graph = ValueDescriptor(
        representation="graph",
        container_backends=frozenset({"numpy", "torch"}),
        devices=frozenset({"cpu", "cuda:0"}),
        rows=3,
        schema=(("edge_index", auxiliary), ("x", primary)),
    )
    tokens = ValueDescriptor(
        representation="tokens",
        container_backends=frozenset({"torch"}),
        devices=frozenset({"cuda:0"}),
        rows=3,
        schema=(("attention_mask", primary), ("input_ids", primary)),
    )
    relations = (
        RoleRelation("same_backend", ("graph", "tokens")),
        RoleRelation("same_device", ("graph", "tokens")),
    )

    assert validate_input_contracts((), relations, {"graph": graph, "tokens": tokens}) == ((), ())


def test_validate_relations_distinguishes_incompatible_and_unverified() -> None:
    base = describe_value(np.ones((2, 3), dtype=np.float32))
    provisions = {
        "base": base,
        "rows": describe_value(np.ones((3, 3), dtype=np.float32)),
        "backend": replace_descriptor(base, container_backends=frozenset({"torch"})),
        "device": replace_descriptor(base, devices=frozenset({"cuda:0"})),
        "dtype_kind": describe_value(np.ones((2, 3), dtype=np.int64)),
        "schema": describe_value(np.ones((2, 4), dtype=np.float32)),
        "concat": describe_value(np.ones((2, 3), dtype=np.float64)),
        "mix": describe_value(np.ones((2, 3), dtype=np.int64)),
        "unknown": ValueDescriptor(representation="dense"),
    }
    relations = (
        RoleRelation(kind="same_rows", roles=("base", "rows")),
        RoleRelation(kind="same_backend", roles=("base", "backend")),
        RoleRelation(kind="same_device", roles=("base", "device")),
        RoleRelation(kind="same_dtype_kind", roles=("base", "dtype_kind")),
        RoleRelation(kind="same_input_schema", roles=("base", "schema")),
        RoleRelation(kind="concat_compatible", roles=("base", "concat")),
        RoleRelation(kind="mix_compatible", roles=("base", "mix")),
        RoleRelation(kind="same_rows", roles=("base", "unknown")),
        RoleRelation(kind="same_backend", roles=("base", "absent")),
    )

    issues, unverified = validate_input_contracts((), relations, provisions)

    assert _codes(issues) == sorted(
        [
            "E_INPUT_RELATION_CONCAT_COMPATIBLE",
            "E_INPUT_RELATION_MIX_COMPATIBLE",
            "E_INPUT_RELATION_SAME_BACKEND",
            "E_INPUT_RELATION_SAME_DEVICE",
            "E_INPUT_RELATION_SAME_DTYPE_KIND",
            "E_INPUT_RELATION_SAME_INPUT_SCHEMA",
            "E_INPUT_RELATION_SAME_ROWS",
        ]
    )
    assert _codes(unverified) == sorted(
        [
            "E_INPUT_RELATION_SAME_BACKEND_UNVERIFIED",
            "E_INPUT_RELATION_SAME_ROWS_UNVERIFIED",
        ]
    )


def test_relation_shape_and_metadata_unknown_branches() -> None:
    base = describe_value(np.ones((2, 3), dtype=np.float32))
    rank_mismatch = ValueDescriptor(
        representation="dense",
        container_backends=base.container_backends,
        dtypes=base.dtypes,
        dtype_kinds=base.dtype_kinds,
        devices=base.devices,
        rank=1,
        shape=(2,),
        rows=2,
    )
    missing_rank = ValueDescriptor(
        representation="dense",
        container_backends=base.container_backends,
        dtypes=base.dtypes,
        dtype_kinds=base.dtype_kinds,
        devices=base.devices,
        rows=2,
    )
    missing_shape = ValueDescriptor(
        representation="dense",
        container_backends=base.container_backends,
        dtypes=base.dtypes,
        dtype_kinds=base.dtype_kinds,
        devices=base.devices,
        rank=2,
        rows=2,
    )
    inconsistent_shape_length = ValueDescriptor(
        representation="dense",
        container_backends=base.container_backends,
        dtypes=base.dtypes,
        dtype_kinds=base.dtype_kinds,
        devices=base.devices,
        rank=2,
        shape=(2,),
        rows=2,
    )
    symbolic_shape = ValueDescriptor(
        representation="dense",
        container_backends=base.container_backends,
        dtypes=base.dtypes,
        dtype_kinds=base.dtype_kinds,
        devices=base.devices,
        rank=2,
        shape=(2, None),
        rows=2,
    )
    different_representation = ValueDescriptor(
        representation="sparse",
        container_backends=base.container_backends,
        dtypes=base.dtypes,
        dtype_kinds=base.dtype_kinds,
        devices=base.devices,
        rank=2,
        shape=(2, 3),
        rows=2,
    )
    empty_structured = ValueDescriptor(representation="structured", rows=2)
    no_device = ValueDescriptor(
        representation="dense",
        container_backends=frozenset({"python"}),
        dtypes=frozenset({"float"}),
        dtype_kinds=frozenset({"float"}),
        rank=2,
        shape=(2, 3),
        rows=2,
    )
    no_dtype = ValueDescriptor(
        representation="dense",
        container_backends=frozenset({"numpy"}),
        devices=frozenset({"cpu"}),
        rank=2,
        shape=(2, 3),
        rows=2,
    )
    provisions = {
        "base": base,
        "rank": rank_mismatch,
        "missing_rank": missing_rank,
        "missing_shape": missing_shape,
        "length": inconsistent_shape_length,
        "symbolic": symbolic_shape,
        "representation": different_representation,
        "empty_a": empty_structured,
        "empty_b": empty_structured,
        "no_device_a": no_device,
        "no_device_b": no_device,
        "no_dtype": no_dtype,
    }
    relations = (
        RoleRelation("same_input_schema", ("base", "rank")),
        RoleRelation("same_input_schema", ("base", "missing_rank")),
        RoleRelation("same_input_schema", ("base", "missing_shape")),
        RoleRelation("same_input_schema", ("base", "length")),
        RoleRelation("same_input_schema", ("base", "symbolic")),
        RoleRelation("same_input_schema", ("base", "representation")),
        RoleRelation("same_input_schema", ("empty_a", "empty_b")),
        RoleRelation("concat_compatible", ("base", "representation")),
        RoleRelation("concat_compatible", ("empty_a", "empty_b")),
        RoleRelation("concat_compatible", ("no_device_a", "no_device_b")),
        RoleRelation("concat_compatible", ("base", "no_dtype")),
        RoleRelation("same_backend", ("base", "missing_rank")),
        RoleRelation("same_backend", ("base", "unknown_backend")),
    )
    provisions["unknown_backend"] = ValueDescriptor(representation="dense", rows=2)

    issues, unverified = validate_input_contracts((), relations, provisions)

    assert "E_INPUT_RELATION_SAME_INPUT_SCHEMA" in _codes(issues)
    assert "E_INPUT_RELATION_CONCAT_COMPATIBLE" in _codes(issues)
    assert "E_INPUT_RELATION_SAME_INPUT_SCHEMA_UNVERIFIED" in _codes(unverified)
    assert "E_INPUT_RELATION_CONCAT_COMPATIBLE_UNVERIFIED" in _codes(unverified)
    assert "E_INPUT_RELATION_SAME_BACKEND_UNVERIFIED" in _codes(unverified)


def test_mix_relation_rejects_nonnumeric_and_integer_and_can_be_unverified() -> None:
    text_a = describe_value(["a", "b"])
    text_b = describe_value(["c", "d"])
    integer_a = describe_value(np.ones((2, 2), dtype=np.int64))
    integer_b = describe_value(np.zeros((2, 2), dtype=np.int64))
    unknown_a = ValueDescriptor(
        representation="dense",
        container_backends=frozenset({"numpy"}),
        devices=frozenset({"cpu"}),
        rank=2,
        shape=(2, 2),
        rows=2,
    )
    unknown_b = replace_descriptor(unknown_a)
    relations = (
        RoleRelation("mix_compatible", ("text_a", "text_b")),
        RoleRelation("mix_compatible", ("integer_a", "integer_b")),
        RoleRelation("mix_compatible", ("unknown_a", "unknown_b")),
    )
    provisions = {
        "text_a": text_a,
        "text_b": text_b,
        "integer_a": integer_a,
        "integer_b": integer_b,
        "unknown_a": unknown_a,
        "unknown_b": unknown_b,
    }

    issues, unverified = validate_input_contracts((), relations, provisions)

    assert _codes(issues) == [
        "E_INPUT_RELATION_MIX_COMPATIBLE",
        "E_INPUT_RELATION_MIX_COMPATIBLE",
    ]
    assert _codes(unverified) == ["E_INPUT_RELATION_MIX_COMPATIBLE_UNVERIFIED"]


def test_missing_required_relation_role_only_reports_required_role_issue() -> None:
    requirements = (InputRoleRequirement("missing"),)
    relations = (RoleRelation("same_rows", ("present", "missing")),)
    provisions = {"present": describe_value(np.ones((1, 1), dtype=np.float32))}

    issues, unverified = validate_input_contracts(requirements, relations, provisions)

    assert _codes(issues) == ["E_INPUT_ROLE_MISSING"]
    assert unverified == ()


def test_optional_missing_role_suppresses_its_relation() -> None:
    requirements = (InputRoleRequirement(role="optional", optional=True),)
    relations = (RoleRelation(kind="same_rows", roles=("present", "optional")),)
    provisions = {"present": describe_value(np.ones((2, 2), dtype=np.float32))}

    assert validate_input_contracts(requirements, relations, provisions) == ((), ())


def test_same_input_schema_handles_structured_keys_and_variable_batch_rows() -> None:
    left = describe_value(
        {
            "input_ids": np.ones((2, 8), dtype=np.int64),
            "attention_mask": np.ones((2, 8), dtype=np.int64),
        }
    )
    compatible = describe_value(
        {
            "attention_mask": np.ones((5, 8), dtype=np.int64),
            "input_ids": np.ones((5, 8), dtype=np.int64),
        }
    )
    incompatible = describe_value({"input_ids": np.ones((5, 8), dtype=np.int64)})
    relation = RoleRelation(kind="same_input_schema", roles=("left", "right"))

    assert validate_input_contracts(
        (), relation_tuple(relation), {"left": left, "right": compatible}
    ) == ((), ())
    issues, unverified = validate_input_contracts(
        (), relation_tuple(relation), {"left": left, "right": incompatible}
    )
    assert _codes(issues) == ["E_INPUT_RELATION_SAME_INPUT_SCHEMA"]
    assert unverified == ()


def test_concat_compatible_recurses_through_structured_schema() -> None:
    left = describe_value(
        {
            "input_ids": np.ones((2, 8), dtype=np.int64),
            "attention_mask": np.ones((2, 8), dtype=np.int64),
        }
    )
    compatible = describe_value(
        {
            "attention_mask": np.ones((5, 8), dtype=np.int64),
            "input_ids": np.ones((5, 8), dtype=np.int64),
        }
    )
    missing_key = describe_value({"input_ids": np.ones((5, 8), dtype=np.int64)})
    relation = RoleRelation("concat_compatible", ("left", "right"))

    assert validate_input_contracts((), (relation,), {"left": left, "right": compatible}) == (
        (),
        (),
    )
    issues, unverified = validate_input_contracts(
        (), (relation,), {"left": left, "right": missing_key}
    )
    assert _codes(issues) == ["E_INPUT_RELATION_CONCAT_COMPATIBLE"]
    assert unverified == ()


def replace_descriptor(descriptor: ValueDescriptor, **changes: object) -> ValueDescriptor:
    payload = descriptor.to_dict()
    payload.update(changes)
    return ValueDescriptor(
        representation=str(payload["representation"]),
        container_backends=frozenset(payload["container_backends"]),
        dtypes=frozenset(payload["dtypes"]),
        dtype_kinds=frozenset(payload["dtype_kinds"]),
        devices=frozenset(payload["devices"]),
        rank=payload["rank"],  # type: ignore[arg-type]
        shape=(
            None if payload["shape"] is None else tuple(payload["shape"])  # type: ignore[arg-type]
        ),
        rows=payload["rows"],  # type: ignore[arg-type]
        schema=descriptor.schema,
    )


def relation_tuple(relation: RoleRelation) -> tuple[RoleRelation, ...]:
    return (relation,)
