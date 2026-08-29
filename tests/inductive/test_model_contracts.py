from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest

from modssc.inductive.deep import TorchModelBundle, build_torch_bundle_from_classifier
from modssc.inductive.methods.simclr_v2 import SimCLRv2Method, SimCLRv2Spec
from modssc.inductive.model_binding import (
    ModelBindingSpec,
    ModelBuildConfig,
    bind_model_to_spec,
)
from modssc.inductive.model_contracts import (
    resolve_bound_component_contracts,
    validate_component_contracts,
)
from modssc.runtime.contracts import (
    ComponentProvision,
    ComponentRelation,
    ComponentRequirement,
    ModelContract,
    ValueDescriptor,
)
from modssc.runtime.method_contracts import resolve_method_execution_contract

torch = pytest.importorskip("torch")


@dataclass(frozen=True)
class _SingleSpec:
    model_bundle: TorchModelBundle | None = None


@dataclass(frozen=True)
class _PairSpec:
    model_bundle_1: TorchModelBundle | None = None
    model_bundle_2: TorchModelBundle | None = None


@dataclass(frozen=True)
class _AuxiliarySpec:
    model_bundle: TorchModelBundle | None = None
    student_bundle: TorchModelBundle | None = None


@dataclass(frozen=True)
class _SharedHeadsSpec:
    shared_bundle: TorchModelBundle | None = None
    head_bundles: tuple[TorchModelBundle, ...] | None = None


def _native_mlp(*, return_features: bool = False, ema: bool = False) -> TorchModelBundle:
    return build_torch_bundle_from_classifier(
        classifier_id="mlp",
        classifier_backend="torch",
        classifier_params={
            "hidden_sizes": (4,),
            "dropout": 0.0,
            "return_features": return_features,
        },
        sample=torch.randn(3, 2),
        num_classes=2,
        seed=7,
        ema=ema,
    )


def _codes(values: tuple[Any, ...]) -> set[str]:
    return {value.code for value in values}


def test_torch_model_bundle_preserves_old_positional_surface() -> None:
    bundle = TorchModelBundle("model", "optimizer", None, None, None, {"key": "value"})

    assert bundle.meta == {"key": "value"}
    assert bundle.contract is None


def test_native_mlp_contract_declares_real_hooks_but_not_mapping_feat() -> None:
    plain = _native_mlp(return_features=False)
    featured = _native_mlp(return_features=True)

    assert plain.contract is not None
    assert plain.contract.outputs == frozenset({"forward_features", "forward_head", "logits"})
    assert plain.contract.input_representations == frozenset({"dense"})
    assert plain.contract.input_dtype_kinds == frozenset({"float"})
    assert plain.contract.input_ranks == frozenset({2})
    assert featured.contract is not None
    assert featured.contract.outputs == frozenset(
        {"feat", "forward_features", "forward_head", "logits"}
    )

    provisions = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=plain),
        ModelBindingSpec.single(),
    )
    issues, unverified = validate_component_contracts(
        (
            ComponentRequirement(
                slot="model_bundle",
                kind="torch_model",
                outputs=frozenset({"feat", "logits"}),
                requires_optimizer=True,
            ),
        ),
        (),
        provisions,
    )

    assert _codes(issues) == {"E_COMPONENT_OUTPUT_MISSING"}
    assert unverified == ()


def test_missing_required_ema_is_a_proven_incompatibility() -> None:
    provisions = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=_native_mlp(ema=False)),
        ModelBindingSpec.single(),
    )

    issues, unverified = validate_component_contracts(
        (
            ComponentRequirement(
                slot="model_bundle",
                kind="torch_model",
                outputs=frozenset({"logits"}),
                requires_optimizer=True,
                requires_ema=True,
            ),
        ),
        (),
        provisions,
    )

    assert _codes(issues) == {"E_COMPONENT_EMA_MISSING"}
    assert unverified == ()


def test_ema_model_must_be_a_distinct_parameter_copy() -> None:
    model = torch.nn.Linear(2, 2)
    alias = TorchModelBundle(
        model=model,
        optimizer=object(),
        ema_model=model,
        contract=ModelContract(outputs=frozenset({"logits"}), source="test"),
    )
    provisions = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=alias),
        ModelBindingSpec.single(),
    )

    issues, unverified = validate_component_contracts(
        (
            ComponentRequirement(
                slot="model_bundle",
                kind="torch_model",
                requires_ema=True,
            ),
        ),
        (),
        provisions,
    )

    assert _codes(issues) == {
        "E_COMPONENT_EMA_OBJECT_ALIAS",
        "E_COMPONENT_EMA_PARAMETERS_SHARED",
    }
    assert unverified == ()


def test_aliased_models_and_shared_parameters_are_reported_separately() -> None:
    bundle = _native_mlp()
    alias_provisions = resolve_bound_component_contracts(
        _PairSpec(model_bundle_1=bundle, model_bundle_2=bundle),
        ModelBindingSpec.pair(),
    )
    relations = (
        ComponentRelation(
            kind="distinct_objects",
            slots=("model_bundle_1", "model_bundle_2"),
        ),
        ComponentRelation(
            kind="disjoint_parameters",
            slots=("model_bundle_1", "model_bundle_2"),
        ),
        ComponentRelation(
            kind="same_device",
            slots=("model_bundle_1", "model_bundle_2"),
        ),
    )

    alias_issues, alias_unverified = validate_component_contracts((), relations, alias_provisions)

    assert _codes(alias_issues) == {
        "E_COMPONENT_OBJECT_ALIAS",
        "E_COMPONENT_PARAMETERS_SHARED",
    }
    assert alias_unverified == ()

    shared_parameter = torch.nn.Parameter(torch.ones(2, 2))

    class _SharedParameterModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = shared_parameter

    explicit = ModelContract(outputs=frozenset({"logits"}), source="test")
    first = TorchModelBundle(_SharedParameterModel(), object(), contract=explicit)
    second = TorchModelBundle(_SharedParameterModel(), object(), contract=explicit)
    shared_provisions = resolve_bound_component_contracts(
        _PairSpec(model_bundle_1=first, model_bundle_2=second),
        ModelBindingSpec.pair(),
    )

    shared_issues, shared_unverified = validate_component_contracts(
        (), relations, shared_provisions
    )

    assert _codes(shared_issues) == {"E_COMPONENT_PARAMETERS_SHARED"}
    assert shared_unverified == ()


@pytest.mark.parametrize("prebound", [True, False])
def test_external_or_prebound_bundle_without_contract_remains_unverified(prebound: bool) -> None:
    class _NoProbeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.forward_calls = 0

        def forward(self, _value: Any) -> Any:
            self.forward_calls += 1
            raise AssertionError("contract resolution must not probe forward")

    model = _NoProbeModel()
    external = TorchModelBundle(model=model, optimizer=object())
    initial = _SingleSpec(model_bundle=external if prebound else None)
    bound = bind_model_to_spec(
        initial,
        None if prebound else ModelBuildConfig(factory=lambda: external),
        binding=ModelBindingSpec.single(),
        X_l=torch.randn(2, 2),
        y_l=torch.tensor([0, 1]),
        default_ema=False,
        seed=0,
    )

    provisions = resolve_bound_component_contracts(bound, ModelBindingSpec.single())
    issues, unverified = validate_component_contracts(
        (
            ComponentRequirement(
                slot="model_bundle",
                kind="torch_model",
                outputs=frozenset({"logits"}),
                requires_optimizer=True,
            ),
        ),
        (),
        provisions,
    )

    assert issues == ()
    assert _codes(unverified) == {"E_COMPONENT_CONTRACT_UNVERIFIED"}
    assert model.forward_calls == 0


def test_shared_head_slots_are_flattened_with_stable_names() -> None:
    shared = _native_mlp(return_features=True)
    heads = (_native_mlp(), _native_mlp(), _native_mlp())
    binding = ModelBindingSpec.shared_heads(
        head_count=3,
        head_classifier_ids=("mlp", "logreg"),
        head_classifier_fallback="logreg",
    )

    provisions = resolve_bound_component_contracts(
        _SharedHeadsSpec(shared_bundle=shared, head_bundles=heads),
        binding,
    )

    assert tuple(provision.slot for provision in provisions) == (
        "head_bundles[0]",
        "head_bundles[1]",
        "head_bundles[2]",
        "shared_bundle",
    )


def test_graphsage_contract_declares_graph_inputs_and_feature_output(monkeypatch) -> None:
    class _FakeSAGEConv(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_channels, out_channels)

        def forward(self, features: Any, _edge_index: Any) -> Any:
            return self.linear(features)

    package = types.ModuleType("torch_geometric")
    package.__path__ = []  # type: ignore[attr-defined]
    nn_module = types.ModuleType("torch_geometric.nn")
    nn_module.SAGEConv = _FakeSAGEConv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_geometric", package)
    monkeypatch.setitem(sys.modules, "torch_geometric.nn", nn_module)
    bundle = build_torch_bundle_from_classifier(
        classifier_id="graphsage_inductive",
        classifier_backend="torch",
        classifier_params={"hidden_channels": 4, "num_layers": 2},
        sample={
            "x": torch.randn(3, 2),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.int64),
        },
        num_classes=2,
        ema=False,
    )

    assert bundle.contract is not None
    assert bundle.contract.outputs == frozenset({"feat", "logits"})
    assert bundle.contract.input_representations == frozenset({"graph"})
    assert bundle.contract.input_dtype_kinds == frozenset({"float"})
    assert bundle.contract.input_ranks == frozenset({2})


def test_component_contract_payload_is_json_serializable_and_hides_identities() -> None:
    provisions = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=_native_mlp(return_features=True, ema=True)),
        ModelBindingSpec.single(),
    )

    payload = provisions[0].to_dict()
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)

    assert '"contract"' in serialized
    assert payload["contract"]["outputs"] == [
        "feat",
        "forward_features",
        "forward_features_ema",
        "forward_head",
        "forward_head_ema",
        "logits",
    ]
    assert "object_id" not in payload
    assert "parameter_ids" not in payload


def test_component_output_alternatives_are_any_of_groups() -> None:
    provisions = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=_native_mlp(return_features=False)),
        ModelBindingSpec.single(),
    )
    accepted = ComponentRequirement(
        slot="model_bundle",
        kind="torch_model",
        output_alternatives=(frozenset({"feat"}), frozenset({"logits"})),
    )
    rejected = ComponentRequirement(
        slot="model_bundle",
        kind="torch_model",
        output_alternatives=(frozenset({"feat"}), frozenset({"projection"})),
    )

    assert validate_component_contracts((accepted,), (), provisions) == ((), ())
    issues, unverified = validate_component_contracts((rejected,), (), provisions)

    assert _codes(issues) == {"E_COMPONENT_OUTPUT_ALTERNATIVE_MISSING"}
    assert unverified == ()


def test_auxiliary_bundle_fields_are_provisioned_outside_the_primary_binding() -> None:
    spec = _AuxiliarySpec(
        model_bundle=_native_mlp(),
        student_bundle=_native_mlp(),
    )

    provisions = resolve_bound_component_contracts(spec, ModelBindingSpec.single())

    assert [provision.slot for provision in provisions] == [
        "model_bundle",
        "student_bundle",
    ]


@pytest.mark.parametrize("primary_slot", ["pretrain_bundle", "finetune_bundle"])
def test_simclrv2_single_native_bundle_without_projection_is_incompatible(
    primary_slot: str,
) -> None:
    bundle = _native_mlp()
    spec = SimCLRv2Spec(**{primary_slot: bundle})
    contract = resolve_method_execution_contract(
        SimCLRv2Method,
        spec,
        SimCLRv2Method.info.capabilities,
        SimCLRv2Method.info.model_binding,
    )
    provisions = resolve_bound_component_contracts(
        spec,
        SimCLRv2Method.info.model_binding,
    )
    inputs = {
        role: ValueDescriptor(
            representation="dense",
            dtype_kinds=frozenset({"float"}),
            rank=2,
        )
        for role in ("fit.X_l", "fit.X_u")
    }

    issues, unverified = validate_component_contracts(
        contract.components,
        contract.component_relations,
        provisions,
        input_provisions=inputs,
        optional_input_roles=(
            requirement.role for requirement in contract.inputs if requirement.optional
        ),
    )

    assert [requirement.slot for requirement in contract.components] == [primary_slot]
    assert [provision.slot for provision in provisions] == [primary_slot]
    assert _codes(issues) == {"E_COMPONENT_OUTPUT_ALTERNATIVE_MISSING"}
    assert unverified == ()


def test_simclrv2_explicit_student_is_provisioned_and_composed() -> None:
    spec = SimCLRv2Spec(
        pretrain_epochs=0,
        finetune_epochs=0,
        distill_epochs=1,
        finetune_bundle=_native_mlp(),
        student_bundle=_native_mlp(),
    )
    contract = resolve_method_execution_contract(
        SimCLRv2Method,
        spec,
        SimCLRv2Method.info.capabilities,
        SimCLRv2Method.info.model_binding,
    )
    provisions = resolve_bound_component_contracts(
        spec,
        SimCLRv2Method.info.model_binding,
    )
    inputs = {
        role: ValueDescriptor(
            representation="dense",
            dtype_kinds=frozenset({"float"}),
            rank=2,
        )
        for role in ("fit.X_l", "fit.X_u")
    }

    issues, unverified = validate_component_contracts(
        contract.components,
        contract.component_relations,
        provisions,
        input_provisions=inputs,
        optional_input_roles=(
            requirement.role for requirement in contract.inputs if requirement.optional
        ),
    )

    assert [provision.slot for provision in provisions] == [
        "finetune_bundle",
        "student_bundle",
    ]
    assert issues == ()
    assert unverified == ()


def test_scheduler_requirement_validates_presence_and_exact_type() -> None:
    plain = _native_mlp()
    lambda_bundle = replace(
        plain,
        scheduler=torch.optim.lr_scheduler.LambdaLR(
            plain.optimizer,
            lr_lambda=lambda _step: 1.0,
        ),
    )
    step_bundle = replace(
        plain,
        scheduler=torch.optim.lr_scheduler.StepLR(plain.optimizer, step_size=1),
    )
    requirement = ComponentRequirement(
        slot="model_bundle",
        kind="torch_model",
        scheduler_types=frozenset({"LambdaLR"}),
    )

    accepted = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=lambda_bundle),
        ModelBindingSpec.single(),
    )
    missing = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=plain),
        ModelBindingSpec.single(),
    )
    wrong = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=step_bundle),
        ModelBindingSpec.single(),
    )

    assert accepted[0].has_scheduler
    assert accepted[0].scheduler_type == "LambdaLR"
    assert validate_component_contracts((requirement,), (), accepted) == ((), ())
    assert _codes(validate_component_contracts((requirement,), (), missing)[0]) == {
        "E_COMPONENT_SCHEDULER_MISSING"
    }
    assert _codes(validate_component_contracts((requirement,), (), wrong)[0]) == {
        "E_COMPONENT_SCHEDULER_TYPE"
    }


def test_model_input_contract_is_checked_against_exact_roles() -> None:
    provisions = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=_native_mlp()),
        ModelBindingSpec.single(),
    )
    requirement = ComponentRequirement(
        slot="model_bundle",
        kind="torch_model",
        input_roles=("fit.X_l", "fit.X_u"),
    )
    inputs = {
        "fit.X_l": ValueDescriptor(
            representation="dense",
            dtype_kinds=frozenset({"float"}),
            rank=2,
        ),
        "fit.X_u": ValueDescriptor(
            representation="tokens",
            dtype_kinds=frozenset({"integer"}),
            rank=3,
        ),
    }

    issues, unverified = validate_component_contracts(
        (requirement,), (), provisions, input_provisions=inputs
    )

    assert _codes(issues) == {
        "E_COMPONENT_INPUT_DTYPE",
        "E_COMPONENT_INPUT_RANK",
        "E_COMPONENT_INPUT_REPRESENTATION",
    }
    assert unverified == ()


def test_optional_component_input_role_may_be_absent() -> None:
    provisions = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=_native_mlp()),
        ModelBindingSpec.single(),
    )
    requirement = ComponentRequirement(
        slot="model_bundle",
        kind="torch_model",
        input_roles=("fit.X_l", "fit.X_u_w"),
    )
    inputs = {
        "fit.X_l": ValueDescriptor(
            representation="dense",
            dtype_kinds=frozenset({"float"}),
            rank=2,
        )
    }

    assert validate_component_contracts(
        (requirement,),
        (),
        provisions,
        input_provisions=inputs,
        optional_input_roles=("fit.X_u_w",),
    ) == ((), ())


def test_graph_model_input_uses_feature_leaf_not_edge_index_dtype() -> None:
    contract = ModelContract(
        outputs=frozenset({"logits"}),
        input_representations=frozenset({"graph"}),
        input_dtype_kinds=frozenset({"float"}),
        input_ranks=frozenset({2}),
        source="test.graph",
    )
    bundle = TorchModelBundle(
        model=torch.nn.Linear(2, 2),
        optimizer=object(),
        contract=contract,
    )
    provisions = resolve_bound_component_contracts(
        _SingleSpec(model_bundle=bundle),
        ModelBindingSpec.single(),
    )
    graph = ValueDescriptor(
        representation="graph",
        dtype_kinds=frozenset({"float", "integer"}),
        rank=None,
        schema=(
            (
                "edge_index",
                ValueDescriptor(
                    representation="dense",
                    dtype_kinds=frozenset({"integer"}),
                    rank=2,
                ),
            ),
            (
                "x",
                ValueDescriptor(
                    representation="dense",
                    dtype_kinds=frozenset({"float"}),
                    rank=2,
                ),
            ),
        ),
    )
    requirement = ComponentRequirement(
        slot="model_bundle",
        kind="torch_model",
        input_roles=("fit.X_l",),
    )

    assert validate_component_contracts(
        (requirement,),
        (),
        provisions,
        input_provisions={"fit.X_l": graph},
    ) == ((), ())


def test_auxiliary_bundle_discovery_handles_plain_objects_sequences_and_none() -> None:
    bundle = SimpleNamespace(
        model=SimpleNamespace(device="mps"),
        optimizer=object(),
        contract=ModelContract(outputs=frozenset({"logits"}), source="test"),
    )
    spec = SimpleNamespace(
        auxiliary=bundle,
        ensemble=(bundle, None),
        ignored="not-a-bundle",
    )

    provisions = resolve_bound_component_contracts(spec, ModelBindingSpec())

    assert [provision.slot for provision in provisions] == [
        "auxiliary",
        "ensemble[0]",
    ]
    assert all(provision.device == "mps" for provision in provisions)
    assert resolve_bound_component_contracts(None, ModelBindingSpec()) == ()
    assert resolve_bound_component_contracts(object(), ModelBindingSpec()) == ()


def test_shared_head_slot_discovery_ignores_absent_or_empty_declared_slots() -> None:
    bundle = SimpleNamespace(model=object(), optimizer=object())
    empty_binding = ModelBindingSpec(
        kind="shared_heads",
        shared_bundle_field=None,
        head_bundles_field=None,
        head_count=2,
    )
    partial_binding = ModelBindingSpec.shared_heads(
        shared_field="shared",
        heads_field="heads",
        head_count=2,
        head_classifier_ids=("mlp",),
        head_classifier_fallback="mlp",
    )

    assert resolve_bound_component_contracts(SimpleNamespace(), empty_binding) == ()
    partial = resolve_bound_component_contracts(
        SimpleNamespace(shared=None, heads=(None, bundle)),
        partial_binding,
    )
    assert [provision.slot for provision in partial] == ["heads[1]"]


def test_component_metadata_handles_parameter_and_buffer_introspection_failures() -> None:
    class BufferedModel:
        def parameters(self) -> tuple[()]:
            return ()

        def buffers(self) -> tuple[Any, ...]:
            return (SimpleNamespace(device="cpu"),)

    class BrokenModel:
        device = " "

        def parameters(self) -> tuple[Any, ...]:
            raise RuntimeError("lazy parameters unavailable")

        def buffers(self) -> tuple[Any, ...]:
            raise TypeError("lazy buffers unavailable")

    class MixedDeviceModel:
        def parameters(self) -> tuple[Any, ...]:
            return (
                SimpleNamespace(device="cpu"),
                SimpleNamespace(device="cuda:0"),
            )

    @dataclass(frozen=True)
    class AuxiliaryModels:
        buffered: Any
        broken: Any
        mixed: Any

    def bundle(model: Any) -> Any:
        return SimpleNamespace(model=model, optimizer=object())

    provisions = resolve_bound_component_contracts(
        AuxiliaryModels(
            buffered=bundle(BufferedModel()),
            broken=bundle(BrokenModel()),
            mixed=bundle(MixedDeviceModel()),
        ),
        ModelBindingSpec(),
    )
    by_slot = {provision.slot: provision for provision in provisions}

    assert by_slot["buffered"].device == "cpu"
    assert by_slot["broken"].device is None
    assert by_slot["mixed"].device is None


def test_component_input_provision_mapping_rejects_invalid_or_duplicate_roles() -> None:
    descriptor = ValueDescriptor(representation="dense")

    with pytest.raises(TypeError, match="roles must be non-empty"):
        validate_component_contracts((), (), (), input_provisions=(("", descriptor),))
    with pytest.raises(TypeError, match="ValueDescriptor"):
        validate_component_contracts(
            (),
            (),
            (),
            input_provisions=(("fit.X", object()),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="duplicate input provision"):
        validate_component_contracts(
            (),
            (),
            (),
            input_provisions=(("fit.X", descriptor), ("fit.X", descriptor)),
        )
    with pytest.raises(TypeError, match="optional input roles"):
        validate_component_contracts((), (), (), optional_input_roles=("",))


@pytest.mark.parametrize(
    ("requirements", "relations", "provisions", "message"),
    [
        ((object(),), (), (), "requirements"),
        ((), (object(),), (), "relations"),
        ((), (), (object(),), "provisions"),
    ],
)
def test_component_validation_rejects_non_contract_values(
    requirements: tuple[Any, ...],
    relations: tuple[Any, ...],
    provisions: tuple[Any, ...],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        validate_component_contracts(requirements, relations, provisions)


def test_component_validation_reports_missing_duplicate_kind_and_optimizer() -> None:
    contract = ModelContract(outputs=frozenset({"logits"}), source="test")
    duplicate = ComponentProvision(
        slot="duplicate",
        kind="torch_model",
        contract=contract,
        has_optimizer=True,
    )
    requirements = (
        ComponentRequirement("missing", "torch_model"),
        ComponentRequirement("classifier", "torch_model"),
        ComponentRequirement(
            "optimizer",
            "torch_model",
            requires_optimizer=True,
        ),
    )
    provisions = (
        duplicate,
        duplicate,
        ComponentProvision("classifier", "classifier", contract),
        ComponentProvision("optimizer", "torch_model", contract),
    )

    issues, unverified = validate_component_contracts(requirements, (), provisions)

    assert _codes(issues) == {
        "E_COMPONENT_KIND_MISMATCH",
        "E_COMPONENT_MISSING",
        "E_COMPONENT_OPTIMIZER_MISSING",
        "E_COMPONENT_SLOT_DUPLICATE",
    }
    assert unverified == ()


def test_unverified_contract_inputs_and_ema_identity_remain_explicit() -> None:
    contract = ModelContract(
        outputs=frozenset({"logits"}),
        input_representations=frozenset({"tokens"}),
        input_dtype_kinds=frozenset({"integer"}),
        input_ranks=frozenset({2}),
        verification="unverified",
        source="external.declaration",
    )
    provision = ComponentProvision(
        slot="model",
        kind="torch_model",
        contract=contract,
        has_ema=True,
    )
    requirement = ComponentRequirement(
        slot="model",
        kind="torch_model",
        input_roles=("fit.tokens", "fit.missing"),
        requires_ema=True,
    )
    tokens = ValueDescriptor(
        representation="tokens",
        schema=(
            (
                "input_ids",
                ValueDescriptor(
                    representation="dense",
                    rank=None,
                ),
            ),
        ),
    )

    issues, unverified = validate_component_contracts(
        (requirement,),
        (),
        (provision,),
        input_provisions={"fit.tokens": tokens},
    )

    assert _codes(issues) == {"E_COMPONENT_INPUT_ROLE_MISSING"}
    assert _codes(unverified) == {
        "E_COMPONENT_CONTRACT_UNVERIFIED",
        "E_COMPONENT_EMA_IDENTITY_UNVERIFIED",
        "E_COMPONENT_EMA_PARAMETERS_UNVERIFIED",
        "E_COMPONENT_INPUT_DTYPE_UNVERIFIED",
        "E_COMPONENT_INPUT_RANK_UNVERIFIED",
    }


def test_unrestricted_model_inputs_and_independent_ema_are_fully_verified() -> None:
    contract = ModelContract(outputs=frozenset({"logits"}), source="test")
    provision = ComponentProvision(
        slot="model",
        kind="torch_model",
        contract=contract,
        object_id=1,
        parameter_ids=frozenset({11}),
        ema_object_id=2,
        ema_parameter_ids=frozenset({22}),
        has_ema=True,
    )
    requirement = ComponentRequirement(
        slot="model",
        kind="torch_model",
        input_roles=("fit.first", "fit.second"),
        requires_ema=True,
    )
    inputs = {
        "fit.first": ValueDescriptor(representation="dense"),
        "fit.second": ValueDescriptor(representation="tokens"),
    }

    assert validate_component_contracts(
        (requirement,),
        (),
        (provision,),
        input_provisions=inputs,
    ) == ((), ())


def test_component_relations_report_every_unproven_or_incompatible_state() -> None:
    contract = ModelContract(outputs=frozenset({"logits"}), source="test")
    first = ComponentProvision(
        "first",
        "torch_model",
        contract,
        object_id=None,
        parameter_ids=frozenset(),
        device=None,
    )
    second = ComponentProvision(
        "second",
        "torch_model",
        contract,
        object_id=2,
        parameter_ids=frozenset({2}),
        device="cpu",
    )
    third = ComponentProvision(
        "third",
        "torch_model",
        contract,
        object_id=3,
        parameter_ids=frozenset({3}),
        device="cuda:0",
    )
    relations = (
        ComponentRelation("distinct_objects", ("first", "second")),
        ComponentRelation("disjoint_parameters", ("first", "second")),
        ComponentRelation("same_device", ("first", "second")),
        ComponentRelation("same_device", ("second", "third")),
        ComponentRelation("same_architecture", ("second", "third")),
        ComponentRelation("distinct_objects", ("first", "missing")),
    )

    issues, unverified = validate_component_contracts(
        (),
        relations,
        (first, second, third),
    )

    assert _codes(issues) == {
        "E_COMPONENT_DEVICE_MISMATCH",
        "E_COMPONENT_RELATION_SLOT_MISSING",
    }
    assert _codes(unverified) == {
        "E_COMPONENT_ARCHITECTURE_UNVERIFIED",
        "E_COMPONENT_RELATION_UNVERIFIED",
    }
