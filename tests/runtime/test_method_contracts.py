from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from modssc.capabilities import MethodCapabilities
from modssc.inductive.methods.adamatch import AdaMatchMethod, AdaMatchSpec
from modssc.inductive.methods.adsh import ADSHMethod, ADSHSpec
from modssc.inductive.methods.co_training import CoTrainingMethod, CoTrainingSpec
from modssc.inductive.methods.comatch import CoMatchMethod, CoMatchSpec
from modssc.inductive.methods.daso import DASOMethod, DASOSpec
from modssc.inductive.methods.deep_co_training import (
    DeepCoTrainingMethod,
    DeepCoTrainingSpec,
)
from modssc.inductive.methods.defixmatch import DeFixMatchMethod, DeFixMatchSpec
from modssc.inductive.methods.fixmatch import FixMatchMethod, FixMatchSpec
from modssc.inductive.methods.flexmatch import FlexMatchMethod, FlexMatchSpec
from modssc.inductive.methods.free_match import FreeMatchMethod, FreeMatchSpec
from modssc.inductive.methods.mean_teacher import MeanTeacherMethod, MeanTeacherSpec
from modssc.inductive.methods.meta_pseudo_labels import (
    MetaPseudoLabelsMethod,
    MetaPseudoLabelsSpec,
)
from modssc.inductive.methods.mixmatch import MixMatchMethod, MixMatchSpec
from modssc.inductive.methods.noisy_student import (
    NoisyStudentMethod,
    NoisyStudentSpec,
)
from modssc.inductive.methods.pi_model import PiModelMethod, PiModelSpec
from modssc.inductive.methods.pseudo_label import PseudoLabelMethod, PseudoLabelSpec
from modssc.inductive.methods.simclr_v2 import SimCLRv2Method, SimCLRv2Spec
from modssc.inductive.methods.softmatch import SoftMatchMethod, SoftMatchSpec
from modssc.inductive.methods.temporal_ensembling import (
    TemporalEnsemblingMethod,
    TemporalEnsemblingSpec,
)
from modssc.inductive.methods.trinet import TriNetMethod, TriNetSpec
from modssc.inductive.methods.uda import UDAMethod, UDASpec
from modssc.inductive.methods.vat import VATMethod, VATSpec
from modssc.inductive.model_binding import ModelBindingSpec
from modssc.inductive.registry import builtin_methods as builtin_inductive_methods
from modssc.inductive.registry import get_method_class as get_inductive_method_class
from modssc.inductive.registry import get_method_info as get_inductive_method_info
from modssc.runtime.contracts import InputRoleRequirement, MethodExecutionContract
from modssc.runtime.method_contracts import (
    fallback_method_execution_contract,
    resolve_method_execution_contract,
    with_inductive_input_roles,
)
from modssc.runtime.method_spec import build_method_spec
from modssc.transductive.registry import builtin_methods as builtin_transductive_methods
from modssc.transductive.registry import get_method_class as get_transductive_method_class
from modssc.transductive.registry import get_method_info as get_transductive_method_info


def _roles(contract: MethodExecutionContract) -> list[str]:
    return [requirement.role for requirement in contract.inputs]


def _component_payload(contract: MethodExecutionContract) -> dict[str, dict[str, Any]]:
    return {requirement.slot: requirement.to_dict() for requirement in contract.components}


def _resolve_native(method_class: type[Any], spec: Any) -> MethodExecutionContract:
    return resolve_method_execution_contract(
        method_class,
        spec,
        method_class.info.capabilities,
        method_class.info.model_binding,
    )


_REPO_ROOT = Path(__file__).resolve().parents[2]
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def test_classmethod_hook_takes_precedence_and_has_explicit_source() -> None:
    capabilities = MethodCapabilities(regime="inductive")
    expected_spec = object()

    class HookMethod:
        seen_spec: Any = None

        @classmethod
        def execution_contract(cls, spec: Any) -> MethodExecutionContract:
            cls.seen_spec = spec
            return MethodExecutionContract(
                base=capabilities,
                inputs=(InputRoleRequirement("custom.input"),),
                source="method-owned value",
            )

    contract = resolve_method_execution_contract(
        HookMethod,
        expected_spec,
        capabilities,
        ModelBindingSpec.single(),
    )

    assert HookMethod.seen_spec is expected_spec
    assert _roles(contract) == ["custom.input"]
    assert contract.components == ()
    assert contract.source == "method.execution_contract"


def test_native_three_argument_hook_receives_capabilities_and_binding() -> None:
    capabilities = MethodCapabilities(regime="inductive")
    binding = ModelBindingSpec.single()
    expected_spec = object()

    class HookMethod:
        seen: tuple[Any, Any, Any] | None = None

        @classmethod
        def execution_contract(
            cls,
            spec: Any,
            received_capabilities: MethodCapabilities,
            model_binding: Any | None,
        ) -> MethodExecutionContract:
            cls.seen = (spec, received_capabilities, model_binding)
            return fallback_method_execution_contract(
                cls,
                received_capabilities,
                model_binding,
            )

    contract = resolve_method_execution_contract(
        HookMethod,
        expected_spec,
        capabilities,
        binding,
    )

    assert HookMethod.seen == (expected_spec, capabilities, binding)
    assert contract.source == "method.execution_contract"


def test_invalid_hook_declarations_fail_instead_of_falling_back() -> None:
    capabilities = MethodCapabilities(regime="inductive")

    class InstanceHook:
        def execution_contract(self, spec: Any) -> MethodExecutionContract:
            raise AssertionError(spec)

    with pytest.raises(TypeError, match="classmethod"):
        resolve_method_execution_contract(InstanceHook, None, capabilities)

    class WrongReturn:
        @classmethod
        def execution_contract(cls, spec: Any) -> object:
            return spec

    with pytest.raises(TypeError, match="must return MethodExecutionContract"):
        resolve_method_execution_contract(WrongReturn, None, capabilities)


def test_inductive_fallback_declares_canonical_augmentation_roles_and_relations() -> None:
    capabilities = MethodCapabilities(
        regime="inductive",
        representations=frozenset({"dense"}),
        requires_unlabeled=True,
        requires_weak_augmentation=True,
        min_strong_augmentations=2,
        required_classifier_outputs=frozenset({"logits", "scores"}),
        backends=frozenset({"torch"}),
        dtypes=frozenset({"float32"}),
    )

    contract = resolve_method_execution_contract(
        type("Method", (), {}),
        None,
        capabilities,
        ModelBindingSpec.single(),
    )

    assert contract.source == "capabilities+model_binding:fallback"
    assert _roles(contract) == [
        "fit.X_l",
        "fit.y_l",
        "fit.X_u",
        "fit.X_u_w",
        "fit.X_u_s.0",
        "fit.X_u_s.1",
    ]
    feature_requirements = [
        requirement for requirement in contract.inputs if requirement.role != "fit.y_l"
    ]
    assert all(
        requirement.representations == frozenset({"dense"}) for requirement in feature_requirements
    )
    assert all(
        requirement.container_backends is None and requirement.dtypes is None
        for requirement in feature_requirements
    )
    assert [(relation.kind, relation.roles) for relation in contract.relations] == [
        ("same_rows", ("fit.X_l", "fit.y_l")),
        (
            "same_rows",
            ("fit.X_u", "fit.X_u_w", "fit.X_u_s.0", "fit.X_u_s.1"),
        ),
        (
            "same_backend",
            (
                "fit.X_l",
                "fit.X_u",
                "fit.X_u_w",
                "fit.X_u_s.0",
                "fit.X_u_s.1",
            ),
        ),
        (
            "same_device",
            (
                "fit.X_l",
                "fit.X_u",
                "fit.X_u_w",
                "fit.X_u_s.0",
                "fit.X_u_s.1",
            ),
        ),
    ]
    assert _component_payload(contract)["model_bundle"]["outputs"] == ["logits"]
    assert _component_payload(contract)["model_bundle"]["input_roles"] == [
        "fit.X_l",
        "fit.X_u",
        "fit.X_u_s.0",
        "fit.X_u_s.1",
        "fit.X_u_w",
    ]


def test_transductive_fallback_requires_node_masks_and_optional_graph_weights() -> None:
    capabilities = MethodCapabilities(
        regime="transductive",
        representations=frozenset({"dense"}),
        target_kinds=frozenset({"class_ids"}),
        requires_unlabeled=True,
        requires_graph=True,
    )

    contract = resolve_method_execution_contract(
        type("Method", (), {}),
        None,
        capabilities,
    )

    assert _roles(contract) == [
        "fit.X",
        "fit.y",
        "fit.masks.train_mask",
        "fit.masks.unlabeled_mask",
        "fit.graph.edge_index",
        "fit.graph.edge_weight",
        "fit.graph.n_nodes",
    ]
    by_role = {requirement.role: requirement for requirement in contract.inputs}
    assert by_role["fit.y"].dtype_kinds == frozenset({"integer"})
    assert not by_role["fit.graph.edge_index"].optional
    assert by_role["fit.graph.edge_weight"].optional
    assert by_role["fit.graph.n_nodes"].optional
    assert len(contract.relations) == 1
    assert contract.relations[0].to_dict() == {
        "kind": "same_rows",
        "roles": [
            "fit.X",
            "fit.y",
            "fit.masks.train_mask",
            "fit.masks.unlabeled_mask",
        ],
    }
    assert contract.components == ()

    no_unlabeled = resolve_method_execution_contract(
        type("Method", (), {}),
        None,
        MethodCapabilities(regime="transductive"),
    )
    assert "fit.masks.unlabeled_mask" not in _roles(no_unlabeled)


def test_inductive_fallback_includes_graph_metadata_when_declared() -> None:
    contract = fallback_method_execution_contract(
        type("Method", (), {}),
        MethodCapabilities(regime="inductive", requires_graph=True),
    )

    assert {
        "fit.graph.edge_index",
        "fit.graph.edge_weight",
        "fit.graph.n_nodes",
    } <= set(_roles(contract))


@pytest.mark.parametrize("kind", ["pair", "teacher_student"])
def test_independent_multi_model_bindings_require_distinct_disjoint_slots(kind: str) -> None:
    capabilities = MethodCapabilities(
        regime="inductive",
        required_classifier_outputs=frozenset({"logits"}),
    )
    binding = (
        ModelBindingSpec.pair(first_field="left", second_field="right")
        if kind == "pair"
        else ModelBindingSpec.teacher_student(
            student_field="student",
            teacher_field="teacher",
        )
    )

    contract = resolve_method_execution_contract(
        type("Method", (), {}),
        None,
        capabilities,
        binding,
    )

    slots = tuple(requirement.slot for requirement in contract.components)
    assert slots == (("left", "right") if kind == "pair" else ("student", "teacher"))
    assert all(requirement.outputs == frozenset({"logits"}) for requirement in contract.components)
    assert [(relation.kind, relation.slots) for relation in contract.component_relations] == [
        ("distinct_objects", slots),
        ("disjoint_parameters", slots),
    ]


def test_generic_multi_phase_fallback_does_not_invent_feature_outputs() -> None:
    capabilities = MethodCapabilities(
        regime="inductive",
        required_classifier_outputs=frozenset({"logits"}),
    )
    shared = resolve_method_execution_contract(
        type("Method", (), {}),
        None,
        capabilities,
        ModelBindingSpec.shared_heads(
            shared_field="backbone",
            heads_field="heads",
            head_count=3,
            head_classifier_ids=("mlp",),
            head_classifier_fallback="mlp",
        ),
    )
    pretrain = resolve_method_execution_contract(
        type("Method", (), {}),
        None,
        capabilities,
        ModelBindingSpec.pretrain_finetune(
            pretrain_field="pretrain",
            finetune_field="finetune",
        ),
    )

    assert {slot: payload["outputs"] for slot, payload in _component_payload(shared).items()} == {
        "backbone": [],
        "heads[0]": ["logits"],
        "heads[1]": ["logits"],
        "heads[2]": ["logits"],
    }
    shared_slots = ("backbone", "heads[0]", "heads[1]", "heads[2]")
    assert [relation.slots for relation in shared.component_relations] == [
        shared_slots,
        shared_slots,
    ]
    assert {slot: payload["outputs"] for slot, payload in _component_payload(pretrain).items()} == {
        "pretrain": [],
        "finetune": ["logits"],
    }
    assert all(
        not payload["output_alternatives"]
        for payload in (
            *_component_payload(shared).values(),
            *_component_payload(pretrain).values(),
        )
    )
    assert pretrain.component_relations == ()


def test_comatch_and_mixmatch_declare_exact_views_and_outputs() -> None:
    comatch = _resolve_native(CoMatchMethod, CoMatchSpec())
    assert _roles(comatch) == [
        "fit.X_l",
        "fit.y_l",
        "fit.X_u_w",
        "fit.X_u_s.0",
        "fit.X_u_s.1",
    ]
    assert _component_payload(comatch)["model_bundle"]["outputs"] == ["feat", "logits"]

    standard = _resolve_native(MixMatchMethod, MixMatchSpec())
    manifold = _resolve_native(
        MixMatchMethod,
        replace(MixMatchSpec(), mixup_manifold=True),
    )
    meta_manifold = _resolve_native(
        MixMatchMethod,
        replace(
            MixMatchSpec(),
            model_bundle=SimpleNamespace(meta={"prefer_manifold_mixup": True}),
        ),
    )
    standard_requirement = standard.components[0]
    assert standard_requirement.outputs == frozenset({"logits"})
    assert standard_requirement.output_alternatives == ()
    expected_alternatives = {
        frozenset({"feat", "forward_head"}),
        frozenset({"forward_features", "forward_head"}),
    }
    assert set(manifold.components[0].output_alternatives) == expected_alternatives
    assert set(meta_manifold.components[0].output_alternatives) == expected_alternatives


def test_teacher_pair_and_shared_head_hooks_declare_component_semantics() -> None:
    mean_teacher = _resolve_native(MeanTeacherMethod, MeanTeacherSpec())
    assert "fit.X_u" not in _roles(mean_teacher)
    assert mean_teacher.components[0].requires_ema
    assert mean_teacher.components[0].outputs == frozenset({"logits"})

    for method_class, spec in (
        (DeepCoTrainingMethod, DeepCoTrainingSpec()),
        (MetaPseudoLabelsMethod, MetaPseudoLabelsSpec()),
    ):
        contract = _resolve_native(method_class, spec)
        slots = tuple(requirement.slot for requirement in contract.components)
        assert all(
            requirement.outputs == frozenset({"logits"}) for requirement in contract.components
        )
        assert [(relation.kind, relation.slots) for relation in contract.component_relations] == [
            ("distinct_objects", slots),
            ("disjoint_parameters", slots),
        ]

    trinet = _resolve_native(TriNetMethod, TriNetSpec())
    shared, *heads = trinet.components
    assert set(shared.output_alternatives) == {
        frozenset({"feat"}),
        frozenset({"features"}),
        frozenset({"embedding"}),
        frozenset({"forward_features"}),
        frozenset({"feature_extractor"}),
    }
    assert shared.outputs == frozenset()
    assert shared.input_roles == ("fit.X_l", "fit.X_u")
    assert all(head.outputs == frozenset({"logits"}) for head in heads)
    assert all(head.input_roles == () for head in heads)


def test_simclrv2_contract_tracks_active_phases_and_unlabeled_input() -> None:
    finetune_only = _resolve_native(
        SimCLRv2Method,
        replace(
            SimCLRv2Spec(),
            pretrain_epochs=0,
            finetune_epochs=1,
            distill_epochs=0,
        ),
    )
    assert _roles(finetune_only) == ["fit.X_l", "fit.y_l"]
    assert not finetune_only.base.requires_unlabeled
    assert [requirement.slot for requirement in finetune_only.components] == ["finetune_bundle"]

    pretrain_only = _resolve_native(
        SimCLRv2Method,
        replace(
            SimCLRv2Spec(),
            pretrain_epochs=1,
            finetune_epochs=0,
            distill_epochs=0,
        ),
    )
    assert "fit.X_u" in _roles(pretrain_only)
    by_role = {requirement.role: requirement for requirement in pretrain_only.inputs}
    assert by_role["fit.X_u"].optional
    assert by_role["fit.X_u_w"].optional
    assert by_role["fit.X_u_s.0"].optional
    assert [requirement.slot for requirement in pretrain_only.components] == ["pretrain_bundle"]
    direct_projection_paths = {
        frozenset({"forward_projection"}),
        frozenset({"projection"}),
        frozenset({"proj"}),
        frozenset({"z"}),
    }
    projected_feature_paths = {
        frozenset({projector, feature})
        for projector in ("projection_head", "projector")
        for feature in (
            "forward_features",
            "feature_extractor",
            "encoder",
            "feat",
            "features",
            "embedding",
        )
    }
    assert set(pretrain_only.components[0].output_alternatives) == (
        direct_projection_paths | projected_feature_paths
    )
    assert set(pretrain_only.components[0].input_roles) == {
        "fit.X_u",
        "fit.X_u_w",
        "fit.X_u_s.0",
    }


def test_simclrv2_contract_follows_single_bundle_fallbacks_and_explicit_student() -> None:
    only_pretrain = SimpleNamespace()
    pretrain_fallback = _resolve_native(
        SimCLRv2Method,
        replace(
            SimCLRv2Spec(),
            pretrain_bundle=only_pretrain,
            finetune_bundle=None,
            student_bundle=None,
        ),
    )
    assert [requirement.slot for requirement in pretrain_fallback.components] == ["pretrain_bundle"]
    assert pretrain_fallback.components[0].outputs == frozenset({"logits"})
    assert pretrain_fallback.components[0].output_alternatives

    only_finetune = SimpleNamespace()
    finetune_fallback = _resolve_native(
        SimCLRv2Method,
        replace(
            SimCLRv2Spec(),
            pretrain_bundle=None,
            finetune_bundle=only_finetune,
            student_bundle=None,
        ),
    )
    assert [requirement.slot for requirement in finetune_fallback.components] == ["finetune_bundle"]
    assert finetune_fallback.components[0].outputs == frozenset({"logits"})
    assert finetune_fallback.components[0].output_alternatives

    explicit_student = _resolve_native(
        SimCLRv2Method,
        replace(
            SimCLRv2Spec(),
            pretrain_epochs=0,
            finetune_epochs=0,
            distill_epochs=1,
            finetune_bundle=SimpleNamespace(),
            student_bundle=SimpleNamespace(),
        ),
    )
    assert [requirement.slot for requirement in explicit_student.components] == [
        "finetune_bundle",
        "student_bundle",
    ]
    assert [relation.kind for relation in explicit_student.component_relations] == [
        "distinct_objects",
        "disjoint_parameters",
    ]

    copied_student = _resolve_native(
        SimCLRv2Method,
        replace(
            SimCLRv2Spec(),
            pretrain_epochs=0,
            finetune_epochs=0,
            distill_epochs=1,
            finetune_bundle=SimpleNamespace(),
            student_bundle=None,
        ),
    )
    assert [requirement.slot for requirement in copied_student.components] == ["finetune_bundle"]


def test_classic_hooks_capture_view_keys_and_conditional_unlabeled_mode() -> None:
    default_cotraining = _resolve_native(CoTrainingMethod, CoTrainingSpec())
    assert {role for role in _roles(default_cotraining) if role.startswith("fit.views.")} == {
        "fit.views.view_a.X_l",
        "fit.views.view_a.X_u",
        "fit.views.view_b.X_l",
        "fit.views.view_b.X_u",
    }

    cotraining = _resolve_native(
        CoTrainingMethod,
        replace(CoTrainingSpec(), view_keys=("words", "links")),
    )
    assert set(_roles(cotraining)) == {
        "fit.X_l",
        "fit.y_l",
        "fit.views.words.X_l",
        "fit.views.words.X_u",
        "fit.views.links.X_l",
        "fit.views.links.X_u",
    }
    assert (
        next(
            requirement for requirement in cotraining.inputs if requirement.role == "fit.X_l"
        ).consumption
        == "alignment_only"
    )

    iterative = _resolve_native(PseudoLabelMethod, PseudoLabelSpec())
    joint = _resolve_native(
        PseudoLabelMethod,
        replace(PseudoLabelSpec(), training_mode="joint_mlp"),
    )
    iterative_unlabeled = next(
        requirement for requirement in iterative.inputs if requirement.role == "fit.X_u"
    )
    joint_unlabeled = next(
        requirement for requirement in joint.inputs if requirement.role == "fit.X_u"
    )
    assert iterative_unlabeled.optional and not iterative_unlabeled.non_empty
    assert not iterative.base.requires_unlabeled
    assert not joint_unlabeled.optional and joint_unlabeled.non_empty
    assert joint.base.requires_unlabeled


@pytest.mark.parametrize(
    ("method_class", "spec"),
    [
        (AdaMatchMethod, AdaMatchSpec()),
        (ADSHMethod, ADSHSpec()),
        (DASOMethod, DASOSpec()),
        (PiModelMethod, PiModelSpec()),
        (TemporalEnsemblingMethod, TemporalEnsemblingSpec()),
        (UDAMethod, UDASpec()),
    ],
)
def test_native_weak_strong_hooks_do_not_require_unused_plain_unlabeled_input(
    method_class: type[Any],
    spec: Any,
) -> None:
    contract = _resolve_native(method_class, spec)

    assert _roles(contract) == [
        "fit.X_l",
        "fit.y_l",
        "fit.X_u_w",
        "fit.X_u_s.0",
    ]
    assert all(
        set(requirement.input_roles) == {"fit.X_l", "fit.X_u_w", "fit.X_u_s.0"}
        for requirement in contract.components
    )


def test_labeled_strong_roles_are_optional_for_consuming_methods() -> None:
    for method_class, spec in (
        (DeFixMatchMethod, DeFixMatchSpec()),
        (MetaPseudoLabelsMethod, MetaPseudoLabelsSpec()),
    ):
        contract = _resolve_native(method_class, spec)
        by_role = {requirement.role: requirement for requirement in contract.inputs}

        assert by_role["fit.X_l_s.0"].optional
        assert "fit.X_l_s.0" in contract.components[0].input_roles
        assert any(
            relation.kind == "same_rows" and relation.roles == ("fit.X_l", "fit.X_l_s.0")
            for relation in contract.relations
        )


def test_optional_plain_and_weak_unlabeled_aliases_match_native_fit_fallbacks() -> None:
    expected = (
        (
            NoisyStudentMethod,
            NoisyStudentSpec(),
            {"fit.X_u", "fit.X_u_w", "fit.X_u_s.0"},
        ),
        (VATMethod, VATSpec(), {"fit.X_u", "fit.X_u_w"}),
        (
            DeepCoTrainingMethod,
            DeepCoTrainingSpec(),
            {"fit.X_u", "fit.X_u_w"},
        ),
    )

    for method_class, spec, aliases in expected:
        contract = _resolve_native(method_class, spec)
        by_role = {requirement.role: requirement for requirement in contract.inputs}

        assert aliases <= set(by_role)
        assert all(by_role[role].optional for role in aliases)
        assert aliases <= set(contract.components[0].input_roles)


@pytest.mark.parametrize(
    ("method_class", "spec"),
    [
        (FixMatchMethod, FixMatchSpec()),
        (FlexMatchMethod, FlexMatchSpec()),
        (FreeMatchMethod, FreeMatchSpec()),
        (SoftMatchMethod, SoftMatchSpec()),
    ],
)
def test_match_hooks_use_weak_strong_and_require_ema_only_for_fixed_steps(
    method_class: type[Any],
    spec: Any,
) -> None:
    epochs = _resolve_native(method_class, spec)
    fixed_steps = _resolve_native(
        method_class,
        replace(spec, training_mode="fixed_steps"),
    )
    assert _roles(epochs) == [
        "fit.X_l",
        "fit.y_l",
        "fit.X_u_w",
        "fit.X_u_s.0",
    ]
    assert not epochs.components[0].requires_ema
    assert not epochs.components[0].requires_scheduler
    assert fixed_steps.components[0].requires_ema
    assert fixed_steps.components[0].requires_scheduler
    assert fixed_steps.components[0].scheduler_types == frozenset({"LambdaLR"})


def test_classifier_scores_remain_legacy_compatibility_without_fictitious_slot() -> None:
    capabilities = MethodCapabilities(
        regime="inductive",
        required_classifier_outputs=frozenset({"scores"}),
    )

    contract = resolve_method_execution_contract(
        type("Method", (), {}),
        None,
        capabilities,
        ModelBindingSpec(),
    )

    assert contract.base.required_classifier_outputs == frozenset({"scores"})
    assert contract.components == ()


def test_every_builtin_default_spec_resolves_to_deterministic_json() -> None:
    resolved: list[tuple[str, str, dict[str, Any]]] = []
    registries = (
        (
            "inductive",
            builtin_inductive_methods(),
            get_inductive_method_class,
            get_inductive_method_info,
        ),
        (
            "transductive",
            builtin_transductive_methods(),
            get_transductive_method_class,
            get_transductive_method_info,
        ),
    )
    for regime, method_ids, get_class, get_info in registries:
        for method_id in method_ids:
            method_class = get_class(method_id)
            method = method_class()
            info = get_info(method_id)
            contract = resolve_method_execution_contract(
                method_class,
                method.spec,
                info.capabilities,
                getattr(info, "model_binding", None),
            )
            payload = contract.to_dict()
            json.dumps(payload, allow_nan=False, sort_keys=True)
            assert payload == contract.to_dict()
            resolved.append((regime, method_id, payload))

    assert len(resolved) == 51
    assert len({(regime, method_id) for regime, method_id, _payload in resolved}) == 51


def test_all_benchmark_cards_build_native_specs_and_execution_contracts() -> None:
    best_root = _REPO_ROOT / "bench" / "configs" / "best"
    reproduction_root = _REPO_ROOT / "bench" / "configs" / "reproductions"
    best_cards = tuple(
        path for path in sorted(best_root.rglob("*.yaml")) if path.name != "regime_manifest.yaml"
    )
    reproduction_cards = tuple(sorted(reproduction_root.glob("*/*.yaml")))

    assert len(best_cards) == 5285
    assert len(reproduction_cards) == 20

    errors: list[str] = []
    for path in (*best_cards, *reproduction_cards):
        relative = path.relative_to(_REPO_ROOT)
        try:
            raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER)
            method = raw["method"]
            regime = method["kind"]
            method_id = method["id"]
            params = method.get("params") or {}
            if regime == "inductive":
                method_class = get_inductive_method_class(method_id)
                info = get_inductive_method_info(method_id)
            elif regime == "transductive":
                method_class = get_transductive_method_class(method_id)
                info = get_transductive_method_info(method_id)
            else:
                raise ValueError(f"unknown method regime {regime!r}")
            spec = build_method_spec(
                method_class,
                params,
                require_spec=True,
                strict=True,
            )
            resolve_method_execution_contract(
                method_class,
                spec,
                info.capabilities,
                getattr(info, "model_binding", None),
            )
        except Exception as exc:  # pragma: no cover - failure report for a card
            errors.append(f"{relative}: {type(exc).__name__}: {exc}")

    assert not errors, "Execution-contract card failures:\n" + "\n".join(errors)


def test_malformed_binding_fails_with_a_stable_error() -> None:
    capabilities = MethodCapabilities(regime="inductive")
    malformed = SimpleNamespace(kind="pair", bundle_fields=("only_one",))

    with pytest.raises(ValueError, match="requires exactly 2 bundle fields"):
        resolve_method_execution_contract(
            type("Method", (), {}),
            None,
            capabilities,
            malformed,
        )


def test_binding_validation_rejects_ambiguous_shapes_before_composition() -> None:
    capabilities = MethodCapabilities(regime="inductive")
    method_class = type("Method", (), {})
    invalid_bindings = (
        (SimpleNamespace(kind="custom"), "unsupported model-binding kind"),
        (SimpleNamespace(kind="single", bundle_fields="model"), "collection of names"),
        (SimpleNamespace(kind="single", bundle_fields=None), "must be iterable"),
        (SimpleNamespace(kind="single", bundle_fields=("",)), "non-empty strings"),
        (
            SimpleNamespace(
                kind="shared_heads",
                shared_bundle_field=None,
                head_bundles_field="heads",
                head_count=2,
            ),
            "shared_bundle_field",
        ),
        (
            SimpleNamespace(
                kind="shared_heads",
                shared_bundle_field="shared",
                head_bundles_field="",
                head_count=2,
            ),
            "head_bundles_field",
        ),
        (
            SimpleNamespace(
                kind="shared_heads",
                shared_bundle_field="shared",
                head_bundles_field="heads",
                head_count=True,
            ),
            "positive head_count",
        ),
    )

    for binding, message in invalid_bindings:
        with pytest.raises((TypeError, ValueError), match=message):
            fallback_method_execution_contract(
                method_class,
                capabilities,
                binding,
            )


def test_fallback_and_exact_role_helpers_validate_their_public_boundaries() -> None:
    capabilities = MethodCapabilities(regime="inductive")
    contract = fallback_method_execution_contract(type("Method", (), {}), capabilities)

    with pytest.raises(TypeError, match="method_class"):
        fallback_method_execution_contract(object(), capabilities)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="capabilities"):
        fallback_method_execution_contract(type("Method", (), {}), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="contract"):
        with_inductive_input_roles(object(), feature_roles=("fit.X_l",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires an inductive contract"):
        with_inductive_input_roles(
            MethodExecutionContract(base=MethodCapabilities(regime="transductive")),
            feature_roles=("fit.X",),
        )
    with pytest.raises(ValueError, match="non-empty"):
        with_inductive_input_roles(contract, feature_roles=(" ",))
    with pytest.raises(ValueError, match="unique"):
        with_inductive_input_roles(contract, feature_roles=("fit.X_l", "fit.X_l"))
    with pytest.raises(ValueError, match="unknown roles"):
        with_inductive_input_roles(
            contract,
            feature_roles=("fit.X_l",),
            optional_feature_roles=("fit.X_u",),
        )
    with pytest.raises(ValueError, match="undeclared roles"):
        with_inductive_input_roles(
            contract,
            feature_roles=("fit.X_l",),
            row_groups=(("fit.X_l", "fit.X_u"),),
        )

    unlabeled_only = with_inductive_input_roles(
        contract,
        feature_roles=("fit.X_u",),
    )
    assert _roles(unlabeled_only) == ["fit.y_l", "fit.X_u"]


def test_hook_signature_and_resolver_types_fail_without_silent_fallback() -> None:
    capabilities = MethodCapabilities(regime="inductive")

    class UnsupportedHook:
        @classmethod
        def execution_contract(cls, first: Any, second: Any) -> MethodExecutionContract:
            raise AssertionError((cls, first, second))

    with pytest.raises(TypeError, match="must accept either"):
        resolve_method_execution_contract(UnsupportedHook, None, capabilities)
    with pytest.raises(TypeError, match="method_class"):
        resolve_method_execution_contract(object(), None, capabilities)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="capabilities"):
        resolve_method_execution_contract(type("Method", (), {}), None, object())  # type: ignore[arg-type]


def test_builtin_inventory_excludes_runtime_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    from modssc.inductive import registry as inductive_registry
    from modssc.transductive import registry as transductive_registry

    monkeypatch.setitem(
        inductive_registry._REGISTRY,
        "runtime_extension",
        inductive_registry.MethodRef("runtime_extension", "extension.module:Method"),
    )
    monkeypatch.setitem(
        transductive_registry._REGISTRY,
        "planned_extension",
        transductive_registry.MethodRef(
            "planned_extension",
            "extension.module:Method",
            status="planned",
        ),
    )

    assert "runtime_extension" not in builtin_inductive_methods(available_only=False)
    assert "planned_extension" not in builtin_transductive_methods(available_only=False)
    assert "pseudo_label" in inductive_registry._debug_registry()
    assert "grand" in transductive_registry._debug_registry()
