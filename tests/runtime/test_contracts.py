from __future__ import annotations

import json

import pytest

from modssc.capabilities import MethodCapabilities
from modssc.runtime.composition import (
    build_execution_contract_report,
    enforce_execution_contract,
    execution_contract_sha256,
)
from modssc.runtime.contracts import (
    ComponentProvision,
    ComponentRelation,
    ComponentRequirement,
    ContractIssue,
    ExecutionContractError,
    ExecutionContractReport,
    InputRoleRequirement,
    MethodExecutionContract,
    ModelContract,
    RoleRelation,
    ValueDescriptor,
)


def test_directional_contract_payload_is_deterministic_and_json_safe() -> None:
    leaf = ValueDescriptor(
        representation=" dense ",
        container_backends=frozenset({"torch"}),
        dtypes=frozenset({"float32"}),
        dtype_kinds=frozenset({"float"}),
        devices=frozenset({"cpu"}),
        rank=2,
        shape=(3, 4),
        rows=3,
    )
    structured = ValueDescriptor(
        representation="tokens",
        container_backends=frozenset({"torch"}),
        dtypes=frozenset({"int64"}),
        dtype_kinds=frozenset({"integer"}),
        rows=3,
        schema=(("input_ids", leaf), ("attention_mask", leaf)),
    )
    base = MethodCapabilities(regime="inductive", requires_unlabeled=True)
    contract = MethodExecutionContract(
        base=base,
        inputs=(
            InputRoleRequirement(
                role="fit.X_l",
                container_backends=frozenset({"torch"}),
                ranks=frozenset({2}),
            ),
        ),
        relations=(RoleRelation("same_rows", ("fit.X_l", "fit.y_l")),),
        components=(
            ComponentRequirement(
                slot="model_bundle",
                kind="torch_model",
                outputs=frozenset({"logits"}),
                requires_optimizer=True,
            ),
        ),
        component_relations=(),
        source="method.execution_contract",
    )
    provision = ComponentProvision(
        slot="model_bundle",
        kind="torch_model",
        contract=ModelContract(
            outputs=frozenset({"feat", "logits"}),
            verification="declared",
            source="native.builder",
        ),
        object_id=7,
        parameter_ids=frozenset({3, 1}),
        has_optimizer=True,
        device="cpu",
    )
    report = ExecutionContractReport(
        method_id="method",
        input_provisions=(("fit.tokens", structured), ("fit.X_l", leaf)),
        component_provisions=(provision,),
        contract=contract,
    )

    assert report.status == "compatible"
    assert report.compatible
    payload = report.to_dict()
    assert list(payload["inputs"]) == ["fit.X_l", "fit.tokens"]
    assert list(payload["inputs"]["fit.tokens"]["schema"]) == [
        "attention_mask",
        "input_ids",
    ]
    assert payload["components"][0]["parameter_count"] == 2
    json.dumps(payload, allow_nan=False)


def test_report_distinguishes_incompatible_from_unverified() -> None:
    unknown = ContractIssue("E_CONTRACT_UNKNOWN", "external component has no contract")
    unverified = ExecutionContractReport(method_id="method", unverified=(unknown,))

    assert unverified.status == "unverified"
    assert not unverified.compatible
    error = ExecutionContractError(unverified)
    assert "is unverified" in str(error)
    assert error.report is unverified

    issue = ContractIssue("E_CONTRACT_ROLE_MISSING", "fit.X_u is missing")
    incompatible = ExecutionContractReport(
        method_id="method",
        issues=(issue,),
        unverified=(unknown,),
    )
    assert incompatible.status == "incompatible"
    assert "E_CONTRACT_ROLE_MISSING" in str(ExecutionContractError(incompatible))


def test_composition_enforces_unverified_only_in_strict_mode_and_hashes_report() -> None:
    contract = MethodExecutionContract(base=MethodCapabilities(regime="inductive"))
    report = build_execution_contract_report(
        method_id="method",
        contract=contract,
        input_provisions=(),
        unverified=(ContractIssue("E_CONTRACT_UNKNOWN", "unknown component"),),
    )

    assert enforce_execution_contract(report, strict=False) is report
    with pytest.raises(ExecutionContractError):
        enforce_execution_contract(report, strict=True)
    digest = execution_contract_sha256(report)
    assert len(digest) == 64
    assert digest == execution_contract_sha256(report)


def test_report_hash_is_independent_of_issue_declaration_order() -> None:
    contract = MethodExecutionContract(base=MethodCapabilities(regime="inductive"))
    first = ContractIssue("E_Z", "last")
    second = ContractIssue("E_A", "first")
    forward = build_execution_contract_report(
        method_id="method",
        contract=contract,
        input_provisions=(),
        issues=(first, second, first),
    )
    reverse = build_execution_contract_report(
        method_id="method",
        contract=contract,
        input_provisions=(),
        issues=(second, first),
    )

    assert forward.issues == (second, first)
    assert execution_contract_sha256(forward) == execution_contract_sha256(reverse)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ValueDescriptor(representation=""), "non-empty"),
        (
            lambda: ValueDescriptor(representation="dense", rank=-1),
            "non-negative",
        ),
        (
            lambda: InputRoleRequirement(role="x", ranks=frozenset()),
            "cannot be empty",
        ),
        (
            lambda: RoleRelation("same_rows", ("x",)),
            "at least two",
        ),
        (
            lambda: ComponentRelation("disjoint_parameters", ("model",)),
            "at least two",
        ),
        (
            lambda: ModelContract(
                outputs=frozenset({"logits"}),
                verification="assumed",  # type: ignore[arg-type]
            ),
            "verification must be",
        ),
    ],
)
def test_invalid_contracts_fail_at_declaration(factory, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_value_descriptor_rejects_invalid_metadata_and_reports_numeric_state() -> None:
    assert ValueDescriptor(representation="dense").numeric is None
    assert (
        ValueDescriptor(
            representation="objects",
            dtype_kinds=frozenset({"text"}),
        ).numeric
        is False
    )

    invalid = (
        (lambda: ValueDescriptor(representation="dense", rows=-1), "rows"),
        (lambda: ValueDescriptor(representation="dense", shape=(2, -1)), "shape"),
        (
            lambda: ValueDescriptor(
                representation="dense",
                schema=(("leaf", object()),),  # type: ignore[arg-type]
            ),
            "ValueDescriptor",
        ),
    )
    for factory, message in invalid:
        with pytest.raises((TypeError, ValueError), match=message):
            factory()


def test_requirement_declarations_validate_every_enum_and_collection_boundary() -> None:
    empty_contract = ModelContract(outputs=None)  # type: ignore[arg-type]
    assert empty_contract.outputs == frozenset()
    assert empty_contract.source is None
    with pytest.raises(TypeError, match="collection of strings"):
        ModelContract(outputs="logits")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be empty"):
        InputRoleRequirement(role="x", representations=frozenset())
    with pytest.raises(ValueError, match="non-negative integers"):
        InputRoleRequirement(role="x", ranks=frozenset({True}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="numeric"):
        InputRoleRequirement(role="x", numeric=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="consumption"):
        InputRoleRequirement(role="x", consumption="read")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind must be"):
        RoleRelation("unknown", ("x", "y"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind must be"):
        ComponentRequirement("model", "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alternatives cannot be empty"):
        ComponentRequirement(
            "model",
            "torch_model",
            output_alternatives=(frozenset(),),
        )
    with pytest.raises(ValueError, match="kind must be"):
        ComponentRelation("unknown", ("left", "right"))  # type: ignore[arg-type]

    role = InputRoleRequirement(role=" x ", model_input=" payload ")
    requirement = ComponentRequirement(
        " model ",
        "torch_model",
        input_roles=(" z ", "a", "a"),
        scheduler_types=frozenset({"LambdaLR"}),
    )
    assert role.role == "x"
    assert role.model_input == "payload"
    assert requirement.slot == "model"
    assert requirement.input_roles == ("a", "z")
    assert requirement.requires_scheduler


def test_component_provision_and_execution_report_reject_wrong_native_types() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        ComponentProvision("model", "unknown", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ModelContract"):
        ComponentProvision("model", "torch_model", object())  # type: ignore[arg-type]

    provision = ComponentProvision(
        " model ",
        "torch_model",
        None,
        scheduler_type=" StepLR ",
        device=" cpu ",
    )
    assert provision.slot == "model"
    assert provision.scheduler_type == "StepLR"
    assert provision.device == "cpu"
    assert provision.has_scheduler
    assert provision.verification == "unverified"
    assert ComponentProvision("other", "torch_model", None).device is None

    capabilities = MethodCapabilities(regime="inductive")
    with pytest.raises(TypeError, match="base"):
        MethodExecutionContract(base=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="inputs"):
        MethodExecutionContract(base=capabilities, inputs=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="issues"):
        ExecutionContractReport(method_id="m", issues=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="input provisions"):
        ExecutionContractReport(
            method_id="m",
            input_provisions=(("fit.X", object()),),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="component_provisions"):
        ExecutionContractReport(
            method_id="m",
            component_provisions=(object(),),  # type: ignore[arg-type]
        )
