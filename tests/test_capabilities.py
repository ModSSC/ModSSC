from __future__ import annotations

import numpy as np
import pytest

from modssc.capabilities import (
    DEFAULT_INDUCTIVE_CAPABILITIES,
    DEFAULT_TRANSDUCTIVE_CAPABILITIES,
    DENSE_TRANSDUCTIVE_CAPABILITIES,
    CapabilityIssue,
    CompatibilityReport,
    IncompatiblePipelineError,
    MethodCapabilities,
    PipelineCapabilities,
    check_pipeline_compatibility,
    materialize_consumed_input_capabilities,
    validate_consumed_input_capabilities,
    validate_pipeline_compatibility,
)
from modssc.graph.artifacts import GraphArtifact, NodeDataset
from modssc.inductive import InductiveDataset
from modssc.inductive.base import MethodInfo as InductiveMethodInfo
from modssc.inductive.methods.co_training import CoTrainingMethod
from modssc.inductive.methods.comatch import CoMatchMethod
from modssc.inductive.methods.fixmatch import FixMatchMethod
from modssc.inductive.methods.s4vm import S4VMMethod
from modssc.inductive.methods.supervised import SupervisedMethod
from modssc.inductive.registry import _debug_registry as debug_inductive_registry
from modssc.inductive.registry import get_method_info as get_inductive_method_info
from modssc.transductive.base import MethodInfo as TransductiveMethodInfo
from modssc.transductive.methods.classic.graph_mincuts import GraphMincutsMethod
from modssc.transductive.methods.gnn.grand import GRANDMethod
from modssc.transductive.registry import _debug_registry as debug_transductive_registry
from modssc.transductive.registry import get_method_info as get_transductive_method_info


def test_compatible_pipeline_normalizes_declared_names() -> None:
    method = MethodCapabilities(
        regime="inductive",
        modalities={" vision "},  # type: ignore[arg-type]
        representations={"tensor"},  # type: ignore[arg-type]
        target_kinds={" class_ids "},  # type: ignore[arg-type]
        min_labeled_classes=2,
        max_labeled_classes=2,
        requires_unlabeled=True,
        min_views=1,
        requires_weak_augmentation=True,
        min_strong_augmentations=1,
        required_classifier_outputs={"logits"},  # type: ignore[arg-type]
        backends={"torch"},  # type: ignore[arg-type]
        devices={"cpu"},  # type: ignore[arg-type]
        dtypes={"float32"},  # type: ignore[arg-type]
        supports_checkpointing=True,
    )
    pipeline = PipelineCapabilities(
        regime="inductive",
        modality=" vision ",
        representation="tensor",
        target_kind=" class_ids ",
        labeled_class_count=2,
        has_unlabeled=True,
        view_count=1,
        has_weak_augmentation=True,
        strong_augmentation_count=1,
        classifier_outputs={"logits", "probabilities"},  # type: ignore[arg-type]
        backend=" torch ",
        device="cpu",
        dtype="float32",
        checkpointing_required=True,
    )

    report = check_pipeline_compatibility(" fixmatch ", method, pipeline)

    assert method.modalities == frozenset({"vision"})
    assert method.target_kinds == frozenset({"class_ids"})
    assert pipeline.modality == "vision"
    assert pipeline.backend == "torch"
    assert report == CompatibilityReport(method_id="fixmatch", issues=())
    assert report.compatible
    assert validate_pipeline_compatibility("fixmatch", method, pipeline).compatible


def test_check_reports_all_incompatibilities_with_stable_codes() -> None:
    method = MethodCapabilities(
        regime="inductive",
        modalities=frozenset({"vision"}),
        representations=frozenset({"tensor"}),
        requires_unlabeled=True,
        requires_graph=True,
        min_views=2,
        requires_weak_augmentation=True,
        min_strong_augmentations=2,
        required_classifier_outputs=frozenset({"logits", "probabilities"}),
        backends=frozenset({"torch"}),
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float32"}),
    )
    pipeline = PipelineCapabilities(
        regime="transductive",
        modality="text",
        representation="tokens",
        classifier_outputs=frozenset({"scores"}),
        backend="numpy",
        device="cpu",
        dtype="float64",
        checkpointing_required=True,
    )

    report = check_pipeline_compatibility("example", method, pipeline)

    assert not report.compatible
    assert [issue.code for issue in report.issues] == [
        "E_CAPABILITY_REGIME",
        "E_CAPABILITY_MODALITY",
        "E_CAPABILITY_REPRESENTATION",
        "E_CAPABILITY_UNLABELED",
        "E_CAPABILITY_GRAPH",
        "E_CAPABILITY_VIEWS",
        "E_CAPABILITY_WEAK_AUGMENTATION",
        "E_CAPABILITY_STRONG_AUGMENTATION",
        "E_CAPABILITY_CLASSIFIER_OUTPUT",
        "E_CAPABILITY_BACKEND",
        "E_CAPABILITY_DEVICE",
        "E_CAPABILITY_DTYPE",
        "E_CAPABILITY_CHECKPOINTING",
    ]
    with pytest.raises(IncompatiblePipelineError) as raised:
        validate_pipeline_compatibility("example", method, pipeline)
    assert raised.value.report is not report
    assert "[E_CAPABILITY_GRAPH] requires a graph" in str(raised.value)


def test_unrestricted_contract_accepts_unspecified_execution_details() -> None:
    method = MethodCapabilities(
        regime="inductive",
        required_classifier_outputs=None,  # type: ignore[arg-type]
    )
    pipeline = PipelineCapabilities(
        regime="inductive",
        modality="custom",
        representation="custom",
    )

    assert pipeline.backend is None
    assert method.required_classifier_outputs == frozenset()
    assert check_pipeline_compatibility("generic", method, pipeline).compatible


def test_target_contract_rejects_non_class_ids_and_non_binary_targets() -> None:
    binary = MethodCapabilities(
        regime="inductive",
        target_kinds=frozenset({"class_ids"}),
        min_labeled_classes=2,
        max_labeled_classes=2,
    )
    matrix = PipelineCapabilities(
        regime="inductive",
        modality="tabular",
        representation="dense",
        target_kind="matrix",
        labeled_class_count=3,
    )

    report = check_pipeline_compatibility("binary", binary, matrix)

    assert [issue.code for issue in report.issues] == [
        "E_CAPABILITY_TARGET_KIND",
        "E_CAPABILITY_CLASS_COUNT",
    ]


def test_target_contract_rejects_too_few_labeled_classes() -> None:
    report = check_pipeline_compatibility(
        "binary",
        MethodCapabilities(
            regime="inductive",
            target_kinds=frozenset({"class_ids"}),
            min_labeled_classes=2,
        ),
        PipelineCapabilities(
            regime="inductive",
            modality="tabular",
            representation="dense",
            target_kind="class_ids",
            labeled_class_count=1,
        ),
    )

    assert [issue.code for issue in report.issues] == ["E_CAPABILITY_CLASS_COUNT"]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: MethodCapabilities(regime="online"), "regime must be one of"),
        (
            lambda: PipelineCapabilities(
                regime="online",
                modality="vision",
                representation="tensor",
            ),
            "regime must be one of",
        ),
        (
            lambda: MethodCapabilities(
                regime="inductive",
                modalities="vision",  # type: ignore[arg-type]
            ),
            "must be a collection of names",
        ),
        (
            lambda: MethodCapabilities(regime="inductive", modalities=frozenset()),
            "cannot be empty",
        ),
        (
            lambda: MethodCapabilities(
                regime="inductive",
                modalities=frozenset({""}),
            ),
            "must be a non-empty string",
        ),
        (
            lambda: MethodCapabilities(regime="inductive", min_views=-1),
            "must be a non-negative integer",
        ),
        (
            lambda: MethodCapabilities(regime="inductive", min_strong_augmentations=True),
            "must be a non-negative integer",
        ),
        (
            lambda: MethodCapabilities(regime="inductive", target_kinds=frozenset()),
            "cannot be empty",
        ),
        (
            lambda: MethodCapabilities(regime="inductive", min_labeled_classes=True),
            "must be a non-negative integer",
        ),
        (
            lambda: MethodCapabilities(
                regime="inductive",
                min_labeled_classes=3,
                max_labeled_classes=2,
            ),
            "cannot exceed",
        ),
        (
            lambda: PipelineCapabilities(
                regime="inductive",
                modality="",
                representation="tensor",
            ),
            "modality must be a non-empty string",
        ),
        (
            lambda: PipelineCapabilities(
                regime="inductive",
                modality="vision",
                representation=" ",
            ),
            "representation must be a non-empty string",
        ),
        (
            lambda: PipelineCapabilities(
                regime="inductive",
                modality="vision",
                representation="tensor",
                backend="",
            ),
            "backend must be a non-empty string",
        ),
        (
            lambda: PipelineCapabilities(
                regime="inductive",
                modality="vision",
                representation="tensor",
                view_count=-1,
            ),
            "must be a non-negative integer",
        ),
        (
            lambda: PipelineCapabilities(
                regime="inductive",
                modality="vision",
                representation="tensor",
                labeled_class_count=True,
            ),
            "must be a non-negative integer",
        ),
        (
            lambda: PipelineCapabilities(
                regime="inductive",
                modality="vision",
                representation="tensor",
                strong_augmentation_count=True,
            ),
            "must be a non-negative integer",
        ),
    ],
)
def test_invalid_capability_declarations_fail_early(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_method_id_must_be_explicit() -> None:
    pipeline = PipelineCapabilities(
        regime="inductive",
        modality="vision",
        representation="tensor",
    )
    with pytest.raises(ValueError, match="method_id must be a non-empty string"):
        check_pipeline_compatibility("", MethodCapabilities(regime="inductive"), pipeline)


def test_method_info_uses_one_shared_contract_for_both_regimes() -> None:
    inductive = InductiveMethodInfo(method_id="i", name="Inductive")
    transductive = TransductiveMethodInfo(method_id="t", name="Transductive")

    assert inductive.capabilities is DEFAULT_INDUCTIVE_CAPABILITIES
    assert transductive.capabilities is DEFAULT_TRANSDUCTIVE_CAPABILITIES
    assert inductive.capabilities.regime == "inductive"
    assert transductive.capabilities.regime == "transductive"
    assert transductive.capabilities.requires_graph


def test_representative_native_methods_declare_semantic_requirements() -> None:
    assert not SupervisedMethod.info.capabilities.requires_unlabeled
    assert CoTrainingMethod.info.capabilities.min_views == 2
    assert CoTrainingMethod.info.capabilities.required_classifier_outputs == frozenset({"scores"})
    assert FixMatchMethod.info.capabilities.requires_weak_augmentation
    assert FixMatchMethod.info.capabilities.min_strong_augmentations == 1
    assert FixMatchMethod.info.capabilities.supports_checkpointing
    assert CoMatchMethod.info.capabilities.requires_weak_augmentation
    assert CoMatchMethod.info.capabilities.min_strong_augmentations == 2
    assert GRANDMethod.info.capabilities.requires_graph
    assert GRANDMethod.info.capabilities.representations == frozenset({"dense"})


def test_dense_gnns_and_graph_optional_tsvm_declare_exact_consumed_inputs() -> None:
    dense_gnns = {
        "appnp",
        "chebnet",
        "gat",
        "gcn",
        "gcnii",
        "grafn",
        "graphhop",
        "graphsage",
        "h_gcn",
        "n_gcn",
        "planetoid",
        "sgc",
    }

    for method_id in dense_gnns:
        assert get_transductive_method_info(method_id).capabilities is (
            DENSE_TRANSDUCTIVE_CAPABILITIES
        )

    tsvm = get_transductive_method_info("tsvm").capabilities
    assert tsvm.representations == frozenset({"dense"})
    assert tsvm.target_kinds == frozenset({"class_ids"})
    assert tsvm.max_labeled_classes == 2
    assert tsvm.requires_unlabeled
    assert not tsvm.requires_graph

    all_transductive = {
        method_id
        for method_id, import_path in debug_transductive_registry().items()
        if import_path.startswith("modssc.")
    }
    for method_id in all_transductive - {"tsvm"}:
        assert get_transductive_method_info(method_id).capabilities.requires_graph
    assert S4VMMethod.info.capabilities.max_labeled_classes == 2
    assert GraphMincutsMethod.info.capabilities.max_labeled_classes == 2


@pytest.mark.parametrize(
    "method_id",
    [
        "adamatch",
        "adsh",
        "daso",
        "defixmatch",
        "fixmatch",
        "flexmatch",
        "free_match",
        "mean_teacher",
        "meta_pseudo_labels",
        "mixmatch",
        "pi_model",
        "softmatch",
        "temporal_ensembling",
        "uda",
    ],
)
def test_methods_requiring_two_ssl_inputs_declare_weak_and_strong_views(
    method_id: str,
) -> None:
    capabilities = get_inductive_method_info(method_id).capabilities

    assert capabilities.requires_weak_augmentation
    assert capabilities.min_strong_augmentations >= 1


def test_every_registered_method_exposes_the_shared_regime_contract() -> None:
    inductive_builtins = {
        method_id
        for method_id, import_path in debug_inductive_registry().items()
        if import_path.startswith("modssc.")
    }
    transductive_builtins = {
        method_id
        for method_id, import_path in debug_transductive_registry().items()
        if import_path.startswith("modssc.")
    }
    for method_id in inductive_builtins:
        capabilities = get_inductive_method_info(method_id).capabilities
        assert isinstance(capabilities, MethodCapabilities)
        assert capabilities.regime == "inductive"
    for method_id in transductive_builtins:
        capabilities = get_transductive_method_info(method_id).capabilities
        assert isinstance(capabilities, MethodCapabilities)
        assert capabilities.regime == "transductive"


def test_exception_preserves_structured_report() -> None:
    report = CompatibilityReport(
        method_id="method",
        issues=(CapabilityIssue(code="E_CAPABILITY_TEST", message="missing test input"),),
    )

    error = IncompatiblePipelineError(report)

    assert error.report is report
    assert str(error) == (
        "Pipeline is incompatible with method 'method': [E_CAPABILITY_TEST] missing test input"
    )


def _graph() -> GraphArtifact:
    return GraphArtifact(
        n_nodes=5,
        edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
    )


def test_consumed_inductive_capabilities_ignore_global_artifacts_and_reserved_views() -> None:
    data = InductiveDataset(
        X_l=np.ones((2, 3), dtype=np.float32),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u=np.ones((3, 3), dtype=np.float32),
        X_u_w=np.ones((3, 3), dtype=np.float32),
        X_u_s=np.ones((3, 3), dtype=np.float32),
        X_u_s_1=np.ones((3, 3), dtype=np.float32),
        views={
            "view_a": object(),
            "view_b": object(),
            "X_u_s_1": np.ones((3, 3), dtype=np.float32),
        },
        graph=_graph(),
    )

    capabilities = materialize_consumed_input_capabilities(
        regime="inductive",
        modality="graph",
        consumed_input=data,
        classifier_outputs={"predictions", "scores", "logits"},
        runtime_backend="article_runtime_that_must_not_override_numpy",
    )

    assert capabilities.has_unlabeled
    assert capabilities.has_graph
    assert capabilities.view_count == 2
    assert capabilities.has_weak_augmentation
    assert capabilities.strong_augmentation_count == 2
    assert capabilities.backend == "numpy"
    assert capabilities.dtype == "float32"
    assert capabilities.target_kind == "class_ids"
    assert capabilities.labeled_class_count == 2


def test_consumed_capabilities_resolve_strong_view_aliases_canonically() -> None:
    data = InductiveDataset(
        X_l=np.ones((2, 3), dtype=np.float32),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u_w=np.ones((3, 3), dtype=np.float32),
        views={
            "X_u_strong0": np.ones((3, 3), dtype=np.float32),
            "X_u_s2": np.ones((3, 3), dtype=np.float32),
            "scientific_view": object(),
        },
    )

    capabilities = materialize_consumed_input_capabilities(
        regime="inductive",
        modality="tabular",
        consumed_input=data,
    )

    assert capabilities.strong_augmentation_count == 2
    assert capabilities.view_count == 1
    assert capabilities.has_unlabeled


def test_consumed_capability_validation_fails_when_graph_was_not_delivered() -> None:
    data = InductiveDataset(
        X_l=np.ones((2, 3), dtype=np.float32),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u=np.ones((1, 3), dtype=np.float32),
    )

    with pytest.raises(IncompatiblePipelineError) as caught:
        validate_consumed_input_capabilities(
            "future_graph_method",
            MethodCapabilities(regime="inductive", requires_graph=True),
            regime="inductive",
            modality="graph",
            consumed_input=data,
        )

    assert [issue.code for issue in caught.value.report.issues] == ["E_CAPABILITY_GRAPH"]


def test_consumed_transductive_capabilities_come_from_fit_visible_data() -> None:
    graph = _graph()
    data = NodeDataset(
        X=np.ones((5, 2), dtype=np.float32),
        y=np.array([0, -1, -1, -1, -1], dtype=np.int64),
        graph=graph,
        masks={
            "labeled_mask": np.array([True, False, False, False, False]),
            "unlabeled_mask": np.array([False, True, True, False, False]),
        },
    )

    capabilities = validate_consumed_input_capabilities(
        "graph_method",
        DEFAULT_TRANSDUCTIVE_CAPABILITIES,
        regime="transductive",
        modality="graph",
        consumed_input=data,
    )

    assert capabilities.has_graph
    assert capabilities.has_unlabeled
    assert capabilities.representation == "dense"
    assert capabilities.target_kind == "class_ids"
    assert capabilities.labeled_class_count == 1


def test_consumed_capabilities_use_observed_dtype_over_runtime_hint() -> None:
    data = InductiveDataset(
        X_l=np.ones((2, 3), dtype=np.float64),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u=np.ones((1, 3), dtype=np.float64),
    )

    capabilities = materialize_consumed_input_capabilities(
        regime="inductive",
        modality="tabular",
        consumed_input=data,
        dtype="float32",
    )

    assert capabilities.dtype == "float64"


@pytest.mark.parametrize(
    ("primary_input", "expected_dtype"),
    [
        (
            {
                "left": np.ones((2, 1), dtype=np.float32),
                "right": np.ones((2, 1), dtype=np.float32),
            },
            "float32",
        ),
        (
            {
                "left": np.ones((2, 1), dtype=np.float32),
                "right": np.ones((2, 1), dtype=np.int64),
            },
            None,
        ),
        ({"metadata": object()}, None),
    ],
)
def test_consumed_capabilities_derive_dtype_from_structured_inputs(
    primary_input: object,
    expected_dtype: str | None,
) -> None:
    capabilities = materialize_consumed_input_capabilities(
        regime="inductive",
        modality="tabular",
        consumed_input=InductiveDataset(
            X_l=primary_input,
            y_l=np.array([0, 1], dtype=np.int64),
        ),
    )

    assert capabilities.dtype == expected_dtype


@pytest.mark.parametrize(
    ("target", "expected_kind", "expected_count"),
    [
        (np.eye(3, dtype=np.float32), "matrix", 3),
        (np.zeros((2, 2, 2), dtype=np.float32), "matrix", None),
        (np.array(["cat", "dog", "cat"]), "labels", 2),
        (np.array([], dtype="U1"), "labels", 0),
    ],
)
def test_consumed_capabilities_describe_non_class_id_targets(
    target: np.ndarray,
    expected_kind: str,
    expected_count: int | None,
) -> None:
    capabilities = materialize_consumed_input_capabilities(
        regime="inductive",
        modality="tabular",
        consumed_input=InductiveDataset(
            X_l=np.ones((max(1, len(target)), 2), dtype=np.float32),
            y_l=target,
        ),
    )

    assert capabilities.target_kind == expected_kind
    assert capabilities.labeled_class_count == expected_count


def test_torch_sparse_input_is_not_reported_as_dense() -> None:
    torch = pytest.importorskip("torch")
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([1.0, 2.0]),
        size=(2, 2),
        check_invariants=True,
    )
    data = InductiveDataset(
        X_l=sparse,
        y_l=torch.tensor([0, 1]),
        X_u=sparse,
    )

    capabilities = materialize_consumed_input_capabilities(
        regime="inductive",
        modality="graph",
        consumed_input=data,
    )

    assert capabilities.representation == "sparse"
    assert capabilities.backend == "torch"
