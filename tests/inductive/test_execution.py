from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import modssc.inductive.execution as execution
from modssc.capabilities import MethodCapabilities
from modssc.data_loader.types import LoadedDataset, Split
from modssc.graph.artifacts import GraphArtifact
from modssc.inductive import (
    DeviceSpec,
    InductiveExecutionConfig,
    InductiveExecutionError,
    InductiveExecutionInput,
    ModelBuildConfig,
    execute_inductive_method,
    requires_torch_inputs,
)
from modssc.inductive.base import MethodInfo
from modssc.inductive.methods.daso import DASOMethod, DASOSpec
from modssc.inductive.methods.simclr_v2 import SimCLRv2Method, SimCLRv2Spec
from modssc.inductive.methods.trinet import TriNetMethod, TriNetSpec
from modssc.preprocess.store import ArtifactStore
from modssc.preprocess.types import PreprocessResult, ResolvedPlan
from modssc.runtime.composition import execution_contract_sha256
from modssc.runtime.contracts import (
    ComponentProvision,
    ComponentRequirement,
    ExecutionContractError,
    InputRoleRequirement,
    MethodExecutionContract,
    ModelContract,
)
from modssc.runtime.execution import ExecutionContext, RunIdentity
from modssc.runtime.outcome import MethodExecutionOutcome, MethodNotEvaluableError
from modssc.sampling.result import SamplingResult
from modssc.sampling.routing import InductiveGraphSamplingPolicy


def _preprocess() -> PreprocessResult:
    train_X = np.arange(20, dtype=np.float32).reshape(10, 2)
    return PreprocessResult(
        dataset=LoadedDataset(
            train=Split(X=train_X, y=np.arange(10, dtype=np.int64) % 2),
            test=Split(
                X=np.arange(8, dtype=np.float32).reshape(4, 2) + 100,
                y=np.array([1, 0, 1, 0], dtype=np.int64),
            ),
            meta={"dataset_fingerprint": "dataset"},
        ),
        plan=ResolvedPlan(steps=()),
        preprocess_fingerprint="preprocess",
        train_artifacts=ArtifactStore(),
        test_artifacts=ArtifactStore(),
    )


def _sampling(*, index_space_stats: bool = True) -> SamplingResult:
    stats = (
        {
            "policy": {"partition_artifact_sha256": "current"},
            "partition_artifact_sha256": "legacy",
        }
        if index_space_stats
        else {}
    )
    return SamplingResult(
        schema_version=1,
        created_at="",
        dataset_fingerprint="dataset",
        split_fingerprint="split",
        plan={},
        indices={
            "train_labeled": np.array([1, 4], dtype=np.int64),
            "train_unlabeled": np.array([5, 8], dtype=np.int64),
            "test": np.array([1, 3], dtype=np.int64),
        },
        refs={"test": "test"},
        stats=stats,
    )


@pytest.mark.parametrize(
    ("index_space", "expected_indices", "expected_size"),
    [
        ("source", np.array([5, 8]), 10),
        ("local", np.array([0, 1]), 2),
    ],
)
def test_native_execution_preserves_all_method_facing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    index_space: str,
    expected_indices: np.ndarray,
    expected_size: int,
) -> None:
    class CapturingMethod:
        info = MethodInfo(method_id="capture", name="Capture")
        diagnostics_ = {"fit": "ok"}
        captured_data = None

        def __init__(self) -> None:
            self.unlabeled_index_space = index_space

        def fit(self, data, *, device, seed):
            assert device == DeviceSpec(device="cpu", dtype="float32")
            assert seed == 3
            type(self).captured_data = data
            return self

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: CapturingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: CapturingMethod.info)

    preprocess = _preprocess()
    view_X = np.arange(30, dtype=np.float32).reshape(10, 3)
    views = SimpleNamespace(
        views={
            "view_a": LoadedDataset(
                train=Split(X=view_X, y=preprocess.dataset.train.y),
                meta={},
            )
        }
    )
    online = SimpleNamespace(seed=71)
    context = ExecutionContext(RunIdentity("0" * 64, 3), tmp_path)
    result = execute_inductive_method(
        InductiveExecutionInput(
            preprocess=preprocess,
            sampling=_sampling(),
            views=views,
            X_u_w=np.full((2, 2), 1.0, dtype=np.float32),
            X_u_s=np.full((2, 2), 2.0, dtype=np.float32),
            X_u_s_1=np.full((2, 2), 3.0, dtype=np.float32),
            online_augmentation=online,
            execution_context=context,
        ),
        InductiveExecutionConfig(
            method_id="capture",
            device=DeviceSpec(device="cpu", dtype="float32"),
            seed=3,
            during_fit_splits=("test",),
        ),
    )

    data = result.data
    assert data is CapturingMethod.captured_data
    np.testing.assert_array_equal(data.X_l, preprocess.dataset.train.X[[1, 4]])
    np.testing.assert_array_equal(data.X_u, preprocess.dataset.train.X[[5, 8]])
    np.testing.assert_array_equal(data.views["view_a"]["X_l"], view_X[[1, 4]])
    np.testing.assert_array_equal(data.views["view_a"]["X_u"], view_X[[5, 8]])
    np.testing.assert_array_equal(data.views["X_u_s_1"], np.full((2, 2), 3.0))
    np.testing.assert_array_equal(data.meta["idx_l"], [1, 4])
    np.testing.assert_array_equal(data.meta["source_idx_l"], [1, 4])
    np.testing.assert_array_equal(data.meta["source_idx_u"], [5, 8])
    np.testing.assert_array_equal(data.meta["idx_u"], expected_indices)
    assert data.meta["ulb_size"] == expected_size
    assert data.meta["partition_sha256"] == "current"
    assert data.meta["online_augmentation"] is online
    assert data.meta["augmentation_seed"] == 71
    np.testing.assert_array_equal(
        data.meta["evaluation_splits"]["test"]["X"],
        preprocess.dataset.test.X[[1, 3]],
    )
    np.testing.assert_array_equal(
        data.meta["evaluation_splits"]["test"]["y"],
        preprocess.dataset.test.y[[1, 3]],
    )
    assert data.execution_context is context
    assert result.resolution["diagnostics"] == {"fit": "ok"}
    assert result.resolution["dtypes"]["X_u_s_1"]["dtype"] == "float32"
    assert result.evaluation_runtime is not None
    assert result.evaluation_runtime.backend == "numpy"
    assert result.evaluation_runtime.device is None
    assert result.method.evaluation_runtime_ is result.evaluation_runtime


def test_native_execution_enforces_the_generic_scientific_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NonConvergedMethod:
        info = MethodInfo(method_id="non_converged", name="Non-converged")

        def fit(self, data, *, device, seed):
            del data, device, seed
            self.fitted = True
            return self

    expected = MethodNotEvaluableError(
        MethodExecutionOutcome(
            status="not_evaluable",
            reason="declared gate",
            diagnostics={"converged": False},
        )
    )

    def enforce(method: object) -> None:
        assert getattr(method, "fitted", False) is True
        raise expected

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: NonConvergedMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: NonConvergedMethod.info)
    monkeypatch.setattr(execution, "enforce_method_execution", enforce)

    with pytest.raises(MethodNotEvaluableError) as raised:
        execute_inductive_method(
            InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            InductiveExecutionConfig(method_id="non_converged"),
        )
    assert raised.value is expected


def test_native_execution_rejects_missing_required_model_bundle_before_fit() -> None:
    with pytest.raises(InductiveExecutionError, match="requires bound model fields") as raised:
        execute_inductive_method(
            InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            InductiveExecutionConfig(method_id="fixmatch"),
        )

    assert raised.value.kind == "model_config"
    assert raised.value.code == "E_INDUCTIVE_MODEL_CONFIG"


def test_native_execution_rejects_invalid_method_index_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidMethod:
        info = MethodInfo(method_id="invalid", name="Invalid")
        unlabeled_index_space = "article-specific"

        def fit(self, data, *, device, seed):  # pragma: no cover - contract fails first
            raise AssertionError((data, device, seed))

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: InvalidMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: InvalidMethod.info)

    with pytest.raises(InductiveExecutionError, match="must be 'source' or 'local'") as caught:
        execute_inductive_method(
            InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            InductiveExecutionConfig(method_id="invalid"),
        )
    assert caught.value.kind == "method_contract"
    assert caught.value.code == "E_INDUCTIVE_METHOD_CONTRACT"


def test_native_execution_rejects_graph_mask_sampling_before_method_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution,
        "get_method_class",
        lambda _method_id: pytest.fail("method lookup must not occur"),
    )
    sampling = SamplingResult(
        schema_version=1,
        created_at="",
        dataset_fingerprint="dataset",
        split_fingerprint="split",
        plan={},
        masks={"labeled": np.array([True])},
    )

    with pytest.raises(InductiveExecutionError) as caught:
        execute_inductive_method(
            InductiveExecutionInput(preprocess=_preprocess(), sampling=sampling),
            InductiveExecutionConfig(method_id="unused"),
        )
    assert caught.value.kind == "graph_sampling"


def test_native_execution_converts_graph_sampling_by_policy_and_carries_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = GraphArtifact(
        n_nodes=10,
        edge_index=np.array([[0, 1, 4, 5], [1, 2, 5, 6]], dtype=np.int64),
        meta={"fingerprint": "native-graph"},
    )

    class GraphMethod:
        info = MethodInfo(
            method_id="graph_capture",
            name="Graph capture",
            capabilities=MethodCapabilities(
                regime="inductive",
                requires_unlabeled=True,
                requires_graph=True,
            ),
        )

        def fit(self, data, *, device, seed):
            del device, seed
            self.data = data
            return self

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: GraphMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: GraphMethod.info)
    sampling = SamplingResult(
        schema_version=1,
        created_at="",
        dataset_fingerprint="dataset",
        split_fingerprint="graph-split",
        plan={},
        masks={
            "train": np.array([True] * 7 + [False] * 3),
            "val": np.array([False] * 7 + [True, False, False]),
            "test": np.array([False] * 8 + [True, True]),
            "labeled": np.array(
                [False, True, False, False, True, False, False, False, False, False]
            ),
            "unlabeled": np.array(
                [True, False, True, True, False, True, True, False, False, False]
            ),
        },
    )

    result = execute_inductive_method(
        InductiveExecutionInput(
            preprocess=_preprocess(),
            sampling=sampling,
            graph=graph,
            graph_sampling_policy=InductiveGraphSamplingPolicy.MASKS_TO_INDICES,
        ),
        InductiveExecutionConfig(method_id="graph_capture"),
    )

    assert result.data.graph is graph
    assert result.data.meta["graph_fingerprint"] == "native-graph"
    assert result.data.meta["input_routing"][0]["policy"] == "masks_to_indices"
    assert result.resolution["pipeline_capabilities"]["has_graph"] is True
    assert result.resolution["input_routing"][0]["code"] == (
        "sampling.graph_masks_to_inductive_indices"
    )


def test_requires_torch_inputs_is_a_native_configuration_contract() -> None:
    assert not requires_torch_inputs(
        InductiveExecutionConfig(
            method_id="classic",
            params={"classifier_backend": "sklearn"},
        )
    )
    assert requires_torch_inputs(
        InductiveExecutionConfig(
            method_id="deep",
            params={"classifier_backend": "torch"},
        )
    )
    assert requires_torch_inputs(
        InductiveExecutionConfig(
            method_id="deep",
            model=ModelBuildConfig(classifier_backend="torch"),
        )
    )
    assert requires_torch_inputs(InductiveExecutionConfig(method_id="forced", requires_torch=True))


def test_non_strict_native_execution_converts_all_augmented_inputs_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class CapturingMethod:
        info = MethodInfo(method_id="capture", name="Capture")
        captured_data = None

        def fit(self, data, *, device, seed):
            del device, seed
            type(self).captured_data = data
            return self

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: CapturingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: CapturingMethod.info)
    result = execute_inductive_method(
        InductiveExecutionInput(
            preprocess=_preprocess(),
            sampling=_sampling(index_space_stats=False),
            X_u_w=np.ones((2, 2), dtype=np.float32),
            X_u_s=np.ones((2, 2), dtype=np.float32),
            X_u_s_1=np.ones((2, 2), dtype=np.float32),
        ),
        InductiveExecutionConfig(method_id="capture", requires_torch=True),
    )

    assert isinstance(result.data.X_l, torch.Tensor)
    assert isinstance(result.data.y_l, torch.Tensor)
    assert isinstance(result.data.X_u, torch.Tensor)
    assert isinstance(result.data.X_u_w, torch.Tensor)
    assert isinstance(result.data.X_u_s, torch.Tensor)
    assert isinstance(result.data.views["X_u_s_1"], torch.Tensor)
    assert result.evaluation_runtime is not None
    assert result.evaluation_runtime.backend == "torch"
    assert result.evaluation_runtime.device == result.data.X_l.device
    assert result.method.evaluation_runtime_ is result.evaluation_runtime


def test_native_execution_preserves_graph_like_feature_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingMethod:
        info = MethodInfo(method_id="capture", name="Capture")

        def fit(self, data, *, device, seed):
            del device, seed
            self.data = data
            return self

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: CapturingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: CapturingMethod.info)
    graph = {
        "x": np.arange(10, dtype=np.float32).reshape(5, 2),
        "edge_index": np.array(
            [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]],
            dtype=np.int64,
        ),
        "edge_weight": np.arange(6, dtype=np.float32) + 1,
        "num_nodes": 5,
    }
    preprocess = PreprocessResult(
        dataset=LoadedDataset(
            train=Split(X=graph, y=np.array([0, 1, 0, 1, 0], dtype=np.int64)),
            meta={"dataset_fingerprint": "graph-dataset"},
        ),
        plan=ResolvedPlan(steps=()),
        preprocess_fingerprint="graph-preprocess",
        train_artifacts=ArtifactStore(),
    )
    sampling = SamplingResult(
        schema_version=1,
        created_at="",
        dataset_fingerprint="graph-dataset",
        split_fingerprint="graph-split",
        plan={},
        indices={
            "train_labeled": np.array([0, 1], dtype=np.int64),
            "train_unlabeled": np.array([2, 3, 4], dtype=np.int64),
        },
    )

    result = execute_inductive_method(
        InductiveExecutionInput(preprocess=preprocess, sampling=sampling),
        InductiveExecutionConfig(method_id="capture"),
    )

    np.testing.assert_array_equal(result.data.X_l["x"], graph["x"][[0, 1]])
    np.testing.assert_array_equal(result.data.X_l["edge_index"], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(result.data.X_l["edge_weight"], [1.0, 2.0])
    np.testing.assert_array_equal(result.data.X_u["x"], graph["x"][[2, 3, 4]])
    np.testing.assert_array_equal(result.data.X_u["edge_index"], [[1, 2], [2, 1]])
    np.testing.assert_array_equal(result.data.X_u["edge_weight"], [5.0, 6.0])
    assert result.data.X_l["num_nodes"] == 2
    assert result.data.X_u["num_nodes"] == 3
    assert result.data.meta["ulb_size"] == 5


def test_execution_contract_report_is_persisted_with_its_canonical_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingMethod:
        info = MethodInfo(method_id="capture", name="Capture")

        def fit(self, data, *, device, seed):
            del data, device, seed
            return self

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: CapturingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: CapturingMethod.info)

    result = execute_inductive_method(
        InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
        InductiveExecutionConfig(method_id="capture"),
    )

    report = execution.ExecutionContractReport(
        method_id="capture",
        input_provisions=tuple(
            execution.materialize_input_contracts(regime="inductive", consumed_input=result.data)
        ),
        contract=execution.resolve_method_execution_contract(
            CapturingMethod,
            None,
            CapturingMethod.info.capabilities,
            CapturingMethod.info.model_binding,
        ),
    )
    assert result.resolution["execution_contract"]["status"] == "compatible"
    assert result.resolution["execution_contract_sha256"] == execution_contract_sha256(report)


def test_incompatible_input_contract_prevents_inductive_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NeverFitMethod:
        info = MethodInfo(method_id="never_fit", name="Never fit")
        fit_calls = 0

        def fit(self, data, *, device, seed):  # pragma: no cover - gate fails first
            del data, device, seed
            type(self).fit_calls += 1
            return self

    contract = MethodExecutionContract(
        base=NeverFitMethod.info.capabilities,
        inputs=(InputRoleRequirement("fit.X_l", ranks=frozenset({3})),),
        source="test",
    )
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: NeverFitMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: NeverFitMethod.info)
    monkeypatch.setattr(
        execution,
        "resolve_method_execution_contract",
        lambda *_args, **_kwargs: contract,
    )

    with pytest.raises(InductiveExecutionError) as caught:
        execute_inductive_method(
            InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            InductiveExecutionConfig(method_id="never_fit"),
        )

    assert caught.value.kind == "execution_contract"
    assert isinstance(caught.value.__cause__, ExecutionContractError)
    assert caught.value.__cause__.report.status == "incompatible"
    assert {issue.code for issue in caught.value.__cause__.report.issues} == {"E_INPUT_RANK"}
    assert NeverFitMethod.fit_calls == 0


def test_incompatible_component_contract_prevents_inductive_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NeverFitMethod:
        info = MethodInfo(method_id="never_fit", name="Never fit")
        fit_calls = 0

        def fit(self, data, *, device, seed):  # pragma: no cover - gate fails first
            del data, device, seed
            type(self).fit_calls += 1
            return self

    contract = MethodExecutionContract(
        base=NeverFitMethod.info.capabilities,
        components=(
            ComponentRequirement(
                slot="model_bundle",
                kind="torch_model",
                outputs=frozenset({"feat"}),
            ),
        ),
        source="test",
    )
    provision = ComponentProvision(
        slot="model_bundle",
        kind="torch_model",
        contract=ModelContract(outputs=frozenset({"logits"}), source="test"),
    )
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: NeverFitMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: NeverFitMethod.info)
    monkeypatch.setattr(
        execution,
        "resolve_method_execution_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        execution,
        "resolve_bound_component_contracts",
        lambda *_args, **_kwargs: (provision,),
    )

    with pytest.raises(InductiveExecutionError) as caught:
        execute_inductive_method(
            InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            InductiveExecutionConfig(method_id="never_fit"),
        )

    assert isinstance(caught.value.__cause__, ExecutionContractError)
    assert {issue.code for issue in caught.value.__cause__.report.issues} == {
        "E_COMPONENT_OUTPUT_MISSING"
    }
    assert NeverFitMethod.fit_calls == 0


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_invalid_inductive_execution_contract_is_closed_before_fit(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    class NeverFitMethod:
        info = MethodInfo(method_id="invalid_contract", name="Invalid contract")
        fit_calls = 0

        def fit(self, data, *, device, seed):  # pragma: no cover - gate fails first
            del data, device, seed
            type(self).fit_calls += 1
            return self

    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: NeverFitMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: NeverFitMethod.info)

    def invalid_contract(**_kwargs):
        raise error_type("malformed contract")

    monkeypatch.setattr(execution, "_resolve_execution_contract", invalid_contract)

    with pytest.raises(InductiveExecutionError) as caught:
        execute_inductive_method(
            InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            InductiveExecutionConfig(method_id="invalid_contract"),
        )

    assert caught.value.kind == "execution_contract"
    assert isinstance(caught.value.__cause__, error_type)
    assert NeverFitMethod.fit_calls == 0


@pytest.mark.parametrize(
    ("method_class", "spec", "target_slot"),
    [
        (DASOMethod, DASOSpec(), "model_bundle"),
        (
            SimCLRv2Method,
            SimCLRv2Spec(pretrain_epochs=1, finetune_epochs=0, distill_epochs=0),
            "pretrain_bundle",
        ),
        (TriNetMethod, TriNetSpec(), "shared_bundle"),
    ],
    ids=("daso", "simclr_v2", "trinet"),
)
def test_scientific_feature_contract_rejections_are_zero_fit_sentinels(
    monkeypatch: pytest.MonkeyPatch,
    method_class: type,
    spec: object,
    target_slot: str,
) -> None:
    resolved = method_class.execution_contract(
        spec,
        method_class.info.capabilities,
        method_class.info.model_binding,
    )
    requirement = next(
        component for component in resolved.components if component.slot == target_slot
    )
    contract = replace(
        resolved,
        inputs=(),
        relations=(),
        components=(requirement,),
        component_relations=(),
    )

    class NeverFitMethod:
        info = MethodInfo(method_id="scientific_never_fit", name="Scientific never fit")
        fit_calls = 0

        def fit(self, data, *, device, seed):  # pragma: no cover - gate fails first
            del data, device, seed
            type(self).fit_calls += 1
            return self

    provision = ComponentProvision(
        slot=target_slot,
        kind="torch_model",
        contract=ModelContract(outputs=frozenset({"logits"}), source="test.logits_only"),
        has_optimizer=True,
    )
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: NeverFitMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: NeverFitMethod.info)
    monkeypatch.setattr(
        execution,
        "resolve_method_execution_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        execution,
        "resolve_bound_component_contracts",
        lambda *_args, **_kwargs: (provision,),
    )

    with pytest.raises(InductiveExecutionError) as caught:
        execute_inductive_method(
            InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            InductiveExecutionConfig(method_id="scientific_never_fit"),
        )

    assert isinstance(caught.value.__cause__, ExecutionContractError)
    assert "E_COMPONENT_OUTPUT_ALTERNATIVE_MISSING" in {
        issue.code for issue in caught.value.__cause__.report.issues
    }
    assert NeverFitMethod.fit_calls == 0


def test_strict_mode_blocks_unverified_component_but_nonstrict_records_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingMethod:
        info = MethodInfo(method_id="counting", name="Counting")
        fit_calls = 0

        def fit(self, data, *, device, seed):
            del data, device, seed
            type(self).fit_calls += 1
            return self

    contract = MethodExecutionContract(
        base=CountingMethod.info.capabilities,
        components=(
            ComponentRequirement(
                slot="model_bundle",
                kind="torch_model",
                outputs=frozenset({"logits"}),
            ),
        ),
        source="test",
    )
    provision = ComponentProvision(
        slot="model_bundle",
        kind="torch_model",
        contract=None,
    )
    monkeypatch.setattr(execution, "get_method_class", lambda _method_id: CountingMethod)
    monkeypatch.setattr(execution, "get_method_info", lambda _method_id: CountingMethod.info)
    monkeypatch.setattr(
        execution,
        "resolve_method_execution_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        execution,
        "resolve_bound_component_contracts",
        lambda *_args, **_kwargs: (provision,),
    )

    with pytest.raises(InductiveExecutionError) as caught:
        execute_inductive_method(
            InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
            InductiveExecutionConfig(method_id="counting", strict=True),
        )

    assert isinstance(caught.value.__cause__, ExecutionContractError)
    assert caught.value.__cause__.report.status == "unverified"
    assert CountingMethod.fit_calls == 0

    result = execute_inductive_method(
        InductiveExecutionInput(preprocess=_preprocess(), sampling=_sampling()),
        InductiveExecutionConfig(method_id="counting", strict=False),
    )
    assert CountingMethod.fit_calls == 1
    assert result.resolution["execution_contract"]["status"] == "unverified"
