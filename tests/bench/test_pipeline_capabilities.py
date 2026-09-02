from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bench import main as bench_main
from bench.schema import ExperimentConfig
from modssc.capabilities import (
    IncompatiblePipelineError,
    MethodCapabilities,
    materialize_pipeline_capabilities,
    validate_pipeline_compatibility,
)
from modssc.data_loader.types import LoadedDataset, Split
from modssc.inductive.base import MethodInfo as InductiveMethodInfo
from modssc.sampling.result import SamplingResult
from modssc.transductive.base import MethodInfo as TransductiveMethodInfo


def _config(*, kind: str = "inductive", model: dict[str, Any] | None = None) -> ExperimentConfig:
    method: dict[str, Any] = {
        "kind": kind,
        "id": "contract_method",
        "device": {"device": "cpu", "dtype": "float32"},
        "params": {},
    }
    if model is not None:
        method["model"] = model
    return ExperimentConfig.from_dict(
        {
            "run": {"name": "contract", "seed": 3, "output_dir": "runs"},
            "dataset": {"id": "toy"},
            "sampling": {"seed": 3, "plan": {"split": {"kind": "holdout"}}},
            "preprocess": {"seed": 3, "cache": False, "plan": {"steps": []}},
            "method": method,
            "evaluation": {"report_splits": ["test"], "metrics": ["accuracy"]},
        }
    )


def _prepared() -> tuple[Any, SamplingResult]:
    dataset = LoadedDataset(
        train=Split(
            X=np.arange(24, dtype=np.float32).reshape(12, 2),
            y=np.arange(12, dtype=np.int64) % 2,
        ),
        meta={},
    )
    pre = SimpleNamespace(dataset=dataset)
    sampling = SamplingResult(
        schema_version=1,
        created_at="",
        dataset_fingerprint="dataset",
        split_fingerprint="split",
        plan={},
        indices={
            "train": np.arange(8, dtype=np.int64),
            "val": np.array([8, 9], dtype=np.int64),
            "test": np.array([10, 11], dtype=np.int64),
            "train_labeled": np.array([0, 1], dtype=np.int64),
            "train_unlabeled": np.arange(2, 8, dtype=np.int64),
        },
    )
    return pre, sampling


def _build(
    *,
    cfg: ExperimentConfig,
    views: Any | None = None,
    graph: Any | None = None,
    X_u_w: Any | None = None,
    X_u_s: Any | None = None,
    X_u_s_1: Any | None = None,
    requires_torch: bool = False,
):
    pre, sampling = _prepared()
    configured_backend = cfg.method.params.get("backend")
    if configured_backend is None:
        configured_backend = cfg.method.params.get("classifier_backend")
    if configured_backend is None and cfg.method.model is not None:
        configured_backend = cfg.method.model.classifier_backend
    return materialize_pipeline_capabilities(
        regime=cfg.method.kind,
        modality="tabular",
        primary_input=pre.dataset.train.X,
        sampling=sampling,
        view_count=0 if views is None else len(views.views),
        has_graph=graph is not None,
        has_weak_augmentation=X_u_w is not None,
        strong_augmentation_count=sum(value is not None for value in (X_u_s, X_u_s_1)),
        configured_backend=configured_backend,
        model_configured=cfg.method.model is not None,
        requires_torch=requires_torch,
        device="cpu",
        dtype=cfg.method.device.dtype,
        checkpointing_required=cfg.run.resume_policy != "never",
    )


def _validate(info: Any, pipeline: Any):
    return validate_pipeline_compatibility(
        info.method_id,
        info.capabilities,
        pipeline,
    )


def test_materialized_two_view_pipeline_satisfies_method_contract() -> None:
    pipeline = _build(
        cfg=_config(),
        views=SimpleNamespace(views={"first": object(), "second": object()}),
    )
    info = InductiveMethodInfo(
        method_id="two_view",
        name="Two view",
        capabilities=MethodCapabilities(
            regime="inductive",
            requires_unlabeled=True,
            min_views=2,
            required_classifier_outputs=frozenset({"scores"}),
        ),
    )

    report = _validate(info, pipeline)

    assert report.compatible
    assert pipeline.modality == "tabular"
    assert pipeline.representation == "dense"
    assert pipeline.view_count == 2
    assert pipeline.has_unlabeled
    assert pipeline.backend == "numpy"


def test_missing_views_is_reported_by_shared_contract() -> None:
    pipeline = _build(cfg=_config())
    info = InductiveMethodInfo(
        method_id="two_view",
        name="Two view",
        capabilities=MethodCapabilities(regime="inductive", min_views=2),
    )

    with pytest.raises(IncompatiblePipelineError) as raised:
        _validate(info, pipeline)

    assert [issue.code for issue in raised.value.report.issues] == ["E_CAPABILITY_VIEWS"]


def test_transductive_graph_requirement_is_reported_by_shared_contract() -> None:
    pipeline = _build(cfg=_config(kind="transductive"))
    info = TransductiveMethodInfo(method_id="graph_method", name="Graph method")

    with pytest.raises(IncompatiblePipelineError) as raised:
        _validate(info, pipeline)

    assert [issue.code for issue in raised.value.report.issues] == ["E_CAPABILITY_GRAPH"]

    compatible = _build(cfg=_config(kind="transductive"), graph=object())
    assert _validate(info, compatible).compatible


def test_weak_strong_and_torch_model_capabilities_come_from_materialized_pipeline() -> None:
    values = np.ones((4, 3), dtype=np.float32)
    pipeline = _build(
        cfg=_config(model={"classifier_id": "mlp", "classifier_backend": "torch"}),
        X_u_w=values,
        X_u_s=values,
        X_u_s_1=values,
        requires_torch=True,
    )
    info = InductiveMethodInfo(
        method_id="augmentation_method",
        name="Augmentation method",
        capabilities=MethodCapabilities(
            regime="inductive",
            requires_weak_augmentation=True,
            min_strong_augmentations=2,
            required_classifier_outputs=frozenset({"logits"}),
            backends=frozenset({"torch"}),
        ),
    )

    assert _validate(info, pipeline).compatible
    assert pipeline.has_weak_augmentation
    assert pipeline.strong_augmentation_count == 2
    assert pipeline.backend == "torch"
    assert "logits" in pipeline.classifier_outputs


def test_pipeline_capability_payload_is_json_serializable() -> None:
    pipeline = _build(cfg=_config())

    payload = pipeline.to_dict()

    assert payload["classifier_outputs"] == ["predictions", "scores"]
    json.dumps(payload)


def test_resume_policy_materializes_checkpoint_requirement() -> None:
    raw = {
        "run": {
            "name": "contract",
            "seed": 3,
            "output_dir": "runs",
            "resume_policy": "required",
        },
        "dataset": {"id": "toy"},
        "sampling": {"seed": 3, "plan": {"split": {"kind": "holdout"}}},
        "preprocess": {"seed": 3, "cache": False, "plan": {"steps": []}},
        "method": {
            "kind": "inductive",
            "id": "contract_method",
            "device": {"device": "cpu", "dtype": "float32"},
            "params": {},
        },
        "evaluation": {"report_splits": ["test"], "metrics": ["accuracy"]},
    }
    pipeline = _build(cfg=ExperimentConfig.from_dict(raw))

    assert pipeline.checkpointing_required
    unsupported = InductiveMethodInfo(
        method_id="without_checkpointing",
        name="Without checkpointing",
        capabilities=MethodCapabilities(regime="inductive"),
    )
    with pytest.raises(IncompatiblePipelineError) as raised:
        _validate(unsupported, pipeline)
    assert [issue.code for issue in raised.value.report.issues] == ["E_CAPABILITY_CHECKPOINTING"]


def test_preflight_keeps_dependency_and_metric_checks_without_pipeline_special_cases() -> None:
    cfg = _config(kind="transductive")

    bench_main._preflight(
        cfg=cfg,
    )
