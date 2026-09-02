from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from modssc.data_augmentation import UnlabeledAugmentationResult
from modssc.data_loader.types import LoadedDataset, Split
from modssc.graph.artifacts import GraphArtifact
from modssc.preprocess.store import ArtifactStore
from modssc.preprocess.types import PreprocessResult, ResolvedPlan
from modssc.runtime import input_routing as input_routing_module
from modssc.runtime import limits as limits_module
from modssc.runtime.checkpoint import (
    CheckpointIntegrityError,
    CheckpointStore,
)
from modssc.runtime.execution import RunIdentity
from modssc.runtime.input_routing import (
    InputRoutingError,
    ScientificInputRequest,
    route_scientific_input,
)
from modssc.runtime.limits import (
    ResourceLimitError,
    ResourceLimits,
    apply_resource_limits,
    resolve_resource_limits,
)
from modssc.sampling import SamplingResult
from modssc.transductive.errors import TransductiveDataError


def _preprocess(*, with_test: bool = False) -> PreprocessResult:
    test = (
        Split(X=np.ones((2, 2), dtype=np.float32), y=np.array([0, 1], dtype=np.int64))
        if with_test
        else None
    )
    return PreprocessResult(
        dataset=LoadedDataset(
            train=Split(
                X=np.arange(10, dtype=np.float32).reshape(5, 2),
                y=np.array([0, 1, 0, 1, 0], dtype=np.int64),
            ),
            test=test,
            meta={"dataset_fingerprint": "dataset", "modality": "graph"},
        ),
        plan=ResolvedPlan(steps=()),
        preprocess_fingerprint="preprocess",
        train_artifacts=ArtifactStore(),
    )


def _sampling() -> SamplingResult:
    return SamplingResult(
        schema_version=1,
        created_at="",
        dataset_fingerprint="dataset",
        split_fingerprint="split",
        plan={},
        masks={
            "train": np.array([True, True, True, False, False]),
            "val": np.array([False, False, False, True, False]),
            "test": np.array([False, False, False, False, True]),
            "labeled": np.array([True, False, False, False, False]),
            "unlabeled": np.array([False, True, True, False, False]),
        },
        stats={"labeled": 1},
    )


def _graph() -> GraphArtifact:
    return GraphArtifact(
        n_nodes=5,
        edge_index=np.empty((2, 0), dtype=np.int64),
    )


def test_scientific_input_request_rejects_invalid_contract_types() -> None:
    preprocess = _preprocess()
    sampling = _sampling()
    with pytest.raises(ValueError, match="regime"):
        ScientificInputRequest(regime="online", preprocess=preprocess, sampling=sampling)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PreprocessResult"):
        ScientificInputRequest(regime="inductive", preprocess=object(), sampling=sampling)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SamplingResult"):
        ScientificInputRequest(regime="inductive", preprocess=preprocess, sampling=object())  # type: ignore[arg-type]


def test_input_routing_rejects_request_policy_alignment_graph_and_mask_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocess = _preprocess()
    sampling = _sampling()
    with pytest.raises(TypeError, match="ScientificInputRequest"):
        route_scientific_input(object())  # type: ignore[arg-type]

    with pytest.raises(InputRoutingError) as policy_error:
        route_scientific_input(
            ScientificInputRequest(
                regime="inductive",
                preprocess=preprocess,
                sampling=sampling,
                inductive_graph_policy="reject",
            )
        )
    assert policy_error.value.kind == "sampling_policy"

    wrong_augmentation = UnlabeledAugmentationResult(
        weak=np.ones((2, 2), dtype=np.float32),
        strong=np.ones((2, 2), dtype=np.float32),
        second_strong=None,
        online=None,
        sample_ids=np.array([2, 1], dtype=np.int64),
    )
    with pytest.raises(InputRoutingError) as alignment_error:
        route_scientific_input(
            ScientificInputRequest(
                regime="inductive",
                preprocess=preprocess,
                sampling=sampling.as_inductive_indices(),
                augmentation=wrong_augmentation,
            )
        )
    assert alignment_error.value.kind == "augmentation_alignment"

    graph_optional = route_scientific_input(
        ScientificInputRequest(
            regime="transductive",
            preprocess=preprocess,
            sampling=sampling,
        )
    )
    assert graph_optional.execution_input.graph is None
    assert graph_optional.to_dict()["has_graph"] is False

    def fail_masks(*args: Any, **kwargs: Any) -> None:
        raise TransductiveDataError("bad masks")

    monkeypatch.setattr(input_routing_module, "masks_from_sampling", fail_masks)
    with pytest.raises(InputRoutingError) as mask_error:
        route_scientific_input(
            ScientificInputRequest(
                regime="transductive",
                preprocess=preprocess,
                sampling=sampling,
                graph=_graph(),
            )
        )
    assert mask_error.value.kind == "mask_contract"


def test_input_routing_covers_plain_inductive_and_test_aware_transductive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling = _sampling()
    inductive = route_scientific_input(
        ScientificInputRequest(
            regime="inductive",
            preprocess=_preprocess(),
            sampling=sampling.as_inductive_indices(),
        )
    )
    assert inductive.to_dict() == {
        "regime": "inductive",
        "sampling_representation": "indices",
        "has_graph": False,
        "augmentation_delivered": False,
        "events": [],
    }

    monkeypatch.setattr(
        input_routing_module,
        "masks_from_sampling",
        lambda *args, **kwargs: {
            "train_mask": np.array([True, True, True, False, False, False, False]),
            "val_mask": np.array([False, False, False, True, False, False, False]),
            "test_mask": np.array([False, False, False, False, False, True, True]),
            "labeled_mask": np.array([True, False, False, False, False, False, False]),
            "unlabeled_mask": np.array([False, True, True, False, False, False, False]),
        },
    )
    transductive = route_scientific_input(
        ScientificInputRequest(
            regime="transductive",
            preprocess=_preprocess(with_test=True),
            sampling=sampling,
            graph=_graph(),
            use_test_split=True,
        )
    )
    assert transductive.to_dict()["sampling_representation"] == "graph_masks"


def _identity() -> RunIdentity:
    return RunIdentity("a" * 64, 1)


def test_checkpoint_prune_covers_absent_staging_pointer_and_invalid_pointer(tmp_path: Path) -> None:
    absent = CheckpointStore(tmp_path / "absent", _identity())
    assert absent.prune() == ()

    no_pointer = CheckpointStore(tmp_path / "no-pointer", _identity())
    record = no_pointer.save(b"state", step=1)
    no_pointer.pointer_path.unlink()
    (no_pointer.generations_dir / ".staging-orphan").mkdir()
    assert no_pointer.prune() == ()
    assert record.generation_dir.is_dir()

    invalid = CheckpointStore(tmp_path / "invalid", _identity())
    invalid.save(b"state", step=1)
    pointer = json.loads(invalid.pointer_path.read_text(encoding="utf-8"))
    pointer["generation"] = "nested/value"
    invalid.pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="generation is invalid"):
        invalid.prune()


def _cuda(*, available: bool = True, props: Any = None, failure: bool = False) -> Any:
    def get_properties(_index: int) -> Any:
        if failure:
            raise RuntimeError("device vanished")
        return props

    return SimpleNamespace(is_available=lambda: available, get_device_properties=get_properties)


def test_limit_profile_detection_covers_import_cuda_names_memory_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_name: str) -> None:
        raise ImportError

    monkeypatch.setattr(limits_module.importlib, "import_module", fail_import)
    assert limits_module._detect_profile() is None

    cases = [
        (SimpleNamespace(cuda=None), None),
        (SimpleNamespace(cuda=_cuda(available=False)), None),
        (SimpleNamespace(cuda=_cuda(failure=True)), None),
        (SimpleNamespace(cuda=_cuda(props=SimpleNamespace(name="NVIDIA H100"))), "h100"),
        (SimpleNamespace(cuda=_cuda(props=SimpleNamespace(name="Tesla V100"))), "v100"),
        (SimpleNamespace(cuda=_cuda(props=SimpleNamespace(name="other", total_memory=0))), None),
        (
            SimpleNamespace(
                cuda=_cuda(props=SimpleNamespace(name="other", total_memory=80 * 1024**3))
            ),
            "h100",
        ),
        (
            SimpleNamespace(
                cuda=_cuda(props=SimpleNamespace(name="other", total_memory=40 * 1024**3))
            ),
            "v100",
        ),
    ]
    for torch_like, expected in cases:
        monkeypatch.setattr(
            limits_module.importlib,
            "import_module",
            lambda _name, value=torch_like: value,
        )
        assert limits_module._detect_profile() == expected


def test_limit_resolution_coercion_clamping_and_malformed_config_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_resource_limits(None) is None
    with pytest.raises(TypeError, match="ResourceLimits"):
        resolve_resource_limits(object())  # type: ignore[arg-type]
    with pytest.raises(ResourceLimitError, match="profile"):
        resolve_resource_limits(ResourceLimits(profile="other"))
    assert resolve_resource_limits(ResourceLimits()) is None

    monkeypatch.setattr(limits_module, "_detect_profile", lambda: None)
    assert resolve_resource_limits(ResourceLimits(profile="AUTO")).profile == "v100"  # type: ignore[union-attr]
    explicit = resolve_resource_limits(
        ResourceLimits(profile="h100", max_method_batch_size=7, max_train_samples=2)
    )
    assert explicit is not None and explicit.max_method_batch_size == 7

    assert limits_module._coerce_int(None) is None
    assert limits_module._coerce_int(True) is None
    assert limits_module._coerce_int(3) == 3
    assert limits_module._coerce_int(3.0) == 3
    assert limits_module._coerce_int(3.5) == 3
    assert limits_module._coerce_int("4") == 4
    assert limits_module._coerce_int(object()) is None

    changes: list[str] = []
    container: dict[str, Any] = {}
    limits_module._clamp_key(container, key="x", limit=None, path="p", changes=changes)
    limits_module._clamp_key(container, key="x", limit=2, path="p", changes=changes)
    limits_module._clamp_key(
        container, key="x", limit=2, path="p", changes=changes, set_if_missing=True
    )
    assert container == {"x": 2}
    limits_module._clamp_key(container, key="x", limit=3, path="p", changes=changes)
    container["bad"] = "not-an-int"
    limits_module._clamp_key(container, key="bad", limit=2, path="p", changes=changes)

    limits_module._clamp_preprocess_steps({}, limit=2, path="p", changes=changes)
    limits_module._clamp_preprocess_steps(
        {"steps": [None, {"params": None}, {"params": {"batch_size": 4}}]},
        limit=2,
        path="p",
        changes=changes,
    )
    assert limits_module._mapping_child({"x": []}, "x") == {}

    with pytest.raises(TypeError, match="config"):
        apply_resource_limits([], limits=None)  # type: ignore[arg-type]
    effective, no_changes, resolved = apply_resource_limits({"dataset": []}, limits=None)
    assert effective == {"dataset": []} and no_changes == [] and resolved is None

    apply_resource_limits({}, limits=ResourceLimits(max_method_batch_size=1))
    apply_resource_limits(
        {"method": {"params": [], "model": {}}},
        limits=ResourceLimits(max_method_batch_size=1),
    )
    nested, _, _ = apply_resource_limits(
        {"method": {"params": {"classifier_params": {"batch_size": 2}}}},
        limits=ResourceLimits(max_method_batch_size=1),
    )
    assert nested["method"]["params"]["classifier_params"]["batch_size"] == 1

    malformed = {
        "dataset": [],
        "preprocess": {"plan": []},
        "views": {"plan": {"views": [None, {"preprocess": []}]}},
        "method": {
            "params": {"classifier_params": []},
            "model": {"classifier_params": []},
        },
        "graph": {"spec": []},
    }
    effective, _, _ = apply_resource_limits(
        malformed,
        limits=ResourceLimits(max_method_batch_size=1),
    )
    assert effective == malformed
