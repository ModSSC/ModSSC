from __future__ import annotations

import numpy as np
import pytest

from modssc.data_loader.types import LoadedDataset, Split
from modssc.preprocess.api import preprocess, resolve_plan
from modssc.preprocess.cache import CacheManager
from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.optional import is_available
from modssc.preprocess.plan import PreprocessPlan, StepConfig


def _tabular_dataset() -> LoadedDataset:
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.array([0, 1] * 5, dtype=np.int64)
    return LoadedDataset(
        train=Split(X=X, y=y, edges=None, masks=None),
        test=None,
        meta={"modality": "tabular", "dataset_fingerprint": "dataset:tabular"},
    )


def _text_dataset() -> LoadedDataset:
    X = ["hello world", "bonjour monde", "hola mundo"]
    y = np.array([0, 1, 0], dtype=np.int64)
    return LoadedDataset(
        train=Split(X=X, y=y, edges=None, masks=None),
        test=None,
        meta={"modality": "text", "dataset_fingerprint": "dataset:text"},
    )


def _graph_dataset() -> LoadedDataset:
    edge_index = np.array(
        [[0, 0, 1, 2, 3, 3, 4, 4, 4, 5], [1, 2, 2, 3, 4, 5, 5, 6, 7, 7]], dtype=np.int64
    )
    X = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.arange(8, dtype=np.int64)
    return LoadedDataset(
        train=Split(X=X, y=y, edges=edge_index, masks=None),
        test=None,
        meta={"modality": "graph", "dataset_fingerprint": "dataset:graph"},
    )


def test_resolve_plan_skips_by_modality() -> None:
    ds = _tabular_dataset()
    plan = PreprocessPlan(
        steps=(
            StepConfig("text.ensure_strings"),
            StepConfig("core.ensure_2d"),
            StepConfig("vision.resize", params={"height": 8, "width": 8}),
        )
    )
    resolved = resolve_plan(ds, plan)
    ids = [s.step_id for s in resolved.steps]
    assert "core.ensure_2d" in ids
    assert "text.ensure_strings" not in ids
    assert "vision.resize" not in ids
    assert resolved.fingerprint.startswith("resolved_plan:")


def test_preprocess_requires_fit_indices_for_fittable_step() -> None:
    ds = _tabular_dataset()
    plan = PreprocessPlan(
        steps=(StepConfig("core.ensure_2d"), StepConfig("core.pca", params={"n_components": 2}))
    )
    with pytest.raises(PreprocessValidationError):
        preprocess(ds, plan, cache=False)


def test_preprocess_tabular_deterministic_and_cache(tmp_path) -> None:
    ds = _tabular_dataset()
    plan = PreprocessPlan(
        steps=(
            StepConfig("core.ensure_2d"),
            StepConfig("core.cast_dtype", params={"dtype": "float32"}),
            StepConfig("core.pca", params={"n_components": 1}),
        )
    )
    fit_idx = np.arange(ds.train.y.shape[0], dtype=np.int64)
    r1 = preprocess(ds, plan, seed=0, fit_indices=fit_idx, cache=True, cache_dir=str(tmp_path))
    r2 = preprocess(ds, plan, seed=0, fit_indices=fit_idx, cache=True, cache_dir=str(tmp_path))

    X1 = np.asarray(r1.dataset.train.X)
    X2 = np.asarray(r2.dataset.train.X)
    assert X1.shape == (10, 1)
    assert np.allclose(X1, X2)

    steps_dir = tmp_path / ds.meta["dataset_fingerprint"] / "steps"
    assert steps_dir.exists()


def test_cast_fp16_step() -> None:
    ds = _tabular_dataset()
    plan = PreprocessPlan(steps=(StepConfig("core.ensure_2d"), StepConfig("core.cast_fp16")))
    result = preprocess(ds, plan, cache=False)

    X = np.asarray(result.dataset.train.X)
    assert X.dtype == np.float16


def test_cache_manager_roundtrip_numpy(tmp_path) -> None:
    cm = CacheManager.for_dataset("dataset:cache")
    cm.root = tmp_path
    step_fp = "step:abc"
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    cm.save_step_outputs(
        step_fingerprint=step_fp,
        split="train",
        produced={"features.X": arr},
        manifest={"step_id": "x", "index": 0, "params": {}, "kind": "transform"},
    )
    out = cm.load_step_outputs(step_fingerprint=step_fp, split="train")
    assert "features.X" in out
    assert np.allclose(out["features.X"], arr)


def test_cache_manager_roundtrip_json(tmp_path) -> None:
    cm = CacheManager.for_dataset("dataset:json")
    cm.root = tmp_path
    step_fp = "step:json"
    cm.save_step_outputs(
        step_fingerprint=step_fp,
        split="train",
        produced={"raw.X": ["a", "b", "c"]},
        manifest={"step_id": "x", "index": 0, "params": {}, "kind": "transform"},
    )
    out = cm.load_step_outputs(step_fingerprint=step_fp, split="train")
    assert out["raw.X"] == ["a", "b", "c"]


def test_cache_sparse_optional_dependency(tmp_path) -> None:
    cm = CacheManager.for_dataset("dataset:sparse")
    cm.root = tmp_path

    step_fp = "step:sparse"

    if is_available("scipy"):
        from scipy import sparse  # type: ignore

        mat = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        cm.save_step_outputs(
            step_fingerprint=step_fp,
            split="train",
            produced={"features.X": mat},
            manifest={"step_id": "x", "index": 0, "params": {}, "kind": "transform"},
        )
        out = cm.load_step_outputs(step_fingerprint=step_fp, split="train")
        assert out["features.X"].shape == (2, 2)
    else:

        class Dummy:  # noqa: D401
            """Pretend scipy sparse object."""

            __module__ = "scipy.sparse.csr"

        with pytest.raises(OptionalDependencyError):
            cm.save_step_outputs(
                step_fingerprint=step_fp,
                split="train",
                produced={"features.X": Dummy()},
                manifest={"step_id": "x", "index": 0, "params": {}, "kind": "transform"},
            )


def test_auto_embedding_text_stub() -> None:
    ds = _text_dataset()
    plan = PreprocessPlan(steps=(StepConfig("embeddings.auto"),))
    r = preprocess(ds, plan, seed=0, cache=False)
    X = np.asarray(r.dataset.train.X)
    assert X.shape == (3, 8)
    assert X.dtype == np.float32
    assert "features.X" in r.train_artifacts


def test_labels_onehot_artifacts() -> None:
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([0, 1, -1, 1, 0, -1], dtype=np.int64)
    ds = LoadedDataset(
        train=Split(X=X, y=y, edges=None, masks=None),
        test=None,
        meta={"modality": "tabular", "dataset_fingerprint": "dataset:labels"},
    )
    plan = PreprocessPlan(steps=(StepConfig("core.ensure_2d"), StepConfig("labels.ensure_onehot")))
    r = preprocess(ds, plan, seed=0, fit_indices=np.arange(6, dtype=np.int64), cache=False)
    oh = r.train_artifacts.require("labels.y_onehot")
    mask = r.train_artifacts.require("labels.is_labeled")
    assert oh.shape[0] == 6
    assert mask.dtype == bool
    assert mask.sum() == 4


def test_graph_sparsify_deterministic() -> None:
    ds = _graph_dataset()
    plan = PreprocessPlan(
        steps=(
            StepConfig("graph.attach_edge_weight"),
            StepConfig("graph.edge_sparsify", params={"keep_fraction": 0.5}),
        )
    )
    r1 = preprocess(ds, plan, seed=123, cache=False)
    r2 = preprocess(ds, plan, seed=123, cache=False)

    e1 = r1.dataset.train.edges
    e2 = r2.dataset.train.edges
    assert isinstance(e1, dict)
    assert isinstance(e2, dict)
    assert np.array_equal(e1["edge_index"], e2["edge_index"])
    assert np.array_equal(e1["edge_weight"], e2["edge_weight"])
