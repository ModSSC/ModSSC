from __future__ import annotations

import numpy as np
import pytest

import modssc.graph.runtime as graph_runtime
from modssc.data_loader.types import LoadedDataset, Split
from modssc.graph.artifacts import GraphArtifact
from modssc.graph.errors import GraphValidationError
from modssc.graph.runtime import build_graph_from_preprocess, summarize_graph
from modssc.graph.specs import GraphBuilderSpec, GraphWeightsSpec
from modssc.preprocess.store import ArtifactStore
from modssc.preprocess.types import PreprocessResult, ResolvedPlan


def _preprocess_result(
    features: np.ndarray,
    *,
    declared_sha256: str | None = None,
    producer_identity: str | None = None,
) -> PreprocessResult:
    artifacts = ArtifactStore()
    artifacts.set("features.vae", features)
    if declared_sha256 is not None:
        artifacts.set(
            "features.vae.info",
            {
                "output": {
                    "content_sha256": declared_sha256,
                    "identity_fingerprint": producer_identity,
                }
            },
        )
    dataset = LoadedDataset(
        train=Split(X=features, y=np.arange(features.shape[0], dtype=np.int64) % 2),
        meta={"dataset_fingerprint": "dataset:fixed"},
    )
    return PreprocessResult(
        dataset=dataset,
        plan=ResolvedPlan(steps=()),
        preprocess_fingerprint="preprocess:semantic-only",
        train_artifacts=artifacts,
    )


def _graph_spec() -> GraphBuilderSpec:
    return GraphBuilderSpec(
        scheme="knn",
        metric="euclidean",
        k=2,
        symmetrize="none",
        weights=GraphWeightsSpec(kind="binary"),
        normalize="none",
        self_loops=False,
        backend="numpy",
        feature_field="features.vae",
    )


def _build(pre: PreprocessResult, *, cache_dir, expected: str | None = None):
    return build_graph_from_preprocess(
        pre,
        spec=_graph_spec(),
        seed=7,
        dataset_fingerprint="dataset:fixed",
        cache=True,
        require_cache_hit=False,
        cache_dir=cache_dir,
        include_test=False,
        expected_preprocess_fingerprint=expected,
    )


def test_graph_cache_identity_changes_when_latent_content_changes(tmp_path) -> None:
    first_features = np.arange(24, dtype=np.float32).reshape(8, 3)
    changed_features = first_features.copy()
    changed_features[4, 1] += 0.25

    first = _build(_preprocess_result(first_features), cache_dir=tmp_path)
    changed = _build(_preprocess_result(changed_features), cache_dir=tmp_path)

    assert first.meta["preprocess_fingerprint"].startswith("preprocess-content-v2:")
    assert first.meta["preprocess_semantic_fingerprint"] == "preprocess:semantic-only"
    assert first.meta["feature_identity_schema_version"] == 2
    assert first.meta["fingerprint"] != changed.meta["fingerprint"]
    assert first.meta["feature_content_sha256"] != changed.meta["feature_content_sha256"]


def test_graph_content_identity_replaces_legacy_semantic_pin(tmp_path) -> None:
    features = np.arange(24, dtype=np.float32).reshape(8, 3)
    pre = _preprocess_result(features)
    first = _build(pre, cache_dir=tmp_path)
    content_bound_pin = str(first.meta["preprocess_fingerprint"])

    pinned = _build(pre, cache_dir=tmp_path, expected=content_bound_pin)
    assert pinned.meta["fingerprint"] == first.meta["fingerprint"]

    with pytest.raises(GraphValidationError, match="preprocessing fingerprint differs"):
        _build(pre, cache_dir=tmp_path, expected="preprocess:semantic-only")


def test_graph_rejects_latent_mutation_after_vae_attestation(tmp_path) -> None:
    features = np.arange(24, dtype=np.float32).reshape(8, 3)
    declared_sha256 = graph_runtime._array_sha256(features)
    pre = _preprocess_result(
        features,
        declared_sha256=declared_sha256,
        producer_identity="vae_output_attested",
    )
    features[2, 2] += 1.0

    with pytest.raises(GraphValidationError, match="changed after preprocessing"):
        _build(pre, cache_dir=tmp_path)


def test_graph_identity_records_validated_vae_producer_identity(tmp_path) -> None:
    features = np.arange(24, dtype=np.float32).reshape(8, 3)
    pre = _preprocess_result(
        features,
        declared_sha256=graph_runtime._array_sha256(features),
        producer_identity="vae_output_attested",
    )

    graph = _build(pre, cache_dir=tmp_path)

    assert (
        graph.meta["feature_identity"]["splits"]["train"]["producer_identity_fingerprint"]
        == "vae_output_attested"
    )


def test_graph_numpy_conversion_and_numeric_hash_contract() -> None:
    calls: list[str] = []

    class TensorLike:
        def detach(self):
            calls.append("detach")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

        def numpy(self):
            calls.append("numpy")
            return np.arange(4, dtype=np.float32)

    np.testing.assert_array_equal(graph_runtime._to_numpy(TensorLike()), np.arange(4))
    np.testing.assert_array_equal(graph_runtime._to_numpy([1, 2]), np.array([1, 2]))
    assert calls == ["detach", "cpu", "numpy"]

    with pytest.raises(GraphValidationError, match="numeric array"):
        graph_runtime._array_sha256(np.array([object()], dtype=object))


def test_feature_lookup_covers_dataset_fallbacks_and_rejects_unknown_split() -> None:
    features = np.arange(6, dtype=np.float32).reshape(3, 2)
    test_features = features + 10
    pre = _preprocess_result(features)
    pre_with_test = PreprocessResult(
        dataset=LoadedDataset(
            train=pre.dataset.train,
            test=Split(X=test_features, y=np.array([0, 1, 0])),
        ),
        plan=pre.plan,
        preprocess_fingerprint=pre.preprocess_fingerprint,
        train_artifacts=ArtifactStore(),
        test_artifacts=ArtifactStore({"features.vae": test_features + 1}),
    )

    assert (
        graph_runtime._feature_from_artifacts(pre_with_test, "features.missing", split="train")
        is pre.dataset.train.X
    )
    np.testing.assert_array_equal(
        graph_runtime._feature_from_artifacts(pre_with_test, "features.vae", split="test"),
        test_features + 1,
    )
    np.testing.assert_array_equal(
        graph_runtime._feature_from_artifacts(pre_with_test, "features.missing", split="test"),
        test_features,
    )
    assert graph_runtime._feature_from_artifacts(pre, "features.missing", split="test") is None
    with pytest.raises(ValueError, match="Unknown split"):
        graph_runtime._feature_from_artifacts(pre, "features.vae", split="validation")


def test_feature_identity_handles_absent_or_unstructured_producer_metadata() -> None:
    features = np.arange(6, dtype=np.float32).reshape(3, 2)
    pre = _preprocess_result(features)
    pre.train_artifacts.set("features.vae.info", "legacy")

    identity = graph_runtime._feature_producer_identity(
        pre,
        key="features.vae",
        split="train",
        feature_array=features,
    )

    assert identity["producer_identity_fingerprint"] is None
    assert identity["shape"] == [3, 2]


def test_content_identity_includes_test_split_attestation() -> None:
    train = np.arange(6, dtype=np.float32).reshape(3, 2)
    test = train[:2] + 10
    train_store = ArtifactStore({"features.vae": train})
    test_store = ArtifactStore(
        {
            "features.vae": test,
            "features.vae.info": {"output": {"identity_fingerprint": "test-producer"}},
        }
    )
    pre = PreprocessResult(
        dataset=LoadedDataset(
            train=Split(X=train, y=np.array([0, 1, 0])),
            test=Split(X=test, y=np.array([1, 0])),
        ),
        plan=ResolvedPlan(steps=()),
        preprocess_fingerprint="semantic",
        train_artifacts=train_store,
        test_artifacts=test_store,
    )

    graph = build_graph_from_preprocess(
        pre,
        spec=GraphBuilderSpec(
            k=1,
            metric="euclidean",
            symmetrize="none",
            weights=GraphWeightsSpec(kind="binary"),
            normalize="none",
            self_loops=False,
            backend="numpy",
            feature_field="features.vae",
        ),
        seed=1,
        dataset_fingerprint="dataset",
        cache=False,
        require_cache_hit=False,
        cache_dir=None,
        include_test=True,
    )

    assert graph.n_nodes == 5
    assert (
        graph.meta["feature_identity"]["splits"]["test"]["producer_identity_fingerprint"]
        == "test-producer"
    )


def test_graph_summary_covers_empty_edgeless_connected_and_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = GraphArtifact(n_nodes=0, edge_index=np.empty((2, 0), dtype=np.int64))
    assert summarize_graph(empty, None)["connected_components"] == 0

    edgeless = GraphArtifact(n_nodes=2, edge_index=np.empty((2, 0), dtype=np.int64))
    summary = summarize_graph(edgeless, {"weights": "legacy", "k": 3})
    assert summary["connected_components"] == 2
    assert summary["largest_component_fraction"] == 0.5
    assert summary["weights"] == {}

    connected = GraphArtifact(
        n_nodes=4,
        edge_index=np.array([[0, 2, 0, 1], [1, 1, 1, 2]], dtype=np.int64),
    )
    assert summarize_graph(connected, {"weights": {"kind": "binary"}})["connected_components"] == 2

    real_import = __import__

    def without_scipy(name, *args, **kwargs):
        if name.startswith("scipy.sparse"):
            raise ImportError("forced scipy fallback")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_scipy)
    fallback = graph_runtime._connected_component_stats(connected)
    assert fallback == {"connected_components": 2, "largest_component_fraction": 0.75}


def test_connected_component_empty_counts_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = GraphArtifact(
        n_nodes=2,
        edge_index=np.array([[0], [1]], dtype=np.int64),
    )
    monkeypatch.setattr(graph_runtime.np, "bincount", lambda *_args, **_kwargs: np.array([]))

    assert graph_runtime._connected_component_stats(graph) == {
        "connected_components": 0,
        "largest_component_fraction": 0.0,
    }
