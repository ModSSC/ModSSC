from __future__ import annotations

import numpy as np

from modssc.graph import (
    GraphBuilderSpec,
    GraphFeaturizerSpec,
    GraphWeightsSpec,
    build_graph,
    graph_to_views,
)
from modssc.graph.artifacts import NodeDataset


def test_graph_to_views_attr_diffusion_struct(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 8)).astype(np.float32)
    y = rng.integers(0, 3, size=(80,), dtype=np.int64)

    gspec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=8,
        symmetrize="mutual",
        self_loops=True,
        normalize="rw",
        weights=GraphWeightsSpec(kind="heat", sigma=0.5),
        backend="numpy",
        chunk_size=32,
    )
    g = build_graph(X, spec=gspec, seed=0, cache=False)

    ds = NodeDataset(X=X, y=y, graph=g, masks={})

    spec = GraphFeaturizerSpec(
        views=("attr", "diffusion", "struct"),
        diffusion_steps=3,
        diffusion_alpha=0.2,
        struct_method="deepwalk",
        struct_dim=16,
        walk_length=15,
        num_walks_per_node=2,
        window_size=3,
        cache=True,
    )

    views_first = graph_to_views(ds, spec=spec, seed=123, cache_dir=tmp_path)
    assert set(views_first.views.keys()) == {"attr", "diffusion", "struct"}
    assert views_first.views["attr"].shape == X.shape
    assert views_first.views["diffusion"].shape == X.shape
    assert views_first.views["struct"].shape == (80, 16)

    views_second = graph_to_views(ds, spec=spec, seed=123, cache=False, cache_dir=tmp_path)
    np.testing.assert_allclose(views_second.views["struct"], views_first.views["struct"])

    import modssc.graph.featurization.api as api_mod

    def boom(*args, **kwargs):
        raise RuntimeError("struct_embeddings should not be called when cache hit")

    monkeypatch.setattr(api_mod, "struct_embeddings", boom)

    views_cached = graph_to_views(ds, spec=spec, seed=123, cache=True, cache_dir=tmp_path)
    assert views_cached.meta["fingerprint"] == views_first.meta["fingerprint"]
