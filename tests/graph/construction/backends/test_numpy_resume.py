from __future__ import annotations

import numpy as np

from modssc.graph.construction.backends.numpy_backend import epsilon_edges_numpy, knn_edges_numpy


def test_knn_numpy_resume(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 4)).astype(np.float32)

    ei1, d1 = knn_edges_numpy(
        X,
        k=4,
        metric="cosine",
        include_self=False,
        chunk_size=7,
        work_dir=tmp_path,
        resume=False,
    )
    assert any(p.name.startswith("knn_") for p in tmp_path.iterdir())

    ei2, d2 = knn_edges_numpy(
        X,
        k=4,
        metric="cosine",
        include_self=False,
        chunk_size=7,
        work_dir=tmp_path,
        resume=True,
    )
    np.testing.assert_array_equal(ei2, ei1)
    np.testing.assert_allclose(d2, d1)


def test_epsilon_numpy_resume(tmp_path) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3)).astype(np.float32)

    ei1, d1 = epsilon_edges_numpy(
        X,
        radius=1.2,
        metric="euclidean",
        include_self=False,
        chunk_size=6,
        work_dir=tmp_path,
        resume=False,
    )
    assert any(p.name.startswith("eps_") for p in tmp_path.iterdir())

    ei2, d2 = epsilon_edges_numpy(
        X,
        radius=1.2,
        metric="euclidean",
        include_self=False,
        chunk_size=6,
        work_dir=tmp_path,
        resume=True,
    )
    np.testing.assert_array_equal(ei2, ei1)
    np.testing.assert_allclose(d2, d1)
