from __future__ import annotations

import builtins
import importlib
from types import ModuleType

import numpy as np
import pytest

from modssc.graph import GraphBuilderSpec, GraphWeightsSpec, build_graph
from modssc.graph.errors import OptionalDependencyError


def test_auto_backend_falls_back_to_numpy_when_sklearn_missing(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3)).astype(np.float32)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ImportError("sklearn intentionally missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=4,
        symmetrize="none",
        self_loops=False,
        normalize="none",
        weights=GraphWeightsSpec(kind="binary"),
        backend="auto",
        chunk_size=8,
    )

    g = build_graph(X, spec=spec, seed=0, cache=False)
    assert g.edge_index.shape[1] == 20 * 4


def test_faiss_backend_raises_optional_dependency_error_when_missing(monkeypatch) -> None:
    import modssc.graph.optional as opt

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> ModuleType:
        if name == "faiss":
            raise ModuleNotFoundError("faiss intentionally missing")
        return real_import_module(name, package)

    monkeypatch.setattr(opt.importlib, "import_module", fake_import_module)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4)).astype(np.float32)

    spec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=3,
        symmetrize="none",
        self_loops=False,
        normalize="none",
        weights=GraphWeightsSpec(kind="binary"),
        backend="faiss",
    )

    with pytest.raises(OptionalDependencyError):
        _ = build_graph(X, spec=spec, seed=0, cache=False)


def test_faiss_backend_runs_if_available() -> None:
    try:
        importlib.import_module("faiss")
    except Exception:
        pytest.skip("faiss is not installed")

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 6)).astype(np.float32)

    spec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=5,
        symmetrize="none",
        self_loops=False,
        normalize="none",
        weights=GraphWeightsSpec(kind="binary"),
        backend="faiss",
        faiss_exact=True,
    )

    g = build_graph(X, spec=spec, seed=0, cache=False)
    assert g.edge_index.shape[1] == 30 * 5
