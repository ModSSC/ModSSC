from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.construction.builder import build_raw_edges
from modssc.graph.errors import GraphValidationError
from modssc.graph.specs import GraphBuilderSpec


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        pytest.skip(f"sklearn unavailable: {exc}")


@pytest.fixture
def sample_data():
    return np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])


def test_knn_missing_k(sample_data):
    spec = MagicMock(spec=GraphBuilderSpec)
    spec.scheme = "knn"
    spec.k = None
    spec.backend = "numpy"
    spec.validate.return_value = None

    with pytest.raises(GraphValidationError, match="k must be set for knn scheme"):
        build_raw_edges(sample_data, spec=spec, seed=42)


def test_knn_unknown_backend(sample_data):
    spec = MagicMock(spec=GraphBuilderSpec)
    spec.scheme = "knn"
    spec.k = 5
    spec.backend = "invalid_backend"
    spec.validate.return_value = None

    with pytest.raises(GraphValidationError, match="Unknown backend"):
        build_raw_edges(sample_data, spec=spec, seed=42)


def test_epsilon_missing_radius(sample_data):
    spec = MagicMock(spec=GraphBuilderSpec)
    spec.scheme = "epsilon"
    spec.radius = None
    spec.backend = "numpy"
    spec.validate.return_value = None

    with pytest.raises(GraphValidationError, match="radius must be set for epsilon scheme"):
        build_raw_edges(sample_data, spec=spec, seed=42)


def test_epsilon_faiss_backend(sample_data):
    spec = MagicMock(spec=GraphBuilderSpec)
    spec.scheme = "epsilon"
    spec.radius = 0.5
    spec.backend = "faiss"
    spec.validate.return_value = None

    with pytest.raises(GraphValidationError, match="faiss backend does not support epsilon scheme"):
        build_raw_edges(sample_data, spec=spec, seed=42)


def test_epsilon_sklearn_backend(sample_data):
    spec = GraphBuilderSpec(scheme="epsilon", radius=0.5, backend="sklearn")

    with patch("modssc.graph.construction.builder.epsilon_edges_sklearn") as mock_sklearn:
        mock_sklearn.return_value = (np.array([[0, 1], [1, 0]]), np.array([0.1, 0.1]))
        build_raw_edges(sample_data, spec=spec, seed=42)
        mock_sklearn.assert_called_once()


def test_epsilon_unknown_backend(sample_data):
    spec = MagicMock(spec=GraphBuilderSpec)
    spec.scheme = "epsilon"
    spec.radius = 0.5
    spec.backend = "invalid_backend"
    spec.validate.return_value = None

    with pytest.raises(GraphValidationError, match="Unknown backend"):
        build_raw_edges(sample_data, spec=spec, seed=42)


def test_anchor_missing_k(sample_data):
    spec = MagicMock(spec=GraphBuilderSpec)
    spec.scheme = "anchor"
    spec.k = None
    spec.backend = "numpy"
    spec.validate.return_value = None

    with pytest.raises(GraphValidationError, match="k must be set for anchor scheme"):
        build_raw_edges(sample_data, spec=spec, seed=42)


def test_anchor_auto_backend(sample_data):
    spec = GraphBuilderSpec(scheme="anchor", k=5, backend="auto")
    with patch("modssc.graph.construction.builder.anchor_edges") as mock_anchor:
        mock_anchor.return_value = (np.array([[0, 1], [1, 0]]), np.array([0.1, 0.1]))
        build_raw_edges(sample_data, spec=spec, seed=42)
        mock_anchor.assert_called_once()


def test_anchor_unknown_backend(sample_data):
    spec = MagicMock(spec=GraphBuilderSpec)
    spec.scheme = "anchor"
    spec.k = 5
    spec.backend = "invalid"
    spec.validate.return_value = None

    with pytest.raises(GraphValidationError, match="Unknown backend"):
        build_raw_edges(sample_data, spec=spec, seed=42)


def test_unknown_scheme(sample_data):
    spec = MagicMock(spec=GraphBuilderSpec)
    spec.scheme = "invalid_scheme"
    spec.backend = "numpy"
    spec.validate.return_value = None

    with pytest.raises(GraphValidationError, match="Unknown scheme"):
        build_raw_edges(sample_data, spec=spec, seed=42)


def test_invalid_edge_index_shape(sample_data):
    spec = GraphBuilderSpec(scheme="knn", k=5, backend="numpy")
    with patch("modssc.graph.construction.builder.knn_edges_numpy") as mock_knn:
        mock_knn.return_value = (np.array([0, 1]), np.array([0.1, 0.1]))
        with pytest.raises(GraphValidationError, match="edge_index must have shape"):
            build_raw_edges(sample_data, spec=spec, seed=42)


def test_invalid_dist_shape(sample_data):
    spec = GraphBuilderSpec(scheme="knn", k=5, backend="numpy")
    with patch("modssc.graph.construction.builder.knn_edges_numpy") as mock_knn:
        mock_knn.return_value = (np.array([[0, 1], [1, 0]]), np.array([0.1]))
        with pytest.raises(GraphValidationError, match="distances must have shape"):
            build_raw_edges(sample_data, spec=spec, seed=42)


def test_pick_backend_auto_no_sklearn():
    GraphBuilderSpec(scheme="knn", k=5, backend="auto")
    with patch.dict("sys.modules", {"sklearn": None}):
        pass

    with patch("builtins.__import__", side_effect=ImportError):
        pass


def test_pick_backend_explicit():
    from modssc.graph.construction.builder import _pick_backend

    spec = GraphBuilderSpec(scheme="knn", k=5, backend="faiss")
    assert _pick_backend(spec) == "faiss"


def test_pick_backend_auto_with_sklearn():
    _require_sklearn()
    from modssc.graph.construction.builder import _pick_backend

    spec = GraphBuilderSpec(scheme="knn", k=5, backend="auto")

    assert _pick_backend(spec) == "sklearn"


def test_pick_backend_auto_without_sklearn():
    from modssc.graph.construction.builder import _pick_backend

    spec = GraphBuilderSpec(scheme="knn", k=5, backend="auto")

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "sklearn":
            raise ImportError("No module named 'sklearn'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        assert _pick_backend(spec) == "numpy"


def test_work_dir_creation(sample_data, tmp_path):
    spec = GraphBuilderSpec(scheme="knn", k=5, backend="sklearn")
    work_dir = tmp_path / "work"
    with patch("modssc.graph.construction.builder.knn_edges_sklearn") as mock_knn:
        mock_knn.return_value = (np.array([[0, 1], [1, 0]]), np.array([0.1, 0.1]))
        build_raw_edges(sample_data, spec=spec, seed=42, work_dir=work_dir)
        assert work_dir.exists()


def test_knn_faiss_backend(sample_data):
    spec = GraphBuilderSpec(scheme="knn", k=5, backend="faiss")
    with patch("modssc.graph.construction.builder.knn_edges_faiss") as mock_faiss:
        mock_faiss.return_value = (np.array([[0, 1], [1, 0]]), np.array([0.1, 0.1]))
        build_raw_edges(sample_data, spec=spec, seed=42)
        mock_faiss.assert_called_once()


def test_knn_numpy_backend(sample_data):
    spec = GraphBuilderSpec(scheme="knn", k=5, backend="numpy")
    with patch("modssc.graph.construction.builder.knn_edges_numpy") as mock_numpy:
        mock_numpy.return_value = (np.array([[0, 1], [1, 0]]), np.array([0.1, 0.1]))
        build_raw_edges(sample_data, spec=spec, seed=42)
        mock_numpy.assert_called_once()


def test_epsilon_numpy_backend(sample_data):
    spec = GraphBuilderSpec(scheme="epsilon", radius=0.5, backend="numpy")
    with patch("modssc.graph.construction.builder.epsilon_edges_numpy") as mock_numpy:
        mock_numpy.return_value = (np.array([[0, 1], [1, 0]]), np.array([0.1, 0.1]))
        build_raw_edges(sample_data, spec=spec, seed=42)
        mock_numpy.assert_called_once()


def test_anchor_auto_backend_resolution(sample_data):
    spec = GraphBuilderSpec(scheme="anchor", k=5, backend="auto")
    with patch("modssc.graph.construction.builder.anchor_edges") as mock_anchor:
        mock_anchor.return_value = (np.array([[0, 1], [1, 0]]), np.array([0.1, 0.1]))
        build_raw_edges(sample_data, spec=spec, seed=42)

        args, kwargs = mock_anchor.call_args
        assert kwargs["backend"] in ("sklearn", "numpy")
