from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.embeddings.auto import (
    AutoEmbeddingStep,
    _first_item,
    _is_audio_path,
    _looks_like_audio,
    _looks_like_text,
)
from modssc.preprocess.steps.graph.attach_edge_weight import AttachEdgeWeightStep
from modssc.preprocess.steps.graph.edge_sparsify import EdgeSparsifyStep
from modssc.preprocess.store import ArtifactStore


@pytest.fixture
def mock_encoder():
    with patch("modssc.preprocess.steps.embeddings.auto.load_encoder") as mock:
        encoder_instance = MagicMock()
        mock.return_value = encoder_instance
        yield encoder_instance


def test_auto_embedding_text(mock_encoder):
    store = ArtifactStore()
    store.set("raw.X", ["hello", "world"])

    step = AutoEmbeddingStep(model_id_text="my-text-model")
    rng = np.random.default_rng(42)

    mock_encoder.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    res = step.transform(store, rng=rng)

    assert "features.X" in res
    assert np.allclose(res["features.X"], [[0.1, 0.2], [0.3, 0.4]])
    step._get_encoder("text")
    mock_encoder.encode.assert_called_once()
    args, kwargs = mock_encoder.encode.call_args
    assert args[0] == ["hello", "world"]


def test_auto_embedding_images(mock_encoder):
    store = ArtifactStore()

    images = np.zeros((2, 3, 32, 32), dtype=np.float32)
    store.set("raw.X", images)

    step = AutoEmbeddingStep(model_id_vision="my-vision-model")
    rng = np.random.default_rng(42)

    mock_encoder.encode.return_value = np.array([[1.0], [2.0]])

    res = step.transform(store, rng=rng)

    assert "features.X" in res
    mock_encoder.encode.assert_called_once()

    assert np.array_equal(mock_encoder.encode.call_args[0][0], images)
    step._get_encoder("vision")


def test_auto_embedding_audio(mock_encoder):
    store = ArtifactStore()

    audio = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    store.set("raw.X", audio)

    step = AutoEmbeddingStep(model_id_audio="my-audio-model")
    rng = np.random.default_rng(42)

    mock_encoder.encode.return_value = np.array([[5.0], [6.0]])

    res = step.transform(store, rng=rng)

    assert "features.X" in res
    mock_encoder.encode.assert_called_once()

    assert isinstance(mock_encoder.encode.call_args[0][0], list)
    step._get_encoder("audio")


def test_auto_embedding_numeric_pass_through():
    store = ArtifactStore()

    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    store.set("raw.X", data)

    step = AutoEmbeddingStep()
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    assert np.allclose(res["features.X"], data)


def test_auto_embedding_numeric_reshape_1d():
    store = ArtifactStore()
    data = np.array([1.0, 2.0])
    store.set("raw.X", data)

    step = AutoEmbeddingStep()
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 1)
    assert np.allclose(res["features.X"], [[1.0], [2.0]])


def test_auto_embedding_numeric_reshape_3d():
    store = ArtifactStore()

    data = np.zeros((2, 1, 1, 1, 2))
    store.set("raw.X", data)

    step = AutoEmbeddingStep()
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 2)


def test_auto_embedding_error_scalar():
    store = ArtifactStore()
    store.set("raw.X", np.array(1.0))

    step = AutoEmbeddingStep()
    rng = np.random.default_rng(42)

    with pytest.raises(PreprocessValidationError, match="Cannot embed scalar"):
        step.transform(store, rng=rng)


def test_auto_embedding_error_object():
    store = ArtifactStore()

    store.set("raw.X", np.array([object(), object()], dtype=object))

    step = AutoEmbeddingStep()
    rng = np.random.default_rng(42)

    with pytest.raises(PreprocessValidationError, match="could not infer modality"):
        step.transform(store, rng=rng)


def test_auto_embedding_list_numeric():
    store = ArtifactStore()

    store.set("raw.X", [1.0, 2.0])

    step = AutoEmbeddingStep()
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)

    assert res["features.X"].shape == (2, 1)


def test_auto_embedding_helpers():
    assert _first_item(np.array([], dtype=np.float32)) is None
    assert _first_item([]) is None
    assert _first_item(123) is None
    assert _is_audio_path("file.wav")

    class BadStr:
        def __str__(self):
            raise ValueError("boom")

    assert _is_audio_path(BadStr()) is False
    assert _looks_like_text(np.array([], dtype=str)) is False
    assert _looks_like_text(np.array(["note"], dtype=str)) is True
    assert _looks_like_audio(["/tmp/a.wav"]) is True


def test_auto_embedding_is_audio_path_exception(monkeypatch):
    import modssc.preprocess.steps.embeddings.auto as auto_mod

    class BadPath:
        def __init__(self, *args, **kwargs):
            raise ValueError("boom")

    monkeypatch.setattr(auto_mod, "Path", BadPath)
    assert auto_mod._is_audio_path("file.wav") is False


def test_auto_embedding_looks_like_text_numpy_branch(monkeypatch):
    import modssc.preprocess.steps.embeddings.auto as auto_mod

    monkeypatch.setattr(auto_mod, "_first_item", lambda X: None)
    assert auto_mod._looks_like_text(np.array(["note"], dtype=str)) is True


def test_auto_embedding_unknown_encoder_kind():
    step = AutoEmbeddingStep()
    with pytest.raises(PreprocessValidationError, match="Unknown encoder kind"):
        step._get_encoder("unknown")


def _load_graph_node2vec_step(monkeypatch):
    import importlib

    import modssc.transductive.optional as optional_mod

    monkeypatch.setattr(optional_mod, "optional_import", lambda *a, **k: MagicMock())
    return importlib.import_module("modssc.preprocess.steps.graph.node2vec").GraphNode2VecStep


def test_attach_edge_weight_existing():
    store = ArtifactStore()
    store.set("graph.edge_weight", np.array([0.5, 0.5]))

    step = AttachEdgeWeightStep(weight=2.0)
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    assert np.allclose(res["graph.edge_weight"], [0.5, 0.5])


def test_attach_edge_weight_create_2_E():
    store = ArtifactStore()

    edge_index = np.array([[0, 1, 2], [1, 2, 0]])
    store.set("graph.edge_index", edge_index)

    step = AttachEdgeWeightStep(weight=0.8)
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    w = res["graph.edge_weight"]
    assert w.shape == (3,)
    assert np.allclose(w, 0.8)


def test_attach_edge_weight_create_E_2():
    store = ArtifactStore()

    edge_index = np.array([[0, 1], [1, 2], [2, 0]])
    store.set("graph.edge_index", edge_index)

    step = AttachEdgeWeightStep(weight=0.8)
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    w = res["graph.edge_weight"]
    assert w.shape == (3,)
    assert np.allclose(w, 0.8)


def test_attach_edge_weight_1d_list():
    store = ArtifactStore()

    store.set("graph.edge_index", [0, 1, 2])

    step = AttachEdgeWeightStep(weight=0.5)
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    w = res["graph.edge_weight"]
    assert w.shape == (3,)
    assert np.allclose(w, 0.5)


def test_edge_sparsify_validation():
    step = EdgeSparsifyStep(keep_fraction=1.5)
    store = ArtifactStore()
    rng = np.random.default_rng(42)
    with pytest.raises(PreprocessValidationError, match="keep_fraction"):
        step.transform(store, rng=rng)

    step = EdgeSparsifyStep(keep_fraction=0.0)
    with pytest.raises(PreprocessValidationError, match="keep_fraction"):
        step.transform(store, rng=rng)


def test_edge_sparsify_logic_2_E():
    store = ArtifactStore()

    edge_index = np.zeros((2, 10), dtype=int)
    edge_weight = np.ones(10, dtype=float)
    store.set("graph.edge_index", edge_index)
    store.set("graph.edge_weight", edge_weight)

    step = EdgeSparsifyStep(keep_fraction=0.5)
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)

    ei = res["graph.edge_index"]
    ew = res["graph.edge_weight"]

    assert ei.shape[0] == 2
    assert ei.shape[1] < 10
    assert ei.shape[1] == ew.shape[0]


def test_edge_sparsify_logic_E_2():
    store = ArtifactStore()

    edge_index = np.zeros((10, 2), dtype=int)
    store.set("graph.edge_index", edge_index)

    step = EdgeSparsifyStep(keep_fraction=0.5)
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)

    ei = res["graph.edge_index"]
    assert ei.shape[1] == 2
    assert ei.shape[0] < 10


def test_edge_sparsify_ensure_one():
    store = ArtifactStore()

    edge_index = np.zeros((2, 2), dtype=int)
    store.set("graph.edge_index", edge_index)

    step = EdgeSparsifyStep(keep_fraction=0.0001)
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    ei = res["graph.edge_index"]

    assert ei.shape[1] >= 1


def test_edge_sparsify_ensure_one_E_2():
    store = ArtifactStore()

    edge_index = np.zeros((3, 2), dtype=int)
    store.set("graph.edge_index", edge_index)

    step = EdgeSparsifyStep(keep_fraction=0.0001)
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    ei = res["graph.edge_index"]

    assert ei.shape[0] >= 1


def test_edge_sparsify_3d():
    store = ArtifactStore()

    edge_index = np.zeros((4, 2, 2), dtype=int)
    store.set("graph.edge_index", edge_index)

    step = EdgeSparsifyStep(keep_fraction=0.5)
    rng = np.random.default_rng(42)

    res = step.transform(store, rng=rng)
    ei = res["graph.edge_index"]

    assert ei.ndim == 3
    assert ei.shape[0] < 4


def test_node2vec_missing_edge_index(monkeypatch):
    GraphNode2VecStep = _load_graph_node2vec_step(monkeypatch)
    step = GraphNode2VecStep()
    store = ArtifactStore()
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="graph.edge_index"):
        step.transform(store, rng=rng)


def test_node2vec_missing_n_nodes(monkeypatch):
    GraphNode2VecStep = _load_graph_node2vec_step(monkeypatch)
    step = GraphNode2VecStep()
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0]]))
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="raw.y or raw.X"):
        step.transform(store, rng=rng)


def test_node2vec_invalid_edge_index_shape(monkeypatch):
    GraphNode2VecStep = _load_graph_node2vec_step(monkeypatch)
    step = GraphNode2VecStep()
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([1, 2, 3]))
    store.set("raw.y", np.array([0, 1, 0]))
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="2D array"):
        step.transform(store, rng=rng)

    store.set("graph.edge_index", np.zeros((3, 3), dtype=np.int64))
    with pytest.raises(PreprocessValidationError, match="shape \\(2, E\\)"):
        step.transform(store, rng=rng)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"num_walks": 0}, "num_walks"),
        ({"walk_length": 1}, "walk_length"),
        ({"window_size": 0}, "window_size"),
        ({"num_negative": 0}, "num_negative"),
        ({"batch_size": 0}, "batch_size"),
        ({"embed_epochs": 0}, "embed_epochs"),
    ],
)
def test_node2vec_invalid_params(monkeypatch, kwargs, match):
    GraphNode2VecStep = _load_graph_node2vec_step(monkeypatch)
    step = GraphNode2VecStep(**kwargs)
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0]]))
    store.set("raw.y", np.array([0, 1]))
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match=match):
        step.transform(store, rng=rng)


def test_node2vec_empty_pairs(monkeypatch):
    GraphNode2VecStep = _load_graph_node2vec_step(monkeypatch)
    step = GraphNode2VecStep()
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0]]))
    store.set("raw.y", np.array([0, 1]))
    rng = np.random.default_rng(0)

    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.node2vec._build_adjacency",
        lambda *a, **k: [[1], [0]],
    )
    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.node2vec._random_walks_node2vec",
        lambda *a, **k: np.array([[0, 1]]),
    )
    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.node2vec._walk_pairs",
        lambda *a, **k: (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
    )

    with pytest.raises(PreprocessValidationError, match="training pairs"):
        step.transform(store, rng=rng)


def test_node2vec_happy_path(monkeypatch):
    GraphNode2VecStep = _load_graph_node2vec_step(monkeypatch)
    step = GraphNode2VecStep(embedding_dim=4, num_walks=1, walk_length=2, window_size=1)
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0], [0, 1]]))
    store.set("raw.X", np.zeros((2, 3), dtype=np.float32))
    rng = np.random.default_rng(0)

    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.node2vec._build_adjacency",
        lambda *a, **k: [[1], [0]],
    )
    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.node2vec._random_walks_node2vec",
        lambda *a, **k: np.array([[0, 1]]),
    )
    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.node2vec._walk_pairs",
        lambda *a, **k: (np.array([0, 1], dtype=np.int64), np.array([1, 0], dtype=np.int64)),
    )
    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.node2vec._sample_negatives",
        lambda *a, **k: np.zeros((2, 1), dtype=np.int64),
    )

    class FakeTensor:
        def __init__(self, data):
            self.data = np.asarray(data)

        @property
        def shape(self):
            return self.data.shape

        def to(self, device):
            return self

        def __getitem__(self, idx):
            return FakeTensor(self.data[idx])

        def __len__(self):
            return int(self.data.shape[0])

        def __mul__(self, other):
            return FakeTensor(self.data * other.data)

        def __add__(self, other):
            return FakeTensor(self.data + other.data)

        def __sub__(self, other):
            return FakeTensor(self.data - other.data)

        def __neg__(self):
            return FakeTensor(-self.data)

        def sum(self, dim=-1):
            return FakeTensor(self.data.sum(axis=dim))

        def unsqueeze(self, dim):
            return FakeTensor(np.expand_dims(self.data, axis=dim))

        def mean(self):
            return FakeTensor(np.array(self.data.mean()))

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.data)

        def detach(self):
            return self

        def backward(self):
            return None

    class FakeEmbedding:
        def __init__(self, n, dim):
            self.weight = FakeTensor(np.zeros((n, dim), dtype=np.float32))

        def to(self, device):
            return self

        def parameters(self):
            return [self.weight]

        def __call__(self, idx):
            data = idx.data if isinstance(idx, FakeTensor) else np.asarray(idx)
            return FakeTensor(self.weight.data[data])

    class FakeAdam:
        def __init__(self, params, lr):
            self.params = params
            self.lr = lr

        def zero_grad(self):
            return None

        def step(self):
            return None

    class FakeTorch:
        long = "long"

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def as_tensor(data, dtype=None, device=None):
            return FakeTensor(data)

        class nn:
            Embedding = FakeEmbedding

            class functional:
                @staticmethod
                def logsigmoid(x):
                    return x if isinstance(x, FakeTensor) else FakeTensor(x)

        class optim:
            Adam = FakeAdam

    monkeypatch.setattr("modssc.preprocess.steps.graph.node2vec.require", lambda *a, **k: FakeTorch)

    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 4)


def test_dgi_ensure_2d_reshape_1d(monkeypatch):
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep(embedding_dim=2, hidden_dim=2, unsup_epochs=1)
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0]]))
    store.set("raw.X", np.array([1.0, 2.0], dtype=np.float32))
    rng = np.random.default_rng(0)

    seen = {}

    def fake_train(X, *args, **kwargs):
        seen["shape"] = X.shape
        return np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr("modssc.preprocess.steps.graph.dgi._train_dgi", fake_train)

    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 2)
    assert seen["shape"] == (2, 1)


def test_dgi_rejects_3d_features():
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep()
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0]]))
    store.set("raw.X", np.zeros((2, 1, 1), dtype=np.float32))
    rng = np.random.default_rng(0)

    with pytest.raises(PreprocessValidationError, match="2D features"):
        step.transform(store, rng=rng)


def test_dgi_requires_numeric_features():
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep()
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0]]))
    store.set("raw.X", ["bad"])
    rng = np.random.default_rng(0)

    with pytest.raises(PreprocessValidationError, match="numeric features"):
        step.transform(store, rng=rng)


def test_dgi_edge_index_transpose_and_empty():
    from modssc.preprocess.steps.graph.dgi import _as_edge_index

    edge_index = np.array([[0, 1], [1, 2], [2, 0]])
    out = _as_edge_index(edge_index, n_nodes=3)
    assert out.shape == (2, 3)
    assert np.array_equal(out, edge_index.T)

    empty = np.zeros((2, 0), dtype=np.int64)
    out_empty = _as_edge_index(empty, n_nodes=3)
    assert out_empty.shape == (2, 0)


def test_dgi_edge_weight_length_mismatch():
    from modssc.preprocess.steps.graph.dgi import _as_edge_weight

    with pytest.raises(PreprocessValidationError, match="length mismatch"):
        _as_edge_weight([0.5], n_edges=2)


def test_dgi_edge_weight_provided(monkeypatch):
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep(embedding_dim=2, hidden_dim=2, unsup_epochs=1)
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 2], [2, 0]]))
    store.set("graph.edge_weight", np.array([0.1, 0.2, 0.3], dtype=np.float32))
    store.set("raw.X", np.zeros((3, 2), dtype=np.float32))
    rng = np.random.default_rng(0)

    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.dgi._train_dgi",
        lambda *a, **k: np.ones((3, 2), dtype=np.float32),
    )

    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (3, 2)


@pytest.mark.parametrize("add_self_loops", [True, False])
def test_train_dgi_torch_runs(add_self_loops):
    pytest.importorskip("torch")
    from modssc.preprocess.steps.graph.dgi import _train_dgi

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    edge_weight = np.ones((3,), dtype=np.float32)

    emb = _train_dgi(
        X,
        edge_index,
        edge_weight,
        embedding_dim=2,
        hidden_dim=2,
        dropout=0.0,
        unsup_epochs=1,
        unsup_lr=0.01,
        add_self_loops=add_self_loops,
        device="cpu",
        seed=0,
    )
    assert emb.shape == (3, 2)
    assert emb.dtype == np.float32


def test_dgi_missing_edge_index():
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep()
    store = ArtifactStore()
    store.set("raw.X", np.zeros((2, 3), dtype=np.float32))
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="graph.edge_index"):
        step.transform(store, rng=rng)


def test_dgi_missing_features():
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep()
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0]]))
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="raw.X or features.X"):
        step.transform(store, rng=rng)


def test_dgi_invalid_edge_index_shape():
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep()
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([1, 2, 3]))
    store.set("raw.X", np.zeros((2, 3), dtype=np.float32))
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="2D array"):
        step.transform(store, rng=rng)

    store.set("graph.edge_index", np.zeros((3, 3), dtype=np.int64))
    with pytest.raises(PreprocessValidationError, match="shape \\(2, E\\)"):
        step.transform(store, rng=rng)


def test_dgi_edge_index_out_of_range():
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep()
    store = ArtifactStore()
    store.set("raw.X", np.zeros((2, 3), dtype=np.float32))
    store.set("graph.edge_index", np.array([[0, 3], [1, 0]]))
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="out of range"):
        step.transform(store, rng=rng)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"unsup_epochs": 0}, "unsup_epochs"),
        ({"unsup_lr": 0.0}, "unsup_lr"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": -0.1}, "dropout"),
    ],
)
def test_dgi_invalid_params(monkeypatch, kwargs, match):
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep(**kwargs)
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0]]))
    store.set("raw.X", np.zeros((2, 3), dtype=np.float32))
    rng = np.random.default_rng(0)

    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.dgi._train_dgi",
        lambda *a, **k: np.zeros((2, 4), dtype=np.float32),
    )
    with pytest.raises(PreprocessValidationError, match=match):
        step.transform(store, rng=rng)


def test_dgi_happy_path(monkeypatch):
    from modssc.preprocess.steps.graph.dgi import GraphDGIStep

    step = GraphDGIStep(embedding_dim=4, hidden_dim=4, unsup_epochs=1)
    store = ArtifactStore()
    store.set("graph.edge_index", np.array([[0, 1], [1, 0]]))
    store.set("raw.X", np.zeros((2, 3), dtype=np.float32))
    rng = np.random.default_rng(0)

    monkeypatch.setattr(
        "modssc.preprocess.steps.graph.dgi._train_dgi",
        lambda *a, **k: np.ones((2, 4), dtype=np.float32),
    )
    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 4)
