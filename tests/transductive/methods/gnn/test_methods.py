from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.transductive.errors import TransductiveValidationError  # noqa: E402
from modssc.transductive.methods.gnn.appnp import APPNPMethod  # noqa: E402
from modssc.transductive.methods.gnn.chebnet import (  # noqa: E402
    ChebNetMethod,
    ChebNetSpec,
    _ChebConv,
)
from modssc.transductive.methods.gnn.common import (  # noqa: E402
    _as_edge_index,
    _as_mask,
    _as_numpy,
    _ensure_2d,
    _labels_to_int,
    coalesce_edges,
    normalize_edge_weight,
    set_torch_seed,
)
from modssc.transductive.methods.gnn.gat import GATMethod, _GATConv  # noqa: E402
from modssc.transductive.methods.gnn.gcn import GCNMethod  # noqa: E402
from modssc.transductive.methods.gnn.gcnii import GCNIIMethod  # noqa: E402
from modssc.transductive.methods.gnn.grand import GRANDMethod, GRANDSpec  # noqa: E402
from modssc.transductive.methods.gnn.graphsage import GraphSAGEMethod  # noqa: E402
from modssc.transductive.methods.gnn.planetoid import PlanetoidMethod, PlanetoidSpec  # noqa: E402
from modssc.transductive.methods.gnn.sgc import SGCMethod, SGCSpec  # noqa: E402


@dataclass
class DummyGraph:
    edge_index: np.ndarray
    edge_weight: np.ndarray


@dataclass
class DummyNodeDataset:
    X: np.ndarray
    y: np.ndarray
    graph: DummyGraph
    masks: dict[str, np.ndarray]

    meta: dict


def make_toy_dataset(n_nodes: int = 30, n_classes: int = 3, seed: int = 0) -> DummyNodeDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_nodes, 8)).astype(np.float32)
    y = rng.integers(0, n_classes, size=(n_nodes,), dtype=np.int64)

    src = np.arange(n_nodes, dtype=np.int64)
    dst = (src + 1) % n_nodes
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)

    train_mask = np.zeros(n_nodes, dtype=bool)

    for c in range(n_classes):
        idx = int(np.flatnonzero(y == c)[0])
        train_mask[idx] = True
    val_mask = np.zeros(n_nodes, dtype=bool)
    val_mask[: max(1, n_nodes // 5)] = True
    val_mask[train_mask] = False

    masks = {"train_mask": train_mask, "val_mask": val_mask}
    return DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks=masks,
        meta={},
    )


@pytest.mark.parametrize(
    "method",
    [
        __import__(
            "modssc.transductive.methods.gnn.gcn", fromlist=["GCNMethod", "GCNSpec"]
        ).GCNMethod(
            spec=__import__("modssc.transductive.methods.gnn.gcn", fromlist=["GCNSpec"]).GCNSpec(
                hidden_dim=8,
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.sgc", fromlist=["SGCMethod", "SGCSpec"]
        ).SGCMethod(
            spec=__import__("modssc.transductive.methods.gnn.sgc", fromlist=["SGCSpec"]).SGCSpec(
                k=2,
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.appnp", fromlist=["APPNPMethod", "APPNPSpec"]
        ).APPNPMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.appnp", fromlist=["APPNPSpec"]
            ).APPNPSpec(
                hidden_dim=16,
                k=3,
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.n_gcn", fromlist=["NGCNMethod", "NGCNSpec"]
        ).NGCNMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.n_gcn", fromlist=["NGCNSpec"]
            ).NGCNSpec(
                hidden_dim=8,
                gcn_layers=2,
                K=2,
                r=1,
                classifier="fc",
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.grafn", fromlist=["GraFNMethod", "GraFNSpec"]
        ).GraFNMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.grafn", fromlist=["GraFNSpec"]
            ).GraFNSpec(
                hidden_dims=(8, 8),
                max_epochs=3,
                patience=1,
                drop_feat_strong=0.2,
                drop_edge_strong=0.2,
                drop_feat_weak=0.1,
                drop_edge_weak=0.1,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.graphhop",
            fromlist=["GraphHopMethod", "GraphHopSpec"],
        ).GraphHopMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.graphhop", fromlist=["GraphHopSpec"]
            ).GraphHopSpec(
                hops=2,
                max_iter=2,
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.h_gcn", fromlist=["HGCNMethod", "HGCNSpec"]
        ).HGCNMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.h_gcn", fromlist=["HGCNSpec"]
            ).HGCNSpec(
                hidden_dim=8,
                weight_embed_dim=4,
                num_layers=3,
                channels=2,
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.chebnet", fromlist=["ChebNetMethod", "ChebNetSpec"]
        ).ChebNetMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.chebnet", fromlist=["ChebNetSpec"]
            ).ChebNetSpec(
                k=2,
                hidden_dim=16,
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.graphsage",
            fromlist=["GraphSAGEMethod", "GraphSAGESpec"],
        ).GraphSAGEMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.graphsage", fromlist=["GraphSAGESpec"]
            ).GraphSAGESpec(
                hidden_dim=16,
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.gat", fromlist=["GATMethod", "GATSpec"]
        ).GATMethod(
            spec=__import__("modssc.transductive.methods.gnn.gat", fromlist=["GATSpec"]).GATSpec(
                head_dim=4,
                heads=2,
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.gcnii", fromlist=["GCNIIMethod", "GCNIISpec"]
        ).GCNIIMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.gcnii", fromlist=["GCNIISpec"]
            ).GCNIISpec(
                hidden_dim=16,
                num_layers=4,
                max_epochs=5,
                patience=2,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.planetoid",
            fromlist=["PlanetoidMethod", "PlanetoidSpec"],
        ).PlanetoidMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.planetoid", fromlist=["PlanetoidSpec"]
            ).PlanetoidSpec(
                embedding_dim=16,
                num_walks=2,
                walk_length=10,
                window_size=2,
                epochs=1,
                batch_size=128,
                sup_batch_size=16,
            )
        ),
        __import__(
            "modssc.transductive.methods.gnn.grand", fromlist=["GRANDMethod", "GRANDSpec"]
        ).GRANDMethod(
            spec=__import__(
                "modssc.transductive.methods.gnn.grand", fromlist=["GRANDSpec"]
            ).GRANDSpec(
                hidden_dim=16,
                prop_steps=2,
                num_samples=2,
                max_epochs=5,
                patience=2,
            )
        ),
    ],
)
def test_gnn_methods_fit_predict(method):
    data = make_toy_dataset()
    method.fit(data, device="cpu", seed=0)
    proba = method.predict_proba(data)

    assert isinstance(proba, np.ndarray)
    assert proba.shape[0] == data.X.shape[0]
    assert proba.shape[1] == int(np.max(data.y)) + 1

    row_sums = proba.sum(axis=1)
    assert np.all(np.isfinite(row_sums))
    assert np.allclose(row_sums, 1.0, atol=1e-4)


def test_grand_early_stopping():
    """Test GRAND with early stopping triggered."""
    data = make_toy_dataset(n_nodes=30, n_classes=3)

    spec = GRANDSpec(hidden_dim=8, prop_steps=1, num_samples=1, max_epochs=10, patience=1, lr=0.01)
    method = GRANDMethod(spec=spec)
    method.fit(data, device="cpu", seed=42)

    proba = method.predict_proba(data)
    assert proba.shape == (30, 3)


def test_grand_no_val_mask():
    """Test GRAND without validation mask (should skip early stopping logic)."""
    data = make_toy_dataset(n_nodes=30, n_classes=3)

    data.masks["val_mask"] = np.zeros(30, dtype=bool)

    spec = GRANDSpec(max_epochs=2, patience=1)
    method = GRANDMethod(spec=spec)
    method.fit(data, device="cpu", seed=42)

    proba = method.predict_proba(data)
    assert proba.shape == (30, 3)


def test_grand_not_fitted():
    """Test RuntimeError when calling predict_proba before fit."""
    method = GRANDMethod()
    data = make_toy_dataset()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.predict_proba(data)


def test_grand_mismatched_nodes():
    """Test ValueError when predicting on a graph with different number of nodes."""
    data_train = make_toy_dataset(n_nodes=20)
    method = GRANDMethod(spec=GRANDSpec(max_epochs=1))
    method.fit(data_train, device="cpu")

    data_test = make_toy_dataset(n_nodes=25)
    with pytest.raises(ValueError, match="GRAND was fitted on n=20 nodes, got n=25"):
        method.predict_proba(data_test)


def test_planetoid_not_fitted():
    """Test RuntimeError when calling predict_proba before fit."""
    method = PlanetoidMethod()
    data = make_toy_dataset()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.predict_proba(data)


def test_planetoid_mismatched_nodes():
    """Test ValueError when predicting on a graph with different number of nodes."""
    data_train = make_toy_dataset(n_nodes=20)
    method = PlanetoidMethod(spec=PlanetoidSpec(epochs=1))
    method.fit(data_train, device="cpu")

    data_test = make_toy_dataset(n_nodes=25)
    with pytest.raises(ValueError, match="Planetoid was fitted on n=20 nodes, got n=25"):
        method.predict_proba(data_test)


def test_planetoid_empty_train_mask():
    """Test ValueError when train_mask is empty."""
    data = make_toy_dataset(n_nodes=20)
    data.masks["train_mask"] = np.zeros(20, dtype=bool)

    method = PlanetoidMethod(spec=PlanetoidSpec(epochs=1))
    with pytest.raises(ValueError, match="train_mask is empty"):
        method.fit(data, device="cpu")


def test_planetoid_large_sup_batch():
    """Test case where train_idx.numel() > sup_batch_size."""
    data = make_toy_dataset(n_nodes=50, n_classes=2)

    data.masks["train_mask"][:20] = True

    spec = PlanetoidSpec(epochs=1, sup_batch_size=10)
    method = PlanetoidMethod(spec=spec)
    method.fit(data, device="cpu")

    proba = method.predict_proba(data)
    assert proba.shape == (50, 2)


def test_planetoid_no_edges():
    """Test ValueError when graph has no edges (no walks generated)."""
    n_nodes = 10
    X = np.zeros((n_nodes, 4), dtype=np.float32)
    y = np.zeros(n_nodes, dtype=np.int64)

    edge_index = np.empty((2, 0), dtype=np.int64)
    edge_weight = np.empty((0,), dtype=np.float32)

    masks = {"train_mask": np.zeros(n_nodes, dtype=bool), "val_mask": np.zeros(n_nodes, dtype=bool)}

    masks["train_mask"][0] = True

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks=masks,
        meta={},
    )

    method = PlanetoidMethod(spec=PlanetoidSpec(epochs=1))
    with pytest.raises(ValueError, match="no training pairs could be generated"):
        method.fit(data, device="cpu")


def test_chebnet_not_fitted():
    """Test RuntimeError when calling predict_proba before fit."""
    method = ChebNetMethod()
    data = make_toy_dataset()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.predict_proba(data)


def test_chebnet_k0():
    """Test ChebNet with k=0."""
    data = make_toy_dataset(n_nodes=20)
    spec = ChebNetSpec(k=0, max_epochs=1)
    method = ChebNetMethod(spec=spec)
    method.fit(data, device="cpu")
    proba = method.predict_proba(data)
    assert proba.shape == (20, 3)


def test_chebnet_k1():
    """Test ChebNet with k=1."""
    data = make_toy_dataset(n_nodes=20)
    spec = ChebNetSpec(k=1, max_epochs=1)
    method = ChebNetMethod(spec=spec)
    method.fit(data, device="cpu")
    proba = method.predict_proba(data)
    assert proba.shape == (20, 3)


def test_chebconv_no_bias():
    """Test _ChebConv with bias=False."""
    conv = _ChebConv(16, 8, k=2, bias=False)
    assert conv.bias is None

    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0])

    out = conv(x, edge_index, edge_weight, n_nodes=10)
    assert out.shape == (10, 8)


def test_sgc_not_fitted():
    method = SGCMethod()
    data = make_toy_dataset()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.predict_proba(data)


def test_sgc_mismatched_nodes():
    data_train = make_toy_dataset(n_nodes=20)
    method = SGCMethod(spec=SGCSpec(max_epochs=1))
    method.fit(data_train, device="cpu")

    data_test = make_toy_dataset(n_nodes=25)
    with pytest.raises(ValueError, match="SGC was fitted on n=20 nodes, got n=25"):
        method.predict_proba(data_test)


def test_gat_not_fitted():
    method = GATMethod()
    data = make_toy_dataset()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.predict_proba(data)


def test_gat_conv_no_bias():
    conv = _GATConv(16, 8, heads=2, bias=False)
    assert conv.bias is None

    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    out = conv(x, edge_index, n_nodes=10)
    assert out.shape == (10, 16)


def test_appnp_not_fitted():
    method = APPNPMethod()
    data = make_toy_dataset()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.predict_proba(data)


def test_gcn_not_fitted():
    method = GCNMethod()
    data = make_toy_dataset()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.predict_proba(data)


def test_gcnii_not_fitted():
    method = GCNIIMethod()
    data = make_toy_dataset()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.predict_proba(data)


def test_graphsage_not_fitted():
    method = GraphSAGEMethod()
    data = make_toy_dataset()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.predict_proba(data)


def test_grand_early_stopping_patience():
    X = np.random.randn(10, 4).astype(np.float32)
    y = np.array([0, 1] * 5)
    adj = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]]
    ).T
    graph = DummyGraph(adj, None)
    dataset = DummyNodeDataset(
        X=X,
        y=y,
        graph=graph,
        masks={
            "train_mask": np.array([True] * 5 + [False] * 5),
            "val_mask": np.array([False] * 5 + [True] * 5),
            "test_mask": np.array([False] * 10),
        },
        meta={},
    )

    spec = GRANDSpec(hidden_dim=4, mlp_dropout=0.0, dropnode=0.0, patience=2, max_epochs=10)
    model = GRANDMethod(spec)
    model.fit(dataset)


def test_common_error_handling():
    X = np.eye(3)
    y = np.array([0, 1, 0])
    adj = np.array([[0, 1], [1, 0]])
    graph = DummyGraph(adj, None)

    dataset = DummyNodeDataset(
        X=X, y=y, graph=graph, masks={"val_mask": np.zeros(3, dtype=bool)}, meta={}
    )
    model = GCNMethod()
    with pytest.raises(ValueError, match="train_mask"):
        model.fit(dataset)

    dataset.masks = {
        "train_mask": np.zeros(3, dtype=bool),
        "val_mask": np.zeros(2, dtype=bool),
    }
    with pytest.raises(TransductiveValidationError, match="val_mask"):
        model.fit(dataset)

    graph_bad = DummyGraph(adj, np.ones(5))
    dataset.graph = graph_bad
    dataset.masks = {"train_mask": np.zeros(3, dtype=bool)}
    with pytest.raises(ValueError, match="edge_weight length mismatch"):
        model.fit(dataset)


def test_common_utils():
    set_torch_seed(42)

    t = torch.tensor([1, 2, 3])
    assert np.array_equal(_as_numpy(t), np.array([1, 2, 3]))

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        normalize_edge_weight(
            edge_index=edge_index, edge_weight=edge_weight, n_nodes=2, mode="invalid"
        )

    ei = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    ew = torch.tensor([1.0, 2.0], dtype=torch.float)
    ei_out, ew_out = coalesce_edges(ei, ew, n_nodes=2)

    assert ei_out.shape == (2, 1)

    assert ei_out[0, 0] == 0
    assert ei_out[1, 0] == 1
    assert ew_out[0] == 3.0

    with pytest.raises(ValueError, match="X must be 2D"):
        _ensure_2d(np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="y has zero columns"):
        _labels_to_int(np.zeros((5, 0)))

    with pytest.raises(ValueError, match="edge_index must be 2D"):
        _as_edge_index(np.zeros(5))

    with pytest.raises(ValueError, match="edge_index must have shape"):
        _as_edge_index(np.zeros((3, 3)))

    with pytest.raises(ValueError, match="mask must have shape"):
        _as_mask(np.zeros(5), n=4, name="mask")


def test_common_error_handling_extended():
    with pytest.raises(ValueError, match="X must be 2D"):
        _ensure_2d(np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="y has zero columns"):
        _labels_to_int(np.zeros((5, 0)))

    with pytest.raises(ValueError, match="edge_index must be 2D"):
        _as_edge_index(np.zeros((2, 2, 2)))

    from modssc.transductive.methods.gnn.common import normalize_edge_weight

    with pytest.raises(ValueError, match="Unknown normalization mode"):
        normalize_edge_weight(
            edge_index=torch.zeros((2, 2), dtype=torch.long),
            edge_weight=torch.ones(2, dtype=torch.float32),
            n_nodes=2,
            mode="unknown",
        )

    with patch("modssc.transductive.methods.gnn.common.validate_node_dataset"):
        dataset = DummyNodeDataset(
            X=np.eye(3),
            y=np.array([0, 1, 0]),
            graph=DummyGraph(
                edge_index=np.array([[0, 1], [1, 0]]),
                edge_weight=np.array([1.0, 1.0, 1.0]),
            ),
            masks={"train_mask": np.zeros(3, dtype=bool)},
            meta={},
        )
        from modssc.transductive.methods.gnn.common import prepare_data

        with pytest.raises(ValueError, match="edge_weight length mismatch"):
            prepare_data(dataset)


def test_spmm():
    from modssc.transductive.methods.gnn.common import spmm

    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 2.0], dtype=torch.float32)

    X = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    n_nodes = 3

    out = spmm(edge_index, edge_weight, X, n_nodes=n_nodes)

    expected = torch.tensor([[0.0, 0.0], [0.5, 1.0], [6.0, 8.0]], dtype=torch.float32)

    assert torch.allclose(out, expected)


def test_accuracy_from_logits():
    from modssc.transductive.methods.gnn.common import accuracy_from_logits

    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    y = torch.tensor([1, 0, 1])
    mask = torch.tensor([True, True, True])

    acc = accuracy_from_logits(logits, y, mask)
    assert acc == 1.0

    assert math.isnan(accuracy_from_logits(logits, y, None))

    assert math.isnan(accuracy_from_logits(logits, y, torch.tensor([], dtype=torch.bool)))

    assert math.isnan(accuracy_from_logits(logits, y, torch.tensor([False, False, False])))


def test_prepare_data_coverage():
    with patch("modssc.transductive.methods.gnn.common.validate_node_dataset"):
        dataset = DummyNodeDataset(
            X=np.eye(3),
            y=np.array([0, 1, 0]),
            graph=DummyGraph(np.array([[0, 1], [1, 0]]), None),
            masks={"train_mask": np.zeros(3, dtype=bool), "val_mask": "bad_mask"},
            meta={},
        )
        from modssc.transductive.methods.gnn.common import prepare_data

        prep = prepare_data(dataset)
        assert prep.val_mask is None

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.manual_seed_all") as mock_cuda_seed,
    ):
        set_torch_seed(42)
        mock_cuda_seed.assert_called_with(42)
