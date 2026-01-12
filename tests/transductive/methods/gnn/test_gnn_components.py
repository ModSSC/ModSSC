from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional torch dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.transductive.methods.gnn.common import PreparedData  # noqa: E402
from modssc.transductive.methods.gnn.grafn import (  # noqa: E402
    GraFNMethod,
    GraFNSpec,
    _drop_edges,
    _drop_features,
    _GCNEncoder,
    _sample_support,
)
from modssc.transductive.methods.gnn.graphhop import (  # noqa: E402
    GraphHopMethod,
    GraphHopSpec,
    _build_adj_list,
    _build_hop_edges,
    _row_normalize_weights,
    _train_lr,
)
from modssc.transductive.methods.gnn.h_gcn import (  # noqa: E402
    HGCNMethod,
    HGCNSpec,
    _coarsen_graph,
    _HGCNNet,
    _MultiChannelGCN,
)
from modssc.transductive.methods.gnn.h_gcn import (
    _build_dense_adjacency as _hgcn_build_dense_adjacency,
)
from modssc.transductive.methods.gnn.h_gcn import (
    _normalize_adjacency as _hgcn_normalize_adjacency,
)
from modssc.transductive.methods.gnn.n_gcn import (  # noqa: E402
    NGCNMethod,
    NGCNSpec,
    _DenseGCN,
    _NGCNNet,
)
from modssc.transductive.methods.gnn.n_gcn import (
    _build_dense_adjacency as _ngcn_build_dense_adjacency,
)


@dataclass(frozen=True)
class DummyGraph:
    edge_index: np.ndarray
    edge_weight: np.ndarray | None = None


@dataclass(frozen=True)
class DummyNodeDataset:
    X: np.ndarray
    y: np.ndarray
    graph: DummyGraph
    masks: dict[str, np.ndarray]
    meta: dict | None = None


def _cycle_graph(n_nodes: int) -> np.ndarray:
    src = np.arange(n_nodes, dtype=np.int64)
    dst = (src + 1) % n_nodes
    return np.stack([src, dst], axis=0)


def _make_dataset(
    *,
    n_nodes: int = 6,
    n_classes: int = 2,
    with_val: bool = True,
    edge_weight: np.ndarray | None = None,
    fill_edge_weight: bool = True,
    edge_index: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    seed: int = 0,
) -> DummyNodeDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_nodes, 4)).astype(np.float32)
    y = (np.arange(n_nodes) % n_classes).astype(np.int64)

    if train_mask is None:
        train_mask = np.zeros(n_nodes, dtype=bool)
        for c in range(n_classes):
            idx = int(np.flatnonzero(y == c)[0])
            train_mask[idx] = True

    masks = {"train_mask": train_mask}
    if with_val:
        val_mask = np.zeros(n_nodes, dtype=bool)
        val_mask[: max(1, n_nodes // 3)] = True
        val_mask[train_mask] = False
        masks["val_mask"] = val_mask

    ei = edge_index if edge_index is not None else _cycle_graph(n_nodes)
    ew = edge_weight
    if fill_edge_weight and edge_weight is None and ei.size:
        ew = np.ones(ei.shape[1], dtype=np.float32)

    return DummyNodeDataset(X=X, y=y, graph=DummyGraph(edge_index=ei, edge_weight=ew), masks=masks)


class _ConstRng:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def random(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.full(shape, self.value, dtype=np.float32)


def test_grafn_drop_features_and_edges_paths():
    X = torch.ones((3, 4), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    out = _drop_features(X, p=0.0, rng=_ConstRng(0.5))
    assert out is X
    out = _drop_features(X, p=0.5, rng=_ConstRng(0.9))
    assert torch.allclose(out, X)

    ei, ew = _drop_edges(edge_index, edge_weight, p=0.0, rng=_ConstRng(0.2))
    assert torch.equal(ei, edge_index)
    ei, ew = _drop_edges(edge_index, edge_weight, p=1.0, rng=_ConstRng(0.0))
    assert torch.equal(ei, edge_index)


def test_grafn_sample_support_requires_labels():
    y = np.array([0, 0, 0, 1], dtype=np.int64)
    train_mask = np.array([True, False, False, False])
    with pytest.raises(ValueError, match="per class"):
        _sample_support(y=y, train_mask=train_mask, n_classes=2, rng=np.random.default_rng(0))


def test_grafn_encoder_requires_layer_sizes():
    with pytest.raises(ValueError, match="layer_sizes"):
        _GCNEncoder((4,))


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (GraFNSpec(tau=0.0), "tau must be"),
        (GraFNSpec(thres=1.5), "thres must be"),
    ],
)
def test_grafn_fit_invalid_hyperparams(spec, match):
    data = _make_dataset()
    with pytest.raises(ValueError, match=match):
        GraFNMethod(spec=spec).fit(data, device="cpu", seed=0)


def test_grafn_fit_empty_train_mask():
    train_mask = np.zeros(6, dtype=bool)
    data = _make_dataset(train_mask=train_mask)
    with pytest.raises(ValueError, match="train_mask is empty"):
        GraFNMethod(spec=GraFNSpec(max_epochs=1, patience=1)).fit(data, device="cpu", seed=0)


def test_grafn_fit_edge_weight_mismatch(monkeypatch):
    dummy_prep = PreparedData(
        X=torch.zeros((2, 3), dtype=torch.float32),
        y=torch.tensor([0, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_weight=torch.ones((2,), dtype=torch.float32),
        train_mask=np.array([True, True]),
        val_mask=None,
        n_nodes=2,
        n_classes=2,
        device="cpu",
    )
    import modssc.transductive.methods.gnn.grafn as grafn

    monkeypatch.setattr(grafn, "prepare_data_cached", lambda *args, **kwargs: dummy_prep)

    edge_index = _cycle_graph(6)
    edge_weight = np.ones(edge_index.shape[1] - 1, dtype=np.float32)
    data = _make_dataset(edge_index=edge_index, edge_weight=edge_weight)
    with pytest.raises(ValueError, match="edge_weight length mismatch"):
        GraFNMethod(spec=GraFNSpec(max_epochs=1, patience=1)).fit(data, device="cpu", seed=0)


def test_grafn_fit_no_epochs_no_val():
    data = _make_dataset(with_val=False, edge_weight=None, fill_edge_weight=False)
    method = GraFNMethod(
        spec=GraFNSpec(hidden_dims=(4,), max_epochs=0, patience=1, drop_feat_strong=0.0)
    )
    method.fit(data, device="cpu", seed=0)


def test_grafn_fit_early_stopping_no_break():
    data = _make_dataset(with_val=True)
    val_mask = np.zeros(data.y.shape[0], dtype=bool)
    val_mask[-1] = True
    val_mask[data.masks["train_mask"]] = False
    data.masks["val_mask"] = val_mask
    method = GraFNMethod(
        spec=GraFNSpec(
            hidden_dims=(4,),
            max_epochs=2,
            patience=2,
            lr=0.0,
            drop_feat_strong=0.0,
            drop_edge_strong=0.0,
            drop_feat_weak=0.0,
            drop_edge_weak=0.0,
        )
    )
    method.fit(data, device="cpu", seed=0)


def test_grafn_fit_early_stopping_break():
    data = _make_dataset(with_val=True)
    val_mask = np.zeros(data.y.shape[0], dtype=bool)
    val_mask[-1] = True
    val_mask[data.masks["train_mask"]] = False
    data.masks["val_mask"] = val_mask
    method = GraFNMethod(
        spec=GraFNSpec(
            hidden_dims=(4,),
            max_epochs=2,
            patience=1,
            lr=0.0,
            drop_feat_strong=0.0,
            drop_edge_strong=0.0,
            drop_feat_weak=0.0,
            drop_edge_weak=0.0,
        )
    )
    method.fit(data, device="cpu", seed=0)


def test_grafn_predict_proba_requires_fit():
    data = _make_dataset()
    method = GraFNMethod(spec=GraFNSpec(max_epochs=0))
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(data)


def test_grafn_predict_proba_node_mismatch():
    data = _make_dataset(n_nodes=6)
    method = GraFNMethod(spec=GraFNSpec(hidden_dims=(4,), max_epochs=1, patience=1))
    method.fit(data, device="cpu", seed=0)

    data2 = _make_dataset(n_nodes=7)
    with pytest.raises(ValueError, match="fitted on n="):
        method.predict_proba(data2)


def test_graphhop_build_adj_list_branches():
    edge_index = np.array([[0, 1, 2, 2], [0, 2, 3, 99]], dtype=np.int64)
    adj = _build_adj_list(edge_index, n_nodes=4, symmetrize=True)
    assert 2 in adj[1]
    adj = _build_adj_list(edge_index, n_nodes=4, symmetrize=False)
    assert 2 in adj[1]


def test_graphhop_build_hop_edges_empty():
    adj = [[] for _ in range(3)]
    hop_edges = _build_hop_edges(adj, n_nodes=3, max_hops=2)
    assert hop_edges[0].shape == (2, 0)


def test_graphhop_row_normalize_empty():
    edge_index = np.empty((2, 0), dtype=np.int64)
    weights = _row_normalize_weights(edge_index, n_nodes=3)
    assert weights.shape == (0,)


def test_graphhop_train_lr_val_mask_none():
    X = torch.zeros((4, 3), dtype=torch.float32)
    labels = torch.eye(2, dtype=torch.float32).repeat(2, 1)
    train_mask = torch.tensor([True, True, False, False])
    model = _train_lr(
        step=0,
        X=X,
        labels=labels,
        train_mask=train_mask,
        val_mask=None,
        y_val=None,
        prev_model=None,
        spec=GraphHopSpec(max_epochs=1),
        device=torch.device("cpu"),
    )
    assert isinstance(model, torch.nn.Module)


def test_graphhop_train_lr_best_state_loaded():
    X = torch.zeros((4, 3), dtype=torch.float32)
    labels = torch.eye(2, dtype=torch.float32).repeat(2, 1)
    train_mask = torch.tensor([True, True, False, False])
    val_mask = torch.tensor([False, False, True, True])
    y_val = torch.eye(2, dtype=torch.float32)
    model = _train_lr(
        step=1,
        X=X,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        y_val=y_val,
        prev_model=None,
        spec=GraphHopSpec(max_epochs=1, patience=1),
        device=torch.device("cpu"),
    )
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (GraphHopSpec(hops=0), "hops must be"),
        (GraphHopSpec(temperature=0.0), "temperature must be"),
    ],
)
def test_graphhop_fit_invalid_hyperparams(spec, match):
    data = _make_dataset()
    with pytest.raises(ValueError, match=match):
        GraphHopMethod(spec=spec).fit(data, device="cpu", seed=0)


def test_graphhop_fit_empty_train_mask():
    train_mask = np.zeros(6, dtype=bool)
    data = _make_dataset(train_mask=train_mask)
    with pytest.raises(ValueError, match="train_mask is empty"):
        GraphHopMethod(spec=GraphHopSpec(max_iter=1, max_epochs=1)).fit(data, device="cpu", seed=0)


def test_graphhop_fit_missing_class():
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    train_mask = np.array([True, False, True, False, True, False])
    data = _make_dataset(train_mask=train_mask, n_nodes=6)
    data = DummyNodeDataset(X=data.X, y=y, graph=data.graph, masks=data.masks)
    with pytest.raises(ValueError, match="per class"):
        GraphHopMethod(spec=GraphHopSpec(max_iter=1, max_epochs=1)).fit(data, device="cpu", seed=0)


def test_graphhop_fit_val_mask_empty():
    data = _make_dataset(with_val=True)
    data.masks["val_mask"][:] = False
    method = GraphHopMethod(spec=GraphHopSpec(max_iter=1, max_epochs=1, hops=1))
    method.fit(data, device="cpu", seed=0)


def test_graphhop_fit_weight_length_errors():
    data = _make_dataset()
    with pytest.raises(ValueError, match="weight_temp must have length"):
        GraphHopMethod(spec=GraphHopSpec(hops=2, weight_temp=(1.0,), max_iter=1, max_epochs=1)).fit(
            data, device="cpu", seed=0
        )
    with pytest.raises(ValueError, match="weight_pred must have length"):
        GraphHopMethod(spec=GraphHopSpec(hops=2, weight_pred=(1.0,), max_iter=1, max_epochs=1)).fit(
            data, device="cpu", seed=0
        )


def test_graphhop_fit_no_edges_branches():
    edge_index = np.empty((2, 0), dtype=np.int64)
    data = _make_dataset(edge_index=edge_index, edge_weight=np.empty((0,), dtype=np.float32))
    method = GraphHopMethod(spec=GraphHopSpec(hops=1, max_iter=1, max_epochs=1))
    method.fit(data, device="cpu", seed=0)


def test_graphhop_fit_requires_label_in_prep(monkeypatch):
    dummy_prep = PreparedData(
        X=torch.zeros((1, 2), dtype=torch.float32),
        y=np.array([], dtype=np.int64),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=torch.empty((0,), dtype=torch.float32),
        train_mask=np.array([True]),
        val_mask=None,
        n_nodes=1,
        n_classes=0,
        device="cpu",
    )

    import modssc.transductive.methods.gnn.graphhop as gh

    monkeypatch.setattr(gh, "prepare_data_cached", lambda *args, **kwargs: dummy_prep)

    data = DummyNodeDataset(
        X=np.zeros((1, 4), dtype=np.float32),
        y=np.zeros((1,), dtype=np.int64),
        graph=DummyGraph(edge_index=np.empty((2, 0), dtype=np.int64)),
        masks={"train_mask": np.array([True])},
    )
    with pytest.raises(ValueError, match="at least one label"):
        GraphHopMethod(spec=GraphHopSpec(max_iter=1, max_epochs=1)).fit(data, device="cpu", seed=0)


def test_graphhop_predict_proba_requires_fit():
    data = _make_dataset()
    with pytest.raises(RuntimeError, match="not fitted"):
        GraphHopMethod().predict_proba(data)


def test_graphhop_predict_proba_node_mismatch():
    data = _make_dataset()
    method = GraphHopMethod(spec=GraphHopSpec(max_iter=1, max_epochs=1))
    method.fit(data, device="cpu", seed=0)
    data2 = _make_dataset(n_nodes=7)
    with pytest.raises(ValueError, match="fitted on n="):
        method.predict_proba(data2)


def test_hgcn_build_dense_adjacency_branches():
    edge_index = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
    edge_weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    adj = _hgcn_build_dense_adjacency(edge_index, edge_weight, n_nodes=3, symmetrize=True)
    assert adj[0, 0] == 0.0
    adj = _hgcn_build_dense_adjacency(edge_index, edge_weight, n_nodes=3, symmetrize=False)
    assert adj.shape == (3, 3)


def test_hgcn_normalize_adjacency_branches():
    adj = torch.eye(3, dtype=torch.float32)
    out = _hgcn_normalize_adjacency(adj, add_self_loops=True)
    assert out.shape == adj.shape
    out = _hgcn_normalize_adjacency(adj, add_self_loops=False)
    assert out.shape == adj.shape


def test_hgcn_multi_channel_weight_embed_branches():
    module = _MultiChannelGCN(4, 3, channels=2, dropout=0.0)
    x = torch.zeros((3, 4), dtype=torch.float32)
    adj = torch.eye(3, dtype=torch.float32)
    node_weights = torch.ones((3,), dtype=torch.long)
    out = module(x, adj, node_weights=node_weights, weight_embed=None)
    assert out.shape[0] == 3


def test_hgcn_net_weight_embed_none():
    model = _HGCNNet(
        4,
        hidden_dim=3,
        out_channels=2,
        num_layers=3,
        channels=1,
        weight_embed_dim=0,
        max_weight=1,
        dropout=0.0,
    )
    assert model.weight_embed is None


def test_hgcn_coarsen_graph_seg_groups():
    adj = np.zeros((4, 4), dtype=np.float32)
    adj[0, 2] = 1.0
    adj[1, 2] = 1.0
    adj[2, 0] = 1.0
    adj[2, 1] = 1.0
    node_weights = np.ones((4,), dtype=np.int64)

    M, adj_next, weights_next = _coarsen_graph(adj, node_weights)
    assert M.shape[0] == 4
    assert (M.sum(axis=0) == 2.0).any()
    assert adj_next.shape[0] == adj_next.shape[1]
    assert weights_next.size > 0


def test_hgcn_coarsen_graph_seg_skip_marked(monkeypatch):
    adj = np.zeros((4, 4), dtype=np.float32)
    adj[0, 2] = 1.0
    adj[1, 2] = 1.0
    adj[2, 0] = 1.0
    adj[2, 1] = 1.0
    node_weights = np.ones((4,), dtype=np.int64)

    class _MarkedArray(np.ndarray):
        def __getitem__(self, idx):
            value = super().__getitem__(idx)
            if isinstance(idx, (int, np.integer)) and int(idx) == 0:
                return True
            return value

    orig_zeros = np.zeros

    def _zeros(shape, dtype=float):
        arr = orig_zeros(shape, dtype=dtype)
        if dtype is bool and (shape == 4 or shape == (4,)):
            return arr.view(_MarkedArray)
        return arr

    import modssc.transductive.methods.gnn.h_gcn as hgcn

    monkeypatch.setattr(hgcn.np, "zeros", _zeros)

    M, adj_next, weights_next = _coarsen_graph(adj, node_weights)
    assert M.shape[0] == 4
    assert adj_next.shape[0] == adj_next.shape[1]
    assert weights_next.size > 0


def test_hgcn_coarsen_graph_unmarked_branch(monkeypatch):
    adj = np.zeros((4, 4), dtype=np.float32)
    adj[0, 2] = 1.0
    adj[1, 2] = 1.0
    adj[2, 0] = 1.0
    adj[2, 1] = 1.0
    node_weights = np.ones((4,), dtype=np.int64)

    class _StickyFalseArray(np.ndarray):
        def __setitem__(self, idx, value):
            if isinstance(idx, (int, np.integer)) and int(idx) == 0:
                return
            return super().__setitem__(idx, value)

    orig_zeros = np.zeros

    def _zeros(shape, dtype=float):
        arr = orig_zeros(shape, dtype=dtype)
        if dtype is bool and (shape == 4 or shape == (4,)):
            return arr.view(_StickyFalseArray)
        return arr

    import modssc.transductive.methods.gnn.h_gcn as hgcn

    monkeypatch.setattr(hgcn.np, "zeros", _zeros)

    M, adj_next, weights_next = _coarsen_graph(adj, node_weights)
    assert M.shape[0] == 4
    assert adj_next.shape[0] == adj_next.shape[1]
    assert weights_next.size > 0


def test_hgcn_fit_invalid_hyperparams():
    data = _make_dataset()
    with pytest.raises(ValueError, match="num_layers"):
        HGCNMethod(spec=HGCNSpec(num_layers=4)).fit(data, device="cpu", seed=0)
    with pytest.raises(ValueError, match="channels"):
        HGCNMethod(spec=HGCNSpec(channels=0)).fit(data, device="cpu", seed=0)
    with pytest.raises(ValueError, match="weight_embed_dim"):
        HGCNMethod(spec=HGCNSpec(weight_embed_dim=0)).fit(data, device="cpu", seed=0)


def test_hgcn_fit_edge_weight_mismatch(monkeypatch):
    dummy_prep = PreparedData(
        X=torch.zeros((2, 3), dtype=torch.float32),
        y=torch.tensor([0, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_weight=torch.ones((2,), dtype=torch.float32),
        train_mask=np.array([True, True]),
        val_mask=None,
        n_nodes=2,
        n_classes=2,
        device="cpu",
    )
    import modssc.transductive.methods.gnn.h_gcn as hgcn

    monkeypatch.setattr(hgcn, "prepare_data_cached", lambda *args, **kwargs: dummy_prep)

    edge_index = _cycle_graph(6)
    edge_weight = np.ones(edge_index.shape[1] - 1, dtype=np.float32)
    data = _make_dataset(edge_index=edge_index, edge_weight=edge_weight)
    with pytest.raises(ValueError, match="edge_weight length mismatch"):
        HGCNMethod(spec=HGCNSpec(max_epochs=1, patience=1)).fit(data, device="cpu", seed=0)


def test_hgcn_fit_edge_weight_none():
    data = _make_dataset(edge_weight=None, fill_edge_weight=False)
    method = HGCNMethod(spec=HGCNSpec(num_layers=3, max_epochs=1, patience=1))
    method.fit(data, device="cpu", seed=0)


def test_hgcn_fit_max_weight_branch(monkeypatch):
    data = _make_dataset()

    def _fake_coarsen(adj, node_weights):
        M = np.eye(adj.shape[0], dtype=np.float32)
        return M, adj, np.zeros_like(node_weights)

    import modssc.transductive.methods.gnn.h_gcn as hgcn

    orig_ones = hgcn.np.ones

    def _ones(shape, dtype=float):
        if shape == data.X.shape[0] or shape == (data.X.shape[0],):
            return np.zeros(shape, dtype=dtype)
        return orig_ones(shape, dtype=dtype)

    monkeypatch.setattr(hgcn, "_coarsen_graph", _fake_coarsen)
    monkeypatch.setattr(hgcn.np, "ones", _ones)
    method = HGCNMethod(spec=HGCNSpec(num_layers=3, max_epochs=1, patience=1))
    method.fit(data, device="cpu", seed=0)


def test_hgcn_predict_proba_requires_fit():
    data = _make_dataset()
    with pytest.raises(RuntimeError, match="not fitted"):
        HGCNMethod().predict_proba(data)


def test_hgcn_predict_proba_node_mismatch():
    data = _make_dataset()
    method = HGCNMethod(spec=HGCNSpec(num_layers=3, max_epochs=1, patience=1))
    method.fit(data, device="cpu", seed=0)
    data2 = _make_dataset(n_nodes=7)
    with pytest.raises(ValueError, match="fitted on n="):
        method.predict_proba(data2)


def test_ngcn_build_dense_adjacency_branches():
    edge_index = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
    edge_weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    adj = _ngcn_build_dense_adjacency(
        edge_index, edge_weight, n_nodes=3, symmetrize=True, add_self_loops=True
    )
    assert adj.shape == (3, 3)
    adj = _ngcn_build_dense_adjacency(
        edge_index, edge_weight, n_nodes=3, symmetrize=False, add_self_loops=False
    )
    assert adj.shape == (3, 3)


def test_ngcn_dense_gcn_layer_dims():
    model = _DenseGCN(3, hidden_dim=4, out_channels=2, num_layers=1, dropout=0.0)
    out = model(torch.eye(3), torch.zeros((3, 3)))
    assert out.shape[0] == 3
    model = _DenseGCN(3, hidden_dim=4, out_channels=2, num_layers=3, dropout=0.0)
    out = model(torch.eye(3), torch.zeros((3, 3)))
    assert out.shape[0] == 3


def test_ngcn_attention_classifier_out_lin():
    x = torch.zeros((3, 4), dtype=torch.float32)
    adj_powers = [torch.eye(3, dtype=torch.float32)]
    model = _NGCNNet(
        4,
        hidden_dim=3,
        gcn_out_dim=5,
        num_layers=1,
        K=1,
        r=1,
        classifier="attention",
        dropout=0.0,
        n_classes=2,
    )
    out = model(adj_powers, x)
    assert out.shape == (3, 2)


def test_ngcn_attention_classifier_out_lin_none():
    x = torch.zeros((3, 4), dtype=torch.float32)
    adj_powers = [torch.eye(3, dtype=torch.float32)]
    model = _NGCNNet(
        4,
        hidden_dim=3,
        gcn_out_dim=2,
        num_layers=1,
        K=1,
        r=1,
        classifier="attention",
        dropout=0.0,
        n_classes=2,
    )
    out = model(adj_powers, x)
    assert out.shape == (3, 2)


def test_ngcn_fit_invalid_hyperparams():
    data = _make_dataset()
    with pytest.raises(ValueError, match="K must be"):
        NGCNMethod(spec=NGCNSpec(K=0)).fit(data, device="cpu", seed=0)
    with pytest.raises(ValueError, match="r must be"):
        NGCNMethod(spec=NGCNSpec(r=0)).fit(data, device="cpu", seed=0)
    with pytest.raises(ValueError, match="gcn_layers must be"):
        NGCNMethod(spec=NGCNSpec(gcn_layers=0)).fit(data, device="cpu", seed=0)
    with pytest.raises(ValueError, match="classifier must be"):
        NGCNMethod(spec=NGCNSpec(classifier="other")).fit(data, device="cpu", seed=0)


def test_ngcn_fit_edge_weight_mismatch(monkeypatch):
    dummy_prep = PreparedData(
        X=torch.zeros((2, 3), dtype=torch.float32),
        y=torch.tensor([0, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_weight=torch.ones((2,), dtype=torch.float32),
        train_mask=np.array([True, True]),
        val_mask=None,
        n_nodes=2,
        n_classes=2,
        device="cpu",
    )
    import modssc.transductive.methods.gnn.n_gcn as ngcn

    monkeypatch.setattr(ngcn, "prepare_data_cached", lambda *args, **kwargs: dummy_prep)

    edge_index = _cycle_graph(6)
    edge_weight = np.ones(edge_index.shape[1] - 1, dtype=np.float32)
    data = _make_dataset(edge_index=edge_index, edge_weight=edge_weight)
    with pytest.raises(ValueError, match="edge_weight length mismatch"):
        NGCNMethod(spec=NGCNSpec(max_epochs=1, patience=1)).fit(data, device="cpu", seed=0)


def test_ngcn_fit_edge_weight_none():
    data = _make_dataset(edge_weight=None, fill_edge_weight=False)
    method = NGCNMethod(spec=NGCNSpec(max_epochs=1, patience=1, K=1, r=1))
    method.fit(data, device="cpu", seed=0)


def test_ngcn_predict_proba_requires_fit():
    data = _make_dataset()
    with pytest.raises(RuntimeError, match="not fitted"):
        NGCNMethod().predict_proba(data)


def test_ngcn_predict_proba_node_mismatch():
    data = _make_dataset()
    method = NGCNMethod(spec=NGCNSpec(max_epochs=1, patience=1, K=1, r=1))
    method.fit(data, device="cpu", seed=0)
    data2 = _make_dataset(n_nodes=7)
    with pytest.raises(ValueError, match="fitted on n="):
        method.predict_proba(data2)
