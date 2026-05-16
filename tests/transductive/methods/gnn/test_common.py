from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)


from modssc.transductive.methods.gnn import common  # noqa: E402


@dataclass
class DummyGraph:
    edge_index: np.ndarray
    edge_weight: np.ndarray | None = None


@dataclass
class DummyDataset:
    X: np.ndarray
    y: np.ndarray
    graph: DummyGraph
    masks: dict[str, np.ndarray]
    meta: dict | None = None


def _make_dataset(n_nodes: int = 4) -> DummyDataset:
    X = np.zeros((n_nodes, 2), dtype=np.float32)
    y = np.arange(n_nodes) % 2
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    masks = {"train_mask": np.array([True, True, False, False])[:n_nodes]}
    return DummyDataset(X=X, y=y, graph=DummyGraph(edge_index=edge_index), masks=masks, meta={})


def test_set_torch_seed_calls_cuda(monkeypatch) -> None:
    calls = {}

    monkeypatch.setattr(common.torch, "manual_seed", lambda v: calls.setdefault("seed", v))
    monkeypatch.setattr(
        common.torch,
        "cuda",
        SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda v: calls.setdefault("cuda", v),
        ),
    )

    common.set_torch_seed(123)
    assert calls["seed"] == 123
    assert calls["cuda"] == 123


def test_as_numpy_tensor_like() -> None:
    class Dummy:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([1, 2, 3])

    out = common._as_numpy(Dummy())
    assert np.array_equal(out, np.array([1, 2, 3]))


def test_ensure_2d_and_labels_to_int() -> None:
    x1 = np.array([1, 2, 3])
    x2 = common._ensure_2d(x1)
    assert x2.shape == (3, 1)

    with pytest.raises(ValueError, match="X must be 2D"):
        common._ensure_2d(np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="y has zero columns"):
        common._labels_to_int(np.zeros((3, 0)))

    y2 = common._labels_to_int(np.array([[0.1, 0.9], [0.7, 0.3]]))
    assert np.array_equal(y2, np.array([1, 0]))


def test_as_edge_index_and_mask_errors() -> None:
    with pytest.raises(ValueError, match="edge_index must be 2D"):
        common._as_edge_index(np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="edge_index must have shape"):
        common._as_edge_index(np.zeros((3, 3)))

    ei = common._as_edge_index(np.array([[0, 1], [1, 2], [2, 3]]))
    assert ei.shape == (2, 3)

    with pytest.raises(ValueError, match="train_mask must have shape"):
        common._as_mask(np.array([True]), 3, name="train_mask")


def test_prepare_data_missing_train_mask() -> None:
    data = _make_dataset()
    data.masks = {}
    with pytest.raises(ValueError, match="data.masks must contain 'train_mask'"):
        common.prepare_data(data)


def test_prepare_data_val_mask_invalid_shape_is_ignored(monkeypatch) -> None:
    data = _make_dataset()
    data.masks["val_mask"] = np.array([True])
    monkeypatch.setattr(common, "validate_node_dataset", lambda *_: None)
    prep = common.prepare_data(data)
    assert prep.val_mask is None


def test_prepare_data_edge_weight_length_mismatch() -> None:
    data = _make_dataset()
    data.graph.edge_weight = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="edge_weight length mismatch"):
        common.prepare_data(data)


def test_prepare_data_accepts_device_instance() -> None:
    data = _make_dataset()
    prep = common.prepare_data(data, device=torch.device("cpu"))
    assert prep.device.type == "cpu"


def test_accuracy_from_logits_edge_cases() -> None:
    logits = torch.zeros((3, 2))
    y = torch.tensor([0, 1, 0])

    assert np.isnan(common.accuracy_from_logits(logits, y, None))

    empty_mask = torch.zeros((0,), dtype=torch.bool)
    assert np.isnan(common.accuracy_from_logits(logits, y, empty_mask))

    mask_false = torch.zeros((3,), dtype=torch.bool)
    assert np.isnan(common.accuracy_from_logits(logits, y, mask_false))


def test_train_fullbatch_tracks_best_state() -> None:
    X = torch.randn(6, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    train_mask = torch.tensor([True, True, True, False, False, False])
    val_mask = torch.tensor([False, False, False, True, True, True])

    model = torch.nn.Linear(3, 2)

    res = common.train_fullbatch(
        model=model,
        forward_fn=lambda: model(X),
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        lr=0.05,
        weight_decay=0.0,
        max_epochs=2,
        patience=5,
        seed=0,
    )

    assert res.best_epoch is not None
    assert res.best_val_acc is not None
    assert res.n_epochs >= 1


def test_optimizer_param_groups_weight_decay_scopes() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model.scale = torch.nn.Parameter(torch.ones(4))
    model.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    groups_all = common._optimizer_param_groups(model, weight_decay=0.1, scope="all")
    assert [id(p) for p in groups_all] == [id(p) for p in model.parameters()]

    groups_none = common._optimizer_param_groups(model, weight_decay=0.1, scope="none")
    assert groups_none[0]["weight_decay"] == 0.0
    assert len(groups_none[0]["params"]) == len(list(model.parameters()))

    groups_zero = common._optimizer_param_groups(model, weight_decay=0.0, scope="first_layer")
    assert groups_zero[0]["weight_decay"] == 0.0

    groups_first = common._optimizer_param_groups(model, weight_decay=0.1, scope="first_layer")
    assert groups_first[0]["weight_decay"] == 0.1
    assert groups_first[1]["weight_decay"] == 0.0
    assert groups_first[0]["params"][0] is model[0].weight

    groups_non_bias = common._optimizer_param_groups(model, weight_decay=0.1, scope="non_bias")
    assert groups_non_bias[0]["weight_decay"] == 0.1
    assert groups_non_bias[1]["weight_decay"] == 0.0
    assert {id(p) for p in groups_non_bias[0]["params"]} == {
        id(model[0].weight),
        id(model[1].weight),
    }
    assert {id(p) for p in groups_non_bias[1]["params"]} == {
        id(model[0].bias),
        id(model[1].bias),
        id(model.scale),
    }
    assert id(model.frozen) not in {id(p) for group in groups_non_bias for p in group["params"]}

    no_weight_model = torch.nn.Module()
    groups_missing = common._optimizer_param_groups(
        no_weight_model, weight_decay=0.1, scope="first_layer"
    )
    assert groups_missing[0]["params"] == []

    with pytest.raises(ValueError, match="weight_decay_scope"):
        common._optimizer_param_groups(model, weight_decay=0.1, scope="bad")


def test_checkpoint_selection_metrics() -> None:
    assert common._is_better_checkpoint(
        selection_metric="val_loss",
        val_loss=0.5,
        val_acc=0.1,
        best_val_loss=None,
        best_val_acc=None,
    )
    assert not common._is_better_checkpoint(
        selection_metric="val_loss",
        val_loss=0.5,
        val_acc=1.0,
        best_val_loss=0.4,
        best_val_acc=0.0,
    )
    assert common._is_better_checkpoint(
        selection_metric="val_acc_then_loss",
        val_loss=0.8,
        val_acc=0.7,
        best_val_loss=0.1,
        best_val_acc=0.6,
    )
    assert common._is_better_checkpoint(
        selection_metric="val_acc_then_loss",
        val_loss=0.1,
        val_acc=0.7,
        best_val_loss=0.2,
        best_val_acc=0.7,
    )
    assert not common._is_better_checkpoint(
        selection_metric="val_acc_then_loss",
        val_loss=0.1,
        val_acc=0.6,
        best_val_loss=0.2,
        best_val_acc=0.7,
    )
    assert common._is_better_checkpoint(
        selection_metric="val_acc_then_loss_reset_any",
        val_loss=0.1,
        val_acc=0.7,
        best_val_loss=0.2,
        best_val_acc=0.7,
    )
    assert common._should_reset_patience(
        selection_metric="val_acc_then_loss_reset_any",
        checkpoint_updated=False,
        val_loss=0.1,
        val_acc=0.6,
        best_stop_val_loss=0.2,
        best_stop_val_acc=0.7,
    )
    assert common._should_reset_patience(
        selection_metric="val_acc_then_loss_reset_any",
        checkpoint_updated=False,
        val_loss=0.3,
        val_acc=0.8,
        best_stop_val_loss=0.2,
        best_stop_val_acc=0.7,
    )
    assert not common._should_reset_patience(
        selection_metric="val_acc_then_loss_reset_any",
        checkpoint_updated=False,
        val_loss=0.3,
        val_acc=0.6,
        best_stop_val_loss=0.2,
        best_stop_val_acc=0.7,
    )
    with pytest.raises(ValueError, match="selection_metric"):
        common._is_better_checkpoint(
            selection_metric="bad",
            val_loss=0.1,
            val_acc=0.1,
            best_val_loss=None,
            best_val_acc=None,
        )


def test_train_fullbatch_selects_by_val_acc_then_loss() -> None:
    y = torch.tensor([0, 1, 0, 1])
    train_mask = torch.tensor([True, True, False, False])
    val_mask = torch.tensor([False, False, True, True])
    model = torch.nn.Linear(1, 2)
    logits_by_call = [
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 1.0], [1.0, 0.0]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 1.0], [1.0, 0.0]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.8, 0.0], [0.0, 0.8]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 1.0], [1.0, 0.0]], requires_grad=True),
    ]
    calls = {"i": 0}

    def forward_fn():
        i = min(calls["i"], len(logits_by_call) - 1)
        calls["i"] += 1
        return logits_by_call[i]

    res = common.train_fullbatch(
        model=model,
        forward_fn=forward_fn,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        lr=0.0,
        weight_decay=0.0,
        max_epochs=3,
        patience=3,
        seed=0,
        weight_decay_scope="none",
        selection_metric="val_acc_then_loss",
    )

    assert res.best_epoch == 1
    assert res.best_val_acc == 1.0


def test_train_fullbatch_reset_any_extends_patience_on_loss_progress() -> None:
    y = torch.tensor([0, 1, 0, 1])
    train_mask = torch.tensor([True, True, False, False])
    val_mask = torch.tensor([False, False, True, True])
    model = torch.nn.Linear(1, 2)
    logits_by_call = [
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [1.0, 0.0], [1.0, 0.0]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [1.0, 0.0], [1.0, 0.0]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 0.2], [0.2, 0.0]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 0.2], [0.2, 0.0]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True),
    ]
    calls = {"i": 0}

    def forward_fn():
        i = min(calls["i"], len(logits_by_call) - 1)
        calls["i"] += 1
        return logits_by_call[i]

    res = common.train_fullbatch(
        model=model,
        forward_fn=forward_fn,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        lr=0.0,
        weight_decay=0.0,
        max_epochs=3,
        patience=1,
        seed=0,
        weight_decay_scope="none",
        selection_metric="val_acc_then_loss_reset_any",
    )

    assert res.n_epochs == 3
    assert res.best_epoch == 2
    assert res.best_val_acc == 1.0
