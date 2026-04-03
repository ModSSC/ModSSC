from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from modssc.transductive.methods.classic import tsvm as tsvm_mod
from modssc.transductive.methods.classic.tsvm import TSVMMethod, TSVMTransductiveSpec


@dataclass(frozen=True)
class DummyGraph:
    edge_index: Any
    edge_weight: Any | None = None


@dataclass(frozen=True)
class DummyNodeDataset:
    X: Any
    y: Any
    graph: DummyGraph
    masks: Mapping[str, Any] | None = None
    meta: Mapping[str, Any] | None = None


def test_tsvm_method_fit_predict_proba() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[rng.choice(np.arange(len(y)), size=10, replace=False)] = True

    edge_index = np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index),
        masks={"train_mask": train_mask},
    )

    spec = TSVMTransductiveSpec(
        max_iter=2,
        epochs_per_iter=1,
        batch_size=16,
        lr=0.1,
        C_l=1.0,
        C_u_max=0.1,
        l2=0.1,
        balance=True,
    )
    model = TSVMMethod(spec)
    model.fit(data, seed=0)

    proba = model.predict_proba(data)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_tsvm_method_all_labeled() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index),
        masks={"train_mask": np.ones(len(y), dtype=bool)},
    )

    model = TSVMMethod(TSVMTransductiveSpec(max_iter=1, epochs_per_iter=1, batch_size=8))
    model.fit(data, seed=0)
    proba = model.predict_proba(data)
    assert proba.shape == (len(y), 2)


def test_tsvm_method_requires_train_mask() -> None:
    X = np.zeros((5, 2), dtype=np.float32)
    y = np.zeros(5, dtype=np.int64)
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    data = DummyNodeDataset(X=X, y=y, graph=DummyGraph(edge_index=edge_index), masks={})

    with pytest.raises(ValueError, match="train_mask"):
        TSVMMethod().fit(data)


def test_tsvm_method_requires_binary_task() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(9, 2)).astype(np.float32)
    y = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[:3] = True

    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index),
        masks={"train_mask": train_mask},
    )

    with pytest.raises(ValueError, match="binary classification"):
        TSVMMethod().fit(data, seed=0)


def test_tsvm_batch_indices_empty() -> None:
    rng = np.random.default_rng(0)
    out = list(tsvm_mod._batch_indices(rng, np.array([], dtype=np.int64), batch_size=4))
    assert out == []


def test_linear_svm_no_active_margin_branch() -> None:
    svm = tsvm_mod._LinearSVM(n_features=1, seed=0)
    svm.w = np.array([10.0], dtype=np.float32)
    svm.b = np.float32(10.0)

    X = np.ones((2, 1), dtype=np.float32)
    y = np.ones((2,), dtype=np.float32)
    svm.fit_sgd(
        X,
        y,
        epochs=1,
        batch_size=2,
        lr=0.1,
        C=1.0,
        l2=0.1,
        rng=np.random.default_rng(0),
    )


def test_tsvm_mask_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    X = np.zeros((5, 2), dtype=np.float32)
    y = np.zeros(5, dtype=np.int64)
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index),
        masks={"train_mask": np.array([True, False, True, False])},
    )

    with pytest.raises(ValueError, match="must match number of nodes"):
        monkeypatch.setattr(tsvm_mod, "validate_node_dataset", lambda *_: None)
        TSVMMethod().fit(data)


def test_tsvm_requires_labeled_samples() -> None:
    X = np.zeros((5, 2), dtype=np.float32)
    y = np.zeros(5, dtype=np.int64)
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index),
        masks={"train_mask": np.zeros(5, dtype=bool)},
    )

    with pytest.raises(ValueError, match="at least 1 labeled"):
        TSVMMethod().fit(data)


def test_tsvm_balance_relabels_single_class(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[:2] = True

    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index),
        masks={"train_mask": train_mask},
    )

    monkeypatch.setattr(
        tsvm_mod._LinearSVM,
        "decision_function",
        lambda self, X: np.ones((X.shape[0],), dtype=np.float32),
    )

    spec = TSVMTransductiveSpec(max_iter=1, epochs_per_iter=1, batch_size=4, C_u_max=0.1)
    TSVMMethod(spec).fit(data, seed=0)


def test_tsvm_no_balance_and_no_rep() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[:2] = True

    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index),
        masks={"train_mask": train_mask},
    )

    spec = TSVMTransductiveSpec(
        max_iter=1,
        epochs_per_iter=1,
        batch_size=4,
        C_l=1e-4,
        C_u_max=1e-3,
        balance=False,
    )
    TSVMMethod(spec).fit(data, seed=0)


def test_tsvm_predict_proba_requires_fit() -> None:
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros(4, dtype=np.int64)
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index),
        masks={"train_mask": np.zeros(4, dtype=bool)},
    )

    with pytest.raises(RuntimeError, match="not fitted"):
        TSVMMethod().predict_proba(data)


def test_tsvm_meta_y_true_branch() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[:4] = True

    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index),
        masks={"train_mask": train_mask},
        meta={"y_true": y.copy()},
    )

    spec = TSVMTransductiveSpec(max_iter=1, epochs_per_iter=1, batch_size=4)
    TSVMMethod(spec).fit(data, seed=0)
