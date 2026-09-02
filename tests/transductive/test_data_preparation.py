from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import modssc.transductive.data as data_module
from modssc.data_loader.types import LoadedDataset, Split
from modssc.graph.artifacts import GraphArtifact
from modssc.transductive.data import (
    build_node_dataset,
    graph_from_dataset,
    masks_from_indices,
    masks_from_sampling,
    prepare_node_data,
)
from modssc.transductive.errors import TransductiveDataError


def _dataset() -> LoadedDataset:
    return LoadedDataset(
        train=Split(
            X=np.arange(12, dtype=np.float32).reshape(4, 3),
            y=np.array([0, 1, 0, 1]),
            edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        ),
        test=Split(
            X=np.arange(6, dtype=np.float32).reshape(2, 3),
            y=np.array([1, 0]),
        ),
        meta={},
    )


def _valid_indices() -> dict[str, np.ndarray]:
    return {
        "train": np.array([0, 2, 3]),
        "val": np.array([1]),
        "test": np.array([0, 1]),
        "train_labeled": np.array([0, 2]),
        "train_unlabeled": np.array([3]),
    }


def _valid_refs() -> dict[str, str]:
    return {
        "train": "train",
        "val": "train",
        "test": "test",
        "train_labeled": "train",
        "train_unlabeled": "train",
    }


def test_masks_from_indices_respects_split_references() -> None:
    masks = masks_from_indices(
        n_train=4,
        n_test=2,
        indices=_valid_indices(),
        refs=_valid_refs(),
    )

    np.testing.assert_array_equal(np.flatnonzero(masks["labeled_mask"]), [0, 2])
    np.testing.assert_array_equal(np.flatnonzero(masks["unlabeled_mask"]), [3])
    np.testing.assert_array_equal(np.flatnonzero(masks["val_mask"]), [1])
    np.testing.assert_array_equal(np.flatnonzero(masks["test_mask"]), [4, 5])


def test_graph_from_dataset_normalizes_edge_pair_layout() -> None:
    graph = graph_from_dataset(_dataset(), n_nodes=4)

    assert graph.edge_index.shape == (2, 3)
    np.testing.assert_array_equal(graph.edge_index[:, 0], [0, 1])


def test_prepare_node_data_physically_separates_fit_labels_from_evaluation_truth() -> None:
    dataset = _dataset()
    graph = GraphArtifact(
        n_nodes=6,
        edge_index=np.array([[0, 1, 4], [1, 2, 5]], dtype=np.int64),
    )
    masks = masks_from_indices(
        n_train=4,
        n_test=2,
        indices={
            "train": np.array([0, 1, 2, 3]),
            "val": np.array([], dtype=np.int64),
            "test": np.array([0, 1]),
            "train_labeled": np.array([0, 2]),
            "train_unlabeled": np.array([1, 3]),
        },
        refs=_valid_refs(),
    )

    prepared = prepare_node_data(
        dataset=dataset,
        graph=graph,
        masks=masks,
        use_test_split=True,
        expected_labeled_count=2,
    )

    np.testing.assert_array_equal(prepared.fit.y, [0, -1, 0, -1, -1, -1])
    assert "y_true" not in prepared.fit.meta
    assert "val_mask" not in prepared.fit.masks
    assert "test_mask" not in prepared.fit.masks
    np.testing.assert_array_equal(prepared.evaluation.y_true, [0, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(
        np.flatnonzero(prepared.fit.masks["train_all_mask"]), [0, 1, 2, 3]
    )
    with pytest.raises(ValueError, match="read-only"):
        prepared.evaluation.y_true[0] = 1


def test_prepare_node_data_supports_graph_optional_methods() -> None:
    masks = masks_from_indices(
        n_train=4,
        n_test=None,
        indices={
            "train": np.array([0, 1, 2, 3]),
            "val": np.array([], dtype=np.int64),
            "test": np.array([], dtype=np.int64),
            "train_labeled": np.array([0, 2]),
            "train_unlabeled": np.array([1, 3]),
        },
        refs={
            "train": "train",
            "val": "train",
            "test": "train",
            "train_labeled": "train",
            "train_unlabeled": "train",
        },
    )

    prepared = prepare_node_data(
        dataset=_dataset(),
        graph=None,
        masks=masks,
        use_test_split=False,
        expected_labeled_count=2,
    )

    assert prepared.fit.graph is None
    np.testing.assert_array_equal(prepared.fit.y, [0, -1, 0, -1])


def test_build_node_dataset_returns_only_fit_visible_data() -> None:
    dataset = _dataset()
    graph = GraphArtifact(
        n_nodes=4,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
    )
    masks = {
        "train_mask": np.array([True, True, True, True]),
        "val_mask": np.zeros(4, dtype=bool),
        "test_mask": np.zeros(4, dtype=bool),
        "unlabeled_mask": np.array([False, True, False, True]),
        "labeled_mask": np.array([True, False, True, False]),
    }

    fit_data = build_node_dataset(
        dataset=dataset,
        graph=graph,
        masks=masks,
        use_test_split=False,
    )

    assert "y_true" not in fit_data.meta
    np.testing.assert_array_equal(fit_data.y, [0, -1, 0, -1])


def test_build_node_dataset_fails_closed_on_labeled_count_mismatch() -> None:
    dataset = _dataset()
    graph = GraphArtifact(
        n_nodes=4,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
    )
    masks = {
        "train_mask": np.array([True, True, True, True]),
        "val_mask": np.zeros(4, dtype=bool),
        "test_mask": np.zeros(4, dtype=bool),
        "unlabeled_mask": np.array([False, True, True, True]),
        "labeled_mask": np.array([True, False, False, False]),
    }

    with pytest.raises(TransductiveDataError) as raised:
        build_node_dataset(
            dataset=dataset,
            graph=graph,
            masks=masks,
            use_test_split=False,
            expected_labeled_count=2,
        )

    assert raised.value.code == "E_TRANSDUCTIVE_LABELED_MASK"


def test_build_node_dataset_requires_complete_mask_contract() -> None:
    dataset = _dataset()
    graph = GraphArtifact(
        n_nodes=4,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
    )

    with pytest.raises(TransductiveDataError) as raised:
        build_node_dataset(
            dataset=dataset,
            graph=graph,
            masks={"labeled_mask": np.array([True, False, False, False])},
            use_test_split=False,
        )

    assert raised.value.code == "E_TRANSDUCTIVE_MASKS"


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("missing_index_key", "E_TRANSDUCTIVE_INDICES"),
        ("unknown_index_key", "E_TRANSDUCTIVE_INDICES"),
        ("missing_ref_key", "E_TRANSDUCTIVE_REFS"),
        ("unknown_ref", "E_TRANSDUCTIVE_REFS"),
        ("training_refers_to_test", "E_TRANSDUCTIVE_REFS"),
        ("negative_index", "E_TRANSDUCTIVE_INDICES"),
        ("out_of_bounds_index", "E_TRANSDUCTIVE_INDICES"),
        ("float_index", "E_TRANSDUCTIVE_INDICES"),
        ("duplicate_index", "E_TRANSDUCTIVE_INDICES"),
        ("train_val_overlap", "E_TRANSDUCTIVE_MASK_OVERLAP"),
        ("labeled_outside_train", "E_TRANSDUCTIVE_MASK_OVERLAP"),
        ("invalid_unlabeled_pool", "E_TRANSDUCTIVE_MASK_OVERLAP"),
        ("missing_test_size", "E_TRANSDUCTIVE_REFS"),
        ("invalid_train_size", "E_TRANSDUCTIVE_SHAPE"),
    ],
)
def test_masks_from_indices_fails_closed(case: str, expected_code: str) -> None:
    indices = _valid_indices()
    refs = _valid_refs()
    n_train: int = 4
    n_test: int | None = 2

    if case == "missing_index_key":
        indices.pop("val")
    elif case == "unknown_index_key":
        indices["mystery"] = np.array([], dtype=np.int64)
    elif case == "missing_ref_key":
        refs.pop("val")
    elif case == "unknown_ref":
        refs["test"] = "nodes"
    elif case == "training_refers_to_test":
        refs["val"] = "test"
    elif case == "negative_index":
        indices["train"] = np.array([-1, 2, 3])
    elif case == "out_of_bounds_index":
        indices["test"] = np.array([0, 2])
    elif case == "float_index":
        indices["val"] = np.array([1.0])
    elif case == "duplicate_index":
        indices["train"] = np.array([0, 2, 2])
    elif case == "train_val_overlap":
        indices["train"] = np.array([0, 1, 2, 3])
    elif case == "labeled_outside_train":
        indices["train_labeled"] = np.array([0, 1])
    elif case == "invalid_unlabeled_pool":
        indices["train_unlabeled"] = np.array([2, 3])
    elif case == "missing_test_size":
        n_test = None
    elif case == "invalid_train_size":
        n_train = True
    else:  # pragma: no cover - guards the parameter table
        raise AssertionError(case)

    with pytest.raises(TransductiveDataError) as raised:
        masks_from_indices(
            n_train=n_train,
            n_test=n_test,
            indices=indices,
            refs=refs,
        )

    assert raised.value.code == expected_code


def test_prepare_node_data_preserves_scipy_sparse_features() -> None:
    sparse = pytest.importorskip("scipy.sparse")
    dataset = LoadedDataset(
        train=Split(
            X=sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3)),
            y=np.array([0, 1, 0, 1]),
        ),
        test=Split(
            X=sparse.csr_matrix(np.arange(6, dtype=np.float32).reshape(2, 3)),
            y=np.array([1, 0]),
        ),
    )
    graph = GraphArtifact(
        n_nodes=6,
        edge_index=np.array([[0, 1, 4], [1, 2, 5]], dtype=np.int64),
    )
    masks = masks_from_indices(
        n_train=4,
        n_test=2,
        indices={
            "train": np.array([0, 1, 2, 3]),
            "val": np.array([], dtype=np.int64),
            "test": np.array([0, 1]),
            "train_labeled": np.array([0, 2]),
            "train_unlabeled": np.array([1, 3]),
        },
        refs=_valid_refs(),
    )

    prepared = prepare_node_data(
        dataset=dataset,
        graph=graph,
        masks=masks,
        use_test_split=True,
    )

    assert sparse.isspmatrix_csr(prepared.fit.X)
    assert prepared.fit.X.shape == (6, 3)
    np.testing.assert_array_equal(
        prepared.fit.X.toarray(),
        np.vstack([dataset.train.X.toarray(), dataset.test.X.toarray()]),
    )


def test_graph_from_dataset_wraps_graph_validation_errors() -> None:
    dataset = LoadedDataset(
        train=Split(
            X=np.zeros((2, 1)),
            y=np.array([0, 1]),
            edges=np.array([[0, 2]], dtype=np.int64),
        )
    )

    with pytest.raises(TransductiveDataError) as raised:
        graph_from_dataset(dataset, n_nodes=2)

    assert raised.value.code == "E_TRANSDUCTIVE_GRAPH"


def test_prepare_node_data_wraps_graph_shape_mismatch() -> None:
    graph = GraphArtifact(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
    )
    masks = {
        "train_mask": np.ones(4, dtype=bool),
        "val_mask": np.zeros(4, dtype=bool),
        "test_mask": np.zeros(4, dtype=bool),
        "unlabeled_mask": np.array([False, True, True, True]),
        "labeled_mask": np.array([True, False, False, False]),
    }

    with pytest.raises(TransductiveDataError) as raised:
        prepare_node_data(
            dataset=_dataset(),
            graph=graph,
            masks=masks,
            use_test_split=False,
        )

    assert raised.value.code == "E_TRANSDUCTIVE_SHAPE"


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (np.array([0.0, 1.0]), "integer class ids"),
        (np.array([[0, 1]]), "shape"),
    ],
)
def test_node_evaluation_truth_is_strictly_one_dimensional_integer(truth, message) -> None:
    masks = {
        "train_mask": np.array([True, False]),
        "val_mask": np.array([False, True]),
        "test_mask": np.zeros(2, dtype=bool),
        "unlabeled_mask": np.array([True, False]),
        "labeled_mask": np.array([False, False]),
    }
    with pytest.raises(TransductiveDataError, match=message):
        data_module.NodeEvaluationData(y_true=truth, masks=masks)


def test_transductive_numpy_conversion_runs_tensor_protocol() -> None:
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
            return np.array([1, 2])

    np.testing.assert_array_equal(data_module.to_numpy(TensorLike()), [1, 2])
    np.testing.assert_array_equal(data_module.to_numpy([1, 2]), [1, 2])
    assert calls == ["detach", "cpu", "numpy"]


def test_scipy_sparse_detection_handles_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def missing_scipy(name: str):
        if name == "scipy.sparse":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(data_module.importlib, "import_module", missing_scipy)
    assert data_module._scipy_sparse() is None


@pytest.mark.parametrize("value", [None, -1])
def test_required_sizes_must_be_non_negative_integers(value) -> None:
    with pytest.raises(TransductiveDataError, match="non-negative|is required"):
        data_module._normalize_size("n_nodes", value)


def test_index_and_mask_shapes_are_validated_before_materialization() -> None:
    with pytest.raises(TransductiveDataError, match="one-dimensional"):
        data_module._normalize_indices(
            "train",
            np.array([[0]]),
            reference="train",
            n_train=1,
            n_test=None,
        )

    masks = {
        "train_mask": np.array([1, 0]),
        "val_mask": np.zeros(2, dtype=bool),
        "test_mask": np.zeros(2, dtype=bool),
        "unlabeled_mask": np.zeros(2, dtype=bool),
        "labeled_mask": np.zeros(2, dtype=bool),
    }
    with pytest.raises(TransductiveDataError, match="bool dtype"):
        data_module._canonical_masks(masks, n_nodes=2)

    masks["train_mask"] = np.zeros(3, dtype=bool)
    with pytest.raises(TransductiveDataError, match="must have shape"):
        data_module._canonical_masks(masks, n_nodes=2)


def test_masks_from_sampling_routes_index_representation() -> None:
    sampling = SimpleNamespace(is_graph=lambda: False, indices=_valid_indices(), refs=_valid_refs())

    masks = masks_from_sampling(sampling, n_train=4, n_test=2)

    np.testing.assert_array_equal(np.flatnonzero(masks["test_mask"]), [4, 5])


def test_graph_from_dataset_accepts_artifacts_and_weighted_mappings() -> None:
    artifact = GraphArtifact(
        n_nodes=4,
        edge_index=np.array([[0], [1]], dtype=np.int64),
    )
    dataset = LoadedDataset(train=Split(X=np.zeros((4, 1)), y=np.arange(4), edges=artifact))
    assert graph_from_dataset(dataset, n_nodes=4) is artifact

    with pytest.raises(TransductiveDataError, match="n_nodes mismatch"):
        graph_from_dataset(dataset, n_nodes=3)

    missing = LoadedDataset(
        train=Split(X=np.zeros((2, 1)), y=np.arange(2), edges={"edge_weight": [1.0]})
    )
    with pytest.raises(TransductiveDataError, match="missing edge_index"):
        graph_from_dataset(missing, n_nodes=2)

    weighted = LoadedDataset(
        train=Split(
            X=np.zeros((2, 1)),
            y=np.arange(2),
            edges={
                "edge_index": np.array([[0], [1]], dtype=np.int64),
                "edge_weight": np.array([0.25]),
            },
        )
    )
    graph = graph_from_dataset(weighted, n_nodes=2)
    np.testing.assert_array_equal(graph.edge_index, [[0], [1]])
    np.testing.assert_allclose(graph.edge_weight, [0.25])


def test_labels_array_closes_materialization_shape_and_value_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_module, "_combine_splits", lambda *_args: object())
    monkeypatch.setattr(
        data_module,
        "to_numpy",
        lambda _value: (_ for _ in ()).throw(ValueError("cannot convert")),
    )
    with pytest.raises(TransductiveDataError, match="failed to materialize labels"):
        data_module._labels_array(np.array([0]), None)

    monkeypatch.undo()
    with pytest.raises(TransductiveDataError, match="must have shape"):
        data_module._labels_array(np.array([[0, 1]]), None)
    with pytest.raises(TransductiveDataError, match="finite integer"):
        data_module._labels_array(np.array([0.0, np.nan]), None)
    np.testing.assert_array_equal(data_module._labels_array(np.array([0.0, 1.0]), None), [0, 1])
    with pytest.raises(TransductiveDataError, match="integer class ids"):
        data_module._labels_array(np.array([0.5, 1.0]), None)
    with pytest.raises(TransductiveDataError, match="integer class ids"):
        data_module._labels_array(np.array(["zero", "one"]), None)


def test_prepare_node_data_rejects_feature_label_mismatch_and_wraps_unknown_shapes() -> None:
    mismatched = LoadedDataset(
        train=Split(X=np.zeros((4, 2)), y=np.array([0, 1, 0])),
    )
    graph = GraphArtifact(
        n_nodes=4,
        edge_index=np.array([[0], [1]], dtype=np.int64),
    )
    masks = {
        "train_mask": np.ones(4, dtype=bool),
        "val_mask": np.zeros(4, dtype=bool),
        "test_mask": np.zeros(4, dtype=bool),
        "unlabeled_mask": np.array([False, True, True, True]),
        "labeled_mask": np.array([True, False, False, False]),
    }
    with pytest.raises(TransductiveDataError, match="feature/label row mismatch"):
        prepare_node_data(
            dataset=mismatched,
            graph=graph,
            masks=masks,
            use_test_split=False,
        )

    malformed = SimpleNamespace(
        train=SimpleNamespace(X=object(), y=np.array([0])),
        test=None,
    )
    with pytest.raises(TransductiveDataError) as caught:
        prepare_node_data(
            dataset=malformed,
            graph=GraphArtifact(
                n_nodes=1,
                edge_index=np.empty((2, 0), dtype=np.int64),
            ),
            masks={key: np.zeros(1, dtype=bool) for key in masks},
            use_test_split=False,
        )
    assert caught.value.code == "E_TRANSDUCTIVE_DATA"
