from __future__ import annotations

import hashlib

import numpy as np
import pytest

import modssc.views.api as views_api
from modssc.data_loader.types import LoadedDataset, Split
from modssc.preprocess.plan import PreprocessPlan, StepConfig
from modssc.views import (
    ColumnSelectSpec,
    ViewSpec,
    ViewsPlan,
    generate_views,
    two_view_random_feature_split,
)
from modssc.views.errors import ViewsValidationError


def _dummy_dataset(n_train: int = 10, n_test: int = 5, n_features: int = 6) -> LoadedDataset:
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(n_train, n_features)).astype(np.float32)
    y_tr = rng.integers(0, 3, size=(n_train,), dtype=np.int64)
    X_te = rng.normal(size=(n_test, n_features)).astype(np.float32)
    y_te = rng.integers(0, 3, size=(n_test,), dtype=np.int64)
    return LoadedDataset(
        train=Split(X=X_tr, y=y_tr, edges=None, masks=None),
        test=Split(X=X_te, y=y_te, edges=None, masks=None),
        meta={},
    )


def test_view_source_content_digest_handles_supported_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SparseStub:
        shape = (2, 2)
        row = np.asarray([0, 1])
        col = np.asarray([1, 0])
        data = np.asarray([2.0, 3.0])

        def tocoo(self):
            return self

    class StableObject:
        def __repr__(self) -> str:
            return "stable-object"

    nested_scalar = np.empty((), dtype=object)
    nested_scalar[()] = ["nested"]
    first = hashlib.sha256()
    views_api._update_content_digest(
        first,
        {
            "nested_scalar": nested_scalar,
            "sequence": [np.asarray([1, 2]), (3, 4)],
            "sparse": SparseStub(),
            "objects": np.asarray(
                [np.int64(1), "text", b"bytes", None, True, 1.5, StableObject()],
                dtype=object,
            ),
        },
    )
    second = hashlib.sha256()
    views_api._update_content_digest(
        second,
        {
            "nested_scalar": nested_scalar,
            "objects": np.asarray(
                [np.int64(1), "text", b"bytes", None, True, 1.5, StableObject()],
                dtype=object,
            ),
            "sparse": SparseStub(),
            "sequence": [np.asarray([1, 2]), (3, 4)],
        },
    )
    assert first.hexdigest() == second.hexdigest()

    trusted = _dummy_dataset(n_features=2)
    trusted = LoadedDataset(
        train=trusted.train,
        test=trusted.test,
        meta={"dataset_content_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        views_api,
        "_dataset_content_sha256",
        lambda _dataset: pytest.fail("trusted content digest must be used"),
    )
    assert views_api._source_dataset_fingerprint(trusted).startswith("dataset:")


def test_generate_views_random_and_complement_is_deterministic() -> None:
    ds = _dummy_dataset(n_features=6)

    plan = two_view_random_feature_split(preprocess=None, fraction=0.5)
    res1 = generate_views(ds, plan=plan, seed=123, cache=False)
    res2 = generate_views(ds, plan=plan, seed=123, cache=False)

    cols_a1 = res1.columns["view_a"]
    cols_a2 = res2.columns["view_a"]
    np.testing.assert_array_equal(cols_a1, cols_a2)

    cols_b = res1.columns["view_b"]
    # disjoint + cover all
    assert np.intersect1d(cols_a1, cols_b).size == 0
    union = np.union1d(cols_a1, cols_b)
    np.testing.assert_array_equal(union, np.arange(6, dtype=np.int64))

    assert res1.views["view_a"].train.X.shape[1] == cols_a1.size
    assert res1.views["view_b"].train.X.shape[1] == cols_b.size
    assert res1.meta == {"n_views": 2}


def test_generate_views_indices_mode() -> None:
    ds = _dummy_dataset(n_features=6)
    plan = ViewsPlan(
        views=(
            ViewSpec(name="v1", columns=ColumnSelectSpec(mode="indices", indices=(0, 2, 4))),
            ViewSpec(name="v2", columns=ColumnSelectSpec(mode="all")),
        )
    )
    res = generate_views(ds, plan=plan, seed=0, cache=False)
    assert res.views["v1"].train.X.shape == (10, 3)
    assert res.views["v2"].train.X.shape == (10, 6)


def test_generate_views_invalid_indices_raises() -> None:
    ds = _dummy_dataset(n_features=3)
    plan = ViewsPlan(
        views=(
            ViewSpec(name="v1", columns=ColumnSelectSpec(mode="indices", indices=(0, 3))),
            ViewSpec(name="v2", columns=ColumnSelectSpec(mode="all")),
        )
    )
    with pytest.raises(ViewsValidationError):
        generate_views(ds, plan=plan, seed=0, cache=False)


def test_generate_views_complement_empty_raises() -> None:
    ds = _dummy_dataset(n_features=1)
    plan = two_view_random_feature_split(preprocess=None, fraction=1.0)
    with pytest.raises(ViewsValidationError):
        generate_views(ds, plan=plan, seed=0, cache=False)


def test_shape_of_handles_invalid_shape() -> None:
    class BadShape:
        shape = ("bad",)

    assert views_api._shape_of(BadShape()) is None
    assert views_api._shape_of(object()) is None


def test_generate_views_missing_meta_attribute() -> None:
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(6, 4)).astype(np.float32)
    y_tr = rng.integers(0, 2, size=(6,), dtype=np.int64)
    X_te = rng.normal(size=(3, 4)).astype(np.float32)
    y_te = rng.integers(0, 2, size=(3,), dtype=np.int64)

    class DatasetNoMeta:
        def __init__(self):
            self.train = Split(X=X_tr, y=y_tr, edges=None, masks=None)
            self.test = Split(X=X_te, y=y_te, edges=None, masks=None)
            self.meta = None

    ds = DatasetNoMeta()
    plan = two_view_random_feature_split(preprocess=None, fraction=0.5)
    res = generate_views(ds, plan=plan, seed=123, cache=False)
    assert set(res.views.keys()) == {"view_a", "view_b"}


def test_generate_views_with_preprocess_core_steps() -> None:
    ds = _dummy_dataset(n_features=4)

    pre = PreprocessPlan(
        steps=(
            StepConfig(step_id="core.ensure_2d"),
            StepConfig(step_id="core.cast_dtype", params={"dtype": "float32"}),
        ),
        output_key="features.X",
    )

    plan = two_view_random_feature_split(preprocess=pre, fraction=0.5)
    res = generate_views(ds, plan=plan, seed=0, cache=False)

    assert res.views["view_a"].train.X.dtype == np.float32
    assert res.views["view_b"].train.X.dtype == np.float32
    assert res.views["view_a"].test is not None


def test_generate_views_selects_native_input_columns_before_preprocess() -> None:
    X_train = np.asarray([["page alpha", "anchor one"], ["page beta", "anchor two"]], dtype=object)
    X_test = np.asarray([["page gamma", "anchor three"]], dtype=object)
    ds = LoadedDataset(
        train=Split(X=X_train, y=np.asarray([0, 1])),
        test=Split(X=X_test, y=np.asarray([1])),
        meta={},
    )
    pre = PreprocessPlan(
        steps=(StepConfig(step_id="core.copy_raw"),),
        output_key="features.X",
    )
    plan = ViewsPlan(
        views=(
            ViewSpec(
                name="page",
                input_columns=ColumnSelectSpec(mode="indices", indices=(0,)),
                preprocess=pre,
            ),
            ViewSpec(
                name="anchor",
                input_columns=ColumnSelectSpec(mode="indices", indices=(1,)),
                preprocess=pre,
            ),
        )
    )

    result = generate_views(ds, plan=plan, seed=0, cache=False)

    assert result.views["page"].train.X[:, 0].tolist() == ["page alpha", "page beta"]
    assert result.views["anchor"].train.X[:, 0].tolist() == ["anchor one", "anchor two"]
    assert result.views["page"].test is not None
    assert result.views["page"].test.X[:, 0].tolist() == ["page gamma"]
    assert result.meta["input_columns"] == {"anchor": [1], "page": [0]}
    page_meta = result.views["page"].meta["views"]["page"]
    assert page_meta["input_columns"] == [0]
    assert page_meta["input_columns_mode"] == "indices"
    assert page_meta["columns"] == [0]
    assert page_meta["columns_mode"] == "all"


def test_generate_views_cache_isolated_by_native_input_columns(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modssc.preprocess.steps.core.copy_raw import CopyRawStep

    monkeypatch.setenv("MODSSC_PREPROCESS_CACHE_DIR", str(tmp_path))
    X_train = np.asarray([["page alpha", "anchor one"], ["page beta", "anchor two"]], dtype=object)
    X_test = np.asarray([["page gamma", "anchor three"]], dtype=object)
    source_fingerprint = "native-columns-source"
    ds = LoadedDataset(
        train=Split(X=X_train, y=np.asarray([0, 1])),
        test=Split(X=X_test, y=np.asarray([1])),
        meta={"dataset_fingerprint": source_fingerprint},
    )
    pre = PreprocessPlan(
        steps=(StepConfig(step_id="core.copy_raw"),),
        output_key="features.X",
    )
    plan = ViewsPlan(
        views=(
            ViewSpec(
                name="fulltext",
                input_columns=ColumnSelectSpec(mode="indices", indices=(0,)),
                preprocess=pre,
            ),
            ViewSpec(
                name="inlinks",
                input_columns=ColumnSelectSpec(mode="indices", indices=(1,)),
                preprocess=pre,
            ),
        )
    )

    first = generate_views(ds, plan=plan, seed=0, cache=True)

    assert first.views["fulltext"].train.X[:, 0].tolist() == ["page alpha", "page beta"]
    assert first.views["inlinks"].train.X[:, 0].tolist() == ["anchor one", "anchor two"]
    assert (
        first.views["fulltext"].meta["preprocess_cache_dir"]
        != first.views["inlinks"].meta["preprocess_cache_dir"]
    )
    assert first.views["fulltext"].meta["dataset_fingerprint"] == source_fingerprint
    assert first.views["inlinks"].meta["dataset_fingerprint"] == source_fingerprint

    def fail_if_transform_runs(*args, **kwargs):
        pytest.fail("core.copy_raw should be replayed from cache")

    monkeypatch.setattr(CopyRawStep, "transform", fail_if_transform_runs)
    replay = generate_views(ds, plan=plan, seed=0, cache=True)

    np.testing.assert_array_equal(replay.views["fulltext"].train.X, first.views["fulltext"].train.X)
    np.testing.assert_array_equal(replay.views["inlinks"].train.X, first.views["inlinks"].train.X)
    assert (
        replay.views["fulltext"].meta["preprocess_cache_dir"]
        == first.views["fulltext"].meta["preprocess_cache_dir"]
    )
    assert (
        replay.views["inlinks"].meta["preprocess_cache_dir"]
        == first.views["inlinks"].meta["preprocess_cache_dir"]
    )


def test_generate_views_cache_isolated_by_source_content(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODSSC_PREPROCESS_CACHE_DIR", str(tmp_path))
    pre = PreprocessPlan(
        steps=(StepConfig(step_id="core.copy_raw"),),
        output_key="features.X",
    )
    plan = ViewsPlan(
        views=(
            ViewSpec(
                name="left",
                input_columns=ColumnSelectSpec(mode="indices", indices=(0,)),
                preprocess=pre,
            ),
            ViewSpec(
                name="right",
                input_columns=ColumnSelectSpec(mode="indices", indices=(1,)),
                preprocess=pre,
            ),
        )
    )
    first_dataset = LoadedDataset(
        train=Split(X=np.asarray([[1, 2], [3, 4]]), y=np.asarray([0, 1])),
        test=None,
        meta={},
    )
    second_dataset = LoadedDataset(
        train=Split(X=np.asarray([[10, 20], [30, 40]]), y=np.asarray([0, 1])),
        test=None,
        meta={},
    )

    first = generate_views(first_dataset, plan=plan, seed=0, cache=True)
    second = generate_views(second_dataset, plan=plan, seed=0, cache=True)

    np.testing.assert_array_equal(second.views["left"].train.X[:, 0], [10, 30])
    np.testing.assert_array_equal(second.views["right"].train.X[:, 0], [20, 40])
    assert (
        first.views["left"].meta["preprocess_cache_dir"]
        != second.views["left"].meta["preprocess_cache_dir"]
    )


def test_generate_native_input_complement_without_preprocess() -> None:
    dataset = LoadedDataset(
        train=Split(
            X=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            y=np.asarray([0, 1]),
        ),
        test=None,
        meta={"dataset_fingerprint": "native-complement-source"},
    )
    plan = ViewsPlan(
        views=(
            ViewSpec(
                name="primary",
                input_columns=ColumnSelectSpec(mode="indices", indices=(0,)),
            ),
            ViewSpec(
                name="complement",
                input_columns=ColumnSelectSpec(
                    mode="complement",
                    complement_of="primary",
                ),
            ),
        )
    )

    result = generate_views(dataset, plan=plan, seed=0, cache=True)

    np.testing.assert_array_equal(
        result.views["primary"].train.X,
        np.asarray([[1.0], [4.0]]),
    )
    np.testing.assert_array_equal(
        result.views["complement"].train.X,
        np.asarray([[2.0, 3.0], [5.0, 6.0]]),
    )
    assert result.meta["input_columns"] == {
        "complement": [1, 2],
        "primary": [0],
    }


def test_generate_views_input_columns_requires_matrix() -> None:
    ds = LoadedDataset(
        train=Split(X=np.asarray(["a", "b"], dtype=object), y=np.asarray([0, 1])),
        test=None,
        meta={},
    )
    plan = ViewsPlan(
        views=(
            ViewSpec(name="a", input_columns=ColumnSelectSpec(mode="indices", indices=(0,))),
            ViewSpec(name="b", input_columns=ColumnSelectSpec(mode="indices", indices=(0,))),
        )
    )
    with pytest.raises(ViewsValidationError, match="input_columns requires train.X"):
        generate_views(ds, plan=plan, seed=0, cache=False)


def test_generate_views_with_dict_features() -> None:
    rng = np.random.default_rng(0)
    X_tr = {
        "x": rng.normal(size=(6, 4)).astype(np.float32),
        "edge_index": np.array([[0, 1], [1, 2]]),
    }
    y_tr = rng.integers(0, 2, size=(6,), dtype=np.int64)
    X_te = {
        "x": rng.normal(size=(3, 4)).astype(np.float32),
        "edge_index": np.array([[0, 1], [1, 2]]),
    }
    y_te = rng.integers(0, 2, size=(3,), dtype=np.int64)
    ds = LoadedDataset(
        train=Split(X=X_tr, y=y_tr, edges=None, masks=None),
        test=Split(X=X_te, y=y_te, edges=None, masks=None),
        meta={},
    )
    plan = ViewsPlan(
        views=(
            ViewSpec(name="v1", columns=ColumnSelectSpec(mode="indices", indices=(0, 2))),
            ViewSpec(name="v2", columns=ColumnSelectSpec(mode="all")),
        )
    )
    res = generate_views(ds, plan=plan, seed=0, cache=False)
    view = res.views["v1"]
    assert isinstance(view.train.X, dict)
    assert view.train.X["x"].shape == (6, 2)
    assert np.array_equal(view.train.X["edge_index"], X_tr["edge_index"])
    assert view.test is not None
    assert view.test.X["x"].shape == (3, 2)


def test_as_numpy_handles_detach_cpu_numpy() -> None:
    class FakeTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    arr = views_api._as_numpy(FakeTensor([1.0, 2.0, 3.0]))
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))


def test_as_numpy_fallback_paths() -> None:
    class OnlyNumpy:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def numpy(self):
            return self._arr

    arr = views_api._as_numpy(OnlyNumpy([1, 2]))
    np.testing.assert_array_equal(arr, np.array([1, 2]))

    arr2 = views_api._as_numpy([3, 4])
    np.testing.assert_array_equal(arr2, np.array([3, 4]))


def test_resolve_columns_errors() -> None:
    resolved: dict[str, np.ndarray] = {}
    n_features_map: dict[str, int] = {}

    with pytest.raises(ViewsValidationError, match="n_features <= 0"):
        views_api._resolve_columns(
            spec=None,
            n_features=0,
            seed=0,
            view_name="v",
            resolved=resolved,
            n_features_map=n_features_map,
        )

    class EmptyIndicesSpec:
        mode = "indices"
        indices = ()

        def validate(self):
            return None

    with pytest.raises(ViewsValidationError, match="empty list"):
        views_api._resolve_columns(
            spec=EmptyIndicesSpec(),
            n_features=3,
            seed=0,
            view_name="v",
            resolved=resolved,
            n_features_map=n_features_map,
        )

    with pytest.raises(ViewsValidationError, match="duplicates"):
        views_api._resolve_columns(
            spec=ColumnSelectSpec(mode="indices", indices=(0, 0)),
            n_features=3,
            seed=0,
            view_name="v",
            resolved=resolved,
            n_features_map=n_features_map,
        )

    with pytest.raises(ViewsValidationError, match="not resolved yet"):
        views_api._resolve_columns(
            spec=ColumnSelectSpec(mode="complement", complement_of="missing"),
            n_features=3,
            seed=0,
            view_name="v",
            resolved=resolved,
            n_features_map=n_features_map,
        )

    resolved["base"] = np.array([0], dtype=np.int64)
    n_features_map["base"] = 3
    with pytest.raises(ViewsValidationError, match="has n_features"):
        views_api._resolve_columns(
            spec=ColumnSelectSpec(mode="complement", complement_of="base"),
            n_features=4,
            seed=0,
            view_name="v",
            resolved=resolved,
            n_features_map=n_features_map,
        )

    class DummySpec:
        mode = "weird"

        def validate(self):
            return None

    with pytest.raises(ViewsValidationError, match="Unhandled"):
        views_api._resolve_columns(
            spec=DummySpec(),
            n_features=3,
            seed=0,
            view_name="v",
            resolved=resolved,
            n_features_map=n_features_map,
        )


def test_generate_views_with_fit_indices() -> None:
    ds = _dummy_dataset(n_features=6)
    plan = ViewsPlan(views=(ViewSpec(name="v1"), ViewSpec(name="v2")))
    res = generate_views(ds, plan=plan, seed=0, cache=False, fit_indices=np.array([0, 1]))
    assert set(res.views.keys()) == {"v1", "v2"}


def test_generate_views_train_dim_error() -> None:
    ds = LoadedDataset(
        train=Split(X=np.array([1.0, 2.0]), y=np.array([0, 1]), edges=None, masks=None),
        test=None,
        meta={},
    )
    plan = ViewsPlan(views=(ViewSpec(name="v1"), ViewSpec(name="v2")))
    with pytest.raises(ViewsValidationError, match="expected train.X to be at least 2D"):
        generate_views(ds, plan=plan, seed=0, cache=False)


def test_generate_views_test_dim_error() -> None:
    train = Split(
        X=np.zeros((2, 2), dtype=np.float32),
        y=np.array([0, 1], dtype=np.int64),
        edges=None,
        masks=None,
    )
    test = Split(X=np.array([1.0, 2.0]), y=np.array([0, 1]), edges=None, masks=None)
    ds = LoadedDataset(train=train, test=test, meta={})
    plan = ViewsPlan(views=(ViewSpec(name="v1"), ViewSpec(name="v2")))
    with pytest.raises(ViewsValidationError, match="expected test.X to be at least 2D"):
        generate_views(ds, plan=plan, seed=0, cache=False)


def test_generate_views_input_columns_test_dim_error() -> None:
    ds = LoadedDataset(
        train=Split(X=np.zeros((2, 2)), y=np.asarray([0, 1])),
        test=Split(X=np.asarray(["a", "b"], dtype=object), y=np.asarray([0, 1])),
        meta={},
    )
    plan = ViewsPlan(
        views=(
            ViewSpec(name="a", input_columns=ColumnSelectSpec(mode="indices", indices=(0,))),
            ViewSpec(name="b", input_columns=ColumnSelectSpec(mode="indices", indices=(1,))),
        )
    )
    with pytest.raises(ViewsValidationError, match="input_columns requires test.X"):
        generate_views(ds, plan=plan, seed=0, cache=False)
