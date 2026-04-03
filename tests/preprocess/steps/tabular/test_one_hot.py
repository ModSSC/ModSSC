import numpy as np
import pytest

from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.steps.tabular import one_hot as one_hot_mod
from modssc.preprocess.store import ArtifactStore


class _DummyEncoderNew:
    def __init__(self, *, handle_unknown: str, sparse_output: bool, dtype: object) -> None:
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.fit_data = None

    def fit(self, X):
        self.fit_data = X
        return self

    def transform(self, X):
        return np.ones((X.shape[0], 2), dtype=np.float32)


class _DummyEncoderLegacy:
    def __init__(self, *, handle_unknown: str, sparse: bool, dtype: object) -> None:
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self.dtype = dtype

    def fit(self, X):
        return self

    def transform(self, X):
        return np.zeros((X.shape[0], 1), dtype=np.float32)


def _patch_require(monkeypatch, encoder_cls):
    dummy_module = type("DummyPreprocess", (), {"OneHotEncoder": encoder_cls})
    monkeypatch.setattr(one_hot_mod, "require", lambda **kwargs: dummy_module)


def test_as_object_array_variants():
    arr = one_hot_mod._as_object_array([1, 2, 3])
    assert arr.shape == (3, 1)

    arr = one_hot_mod._as_object_array(np.array([[1, 2], [3, 4]], dtype=object))
    assert arr.shape == (2, 2)

    arr = one_hot_mod._as_object_array(i for i in [1, 2, 3])
    assert arr.shape == (3, 1)

    arr = one_hot_mod._as_object_array(np.zeros((2, 2, 2)))
    assert arr.shape == (2, 4)

    with pytest.raises(PreprocessValidationError, match="must not be a scalar"):
        one_hot_mod._as_object_array(np.array(1))


def test_is_missing_tokens():
    assert one_hot_mod._is_missing(None)
    assert one_hot_mod._is_missing(float("nan"))
    assert one_hot_mod._is_missing(b"NA")
    assert one_hot_mod._is_missing("  null ")
    assert not one_hot_mod._is_missing("value")


def test_numeric_and_categorical_helpers():
    values = np.array([1, "2", None, "NA"], dtype=object)
    assert one_hot_mod._is_numeric_column(values)

    numeric = one_hot_mod._to_numeric_column(values)
    assert np.isnan(numeric[2])
    assert np.isnan(numeric[3])
    assert numeric[0] == 1.0
    assert numeric[1] == 2.0

    values_bad = np.array([1, "nope"], dtype=object)
    assert not one_hot_mod._is_numeric_column(values_bad)

    cats = one_hot_mod._to_categorical_column(values)
    assert cats[2] == "__missing__"
    assert cats[1] == "2"


def test_tabular_one_hot_fit_transform_mixed(monkeypatch):
    _patch_require(monkeypatch, _DummyEncoderNew)

    store = ArtifactStore({"raw.X": np.array([[1, "a"], [2, "b"], [3, "a"]], dtype=object)})
    step = one_hot_mod.TabularOneHotStep()
    rng = np.random.default_rng(0)

    step.fit(store, fit_indices=np.array([0, 1, 2], dtype=np.int64), rng=rng)
    assert step._numeric_cols == [0]
    assert step._categorical_cols == [1]

    res = step.transform(store, rng=rng)
    features = res["features.X"]
    assert features.shape == (3, 3)
    assert features.dtype == np.float32


def test_tabular_one_hot_legacy_encoder(monkeypatch):
    _patch_require(monkeypatch, _DummyEncoderLegacy)

    store = ArtifactStore({"raw.X": np.array([["red"], ["blue"]], dtype=object)})
    step = one_hot_mod.TabularOneHotStep()
    rng = np.random.default_rng(0)

    step.fit(store, fit_indices=np.array([0, 1], dtype=np.int64), rng=rng)
    assert step._encoder is not None

    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 1)


def test_tabular_one_hot_numeric_only(monkeypatch):
    _patch_require(monkeypatch, _DummyEncoderNew)

    store = ArtifactStore({"raw.X": np.array([[1, 2], [3, 4]], dtype=object)})
    step = one_hot_mod.TabularOneHotStep()
    rng = np.random.default_rng(0)

    step.fit(store, fit_indices=np.array([0, 1], dtype=np.int64), rng=rng)
    assert step._categorical_cols == []
    assert step._encoder is None

    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 2)


def test_tabular_one_hot_fit_invalid_indices(monkeypatch):
    _patch_require(monkeypatch, _DummyEncoderNew)

    store = ArtifactStore({"raw.X": np.array([[1, "a"]], dtype=object)})
    step = one_hot_mod.TabularOneHotStep()
    rng = np.random.default_rng(0)

    with pytest.raises(PreprocessValidationError, match="fit_indices must be 1D"):
        step.fit(store, fit_indices=np.array([[0]], dtype=np.int64), rng=rng)


def test_tabular_one_hot_optional_dependency_error(monkeypatch):
    def _boom(**kwargs):
        raise OptionalDependencyError(extra="preprocess-sklearn", purpose="One-hot encoding")

    monkeypatch.setattr(one_hot_mod, "require", _boom)

    store = ArtifactStore({"raw.X": np.array([[1, "a"]], dtype=object)})
    step = one_hot_mod.TabularOneHotStep()
    rng = np.random.default_rng(0)

    with pytest.raises(OptionalDependencyError):
        step.fit(store, fit_indices=np.array([0], dtype=np.int64), rng=rng)


def test_tabular_one_hot_transform_errors():
    store = ArtifactStore({"raw.X": np.array([[1]], dtype=object)})
    step = one_hot_mod.TabularOneHotStep()
    rng = np.random.default_rng(0)

    with pytest.raises(PreprocessValidationError, match="called before fit"):
        step.transform(store, rng=rng)

    step._categorical_cols = [0]
    step._numeric_cols = []
    step._encoder = None
    with pytest.raises(PreprocessValidationError, match="missing fitted encoder"):
        step.transform(store, rng=rng)


def test_tabular_one_hot_transform_no_parts(monkeypatch):
    store = ArtifactStore({"raw.X": np.array([[1]], dtype=object)})
    step = one_hot_mod.TabularOneHotStep()
    step._numeric_cols = [0]
    step._categorical_cols = []
    rng = np.random.default_rng(0)

    def _as_object_array_and_clear(raw):
        step._numeric_cols = []
        step._categorical_cols = []
        return np.asarray(raw, dtype=object)

    monkeypatch.setattr(one_hot_mod, "_as_object_array", _as_object_array_and_clear)

    with pytest.raises(PreprocessValidationError, match="produced no features"):
        step.transform(store, rng=rng)
