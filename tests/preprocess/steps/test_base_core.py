import numpy as np
import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.base import FittableStep, TransformStep, fit_subset, get_X, set_X
from modssc.preprocess.steps.core.copy_raw import CopyRawStep
from modssc.preprocess.steps.core.ensure_2d import Ensure2DStep
from modssc.preprocess.steps.core.pca import PcaStep
from modssc.preprocess.steps.core.random_projection import RandomProjectionStep
from modssc.preprocess.store import ArtifactStore


def test_get_X_prefer_features():
    store = ArtifactStore()
    store.set("raw.X", "raw")
    store.set("features.X", "features")
    assert get_X(store) == "features"


def test_get_X_fallback_raw():
    store = ArtifactStore()
    store.set("raw.X", "raw")
    assert get_X(store) == "raw"


def test_set_X_default():
    store = ArtifactStore()
    res = set_X(store, "val")
    assert res == {"features.X": "val"}


def test_set_X_custom_key():
    store = ArtifactStore()
    res = set_X(store, "val", key="custom.X")
    assert res == {"custom.X": "val"}


def test_fit_subset_invalid_indices():
    X = np.array([1, 2, 3])

    indices = np.array([[0]])
    with pytest.raises(PreprocessValidationError, match="fit_indices must be 1D"):
        fit_subset(X, fit_indices=indices)


def test_fit_subset_scalar_X():
    X = np.array(1)
    indices = np.array([0])
    with pytest.raises(PreprocessValidationError, match="Cannot subset scalar X"):
        fit_subset(X, fit_indices=indices)


def test_transform_step_abstract():
    step = TransformStep()
    store = ArtifactStore()
    rng = np.random.default_rng()
    with pytest.raises(NotImplementedError):
        step.transform(store, rng=rng)


def test_fittable_step_abstract():
    step = FittableStep()
    store = ArtifactStore()
    rng = np.random.default_rng()
    with pytest.raises(NotImplementedError):
        step.fit(store, fit_indices=np.array([]), rng=rng)


def test_ensure_2d_scalar_error():
    step = Ensure2DStep()
    store = ArtifactStore()
    store.set("raw.X", np.array(1))
    rng = np.random.default_rng()
    with pytest.raises(PreprocessValidationError, match="X must not be a scalar"):
        step.transform(store, rng=rng)


def test_ensure_2d_1d_reshape():
    step = Ensure2DStep()
    store = ArtifactStore()
    store.set("raw.X", np.array([1, 2, 3]))
    rng = np.random.default_rng()
    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (3, 1)


def test_ensure_2d_3d_reshape():
    step = Ensure2DStep()
    store = ArtifactStore()

    store.set("raw.X", np.zeros((2, 2, 2)))
    rng = np.random.default_rng()
    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 4)


def test_ensure_2d_pass_through():
    """Test Ensure2DStep with already 2D input."""
    step = Ensure2DStep()
    store = ArtifactStore()
    store.set("raw.X", np.zeros((3, 2)))
    rng = np.random.default_rng()
    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (3, 2)


def test_ensure_2d_object_dtype_error():
    step = Ensure2DStep()
    store = ArtifactStore()
    store.set("raw.X", np.array(["a", "b"], dtype=object))
    rng = np.random.default_rng()
    with pytest.raises(PreprocessValidationError, match="expects numeric arrays"):
        step.transform(store, rng=rng)


def test_pca_fit_invalid_dim():
    step = PcaStep()
    store = ArtifactStore()

    store.set("features.X", np.array([1, 2, 3]))
    rng = np.random.default_rng()
    with pytest.raises(PreprocessValidationError, match="PCA expects a 2D features matrix"):
        step.fit(store, fit_indices=np.array([0, 1]), rng=rng)


def test_pca_fit_empty_selection():
    step = PcaStep()
    store = ArtifactStore()
    store.set("features.X", np.zeros((5, 2)))
    rng = np.random.default_rng()
    with pytest.raises(PreprocessValidationError, match="Cannot fit PCA on empty selection"):
        step.fit(store, fit_indices=np.array([], dtype=int), rng=rng)


def test_pca_transform_before_fit():
    step = PcaStep()
    store = ArtifactStore()
    rng = np.random.default_rng()
    with pytest.raises(PreprocessValidationError, match="called before fit"):
        step.transform(store, rng=rng)


def test_pca_fit_transform_no_center():
    step = PcaStep(n_components=1, center=False)
    store = ArtifactStore()
    X = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    store.set("features.X", X)
    rng = np.random.default_rng()

    step.fit(store, fit_indices=np.array([0, 1]), rng=rng)

    np.testing.assert_array_equal(step.mean_, np.zeros(2))

    res = step.transform(store, rng=rng)
    Z = res["features.X"]
    assert Z.shape == (2, 1)


def test_pca_fit_transform_with_nan():
    step = PcaStep(n_components=1)
    store = ArtifactStore()
    X = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)
    store.set("features.X", X)
    rng = np.random.default_rng()

    step.fit(store, fit_indices=np.array([0, 1]), rng=rng)

    store.set("features.X", X)
    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 1)


def test_copy_raw_step():
    step = CopyRawStep()
    store = ArtifactStore()
    store.set("raw.X", np.array([1.0, 2.0], dtype=np.float32))
    rng = np.random.default_rng()
    res = step.transform(store, rng=rng)
    assert res["features.X"] is store.require("raw.X")


def test_random_projection_fit_transform():
    step = RandomProjectionStep(n_components=2, normalize=True)
    store = ArtifactStore()

    X = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32
    )
    store.set("features.X", X)

    rng = np.random.default_rng(42)

    step.fit(store, fit_indices=np.array([0, 1, 2]), rng=rng)

    assert step.W_ is not None
    assert step.W_.shape == (4, 2)

    result = step.transform(store, rng=rng)
    Z = result["features.X"]

    assert Z.shape == (3, 2)

    expected_Z = X @ step.W_
    np.testing.assert_array_almost_equal(Z, expected_Z)


def test_random_projection_no_normalize():
    step = RandomProjectionStep(n_components=2, normalize=False)
    store = ArtifactStore()
    X = np.zeros((2, 4), dtype=np.float32)
    store.set("features.X", X)
    rng = np.random.default_rng(42)

    step.fit(store, fit_indices=np.array([0, 1]), rng=rng)

    assert step.W_.shape == (4, 2)


def test_random_projection_invalid_input_dim():
    step = RandomProjectionStep(n_components=2)
    store = ArtifactStore()

    store.set("features.X", np.array([1, 2, 3]))
    rng = np.random.default_rng(42)

    with pytest.raises(PreprocessValidationError, match="expects 2D features.X"):
        step.fit(store, fit_indices=np.array([0]), rng=rng)


def test_random_projection_invalid_n_components():
    step = RandomProjectionStep(n_components=0)
    store = ArtifactStore()
    store.set("features.X", np.zeros((2, 2)))
    rng = np.random.default_rng(42)

    with pytest.raises(PreprocessValidationError, match="n_components must be > 0"):
        step.fit(store, fit_indices=np.array([0]), rng=rng)


def test_random_projection_transform_before_fit():
    step = RandomProjectionStep(n_components=2)
    store = ArtifactStore()
    rng = np.random.default_rng(42)

    with pytest.raises(PreprocessValidationError, match="called before fit"):
        step.transform(store, rng=rng)
