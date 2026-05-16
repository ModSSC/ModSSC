from __future__ import annotations

import numpy as np
import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.core.row_normalize import RowNormalizeStep
from modssc.preprocess.store import ArtifactStore


def _store_with_x(x):
    store = ArtifactStore()
    store.set("features.X", x)
    return store


def test_row_normalize_dense_l1_and_zero_row() -> None:
    step = RowNormalizeStep()
    store = _store_with_x(np.array([[1.0, 2.0, -1.0], [0.0, 0.0, 0.0]], dtype=np.float64))

    out = step.transform(store, rng=np.random.default_rng(0))["features.X"]

    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], np.array([0.25, 0.5, -0.25], dtype=np.float32))
    np.testing.assert_array_equal(out[1], np.zeros(3, dtype=np.float32))


def test_row_normalize_sparse_l1_and_zero_row() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")
    mat = scipy_sparse.csr_matrix([[0.0, 2.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    step = RowNormalizeStep()

    out = step.transform(_store_with_x(mat), rng=np.random.default_rng(0))["features.X"]

    assert scipy_sparse.isspmatrix_csr(out)
    np.testing.assert_allclose(out.toarray(), np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 0.0]]))


def test_row_normalize_invalid_norm_and_eps() -> None:
    with pytest.raises(PreprocessValidationError, match="norm='l1'"):
        RowNormalizeStep(norm="l2").transform(
            _store_with_x(np.ones((2, 2))), rng=np.random.default_rng(0)
        )

    with pytest.raises(PreprocessValidationError, match="eps must be"):
        RowNormalizeStep(eps=0.0).transform(
            _store_with_x(np.ones((2, 2))), rng=np.random.default_rng(0)
        )


def test_row_normalize_dense_invalid_shape_and_dtype() -> None:
    with pytest.raises(PreprocessValidationError, match="expects 2D"):
        RowNormalizeStep().transform(_store_with_x(np.ones(2)), rng=np.random.default_rng(0))

    with pytest.raises(PreprocessValidationError, match="numeric"):
        RowNormalizeStep().transform(
            _store_with_x(np.array([["x", "y"]], dtype=object)), rng=np.random.default_rng(0)
        )
