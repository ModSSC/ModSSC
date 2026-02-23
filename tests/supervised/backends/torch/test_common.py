from __future__ import annotations

import numpy as np
import pytest

from modssc.supervised.backends.torch.common import (
    TorchArgmaxPredictMixin,
    TorchNumpyProbaClassifierBase,
    TorchNumpyProbaPredictMixin,
    _RequiresScoresMixin,
    predict_from_scores,
)


class _Scores:
    def __init__(self, idx: np.ndarray):
        self._idx = idx

    def argmax(self, *, dim: int):
        assert dim == 1
        return self._idx


def test_predict_from_scores_raises_when_not_fitted():
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        predict_from_scores(classes_t=None, scores=_Scores(np.asarray([0, 1], dtype=np.int64)))


def test_requires_scores_mixin_default_raises():
    with pytest.raises(NotImplementedError):
        _RequiresScoresMixin()._scores(np.zeros((1, 2), dtype=np.float32))


def test_torch_argmax_predict_mixin_predicts():
    class _Cls(TorchArgmaxPredictMixin):
        def __init__(self):
            self._classes_t = np.asarray([10, 20, 30], dtype=np.int64)

        def _scores(self, X):  # noqa: ARG002
            return _Scores(np.asarray([2, 0], dtype=np.int64))

    pred = _Cls().predict(np.zeros((2, 3), dtype=np.float32))
    assert np.array_equal(pred, np.asarray([30, 10], dtype=np.int64))


def test_torch_numpy_proba_predict_mixin_default_predict_proba_raises():
    with pytest.raises(NotImplementedError, match="Subclasses must implement predict_proba"):
        TorchNumpyProbaPredictMixin().predict_proba(np.zeros((1, 2), dtype=np.float32))


def test_torch_numpy_proba_classifier_base_default_predict_proba_raises():
    with pytest.raises(NotImplementedError, match="Subclasses must implement predict_proba"):
        TorchNumpyProbaClassifierBase().predict_proba(np.zeros((1, 2), dtype=np.float32))


def test_torch_numpy_proba_predict_mixin_argmax_axis_fallback():
    class _AxisOnlyProbs:
        def argmax(self, dim=None, axis=None):
            if dim is not None:
                raise RuntimeError("dim is not supported")
            assert axis == 1
            return [1, 0]

    class _Cls(TorchNumpyProbaPredictMixin):
        def predict_proba(self, X):  # noqa: ARG002
            return _AxisOnlyProbs()

    pred = _Cls().predict(np.zeros((2, 3), dtype=np.float32))
    assert np.array_equal(pred, np.asarray([1, 0], dtype=np.int64))


def test_torch_numpy_proba_predict_mixin_array_fallback():
    class _ArrayOnlyProbs:
        def __array__(self, dtype=None):
            arr = np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
            return arr.astype(dtype) if dtype is not None else arr

    class _Cls(TorchNumpyProbaPredictMixin):
        def predict_proba(self, X):  # noqa: ARG002
            return _ArrayOnlyProbs()

    pred = _Cls().predict(np.zeros((2, 3), dtype=np.float32))
    assert np.array_equal(pred, np.asarray([1, 0], dtype=np.int64))
