import math

import numpy as np
import pytest

from modssc.evaluation.metrics import (
    _pred_indices_for_classes,
    accuracy,
    balanced_accuracy,
    compute_metrics,
    evaluate,
    labels_1d,
    list_metrics,
    macro_f1,
    predict_labels,
    to_numpy,
)


class _TorchLike:
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr
        self.detached = False
        self.cpu_called = False
        self.numpy_called = False

    def detach(self) -> "_TorchLike":
        self.detached = True
        return self

    def cpu(self) -> "_TorchLike":
        self.cpu_called = True
        return self

    def numpy(self) -> np.ndarray:
        self.numpy_called = True
        return self.arr


def test_to_numpy_numpy_identity():
    arr = np.array([1, 2, 3])
    assert to_numpy(arr) is arr


def test_to_numpy_torch_like():
    arr = np.array([1, 2, 3])
    obj = _TorchLike(arr)
    out = to_numpy(obj)
    assert out is arr
    assert obj.detached is True
    assert obj.cpu_called is True
    assert obj.numpy_called is True


def test_to_numpy_list_fallback():
    out = to_numpy([1, 2, 3])
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1, 2, 3]


def test_labels_1d_one_hot():
    y = np.array([[0, 1], [1, 0], [0, 1]])
    out = labels_1d(y)
    assert out.tolist() == [1, 0, 1]


def test_labels_1d_flatten():
    y = np.array([2, 1, 0])
    out = labels_1d(y)
    assert out.shape == (3,)
    assert out.tolist() == [2, 1, 0]


def test_predict_labels_1d():
    scores = np.array([1.0, 0.0, 2.0])
    out = predict_labels(scores)
    assert out.tolist() == [1, 0, 2]
    assert np.issubdtype(out.dtype, np.integer)


def test_predict_labels_2d():
    scores = np.array([[0.1, 0.9], [0.8, 0.2]])
    out = predict_labels(scores)
    assert out.tolist() == [1, 0]


def test_accuracy_empty():
    assert math.isnan(accuracy(np.array([]), np.array([])))


def test_accuracy_simple():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 0])
    assert accuracy(y_true, y_pred) == 0.5


def test_balanced_accuracy_empty():
    assert math.isnan(balanced_accuracy(np.array([]), np.array([])))


def test_balanced_accuracy_multiclass():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert balanced_accuracy(y_true, y_pred) == 0.75


def test_balanced_accuracy_skips_nan_label():
    y_true = np.array([0.0, np.nan])
    y_pred = np.array([0.0, 0.0])
    assert balanced_accuracy(y_true, y_pred) == 1.0


def test_balanced_accuracy_all_nan():
    y_true = np.array([np.nan, np.nan])
    y_pred = np.array([0.0, 1.0])
    assert math.isnan(balanced_accuracy(y_true, y_pred))


def test_macro_f1_empty():
    assert math.isnan(macro_f1(np.array([]), np.array([])))


def test_macro_f1_normal():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert macro_f1(y_true, y_pred) == pytest.approx(0.7333333333333334)


def test_macro_f1_tp_zero_shortcut():
    y_true = np.array([0, 0])
    y_pred = np.array([1, 1])
    assert macro_f1(y_true, y_pred) == 0.0


def test_macro_f1_prec_rec_zero():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    assert macro_f1(y_true, y_pred) == 0.0


def test_list_metrics_sorted():
    assert list_metrics() == ["accuracy", "balanced_accuracy", "macro_f1"]


def test_compute_metrics_ok():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 0])
    res = compute_metrics(y_true, y_pred, ["accuracy", "macro_f1", "balanced_accuracy"])
    assert res["accuracy"] == pytest.approx(2 / 3)
    assert res["macro_f1"] == pytest.approx(2 / 3)
    assert res["balanced_accuracy"] == pytest.approx(0.75)


def test_compute_metrics_unknown():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    with pytest.raises(ValueError) as excinfo:
        compute_metrics(y_true, y_pred, ["bogus"])
    msg = str(excinfo.value)
    assert "Unknown metric: bogus" in msg
    assert "Available" in msg


def test_evaluate_one_hot_scores():
    y_true = np.array([[1, 0], [0, 1], [1, 0]])
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    res = evaluate(y_true, scores, ["accuracy", "balanced_accuracy", "macro_f1"])
    assert res["accuracy"] == 1.0
    assert res["balanced_accuracy"] == 1.0
    assert res["macro_f1"] == 1.0


def test_evaluate_labels_direct():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 0, 1])
    res = evaluate(y_true, y_pred, ["accuracy"])
    assert res["accuracy"] == pytest.approx(2 / 3)


def test_pred_indices_empty_classes():
    y_pred = np.array([1, 2])
    out = _pred_indices_for_classes(y_pred, np.array([], dtype=np.int64))
    assert np.all(out == -1)


def test_pred_indices_fallback_mapping(monkeypatch):
    import modssc.evaluation.metrics as metrics

    def boom(*args, **kwargs):
        raise TypeError("nope")

    monkeypatch.setattr(metrics.np, "searchsorted", boom)
    classes = np.array([1, 2], dtype=np.int64)
    y_pred = np.array([2, 3], dtype=np.int64)
    out = _pred_indices_for_classes(y_pred, classes)
    assert out.tolist() == [1, -1]


def test_balanced_accuracy_mask_exception():
    y_true = np.array(["a", "b"], dtype=object)
    y_pred = np.array(["a", "b"], dtype=object)
    assert balanced_accuracy(y_true, y_pred) == 1.0


def test_balanced_accuracy_mask_mismatch(monkeypatch):
    import modssc.evaluation.metrics as metrics

    def fake_isnan(arr):
        return np.array([True])

    monkeypatch.setattr(metrics.np, "isnan", fake_isnan)
    y_true = np.array([[0.0, 1.0]])
    y_pred = np.array([[0.0, 1.0]])
    assert balanced_accuracy(y_true, y_pred) == 1.0


def test_balanced_accuracy_empty_classes(monkeypatch):
    import modssc.evaluation.metrics as metrics

    def fake_unique(arr, return_inverse=False):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    monkeypatch.setattr(metrics.np, "unique", fake_unique)
    assert math.isnan(balanced_accuracy(np.array([0, 1]), np.array([0, 1])))


def test_macro_f1_empty_classes(monkeypatch):
    import modssc.evaluation.metrics as metrics

    def fake_unique(arr, return_inverse=False):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    monkeypatch.setattr(metrics.np, "unique", fake_unique)
    assert math.isnan(macro_f1(np.array([0, 1]), np.array([0, 1])))
