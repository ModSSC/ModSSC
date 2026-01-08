import numpy as np
import pytest

from modssc.transductive.operators.clamp import labels_to_onehot


def test_labels_to_onehot_invalid_n_classes():
    with pytest.raises(ValueError, match="n_classes must be positive"):
        labels_to_onehot([0, 1], n_classes=0)


def test_labels_to_onehot_float_integral():
    y = np.array([0.0, 1.0], dtype=np.float32)
    out = labels_to_onehot(y, n_classes=2)
    assert out.shape == (2, 2)
    assert np.allclose(out, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


@pytest.mark.parametrize(
    "y, match",
    [
        (np.array([0.0, np.nan]), "finite integer class ids"),
        (np.array([0.0, 1.5]), "integer class ids"),
    ],
)
def test_labels_to_onehot_float_errors(y, match):
    with pytest.raises(ValueError, match=match):
        labels_to_onehot(y, n_classes=2)


def test_labels_to_onehot_out_of_bounds():
    with pytest.raises(ValueError, match=r"outside \[0, n_classes\)"):
        labels_to_onehot(np.array([0, 2]), n_classes=2)
