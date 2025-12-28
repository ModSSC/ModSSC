import numpy as np

from modssc.preprocess.steps.labels.ensure_onehot import EnsureOneHotLabelsStep
from modssc.preprocess.store import ArtifactStore


def test_ensure_onehot_all_unlabeled():
    """Test EnsureOneHotLabelsStep when all labels are unlabeled."""
    step = EnsureOneHotLabelsStep(unlabeled_value=-1)
    store = ArtifactStore()

    y = np.array([-1, -1, -1], dtype=np.int32)
    store.set("raw.y", y)

    rng = np.random.default_rng(42)
    result = step.transform(store, rng=rng)

    onehot = result["labels.y_onehot"]
    is_labeled = result["labels.is_labeled"]

    assert not is_labeled.any()

    assert onehot.shape == (3, 0)
    assert onehot.size == 0


def test_ensure_onehot_mixed():
    """Test EnsureOneHotLabelsStep with mixed labeled and unlabeled data."""
    step = EnsureOneHotLabelsStep(unlabeled_value=-1)
    store = ArtifactStore()

    y = np.array([0, -1, 1, 0], dtype=np.int32)
    store.set("raw.y", y)

    rng = np.random.default_rng(42)
    result = step.transform(store, rng=rng)

    onehot = result["labels.y_onehot"]
    is_labeled = result["labels.is_labeled"]

    expected_labeled = np.array([True, False, True, True])
    np.testing.assert_array_equal(is_labeled, expected_labeled)

    expected_onehot = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype="float32")

    np.testing.assert_array_equal(onehot, expected_onehot)
