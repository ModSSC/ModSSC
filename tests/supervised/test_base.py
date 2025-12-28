import numpy as np
import pytest

from modssc.supervised.base import BaseSupervisedClassifier
from modssc.supervised.errors import NotSupportedError


class ConcreteClassifier(BaseSupervisedClassifier):
    pass


def test_base_not_implemented():
    clf = ConcreteClassifier()
    with pytest.raises(NotImplementedError):
        clf.fit(None, None)
    with pytest.raises(NotImplementedError):
        clf.predict(None)


def test_base_predict_proba_not_supported():
    clf = ConcreteClassifier()
    assert not clf.supports_proba
    with pytest.raises(NotSupportedError):
        clf.predict_proba(None)


def test_base_predict_scores_fallback():
    class MockClassifier(BaseSupervisedClassifier):
        def predict(self, X):
            return np.array(["a", "b"])

    clf = MockClassifier()
    clf.classes_ = np.array(["a", "b", "c"])

    scores = clf.predict_scores(None)
    expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(scores, expected)


def test_base_predict_scores_proba():
    class ProbaClassifier(BaseSupervisedClassifier):
        @property
        def supports_proba(self):
            return True

        def predict_proba(self, X):
            return np.array([[0.1, 0.9]])

    clf = ProbaClassifier()
    scores = clf.predict_scores(None)
    np.testing.assert_array_equal(scores, np.array([[0.1, 0.9]]))


def test_base_predict_scores_no_classes():
    class MockClassifier(BaseSupervisedClassifier):
        def predict(self, X):
            return np.array([0])

    clf = MockClassifier()
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf.predict_scores(None)


def test_base_decode_no_classes():
    clf = ConcreteClassifier()
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf._decode(np.array([0]))


def test_base_decode_success():
    clf = ConcreteClassifier()
    clf.classes_ = np.array(["a", "b"])
    res = clf._decode(np.array([1, 0]))
    np.testing.assert_array_equal(res, np.array(["b", "a"]))


def test_base_n_classes():
    clf = ConcreteClassifier()
    assert clf.n_classes_ == 0
    clf.classes_ = np.array([1, 2])
    assert clf.n_classes_ == 2


def test_base_classifier_decode_error():
    class MockClassifier(BaseSupervisedClassifier):
        def fit(self, X, y):
            pass

        def predict(self, X):
            pass

        def predict_proba(self, X):
            pass

    clf = MockClassifier()
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        clf._decode(np.array([0]))


def test_base_classifier_decode_success():
    class MockClassifier(BaseSupervisedClassifier):
        def fit(self, X, y):
            pass

        def predict(self, X):
            pass

        def predict_proba(self, X):
            pass

    clf = MockClassifier()
    clf.classes_ = np.array(["a", "b"])
    y_enc = np.array([0, 1, 0])
    y_dec = clf._decode(y_enc)
    np.testing.assert_array_equal(y_dec, np.array(["a", "b", "a"]))


def test_base_classifier_set_classes():
    class MockClassifier(BaseSupervisedClassifier):
        def fit(self, X, y):
            return self._set_classes_from_y(y)

        def predict(self, X):
            pass

        def predict_proba(self, X):
            pass

    clf = MockClassifier()
    y = ["a", "b", "a"]
    y_enc = clf.fit(None, y)
    np.testing.assert_array_equal(y_enc, np.array([0, 1, 0]))
