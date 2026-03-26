from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.tri_training import TriTrainingMethod, TriTrainingSpec


def test_tri_training_validation_error_class_mismatch():
    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    model._backend = "numpy"

    clf1 = MagicMock()
    clf1.classes_ = np.array([0, 1])
    clf1.predict_proba.return_value = np.zeros((10, 2))

    clf2 = MagicMock()
    clf2.classes_ = None
    clf2.classes_t_ = None
    clf2.predict_proba.return_value = np.zeros((10, 3))

    model._clfs = [clf1, clf2]

    with (
        patch(
            "modssc.inductive.methods.tri_training.predict_scores",
            side_effect=lambda clf, X, backend: clf.predict_proba(X),
        ),
        pytest.raises(
            InductiveValidationError, match="TriTraining classifiers disagree on class counts"
        ),
    ):
        model.predict_proba(np.zeros((10, 5)))
