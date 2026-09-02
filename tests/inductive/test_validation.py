from __future__ import annotations

import importlib

import numpy as np
import pytest

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.types import InductiveDataset
from modssc.inductive.validation import validate_inductive_dataset


def _assert_module_importable(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing.startswith("modssc"):
            raise
        pytest.skip(f"Optional dependency missing while importing {module_name}: {missing}")
    except Exception as exc:
        if exc.__class__.__name__ == "OptionalDependencyError" or 'pip install "modssc[' in str(
            exc
        ):
            pytest.skip(f"Optional dependency missing while importing {module_name}: {exc}")
        raise


def test_module_importable() -> None:
    _assert_module_importable("modssc.inductive.validation")


def test_fourth_unlabeled_view_must_match_labeled_feature_dimension() -> None:
    data = InductiveDataset(
        X_l=np.zeros((2, 3), dtype=np.float32),
        y_l=np.array([0, 1]),
        X_u_s_1=np.zeros((4, 2), dtype=np.float32),
    )

    with pytest.raises(InductiveValidationError, match="X_u_s_1"):
        validate_inductive_dataset(data)


def test_fourth_unlabeled_view_accepts_matching_feature_dimension() -> None:
    data = InductiveDataset(
        X_l=np.zeros((2, 3), dtype=np.float32),
        y_l=np.array([0, 1]),
        X_u_s_1=np.zeros((4, 3), dtype=np.float32),
    )

    validate_inductive_dataset(data)
