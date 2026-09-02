from __future__ import annotations

import importlib

import pytest

from modssc.inductive.adapters.torch import to_torch_dataset
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.types import InductiveDataset


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
    _assert_module_importable("modssc.inductive.adapters.torch")


def test_fourth_unlabeled_torch_view_must_match_labeled_feature_dimension() -> None:
    torch = pytest.importorskip("torch")
    data = InductiveDataset(
        X_l=torch.zeros((2, 3)),
        y_l=torch.tensor([0, 1]),
        X_u_s_1=torch.zeros((4, 2)),
    )

    with pytest.raises(InductiveValidationError, match="X_u_s_1"):
        to_torch_dataset(data)
