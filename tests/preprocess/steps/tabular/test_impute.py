from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.steps.tabular import impute
from modssc.preprocess.steps.tabular.impute import TabularImputeStep
from modssc.preprocess.store import ArtifactStore


class _FakeImputer:
    def __init__(self, strategy, fill_value=None, add_indicator=False):
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.fit_called = False

    def fit(self, X):
        self.fit_called = True
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0)


def test_tabular_impute_fit_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(impute, "require", lambda **_: SimpleNamespace(SimpleImputer=_FakeImputer))
    store = ArtifactStore({"raw.X": np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)})
    step = TabularImputeStep(strategy="mean")
    step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))
    out = step.transform(store, rng=np.random.default_rng(1))
    assert out["features.X"].shape == (2, 2)


def test_tabular_impute_transform_requires_fit() -> None:
    store = ArtifactStore({"raw.X": np.array([[1.0, 2.0]], dtype=np.float32)})
    step = TabularImputeStep()
    with pytest.raises(PreprocessValidationError, match="before fit"):
        step.transform(store, rng=np.random.default_rng(0))


def test_tabular_impute_optional_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_kwargs):
        raise OptionalDependencyError("missing")

    monkeypatch.setattr(impute, "require", _boom)
    store = ArtifactStore({"raw.X": np.array([[1.0, 2.0]], dtype=np.float32)})
    step = TabularImputeStep()
    with pytest.raises(OptionalDependencyError, match="missing"):
        step.fit(store, fit_indices=np.array([0]), rng=np.random.default_rng(0))
