from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.steps.tabular import standard_scaler
from modssc.preprocess.steps.tabular.standard_scaler import TabularStandardScalerStep
from modssc.preprocess.store import ArtifactStore


class _FakeScaler:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        arr = np.asarray(X, dtype=np.float32)
        self.mean_ = arr.mean(axis=0)
        self.scale_ = arr.std(axis=0) + 1e-6
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float32)
        return (arr - self.mean_) / self.scale_


class _FakeIloc:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def __getitem__(self, idx):
        return _FakeIlocResult(self._arr[idx])


class _FakeIlocResult:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def to_numpy(self):
        return self._arr


class _FakeDF:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr
        self.iloc = _FakeIloc(arr)


def test_tabular_standard_scaler_fit_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        standard_scaler,
        "require",
        lambda **_: SimpleNamespace(StandardScaler=_FakeScaler),
    )
    store = ArtifactStore({"raw.X": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})
    step = TabularStandardScalerStep(with_mean=True, with_std=True)
    step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))
    out = step.transform(store, rng=np.random.default_rng(1))
    assert out["features.X"].dtype == np.float32


def test_tabular_standard_scaler_dataframe_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        standard_scaler,
        "require",
        lambda **_: SimpleNamespace(StandardScaler=_FakeScaler),
    )
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    store = ArtifactStore({"raw.X": _FakeDF(arr)})
    step = TabularStandardScalerStep(with_mean=True, with_std=True)
    step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))
    # Switch to array input for transform; fit covered the iloc branch.
    store.set("raw.X", arr)
    out = step.transform(store, rng=np.random.default_rng(1))
    assert out["features.X"].shape == (2, 2)


def test_tabular_standard_scaler_transform_requires_fit() -> None:
    store = ArtifactStore({"raw.X": np.array([[1.0, 2.0]], dtype=np.float32)})
    step = TabularStandardScalerStep()
    with pytest.raises(PreprocessValidationError, match="before fit"):
        step.transform(store, rng=np.random.default_rng(0))


def test_tabular_standard_scaler_optional_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_kwargs):
        raise OptionalDependencyError("missing")

    monkeypatch.setattr(standard_scaler, "require", _boom)
    store = ArtifactStore({"raw.X": np.array([[1.0, 2.0]], dtype=np.float32)})
    step = TabularStandardScalerStep()
    with pytest.raises(OptionalDependencyError, match="missing"):
        step.fit(store, fit_indices=np.array([0]), rng=np.random.default_rng(0))
