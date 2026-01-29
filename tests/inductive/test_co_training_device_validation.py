from __future__ import annotations

import pytest
import torch

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.co_training import CoTrainingMethod, CoTrainingSpec
from modssc.inductive.types import DeviceSpec, InductiveDataset


def _make_view(device: str):
    X_l = torch.zeros((2, 2), device=device)
    X_u = torch.zeros((2, 2), device=device)
    return {"X_l": X_l, "X_u": X_u}


def test_co_training_view_device_mismatch():
    X_l = torch.zeros((2, 2))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    views = {"view_a": _make_view("cpu"), "view_b": _make_view("meta")}
    data = InductiveDataset(X_l=X_l, y_l=y_l, X_u=None, views=views)
    spec = CoTrainingSpec(classifier_backend="torch")
    method = CoTrainingMethod(spec)
    with pytest.raises(InductiveValidationError, match="views must be on the same device"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_co_training_y_l_to_failure(monkeypatch: pytest.MonkeyPatch):
    X_l = torch.zeros((2, 2), device="meta")
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    views = {"view_a": _make_view("meta"), "view_b": _make_view("meta")}
    data = InductiveDataset(X_l=X_l, y_l=y_l, X_u=None, views=views)
    spec = CoTrainingSpec(classifier_backend="torch")
    method = CoTrainingMethod(spec)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(torch.Tensor, "to", _boom)
    with pytest.raises(InductiveValidationError, match="y_l must be on the same device"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_co_training_y_l_device_still_mismatch(monkeypatch: pytest.MonkeyPatch):
    X_l = torch.zeros((2, 2), device="meta")
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    views = {"view_a": _make_view("meta"), "view_b": _make_view("meta")}
    data = InductiveDataset(X_l=X_l, y_l=y_l, X_u=None, views=views)
    spec = CoTrainingSpec(classifier_backend="torch")
    method = CoTrainingMethod(spec)

    def _noop(self, *args, **kwargs):
        return self

    monkeypatch.setattr(torch.Tensor, "to", _noop)
    with pytest.raises(InductiveValidationError, match="y_l must be on the same device"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
