from __future__ import annotations

import pytest
import torch

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.supervised import SupervisedMethod, SupervisedSpec
from modssc.inductive.types import DeviceSpec

from ..conftest import make_numpy_dataset, make_torch_dataset


def test_supervised_method_numpy_fit_predict() -> None:
    data = make_numpy_dataset()
    method = SupervisedMethod(SupervisedSpec())
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    pred = method.predict(data.X_l)
    assert proba.shape[0] == data.X_l.shape[0]
    assert pred.shape[0] == data.X_l.shape[0]


def test_supervised_method_torch_fit_predict() -> None:
    data = make_torch_dataset()
    method = SupervisedMethod(SupervisedSpec(classifier_backend="torch"))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    pred = method.predict(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    assert int(pred.shape[0]) == int(data.X_l.shape[0])


def test_supervised_method_predict_errors() -> None:
    data = make_numpy_dataset()
    method = SupervisedMethod(SupervisedSpec())
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict(data.X_l)

    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    method._backend = ""
    with pytest.raises(InductiveValidationError, match="backend mismatch"):
        method.predict(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
