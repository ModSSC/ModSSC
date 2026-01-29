from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from modssc.inductive.adapters import to_numpy_dataset, to_torch_dataset
from modssc.inductive.adapters.numpy import _require_numpy_views
from modssc.inductive.adapters.numpy import _suggest_step as _numpy_suggest_step
from modssc.inductive.adapters.torch import (
    _check_2d,
    _check_feature_dim,
    _check_same_device,
    _require_tensor,
    _require_views,
)
from modssc.inductive.adapters.torch import _suggest_step as _torch_suggest_step
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.types import DeviceSpec, InductiveDataset
from modssc.inductive.validation import (
    _as_numpy,
    _require_2d,
    _require_y,
    validate_inductive_dataset,
)

from .conftest import DummyDataset, make_numpy_dataset, make_torch_dataset


class _DetachArray:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _CpuArray:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _NumpyOnly:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def numpy(self):
        return self._arr


def test_as_numpy_variants():
    arr = np.ones((2, 2), dtype=np.float32)
    assert np.array_equal(_as_numpy(_DetachArray(arr)), arr)
    assert np.array_equal(_as_numpy(_CpuArray(arr)), arr)
    assert np.array_equal(_as_numpy(_NumpyOnly(arr)), arr)
    assert np.array_equal(_as_numpy([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))


def test_require_2d_and_y_helpers():
    _require_2d([[1, 2]], name="X_l")
    with pytest.raises(InductiveValidationError):
        _require_2d([1, 2, 3], name="X_l")
    _require_y([1, 2], n=2)
    with pytest.raises(InductiveValidationError):
        _require_y([[1, 2]], n=2)
    with pytest.raises(InductiveValidationError):
        _require_y(np.zeros((2, 2, 2)), n=2)
    with pytest.raises(InductiveValidationError):
        _require_y([1, 2, 3], n=2)


def test_validate_inductive_dataset_success():
    data = make_numpy_dataset()
    validate_inductive_dataset(data)
    data_ssl = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u.copy(),
        X_u_s=data.X_u.copy(),
    )
    validate_inductive_dataset(data_ssl)


def test_validate_inductive_dataset_errors():
    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(None)

    data = make_numpy_dataset()
    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(replace(data, X_l=np.array([1, 2, 3])))
    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(replace(data, y_l=np.array([[1, 2]])))

    bad_u = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(replace(data, X_u=bad_u))

    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(replace(data, X_u_w=np.zeros((2, 2)), X_u_s=np.zeros((3, 2))))

    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(replace(data, X_u_w=np.zeros((2, 3))))
    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(replace(data, X_u_s=np.zeros((2, 3))))

    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(replace(data, views=[("v1", data.X_l)]))
    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(replace(data, views={1: data.X_l}))

    with pytest.raises(InductiveValidationError):
        validate_inductive_dataset(replace(data, meta=[("k", 1)]))


def test_to_numpy_dataset_requires_numpy():
    torch_data = make_torch_dataset()
    with pytest.raises(InductiveValidationError):
        to_numpy_dataset(torch_data)


def test_to_numpy_dataset_views_and_meta():
    data = make_numpy_dataset()
    views = {"v1": data.X_l.copy()}
    assert _numpy_suggest_step("y_l") == "labels.to_numpy"
    assert _numpy_suggest_step("X_l") == "core.to_numpy"
    out = to_numpy_dataset(
        DummyDataset(
            X_l=data.X_l,
            y_l=data.y_l,
            X_u=data.X_u,
            views=views,
            meta={"a": 1},
        )
    )
    assert out.views and "v1" in out.views

    with pytest.raises(InductiveValidationError):
        to_numpy_dataset(
            DummyDataset(
                X_l=data.X_l,
                y_l=data.y_l,
                X_u=data.X_u,
                views={"v1": torch.tensor([[1.0, 2.0]])},
            )
        )
    with pytest.raises(InductiveValidationError):
        _require_numpy_views([("v1", data.X_l)])
    with pytest.raises(InductiveValidationError):
        _require_numpy_views({1: data.X_l})


def test_to_torch_dataset_none():
    with pytest.raises(InductiveValidationError, match="data must not be None"):
        to_torch_dataset(None)


def test_to_torch_dataset_success_auto_device():
    data = make_torch_dataset()
    out = to_torch_dataset(data, device=DeviceSpec(device="auto"), require_same_device=False)
    assert out.X_l.shape[0] == 4
    views = {"v1": data.X_l.clone()}
    out2 = to_torch_dataset(
        DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=data.X_u, views=views),
        device=DeviceSpec(device="auto"),
        require_same_device=False,
    )
    assert out2.views and "v1" in out2.views


def test_to_torch_dataset_device_mismatch():
    X_l = torch.zeros((2, 2))
    y_l = torch.zeros((2,), dtype=torch.int64)
    X_u = torch.zeros((2, 2), device="meta")
    data = InductiveDataset(X_l=X_l, y_l=y_l, X_u=X_u)
    with pytest.raises(InductiveValidationError):
        to_torch_dataset(data, device=DeviceSpec(device="cpu"), require_same_device=True)
    with pytest.raises(InductiveValidationError):
        to_torch_dataset(data, device=DeviceSpec(device="cpu"), require_same_device=False)


def test_to_torch_dataset_views_and_shapes_errors():
    data = make_torch_dataset()
    with pytest.raises(InductiveValidationError):
        to_torch_dataset(
            DummyDataset(
                X_l=data.X_l,
                y_l=data.y_l,
                X_u=torch.zeros((2, 3)),
                X_u_w=data.X_u_w,
                X_u_s=data.X_u_s,
            )
        )

    with pytest.raises(InductiveValidationError):
        to_torch_dataset(
            DummyDataset(
                X_l=data.X_l,
                y_l=data.y_l,
                X_u=data.X_u,
                X_u_w=torch.zeros((2, 2)),
                X_u_s=torch.zeros((3, 2)),
            )
        )

    with pytest.raises(InductiveValidationError):
        to_torch_dataset(
            DummyDataset(
                X_l=data.X_l,
                y_l=data.y_l,
                X_u=data.X_u,
                views={1: data.X_l},
            )
        )

    with pytest.raises(InductiveValidationError):
        to_torch_dataset(
            DummyDataset(
                X_l=data.X_l,
                y_l=data.y_l,
                X_u=data.X_u,
                views={"v1": np.zeros((2, 2))},
            )
        )


def test_to_torch_dataset_tensor_requirements():
    data = make_numpy_dataset()
    with pytest.raises(InductiveValidationError):
        to_torch_dataset(data)

    assert _torch_suggest_step("y_l") == "labels.to_torch"
    assert _torch_suggest_step("X_l") == "core.to_torch"
    with pytest.raises(InductiveValidationError):
        to_torch_dataset(
            DummyDataset(X_l=torch.zeros((2,)), y_l=torch.zeros((2,), dtype=torch.int64))
        )
    with pytest.raises(InductiveValidationError):
        to_torch_dataset(
            DummyDataset(
                X_l=torch.zeros((2, 2)),
                y_l=torch.zeros((2, 2, 2), dtype=torch.int64),
            )
        )
    with pytest.raises(InductiveValidationError):
        to_torch_dataset(
            DummyDataset(
                X_l=torch.zeros((2, 2)),
                y_l=torch.zeros((3,), dtype=torch.int64),
            )
        )

    _check_same_device([torch.zeros((1, 2)), "not-a-tensor"])
    with pytest.raises(InductiveValidationError):
        _require_views([("v1", torch.zeros((2, 2)))])


def test_to_torch_dataset_dict_inputs():
    X_l = {"x": torch.zeros((2, 3)), "meta": "keep"}
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    X_u = {"x": torch.ones((2, 3))}
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u=X_u)
    out = to_torch_dataset(data, device=DeviceSpec(device="auto"), require_same_device=True)
    assert out.X_l["x"].shape == (2, 3)
    assert out.X_u is not None


def test_to_torch_dataset_meta_idx_conversion():
    X_l = {"x": torch.zeros((2, 3))}
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    meta = {
        "idx_u": torch.tensor([0, 1, 2], dtype=torch.int32),
        "unlabeled_idx": [1, 2],
    }
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u=None, meta=meta)
    out = to_torch_dataset(data, device=DeviceSpec(device="auto"), require_same_device=False)
    assert isinstance(out.meta["idx_u"], torch.Tensor)
    assert out.meta["idx_u"].dtype == torch.int64
    assert out.meta["idx_u"].device == X_l["x"].device
    assert isinstance(out.meta["unlabeled_idx"], torch.Tensor)
    assert out.meta["unlabeled_idx"].dtype == torch.int64
    assert out.meta["unlabeled_idx"].device == X_l["x"].device


def test_to_torch_dataset_meta_idx_no_device():
    class _TensorDict(dict):
        device = None

        @property
        def shape(self):
            return (2, 3)

    X_l = _TensorDict(t=torch.zeros((2, 3)))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    meta = {"idx_u": [0, 1]}
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u=None, meta=meta)
    out = to_torch_dataset(data, device=DeviceSpec(device="auto"), require_same_device=False)
    assert out.meta["idx_u"] == [0, 1]


def test_require_tensor_dict_without_tensor():
    with pytest.raises(InductiveValidationError, match="core.to_torch"):
        _require_tensor({"meta": "no-tensor"}, name="X_l")


def test_check_2d_and_feature_dim_dict_without_x():
    _check_2d({"foo": torch.zeros((2, 2))}, name="X_l")
    _check_feature_dim({"foo": torch.zeros((2, 2))}, n_features=2, name="X_u")
