from __future__ import annotations

import importlib

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

import modssc.inductive.methods.defixmatch as defixmatch
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.defixmatch import DeFixMatchMethod, DeFixMatchSpec
from modssc.inductive.types import DeviceSpec, InductiveDataset


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
    _assert_module_importable("modssc.inductive.methods.defixmatch")


class _FlattenNet(torch.nn.Module):
    def __init__(self, in_shape: tuple[int, int, int] = (3, 4, 4), n_classes: int = 2) -> None:
        super().__init__()
        in_dim = 1
        for dim in in_shape:
            in_dim *= int(dim)
        self.fc = torch.nn.Linear(in_dim, n_classes, bias=False)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        return self.fc(x.reshape(int(x.shape[0]), -1))


def _make_bundle(model: torch.nn.Module) -> TorchModelBundle:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TorchModelBundle(model=model, optimizer=optimizer)


def test_defixmatch_non_batch_shape_helpers() -> None:
    x = torch.randn((4, 3, 4, 4), dtype=torch.float32)
    assert defixmatch._non_batch_shape(x) == (3, 4, 4)
    assert defixmatch._non_batch_shape({"x": x}) == (3, 4, 4)

    with pytest.raises(InductiveValidationError, match="Expected torch.Tensor inputs"):
        defixmatch._non_batch_shape({"x": [1, 2, 3]})


def test_defixmatch_fit_accepts_image_strong_views() -> None:
    X_l = torch.randn((4, 3, 4, 4), dtype=torch.float32)
    y_l = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    X_u_w = torch.randn((4, 3, 4, 4), dtype=torch.float32)
    X_u_s = torch.randn((4, 3, 4, 4), dtype=torch.float32)
    X_l_s = torch.randn((4, 3, 4, 4), dtype=torch.float32)
    data = InductiveDataset(
        X_l=X_l,
        y_l=y_l,
        X_u_w=X_u_w,
        X_u_s=X_u_s,
        views={"X_l_s": X_l_s},
    )

    spec = DeFixMatchSpec(
        model_bundle=_make_bundle(_FlattenNet()),
        batch_size=2,
        max_epochs=1,
        lambda_u=0.0,
    )

    method = DeFixMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(X_l)
    assert proba.shape == (4, 2)


def test_defixmatch_fit_rejects_strong_view_rank_mismatch() -> None:
    X_l = torch.randn((4, 3, 4, 4), dtype=torch.float32)
    y_l = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    X_u_w = torch.randn((4, 3, 4, 4), dtype=torch.float32)
    X_u_s = torch.randn((4, 3, 4, 4), dtype=torch.float32)
    bad_X_l_s = torch.randn((4, 48), dtype=torch.float32)
    data = InductiveDataset(
        X_l=X_l,
        y_l=y_l,
        X_u_w=X_u_w,
        X_u_s=X_u_s,
        views={"X_l_s": bad_X_l_s},
    )

    spec = DeFixMatchSpec(
        model_bundle=_make_bundle(_FlattenNet()),
        batch_size=2,
        max_epochs=1,
        lambda_u=0.0,
    )

    with pytest.raises(InductiveValidationError, match="same rank as X_l"):
        DeFixMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)
