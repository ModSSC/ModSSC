from __future__ import annotations

import pytest
import torch

import modssc.inductive.methods.setred as setred
import modssc.inductive.methods.tsvm as tsvm
import modssc.inductive.methods.vat as vat
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.mixmatch import _mixup


def test_mixup_dict_and_errors():
    gen = torch.Generator().manual_seed(0)
    X = {"x": torch.arange(4, dtype=torch.float32).view(2, 2), "meta": "keep"}
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    out_x, out_y = _mixup(X, y, alpha=0.0, generator=gen)
    assert isinstance(out_x, dict)
    assert out_x["meta"] == "keep"
    assert torch.allclose(out_x["x"], X["x"])
    assert torch.allclose(out_y, y)

    with pytest.raises(InductiveValidationError, match="torch.Tensor labels"):
        _mixup(torch.zeros((2, 2)), [0, 1], alpha=0.5, generator=gen)

    with pytest.raises(InductiveValidationError, match="torch.Tensor inputs"):
        _mixup({"x": [[1.0, 2.0]]}, torch.zeros((1, 2)), alpha=0.5, generator=gen)


def test_get_torch_x_helpers():
    x = torch.zeros((2, 2))
    assert setred._get_torch_x({"x": x}) is x
    assert tsvm._get_torch_x({"x": x}) is x


def test_vat_graph_helpers():
    x = torch.zeros((2, 2))
    d = {"x": x, "meta": "keep"}
    assert vat._get_x_tensor(d) is x

    with pytest.raises(InductiveValidationError, match="must include key 'x'"):
        vat._get_x_tensor({"meta": "nope"})

    delta = torch.ones_like(x)
    out = vat._add_to_x(d, delta)
    assert isinstance(out, dict)
    assert torch.allclose(out["x"], x + delta)

    out2 = vat._add_to_x(x, delta)
    assert torch.allclose(out2, x + delta)
