from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.types import InductiveDataset


@dataclass
class DummyDataset:
    X_l: Any
    y_l: Any
    X_u: Any | None = None
    X_u_w: Any | None = None
    X_u_s: Any | None = None
    views: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


class SimpleNet(torch.nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.feat = torch.nn.Linear(in_dim, in_dim, bias=False)
        self.bn = torch.nn.BatchNorm1d(in_dim)
        self.fc = torch.nn.Linear(in_dim, n_classes, bias=False)

    def forward(self, x, only_fc: bool = False):
        if only_fc:
            return self.fc(x)
        feat = self.bn(self.feat(x))
        logits = self.fc(feat)
        return {"logits": logits, "feat": feat}


class TupleNet(torch.nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, n_classes, bias=False)

    def forward(self, x):
        logits = self.fc(x)
        return (logits,)


def make_numpy_dataset(n_l: int = 4, n_u: int = 4) -> InductiveDataset:
    X_l = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y_l = np.array([0, 0, 1, 1], dtype=np.int64)
    if n_l != 4:
        X_l = X_l[:n_l]
        y_l = y_l[:n_l]
    X_u = np.array([[0.1, 0.2], [0.9, 0.8], [0.2, 0.1], [0.8, 0.9]], dtype=np.float32)
    if n_u != 4:
        X_u = X_u[:n_u]
    return InductiveDataset(X_l=X_l, y_l=y_l, X_u=X_u)


def make_torch_dataset(n_l: int = 4, n_u: int = 4, device: str = "cpu") -> InductiveDataset:
    X_l = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
    y_l = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int64)
    if n_l != 4:
        X_l = X_l[:n_l]
        y_l = y_l[:n_l]
    X_u = torch.tensor([[0.1, 0.2], [0.9, 0.8], [0.2, 0.1], [0.8, 0.9]], device=device)
    if n_u != 4:
        X_u = X_u[:n_u]
    return InductiveDataset(X_l=X_l, y_l=y_l, X_u=X_u)


def make_torch_ssl_dataset(n_l: int = 4, n_u: int = 4, device: str = "cpu") -> InductiveDataset:
    base = make_torch_dataset(n_l=n_l, n_u=n_u, device=device)
    noise = torch.zeros_like(base.X_u)
    X_u_w = base.X_u + noise
    X_u_s = base.X_u + 0.01
    return InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u=base.X_u,
        X_u_w=X_u_w,
        X_u_s=X_u_s,
    )


def make_model_bundle(n_classes: int = 2) -> TorchModelBundle:
    model = SimpleNet(n_classes=n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema_model = copy.deepcopy(model)
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)
