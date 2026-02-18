from __future__ import annotations

import torch

from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.methods.adsh import ADSHMethod, ADSHSpec


class _DictModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        logits = self.fc(x)
        return {"logits": logits}


def _make_method() -> ADSHMethod:
    model = _DictModel()
    bundle = TorchModelBundle(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        ema_model=None,
    )
    method = ADSHMethod(ADSHSpec(model_bundle=bundle, batch_size=1, max_epochs=1))
    method._bundle = bundle
    method._backend = "torch"
    return method


def test_adsh_predict_proba_dict():
    method = _make_method()
    X = {"x": torch.zeros((2, 2))}
    proba = method.predict_proba(X)
    assert proba.shape == (2, 2)


def test_adsh_predict_proba_empty_dict():
    method = _make_method()
    X = {"x": torch.zeros((0, 2))}
    proba = method.predict_proba(X)
    assert proba.shape == (0, 0)
