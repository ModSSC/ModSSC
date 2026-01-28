from __future__ import annotations

import pytest
import torch

import modssc.inductive.methods.deep_co_training as dct
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.types import DeviceSpec

from .conftest import DummyDataset, make_numpy_dataset, make_torch_dataset


class _LinearLogits(torch.nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, n_classes, bias=False)

    def forward(self, x):
        return self.fc(x)


class _BadLogits1D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((int(x.shape[0]),), device=x.device)


class _BadBatchLogits(torch.nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.n_classes = int(n_classes)

    def forward(self, x):
        batch = max(0, int(x.shape[0]) - 1)
        return torch.zeros((batch, self.n_classes), device=x.device)


class _GradGoodBadDim(torch.nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, n_classes, bias=False)

    def forward(self, x):
        if x.requires_grad:
            return self.fc(x)
        return torch.zeros((int(x.shape[0]),), device=x.device)


class _GradGoodBadBatch(torch.nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, n_classes, bias=False)
        self.n_classes = int(n_classes)

    def forward(self, x):
        if x.requires_grad:
            return self.fc(x)
        batch = max(0, int(x.shape[0]) - 1)
        return torch.zeros((batch, self.n_classes), device=x.device)


class _GradGoodBadClasses(torch.nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, n_classes, bias=False)
        self.n_classes = int(n_classes)

    def forward(self, x):
        if x.requires_grad:
            return self.fc(x)
        return torch.zeros((int(x.shape[0]), self.n_classes + 1), device=x.device)


class _SentinelLogits(torch.nn.Module):
    def __init__(
        self,
        *,
        bad_kind: str,
        sentinel: float = 9.0,
        in_dim: int = 2,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, n_classes, bias=False)
        self.bad_kind = bad_kind
        self.sentinel = float(sentinel)
        self.n_classes = int(n_classes)

    def forward(self, x):
        if int(x.numel()) > 0 and bool((x == self.sentinel).all()):
            if self.bad_kind == "ndim":
                return torch.zeros((int(x.shape[0]),), device=x.device)
            if self.bad_kind == "batch":
                batch = max(0, int(x.shape[0]) - 1)
                return torch.zeros((batch, self.n_classes), device=x.device)
            if self.bad_kind == "classes":
                return torch.zeros((int(x.shape[0]), self.n_classes + 1), device=x.device)
        return self.fc(x)


class _SharedNet(torch.nn.Module):
    def __init__(self, shared: torch.nn.Module) -> None:
        super().__init__()
        self.shared = shared

    def forward(self, x):
        return self.shared(x)


def _make_bundle(model: torch.nn.Module) -> TorchModelBundle:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TorchModelBundle(model=model, optimizer=optimizer)


def _make_valid_spec(
    model1: torch.nn.Module | None = None,
    model2: torch.nn.Module | None = None,
    **overrides,
) -> dct.DeepCoTrainingSpec:
    model1 = model1 or _LinearLogits()
    model2 = model2 or _LinearLogits()
    params = {
        "model_bundle_1": _make_bundle(model1),
        "model_bundle_2": _make_bundle(model2),
        "batch_size": 2,
        "max_epochs": 1,
    }
    params.update(overrides)
    return dct.DeepCoTrainingSpec(**params)


def test_deep_co_training_entropy_js_divergence() -> None:
    probs = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32)
    ent = dct._entropy(probs)
    assert ent.shape == (2,)
    js = dct._js_divergence(probs, probs)
    assert float(js) == pytest.approx(0.0, abs=1e-6)
    logits = torch.zeros_like(probs)
    loss = dct._soft_cross_entropy(probs, logits)
    assert loss.shape == ()


def test_deep_co_training_soft_cross_entropy_shape_mismatch() -> None:
    with pytest.raises(InductiveValidationError, match="Target distribution shape mismatch"):
        dct._soft_cross_entropy(torch.zeros((2, 2)), torch.zeros((2, 3)))


def test_deep_co_training_fgsm_adversarial_variants() -> None:
    model = _LinearLogits()
    x_l = torch.randn((2, 2))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u = torch.randn((3, 2))

    adv = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        x_u,
        epsilon=0.1,
        freeze_bn=False,
        clip_min=-0.2,
        clip_max=0.2,
    )
    assert adv.shape == (5, 2)
    assert float(adv.max()) <= 0.2 + 1e-6
    assert float(adv.min()) >= -0.2 - 1e-6

    adv_min = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        x_u,
        epsilon=0.1,
        freeze_bn=False,
        clip_min=-0.1,
        clip_max=None,
    )
    assert adv_min.shape == (5, 2)

    adv_max = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        x_u,
        epsilon=0.1,
        freeze_bn=False,
        clip_min=None,
        clip_max=0.1,
    )
    assert adv_max.shape == (5, 2)

    adv_none = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        None,
        epsilon=0.05,
        freeze_bn=False,
        clip_min=None,
        clip_max=None,
    )
    assert adv_none.shape == (2, 2)

    x_l_empty = torch.zeros((0, 2))
    y_l_empty = torch.zeros((0,), dtype=torch.int64)
    adv_unlabeled = dct._fgsm_adversarial(
        model,
        x_l_empty,
        y_l_empty,
        x_u,
        epsilon=0.05,
        freeze_bn=False,
        clip_min=None,
        clip_max=None,
    )
    assert adv_unlabeled.shape == (3, 2)


def test_deep_co_training_fgsm_adversarial_errors() -> None:
    x_l = torch.randn((2, 2))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u = torch.randn((2, 2))

    with pytest.raises(InductiveValidationError, match="2D"):
        dct._fgsm_adversarial(
            _BadLogits1D(),
            x_l,
            y_l,
            x_u,
            epsilon=0.1,
            freeze_bn=False,
            clip_min=None,
            clip_max=None,
        )

    with pytest.raises(InductiveValidationError, match="batch size"):
        dct._fgsm_adversarial(
            _BadBatchLogits(),
            x_l,
            y_l,
            x_u,
            epsilon=0.1,
            freeze_bn=False,
            clip_min=None,
            clip_max=None,
        )

    with pytest.raises(InductiveValidationError, match="Labeled batch size mismatch"):
        dct._fgsm_adversarial(
            _LinearLogits(),
            x_l,
            y_l[:1],
            x_u,
            epsilon=0.1,
            freeze_bn=False,
            clip_min=None,
            clip_max=None,
        )


def test_deep_co_training_fgsm_adversarial_missing_x_dict() -> None:
    class _DictLike(dict):
        @property
        def shape(self):
            return (self["feat"].shape[0],)

    model = _LinearLogits()
    x_l = _DictLike({"feat": torch.zeros((2, 2))})
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(InductiveValidationError, match="include key 'x'"):
        dct._fgsm_adversarial(
            model,
            x_l,
            y_l,
            None,
            epsilon=0.1,
            freeze_bn=False,
            clip_min=None,
            clip_max=None,
        )


def test_deep_co_training_fgsm_adversarial_missing_x_in_unlabeled() -> None:
    x_l = torch.randn((2, 2))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u = {"feat": torch.zeros((2, 2))}
    with pytest.raises(InductiveValidationError, match="x_u dict inputs must include key 'x'"):
        dct._fgsm_adversarial(
            _LinearLogits(),
            x_l,
            y_l,
            x_u,
            epsilon=0.1,
            freeze_bn=False,
            clip_min=None,
            clip_max=None,
        )


def test_deep_co_training_fgsm_adversarial_grad_none(monkeypatch) -> None:
    model = _LinearLogits()
    x_l = torch.randn((2, 2), requires_grad=True)
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u = torch.randn((2, 2), requires_grad=True)

    # Force autograd.grad to return [None]
    def _grad(*args, **kwargs):
        return (None,)

    monkeypatch.setattr(torch.autograd, "grad", _grad)

    adv = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        x_u,
        epsilon=0.1,
        freeze_bn=False,
        clip_min=None,
        clip_max=None,
    )
    # If grad is None, we expect 0.0 update, so adv should equal x_all
    x_all = torch.cat([x_l, x_u], dim=0)
    assert torch.allclose(adv, x_all)


def test_deep_co_training_check_models_errors() -> None:
    method = dct.DeepCoTrainingMethod()
    model = _LinearLogits()
    with pytest.raises(InductiveValidationError, match="distinct models"):
        method._check_models(model, model)

    shared = torch.nn.Linear(2, 2, bias=False)
    model1 = _SharedNet(shared)
    model2 = _SharedNet(shared)
    with pytest.raises(InductiveValidationError, match="share parameters"):
        method._check_models(model1, model2)


def test_deep_co_training_fit_requires_data() -> None:
    method = dct.DeepCoTrainingMethod()
    with pytest.raises(InductiveValidationError, match="data must not be None"):
        method.fit(None, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_requires_torch_backend() -> None:
    data = make_numpy_dataset()
    method = dct.DeepCoTrainingMethod()
    with pytest.raises(InductiveValidationError, match="requires torch tensors"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_requires_unlabeled() -> None:
    base = make_torch_dataset()
    data = DummyDataset(X_l=base.X_l, y_l=base.y_l)
    method = dct.DeepCoTrainingMethod()
    with pytest.raises(InductiveValidationError, match="requires X_u"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "data, match",
    [
        (
            DummyDataset(
                X_l=torch.zeros((0, 2)),
                y_l=torch.zeros((0,), dtype=torch.int64),
                X_u=make_torch_dataset().X_u,
            ),
            "X_l must be non-empty",
        ),
        (
            DummyDataset(
                X_l=make_torch_dataset().X_l,
                y_l=make_torch_dataset().y_l,
                X_u=torch.zeros((0, 2)),
            ),
            "X_u must be non-empty",
        ),
    ],
)
def test_deep_co_training_fit_empty_inputs(data, match) -> None:
    spec = _make_valid_spec()
    method = dct.DeepCoTrainingMethod(spec)
    with pytest.raises(InductiveValidationError, match=match):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_bad_label_dtype() -> None:
    base = make_torch_dataset()
    data = DummyDataset(X_l=base.X_l, y_l=base.y_l.to(torch.float32), X_u=base.X_u)
    method = dct.DeepCoTrainingMethod()
    with pytest.raises(InductiveValidationError, match="y_l must be int64"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_requires_tensor_labels(monkeypatch) -> None:
    base = make_torch_dataset()
    data = DummyDataset(X_l=base.X_l, y_l=[0, 1], X_u=base.X_u)
    monkeypatch.setattr(dct, "ensure_torch_data", lambda _data, device: data)
    method = dct.DeepCoTrainingMethod(_make_valid_spec())
    with pytest.raises(InductiveValidationError, match="y_l must be a torch.Tensor"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_requires_bundles() -> None:
    data = make_torch_dataset()
    method = dct.DeepCoTrainingMethod()
    with pytest.raises(InductiveValidationError, match="model_bundle_1 and model_bundle_2"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"batch_size": 0}, "batch_size must be"),
        ({"max_epochs": 0}, "max_epochs must be"),
        ({"lambda_cot": -1.0}, "lambda_cot must be"),
        ({"lambda_dif": -1.0}, "lambda_dif must be"),
        ({"adv_eps": -0.1}, "adv_eps must be"),
        ({"adv_clip_min": 1.0, "adv_clip_max": 0.0}, "adv_clip_min must be"),
    ],
)
def test_deep_co_training_fit_invalid_specs(overrides, match) -> None:
    data = make_torch_dataset()
    spec = _make_valid_spec(**overrides)
    method = dct.DeepCoTrainingMethod(spec)
    with pytest.raises(InductiveValidationError, match=match):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_mismatched_labeled_batches(monkeypatch) -> None:
    data = make_torch_dataset()
    spec = _make_valid_spec(batch_size=4)
    method = dct.DeepCoTrainingMethod(spec)

    x1 = data.X_l[:1]
    y1 = data.y_l[:1]
    x2 = data.X_l[:2]
    y2 = data.y_l[:2]
    calls = {"count": 0}

    def fake_cycle_batches(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return iter([(x1, y1)])
        return iter([(x2, y2)])

    monkeypatch.setattr(dct, "cycle_batches", fake_cycle_batches)

    with pytest.raises(InductiveValidationError, match="Labeled batch sizes must match"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_bad_logits_dim() -> None:
    data = make_torch_dataset()
    spec = _make_valid_spec(model1=_GradGoodBadDim(), model2=_LinearLogits())
    method = dct.DeepCoTrainingMethod(spec)
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_bad_logits_batch_model1() -> None:
    data = make_torch_dataset()
    spec = _make_valid_spec(model1=_GradGoodBadBatch(), model2=_LinearLogits())
    method = dct.DeepCoTrainingMethod(spec)
    with pytest.raises(InductiveValidationError, match="Model1 logits batch size mismatch"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_bad_logits_batch_model2() -> None:
    data = make_torch_dataset()
    spec = _make_valid_spec(model1=_LinearLogits(), model2=_GradGoodBadBatch())
    method = dct.DeepCoTrainingMethod(spec)
    with pytest.raises(InductiveValidationError, match="Model2 logits batch size mismatch"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_bad_logits_class_mismatch() -> None:
    data = make_torch_dataset()
    spec = _make_valid_spec(model1=_LinearLogits(), model2=_GradGoodBadClasses())
    method = dct.DeepCoTrainingMethod(spec)
    with pytest.raises(InductiveValidationError, match="Models must agree on class count"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_fit_no_detach_zero_weights() -> None:
    data = make_torch_dataset()
    spec = _make_valid_spec(detach_target=False, lambda_cot=0.0, lambda_dif=0.0)
    method = dct.DeepCoTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def _fake_fgsm_returning(sentinel: float):
    def _fake(_model, x_l, _y_l, x_u, **_kwargs):
        x_all = x_l if x_u is None else torch.cat([x_l, x_u], dim=0)
        return torch.full_like(x_all, float(sentinel))

    return _fake


@pytest.mark.parametrize(
    "bad_model, good_model, match",
    [
        (_SentinelLogits(bad_kind="ndim"), _LinearLogits(), "Model logits must be 2D"),
        (
            _LinearLogits(),
            _SentinelLogits(bad_kind="batch"),
            "Model2 adversarial logits batch mismatch",
        ),
        (
            _SentinelLogits(bad_kind="batch"),
            _LinearLogits(),
            "Model1 adversarial logits batch mismatch",
        ),
        (_LinearLogits(), _SentinelLogits(bad_kind="classes"), "Model2 logits class mismatch"),
        (_SentinelLogits(bad_kind="classes"), _LinearLogits(), "Model1 logits class mismatch"),
    ],
)
def test_deep_co_training_fit_adversarial_logits_errors(
    monkeypatch, bad_model, good_model, match
) -> None:
    data = make_torch_dataset(n_l=2, n_u=2)
    spec = _make_valid_spec(model1=bad_model, model2=good_model, batch_size=2)
    method = dct.DeepCoTrainingMethod(spec)
    monkeypatch.setattr(dct, "_fgsm_adversarial", _fake_fgsm_returning(9.0))
    with pytest.raises(InductiveValidationError, match=match):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_predict_requires_fit() -> None:
    method = dct.DeepCoTrainingMethod()
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(torch.zeros((1, 2)))


def test_deep_co_training_predict_proba_backend_mismatch() -> None:
    method = dct.DeepCoTrainingMethod()
    method._bundle1 = _make_bundle(_LinearLogits())
    method._bundle2 = _make_bundle(_LinearLogits())
    with pytest.raises(InductiveValidationError, match="requires torch tensors"):
        method.predict_proba(make_numpy_dataset().X_l)


def test_deep_co_training_predict_proba_non_tensor() -> None:
    method = dct.DeepCoTrainingMethod()
    method._bundle1 = _make_bundle(_LinearLogits())
    method._bundle2 = _make_bundle(_LinearLogits())
    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="predict_proba requires torch.Tensor"):
        method.predict_proba([[0.0, 1.0]])


def test_deep_co_training_predict_proba_bad_logits() -> None:
    method = dct.DeepCoTrainingMethod()
    method._bundle1 = _make_bundle(_BadLogits1D())
    method._bundle2 = _make_bundle(_BadLogits1D())
    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method.predict_proba(torch.zeros((2, 2)))


def test_deep_co_training_predict_proba_class_mismatch() -> None:
    method = dct.DeepCoTrainingMethod()
    method._bundle1 = _make_bundle(_LinearLogits(n_classes=2))
    method._bundle2 = _make_bundle(_LinearLogits(n_classes=3))
    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="Models must agree on class count"):
        method.predict_proba(torch.zeros((2, 2)))


def test_deep_co_training_predict_proba_eval_models() -> None:
    data = make_torch_dataset()
    spec = _make_valid_spec()
    method = dct.DeepCoTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    method._bundle1.model.eval()
    method._bundle2.model.eval()
    proba = method.predict_proba(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])


def test_deep_co_training_fit_predict_roundtrip() -> None:
    data = make_torch_dataset()
    spec = _make_valid_spec()
    method = dct.DeepCoTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    model1 = method._bundle1.model
    model2 = method._bundle2.model
    model1.train()
    model2.train()

    proba = method.predict_proba(data.X_l)
    pred = method.predict(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    assert int(pred.shape[0]) == int(data.X_l.shape[0])
    assert model1.training is True
    assert model2.training is True


def test_deep_co_training_fgsm_adversarial_dict_clip():
    class _DictLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["x"]
            return self.fc(x)

    model = _DictLinear()
    x_l = {"x": torch.randn((2, 2))}
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u = {"x": torch.randn((1, 2))}
    adv = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        x_u,
        epsilon=0.1,
        freeze_bn=False,
        clip_min=-0.1,
        clip_max=0.1,
    )
    assert isinstance(adv, dict)
    assert adv["x"].shape[0] == 3


def test_deep_co_training_fgsm_adversarial_dict_empty_xu():
    class _DictLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["x"]
            return self.fc(x)

    model = _DictLinear()
    x_l = {"x": torch.randn((2, 2))}
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u = {"x": torch.zeros((0, 2))}
    adv = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        x_u,
        epsilon=0.05,
        freeze_bn=False,
        clip_min=None,
        clip_max=None,
    )
    assert isinstance(adv, dict)
    assert adv["x"].shape[0] == 2


def test_deep_co_training_fgsm_adversarial_empty_xu_dict_with_tensor_xl():
    model = _LinearLogits()
    x_l = torch.randn((2, 2))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u = {"x": torch.zeros((0, 2))}
    adv = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        x_u,
        epsilon=0.05,
        freeze_bn=False,
        clip_min=None,
        clip_max=None,
    )
    assert adv.shape == (2, 2)


def test_deep_co_training_fgsm_adversarial_empty_xu_tensor():
    model = _LinearLogits()
    x_l = torch.randn((2, 2))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u = torch.zeros((0, 2))
    adv = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        x_u,
        epsilon=0.05,
        freeze_bn=False,
        clip_min=None,
        clip_max=None,
    )
    assert adv.shape == (2, 2)


def test_deep_co_training_fgsm_adversarial_empty_xl_no_u(monkeypatch):
    class _DictLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["x"]
            return self.fc(x)

    def _loss(*_args, **_kwargs):
        return torch.tensor(0.0, requires_grad=True)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", _loss)

    model = _DictLinear()
    x_l = {"x": torch.zeros((0, 2))}
    y_l = torch.zeros((0,), dtype=torch.int64)
    adv = dct._fgsm_adversarial(
        model,
        x_l,
        y_l,
        None,
        epsilon=0.05,
        freeze_bn=False,
        clip_min=None,
        clip_max=None,
    )
    assert isinstance(adv, dict)


def test_deep_co_training_fit_dict_len_paths(monkeypatch):
    X_l = {"x": torch.zeros((2, 2))}
    X_u = {"x": torch.zeros((2, 2))}
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u=X_u)

    spec = _make_valid_spec(batch_size=1, max_epochs=1)
    method = dct.DeepCoTrainingMethod(spec)

    monkeypatch.setattr(dct, "ensure_torch_data", lambda _data, device: data)
    monkeypatch.setattr(
        dct,
        "cycle_batch_indices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )

    with pytest.raises(RuntimeError, match="stop"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_deep_co_training_predict_proba_dict_input():
    class _DictLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["x"]
            return self.fc(x)

    method = dct.DeepCoTrainingMethod()
    method._bundle1 = _make_bundle(_DictLinear())
    method._bundle2 = _make_bundle(_DictLinear())
    method._backend = "torch"
    X = {"x": torch.zeros((2, 2), dtype=torch.float32)}
    proba = method.predict_proba(X)
    assert int(proba.shape[0]) == 2
