from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

import modssc.inductive.methods.simclr_v2 as simclr_v2
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.simclr_v2 import SimCLRv2Method, SimCLRv2Spec
from modssc.inductive.types import DeviceSpec, InductiveDataset

from .conftest import make_numpy_dataset, make_torch_ssl_dataset


class _ProjectorNet(torch.nn.Module):
    def __init__(self, in_dim: int = 2, proj_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(in_dim, proj_dim, bias=False)
        self.projector = torch.nn.Linear(proj_dim, proj_dim, bias=False)
        self.classifier = torch.nn.Linear(proj_dim, n_classes, bias=False)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projector(feat)
        logits = self.classifier(feat)
        return {"proj": proj, "logits": logits, "feat": feat}

    def forward_features(self, x):
        return self.encoder(x)

    def forward_projection(self, x):
        return self.projector(self.encoder(x))

    def forward_head(self, feats):
        return self.classifier(feats)

    def forward_logits(self, x):
        return self.classifier(self.encoder(x))


class _BadProjection1D(torch.nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, n_classes, bias=False)

    def forward(self, x):
        batch = int(x.shape[0])
        return {
            "proj": torch.zeros((batch,), device=x.device),
            "logits": self.fc(x),
        }


class _ShapeSwitchProjector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch = int(x.shape[0])
        if float(x.mean()) > 0:
            proj = torch.zeros((batch, 3), device=x.device)
        else:
            proj = torch.zeros((batch, 2), device=x.device)
        return {"proj": proj, "logits": torch.zeros((batch, 2), device=x.device)}


class _BadLogits1D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((int(x.shape[0]),), device=x.device)


class _OneClassNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return {"logits": self.fc(x)}


class _ConditionalLogitsNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 1, bias=False)
        self.fc2 = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x):
        if float(x.mean()) > 0:
            return {"logits": self.fc1(x).squeeze(1)}
        return {"logits": self.fc2(x)}


def _make_bundle(model: torch.nn.Module, *, meta: dict | None = None) -> TorchModelBundle:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TorchModelBundle(model=model, optimizer=optimizer, meta=meta)


def _make_dataset(
    *,
    X_l: torch.Tensor,
    y_l: torch.Tensor,
    X_u: torch.Tensor | None = None,
    X_u_w: torch.Tensor | None = None,
    X_u_s: torch.Tensor | None = None,
) -> InductiveDataset:
    return InductiveDataset(X_l=X_l, y_l=y_l, X_u=X_u, X_u_w=X_u_w, X_u_s=X_u_s)


def test_simclr_v2_helper_branches():
    x = torch.randn(2, 2)
    assert simclr_v2._as_tensor(x, name="x") is x

    assert simclr_v2._tensor_from_output(x, keys=("feat",), name="tensor") is x
    assert simclr_v2._tensor_from_output({"feat": x}, keys=("feat",), name="mapping") is x
    assert simclr_v2._tensor_from_output((x,), keys=("feat",), name="tuple") is x

    model = _ProjectorNet()
    out_meta = simclr_v2._forward_features(model, {"forward_features": lambda t: t + 1.0}, x)
    assert torch.allclose(out_meta, x + 1.0)
    out_meta_skip = simclr_v2._forward_features(model, {"forward_features": 1}, x)
    assert out_meta_skip.shape == (2, 2)
    out_fallback = simclr_v2._forward_features(model, None, x)
    assert out_fallback.shape == (2, 2)

    proj_meta = simclr_v2._forward_projection(model, {"forward_projection": lambda t: t + 2.0}, x)
    assert torch.allclose(proj_meta, x + 2.0)
    proj_head = simclr_v2._forward_projection(
        model,
        {"forward_features": lambda t: t + 1.0, "projection_head": lambda f: f * 2.0},
        x,
    )
    assert torch.allclose(proj_head, (x + 1.0) * 2.0)
    proj_meta_skip = simclr_v2._forward_projection(model, {"projection_head": 1}, x)
    assert proj_meta_skip.shape == (2, 2)
    proj_fallback = simclr_v2._forward_projection(model, None, x)
    assert proj_fallback.shape == (2, 2)

    logits_meta = simclr_v2._forward_logits(model, {"forward_logits": lambda t: {"logits": t}}, x)
    assert torch.allclose(logits_meta, x)
    logits_head = simclr_v2._forward_logits(
        model,
        {"forward_features": lambda t: t + 1.0, "forward_head": lambda f: f * 2.0},
        x,
    )
    assert torch.allclose(logits_head, (x + 1.0) * 2.0)
    logits_direct = simclr_v2._forward_logits(model, {"head": lambda t: t + 3.0}, x)
    assert torch.allclose(logits_direct, x + 3.0)
    logits_meta_skip = simclr_v2._forward_logits(model, {"head": 1}, x)
    assert logits_meta_skip.shape == (2, 2)
    logits_fallback = simclr_v2._forward_logits(model, None, x)
    assert logits_fallback.shape == (2, 2)

    class _MetaNet(torch.nn.Module):
        def forward_features(self, t):
            return t + 1.0

    class _MetaNetNoCall(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_features = 1

    def passthrough(t):
        return t

    source = _MetaNet()
    target = _MetaNet()
    meta = {"forward_features": source.forward_features, "value": 7, "callable": passthrough}
    rebound = simclr_v2._rebind_meta(meta, source=source, target=target)
    assert rebound["forward_features"].__self__ is target
    assert rebound["value"] == 7
    assert rebound["callable"] is passthrough

    meta_no_call = {"forward_features": source.forward_features}
    rebound_no_call = simclr_v2._rebind_meta(meta_no_call, source=source, target=_MetaNetNoCall())
    assert rebound_no_call["forward_features"].__self__ is source

    sentinel = object()
    assert simclr_v2._rebind_meta(sentinel, source=source, target=target) is sentinel


def test_simclr_v2_helper_errors():
    with pytest.raises(InductiveValidationError, match="must be a torch.Tensor"):
        simclr_v2._as_tensor([1, 2], name="x")

    with pytest.raises(InductiveValidationError, match="mapping with keys"):
        simclr_v2._tensor_from_output({"bad": 1}, keys=("feat",), name="bad")

    with pytest.raises(InductiveValidationError, match="temperature must be > 0"):
        simclr_v2._nt_xent_loss(torch.zeros((2, 2)), temperature=0.0)
    with pytest.raises(InductiveValidationError, match="Projection outputs must be 2D"):
        simclr_v2._nt_xent_loss(torch.zeros((2,)), temperature=0.5)
    with pytest.raises(InductiveValidationError, match="Contrastive batch must be even"):
        simclr_v2._nt_xent_loss(torch.zeros((3, 2)), temperature=0.5)
    loss = simclr_v2._nt_xent_loss(torch.zeros((4, 2)), temperature=0.5)
    assert float(loss) >= 0.0

    with pytest.raises(InductiveValidationError, match="distill_temperature must be > 0"):
        simclr_v2._distill_loss(
            torch.zeros((2, 2)), torch.zeros((2, 2)), temperature=0.0, detach_target=True
        )
    loss_detach = simclr_v2._distill_loss(
        torch.zeros((2, 2)), torch.zeros((2, 2)), temperature=1.0, detach_target=True
    )
    loss_keep = simclr_v2._distill_loss(
        torch.zeros((2, 2)), torch.zeros((2, 2)), temperature=1.0, detach_target=False
    )
    assert float(loss_detach) >= 0.0
    assert float(loss_keep) >= 0.0

    model = torch.nn.Linear(2, 2)
    with pytest.raises(InductiveValidationError, match="must be distinct"):
        simclr_v2._check_distill_models(model, model)

    shared = torch.nn.Linear(2, 2)
    student = torch.nn.Sequential(shared)
    teacher = torch.nn.Sequential(shared)
    with pytest.raises(InductiveValidationError, match="must not share parameters"):
        simclr_v2._check_distill_models(student, teacher)

    simclr_v2._check_distill_models(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))


def test_simclr_v2_fit_full_pipeline():
    data = make_torch_ssl_dataset()
    pretrain_bundle = _make_bundle(_ProjectorNet())
    finetune_bundle = _make_bundle(_ProjectorNet())
    student_bundle = _make_bundle(_ProjectorNet())
    spec = SimCLRv2Spec(
        pretrain_bundle=pretrain_bundle,
        finetune_bundle=finetune_bundle,
        student_bundle=student_bundle,
        batch_size=2,
        pretrain_epochs=1,
        finetune_epochs=1,
        distill_epochs=1,
        use_labeled_in_distill=True,
    )
    method = SimCLRv2Method(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._bundle is student_bundle
    proba = method.predict_proba(data.X_l)
    pred = method.predict(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    assert int(pred.shape[0]) == int(data.X_l.shape[0])
    method._bundle.model.eval()
    proba_eval = method.predict_proba(data.X_l)
    assert int(proba_eval.shape[0]) == int(data.X_l.shape[0])


def test_simclr_v2_fit_distill_no_student_bundle():
    data = make_torch_ssl_dataset()
    finetune_bundle = _make_bundle(_ProjectorNet())
    spec = SimCLRv2Spec(
        finetune_bundle=finetune_bundle,
        batch_size=2,
        pretrain_epochs=0,
        finetune_epochs=1,
        distill_epochs=1,
        use_labeled_in_distill=False,
    )
    method = SimCLRv2Method(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=1)
    assert method._bundle is finetune_bundle


def test_simclr_v2_fit_pretrain_only_uses_x_u():
    data = make_torch_ssl_dataset()
    data = _make_dataset(X_l=data.X_l, y_l=data.y_l, X_u=data.X_u)
    pretrain_bundle = _make_bundle(_ProjectorNet())
    spec = SimCLRv2Spec(
        pretrain_bundle=pretrain_bundle,
        batch_size=2,
        pretrain_epochs=1,
        finetune_epochs=0,
        distill_epochs=0,
        use_labeled_in_distill=False,
    )
    method = SimCLRv2Method(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=2)
    assert method._bundle is pretrain_bundle


@pytest.mark.parametrize(
    "spec_kwargs,data,match",
    [
        ({"pretrain_epochs": 1}, None, "data must not be None"),
        ({}, make_numpy_dataset(), "requires torch tensors"),
        ({"batch_size": 0}, make_torch_ssl_dataset(), "batch_size must be >= 1"),
        ({"pretrain_epochs": -1}, make_torch_ssl_dataset(), "pretrain_epochs must be >= 0"),
        ({"finetune_epochs": -1}, make_torch_ssl_dataset(), "finetune_epochs must be >= 0"),
        ({"distill_epochs": -1}, make_torch_ssl_dataset(), "distill_epochs must be >= 0"),
        (
            {"pretrain_epochs": 0, "finetune_epochs": 0, "distill_epochs": 0},
            make_torch_ssl_dataset(),
            "At least one of pretrain_epochs",
        ),
        ({"alpha": 1.5}, make_torch_ssl_dataset(), "alpha must be in"),
        ({"temperature": 0.0}, make_torch_ssl_dataset(), "temperature must be > 0"),
        ({"distill_temperature": 0.0}, make_torch_ssl_dataset(), "distill_temperature must be > 0"),
        (
            {"pretrain_epochs": 1},
            _make_dataset(
                X_l=torch.zeros((2, 2)),
                y_l=torch.zeros((2,), dtype=torch.int64),
            ),
            "requires unlabeled data",
        ),
        (
            {"pretrain_epochs": 1},
            _make_dataset(
                X_l=torch.zeros((2, 2)),
                y_l=torch.zeros((2,), dtype=torch.int64),
                X_u=torch.empty((0, 2)),
            ),
            "X_u must be non-empty",
        ),
        (
            {"finetune_epochs": 1, "pretrain_epochs": 0, "distill_epochs": 0},
            _make_dataset(
                X_l=torch.empty((0, 2)),
                y_l=torch.empty((0,), dtype=torch.int64),
            ),
            "X_l must be non-empty",
        ),
        (
            {"finetune_epochs": 1, "pretrain_epochs": 0, "distill_epochs": 0},
            _make_dataset(
                X_l=torch.zeros((2, 2)),
                y_l=torch.tensor([0, 1], dtype=torch.int32),
            ),
            "y_l must be int64",
        ),
        (
            {
                "pretrain_epochs": 1,
                "finetune_epochs": 0,
                "distill_epochs": 0,
                "use_labeled_in_distill": False,
            },
            make_torch_ssl_dataset(),
            "pretrain_bundle or finetune_bundle must be provided",
        ),
        (
            {"pretrain_epochs": 0, "finetune_epochs": 1, "distill_epochs": 0},
            make_torch_ssl_dataset(),
            "finetune_bundle or pretrain_bundle must be provided",
        ),
    ],
)
def test_simclr_v2_fit_validation_errors(spec_kwargs, data, match):
    spec = SimCLRv2Spec(**spec_kwargs)
    method = SimCLRv2Method(spec)
    with pytest.raises(InductiveValidationError, match=match):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_simclr_v2_unlabeled_row_mismatch_hits_check(monkeypatch):
    data = make_torch_ssl_dataset()
    bad = _make_dataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u_w=data.X_u_w,
        X_u_s=torch.zeros((int(data.X_u_w.shape[0]) + 1, 2)),
    )
    spec = SimCLRv2Spec(
        pretrain_bundle=_make_bundle(_ProjectorNet()),
        batch_size=2,
        pretrain_epochs=1,
        finetune_epochs=0,
        distill_epochs=0,
        use_labeled_in_distill=False,
    )
    method = SimCLRv2Method(spec)
    monkeypatch.setattr(simclr_v2, "ensure_torch_data", lambda _data, device: bad)
    with pytest.raises(
        InductiveValidationError, match="X_u_w and X_u_s must have the same number of rows"
    ):
        method.fit(bad, device=DeviceSpec(device="cpu"), seed=0)


def test_simclr_v2_pretrain_projection_errors():
    data = make_torch_ssl_dataset()
    spec = SimCLRv2Spec(
        pretrain_bundle=_make_bundle(_BadProjection1D()),
        batch_size=2,
        pretrain_epochs=1,
        finetune_epochs=0,
        distill_epochs=0,
        use_labeled_in_distill=False,
    )
    method = SimCLRv2Method(spec)
    with pytest.raises(InductiveValidationError, match="Projection outputs must be 2D"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    mismatch = _make_dataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u_w=-torch.ones((4, 2)),
        X_u_s=torch.ones((4, 2)),
    )
    spec_mismatch = SimCLRv2Spec(
        pretrain_bundle=_make_bundle(_ShapeSwitchProjector()),
        batch_size=2,
        pretrain_epochs=1,
        finetune_epochs=0,
        distill_epochs=0,
        use_labeled_in_distill=False,
    )
    method_mismatch = SimCLRv2Method(spec_mismatch)
    with pytest.raises(
        InductiveValidationError, match="Projection outputs must have the same shape"
    ):
        method_mismatch.fit(mismatch, device=DeviceSpec(device="cpu"), seed=0)


def test_simclr_v2_finetune_errors():
    data = make_torch_ssl_dataset()
    spec_bad_logits = SimCLRv2Spec(
        finetune_bundle=_make_bundle(_BadLogits1D()),
        batch_size=2,
        pretrain_epochs=0,
        finetune_epochs=1,
        distill_epochs=0,
    )
    method = SimCLRv2Method(spec_bad_logits)
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    bad_labels = _make_dataset(
        X_l=torch.zeros((2, 2)),
        y_l=torch.tensor([0, 1], dtype=torch.int64),
    )
    spec_bad_labels = SimCLRv2Spec(
        finetune_bundle=_make_bundle(_OneClassNet()),
        batch_size=2,
        pretrain_epochs=0,
        finetune_epochs=1,
        distill_epochs=0,
    )
    method_bad_labels = SimCLRv2Method(spec_bad_labels)
    with pytest.raises(InductiveValidationError, match="y_l labels must be within"):
        method_bad_labels.fit(bad_labels, device=DeviceSpec(device="cpu"), seed=0)


def test_simclr_v2_distill_errors():
    data = make_torch_ssl_dataset()
    spec_bad_logits = SimCLRv2Spec(
        finetune_bundle=_make_bundle(_ProjectorNet()),
        student_bundle=_make_bundle(_BadLogits1D()),
        batch_size=2,
        pretrain_epochs=0,
        finetune_epochs=1,
        distill_epochs=1,
        use_labeled_in_distill=False,
    )
    method_bad_logits = SimCLRv2Method(spec_bad_logits)
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method_bad_logits.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    spec_mismatch = SimCLRv2Spec(
        finetune_bundle=_make_bundle(_ProjectorNet(n_classes=3)),
        student_bundle=_make_bundle(_ProjectorNet(n_classes=2)),
        batch_size=2,
        pretrain_epochs=0,
        finetune_epochs=1,
        distill_epochs=1,
        use_labeled_in_distill=False,
    )
    method_mismatch = SimCLRv2Method(spec_mismatch)
    with pytest.raises(InductiveValidationError, match="logits shape mismatch"):
        method_mismatch.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    X_l = torch.ones((4, 2))
    y_l = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    X_u = -torch.ones((4, 2))
    data_cond = _make_dataset(X_l=X_l, y_l=y_l, X_u=X_u, X_u_w=X_u, X_u_s=X_u)
    spec_bad_labeled = SimCLRv2Spec(
        finetune_bundle=_make_bundle(_ProjectorNet()),
        student_bundle=_make_bundle(_ConditionalLogitsNet()),
        batch_size=2,
        pretrain_epochs=0,
        finetune_epochs=1,
        distill_epochs=1,
        use_labeled_in_distill=True,
    )
    method_bad_labeled = SimCLRv2Method(spec_bad_labeled)
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method_bad_labeled.fit(data_cond, device=DeviceSpec(device="cpu"), seed=0)

    data_range = _make_dataset(
        X_l=torch.zeros((2, 2)),
        y_l=torch.tensor([0, 1], dtype=torch.int64),
        X_u=torch.zeros((2, 2)),
        X_u_w=torch.zeros((2, 2)),
        X_u_s=torch.zeros((2, 2)),
    )
    spec_range = SimCLRv2Spec(
        finetune_bundle=_make_bundle(_OneClassNet()),
        student_bundle=_make_bundle(_OneClassNet()),
        batch_size=2,
        pretrain_epochs=0,
        finetune_epochs=0,
        distill_epochs=1,
        use_labeled_in_distill=True,
    )
    method_range = SimCLRv2Method(spec_range)
    with pytest.raises(InductiveValidationError, match="y_l labels must be within"):
        method_range.fit(data_range, device=DeviceSpec(device="cpu"), seed=0)


def test_simclr_v2_distill_requires_unlabeled_inside_block():
    class _SpecSwitch:
        def __init__(self, *, finetune_bundle: TorchModelBundle):
            self.pretrain_bundle = None
            self.finetune_bundle = finetune_bundle
            self.student_bundle = None
            self.temperature = 0.5
            self.distill_temperature = 1.0
            self.alpha = 0.5
            self.batch_size = 2
            self.pretrain_epochs = 0
            self.finetune_epochs = 1
            self.transfer_pretrain = True
            self.use_labeled_in_distill = False
            self.freeze_bn = False
            self.detach_target = True
            self._distill_calls = 0

        @property
        def distill_epochs(self) -> int:
            self._distill_calls += 1
            return 0 if self._distill_calls <= 3 else 1

    X_l = torch.zeros((2, 2))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    data = _make_dataset(X_l=X_l, y_l=y_l)
    spec = _SpecSwitch(finetune_bundle=_make_bundle(_ProjectorNet()))
    method = SimCLRv2Method(spec)
    with pytest.raises(InductiveValidationError, match="distill requires unlabeled data"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_simclr_v2_predict_errors():
    method = SimCLRv2Method()
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(torch.zeros((2, 2)))

    bundle = _make_bundle(_ProjectorNet())
    method._bundle = bundle
    with pytest.raises(InductiveValidationError, match="predict_proba requires torch tensors"):
        method.predict_proba(np.zeros((2, 2), dtype=np.float32))

    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="predict_proba requires torch.Tensor"):
        method.predict_proba(np.zeros((2, 2), dtype=np.float32))

    bad_bundle = _make_bundle(_BadLogits1D())
    method._bundle = bad_bundle
    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method.predict_proba(torch.zeros((2, 2)))
