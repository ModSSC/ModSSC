from __future__ import annotations

import copy

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

import modssc.inductive.methods.daso as daso
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.daso import DASOMethod, DASOSpec
from modssc.inductive.types import DeviceSpec, InductiveDataset

from .conftest import make_numpy_dataset, make_torch_ssl_dataset


class _DASONet(torch.nn.Module):
    def __init__(self, in_dim: int = 2, feat_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.feat = torch.nn.Linear(in_dim, feat_dim, bias=False)
        self.fc = torch.nn.Linear(feat_dim, n_classes, bias=False)

    def forward(self, x):
        feat = self.feat(x)
        logits = self.fc(feat)
        return {"logits": logits, "feat": feat}


class _TupleNet(torch.nn.Module):
    def __init__(self, in_dim: int = 2, feat_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.feat = torch.nn.Linear(in_dim, feat_dim, bias=False)
        self.fc = torch.nn.Linear(feat_dim, n_classes, bias=False)

    def forward(self, x):
        feat = self.feat(x)
        logits = self.fc(feat)
        return logits, feat


class _BadLogits1D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((int(x.shape[0]),), device=x.device)


class _BadLogitsMap1D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.zeros((int(x.shape[0]),), device=x.device)
        feat = torch.zeros((int(x.shape[0]), 2), device=x.device)
        return {"logits": logits, "feat": feat}


class _LogitsOnlyNet(torch.nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((int(x.shape[0]), self.n_classes), device=x.device)


class _LogitsMapNet(torch.nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.zeros((int(x.shape[0]), self.n_classes), device=x.device)
        return {"logits": logits}


class _FeatOnlyMap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return {"feat": x}


class _TupleBadFeatNet(torch.nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.zeros((int(x.shape[0]), self.n_classes), device=x.device)
        return logits, "bad"


class _BadBatchNet(torch.nn.Module):
    def __init__(self, n_classes: int = 2, feat_dim: int = 2) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.feat_dim = int(feat_dim)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rows = max(int(x.shape[0]) - 1, 0)
        logits = torch.zeros((rows, self.n_classes), device=x.device)
        feat = torch.zeros((rows, self.feat_dim), device=x.device)
        return {"logits": logits, "feat": feat}


class _VarShapeNet(torch.nn.Module):
    def __init__(self, class_counts: list[int], feat_dims: list[int]) -> None:
        super().__init__()
        self.class_counts = class_counts
        self.feat_dims = feat_dims
        self.call = 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        idx = min(self.call, len(self.class_counts) - 1)
        self.call += 1
        n_classes = int(self.class_counts[idx])
        feat_dim = int(self.feat_dims[idx])
        logits = torch.zeros((int(x.shape[0]), n_classes), device=x.device)
        feat = torch.zeros((int(x.shape[0]), feat_dim), device=x.device)
        return {"logits": logits, "feat": feat}


def _make_base_spec(model: torch.nn.Module, *, ema: bool = False, **overrides) -> DASOSpec:
    base = DASOSpec(
        model_bundle=_make_bundle(model, ema=ema),
        batch_size=4,
        max_epochs=1,
        pretrain_steps=0,
        dist_update_period=1,
        queue_size=1,
        use_ema=False,
        dist_aware=True,
        p_cutoff=0.0,
    )
    return DASOSpec(**{**base.__dict__, **overrides})


def _make_bundle(model: torch.nn.Module, *, ema: bool = False) -> TorchModelBundle:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema_model = copy.deepcopy(model) if ema else None
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)


def _make_dataset(*, y: torch.Tensor | None = None) -> InductiveDataset:
    base = make_torch_ssl_dataset()
    y_l = y if y is not None else base.y_l
    return InductiveDataset(
        X_l=base.X_l,
        y_l=y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )


def test_daso_helper_functions():
    x = torch.randn(2, 2)
    feats = daso._flatten_features(x, name="feat")
    assert feats.shape == (2, 2)
    feats_1d = daso._flatten_features(torch.randn(2), name="feat_1d")
    assert feats_1d.shape == (1, 2)
    feats_3d = daso._flatten_features(torch.randn(2, 1, 2), name="feat_3d")
    assert feats_3d.shape == (2, 2)

    with pytest.raises(InductiveValidationError, match="must be a torch.Tensor"):
        daso._flatten_features([1, 2], name="feat")
    with pytest.raises(InductiveValidationError, match="must include a batch dimension"):
        daso._flatten_features(torch.tensor(1.0), name="feat")
    with pytest.raises(InductiveValidationError, match="batch dimension mismatch"):
        daso._flatten_features(torch.randn(2, 2), name="feat", batch=3)

    sims = daso._cosine_similarity(torch.randn(2, 2), torch.randn(3, 2))
    assert sims.shape == (2, 3)
    with pytest.raises(InductiveValidationError, match="requires torch.Tensor inputs"):
        daso._cosine_similarity(torch.randn(2, 2), np.zeros((2, 2)))


def test_daso_forward_helpers_and_ema_checks():
    x = torch.randn(2, 2)
    bundle = _make_bundle(_DASONet())
    logits, feats = daso._forward_logits_features(bundle, x)
    assert logits.shape == (2, 2)
    assert feats.shape == (2, 2)

    meta = {
        "forward_features": lambda t: t + 1.0,
        "forward_head": lambda f: f * 2.0,
    }
    model_meta = _DASONet()
    optimizer_meta = torch.optim.SGD(model_meta.parameters(), lr=0.1)
    bundle_meta = TorchModelBundle(model=model_meta, optimizer=optimizer_meta, meta=meta)
    logits_meta, feats_meta = daso._forward_logits_features(bundle_meta, x)
    assert torch.allclose(feats_meta, x + 1.0)
    assert torch.allclose(logits_meta, (x + 1.0) * 2.0)

    tuple_bundle = _make_bundle(_TupleNet())
    logits_t, feats_t = daso._forward_logits_features(tuple_bundle, x)
    assert logits_t.shape == (2, 2)
    assert feats_t.shape == (2, 2)

    with pytest.raises(InductiveValidationError, match="forward_features must return"):
        bad_meta = {"forward_features": lambda _t: "bad", "forward_head": lambda f: f}
        daso._forward_logits_features(
            TorchModelBundle(model=_DASONet(), optimizer=bundle.optimizer, meta=bad_meta), x
        )

    class _FeatOnly(torch.nn.Module):
        def forward(self, x):
            return {"feat": x}

    with pytest.raises(InductiveValidationError, match="Model output must include logits"):
        daso._forward_logits_features(
            TorchModelBundle(model=_FeatOnly(), optimizer=bundle.optimizer, meta=None), x
        )

    feats_direct = daso._forward_features(bundle, x)
    assert feats_direct.shape == (2, 2)
    feats_meta = daso._forward_features(bundle_meta, x)
    assert torch.allclose(feats_meta, x + 1.0)
    feats_ema = daso._forward_features(bundle_meta, x, model_override=_DASONet())
    assert feats_ema.shape == (2, 2)

    with pytest.raises(InductiveValidationError, match="ema_model must be distinct"):
        daso._check_ema(bundle.model, bundle.model)
    with pytest.raises(InductiveValidationError, match="parameter count"):
        daso._check_ema(
            torch.nn.Linear(2, 2),
            torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)),
        )
    with pytest.raises(InductiveValidationError, match="parameter shapes"):
        daso._check_ema(torch.nn.Linear(2, 2), torch.nn.Linear(3, 2))
    try:
        teacher_meta = torch.nn.Linear(2, 2, device="meta")
    except Exception:  # pragma: no cover - meta device may be unavailable
        teacher_meta = None
    if teacher_meta is not None:
        with pytest.raises(InductiveValidationError, match="same device"):
            daso._check_ema(torch.nn.Linear(2, 2), teacher_meta)

    teacher = torch.nn.Linear(2, 2)
    student = torch.nn.Linear(2, 2)
    daso._init_ema(student, teacher)
    before = teacher.weight.clone()
    with torch.no_grad():
        student.weight.add_(1.0)
    daso._update_ema(student, teacher, decay=0.0)
    assert not torch.allclose(before, teacher.weight)

    class _BadOut(torch.nn.Module):
        def forward(self, _x):
            return {"oops": torch.zeros((1,))}

    with pytest.raises(InductiveValidationError, match="requires feature representations"):
        daso._forward_features(
            TorchModelBundle(model=_BadOut(), optimizer=bundle.optimizer), torch.randn(2, 2)
        )


def test_daso_forward_logits_features_meta_nonmapping():
    model = _LogitsOnlyNet()
    bundle = TorchModelBundle(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), meta="bad"
    )
    logits, feats = daso._forward_logits_features(bundle, torch.randn(2, 2))
    assert torch.allclose(logits, feats)


def test_daso_forward_logits_features_forward_head_branch():
    meta = {"forward_head": lambda f: f * 2.0}
    model = _FeatOnlyMap()
    bundle = TorchModelBundle(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), meta=meta
    )
    x = torch.randn(2, 2)
    logits, feats = daso._forward_logits_features(bundle, x)
    assert torch.allclose(feats, x)
    assert torch.allclose(logits, x * 2.0)


def test_daso_forward_logits_features_mapping_no_feat():
    model = _LogitsMapNet()
    bundle = TorchModelBundle(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    logits, feats = daso._forward_logits_features(bundle, torch.randn(2, 2))
    assert torch.allclose(logits, feats)


def test_daso_forward_logits_features_forward_features_fallback_error():
    meta = {"forward_features": lambda _t: "bad"}
    model = _LogitsMapNet()
    bundle = TorchModelBundle(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), meta=meta
    )
    with pytest.raises(InductiveValidationError, match="forward_features must return"):
        daso._forward_logits_features(bundle, torch.randn(2, 2))


def test_daso_forward_logits_features_tuple_fallback_features():
    meta = {"forward_features": lambda t: t + 1.0}
    model = _TupleBadFeatNet()
    bundle = TorchModelBundle(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), meta=meta
    )
    x = torch.randn(2, 2)
    logits, feats = daso._forward_logits_features(bundle, x)
    assert logits.shape == (2, 2)
    assert torch.allclose(feats, x + 1.0)


def test_daso_forward_features_meta_errors_and_tuple_path():
    meta = {"forward_features": lambda _t: "bad"}
    model = _LogitsOnlyNet()
    bundle = TorchModelBundle(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), meta=meta
    )
    with pytest.raises(InductiveValidationError, match="forward_features must return"):
        daso._forward_features(bundle, torch.randn(2, 2))

    meta_ema = {"forward_features_ema": lambda _t: "bad"}
    model_ema = _LogitsOnlyNet()
    bundle_ema = TorchModelBundle(
        model=model_ema, optimizer=torch.optim.SGD(model_ema.parameters(), lr=0.1), meta=meta_ema
    )
    with pytest.raises(InductiveValidationError, match="forward_features_ema must return"):
        daso._forward_features(bundle_ema, torch.randn(2, 2), model_override=_LogitsOnlyNet())

    class _TupleFeatNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x, x + 1.0

    model_tuple = _TupleFeatNet()
    bundle_tuple = TorchModelBundle(
        model=model_tuple, optimizer=torch.optim.SGD(model_tuple.parameters(), lr=0.1)
    )
    x = torch.randn(2, 2)
    feats = daso._forward_features(bundle_tuple, x)
    assert torch.allclose(feats, x + 1.0)


def test_daso_fit_paths():
    data = _make_dataset()
    bundle = _make_bundle(_DASONet(), ema=True)
    spec = DASOSpec(
        model_bundle=bundle,
        batch_size=2,
        max_epochs=1,
        pretrain_steps=1,
        dist_update_period=1,
        queue_size=1,
        use_ema=True,
        dist_aware=True,
        hard_label=True,
        use_cat=True,
        detach_target=True,
    )
    DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    y_zeros = torch.zeros_like(data.y_l)
    data_zero = _make_dataset(y=y_zeros)
    spec2 = DASOSpec(
        model_bundle=_make_bundle(_DASONet()),
        batch_size=2,
        max_epochs=1,
        pretrain_steps=0,
        dist_update_period=10,
        queue_size=2,
        use_ema=False,
        dist_aware=False,
        hard_label=False,
        use_cat=False,
        detach_target=False,
    )
    method = DASOMethod(spec2).fit(data_zero, device=DeviceSpec(device="cpu"), seed=1)
    proba = method.predict_proba(data_zero.X_l)
    pred = method.predict(data_zero.X_l)
    assert int(proba.shape[0]) == int(data_zero.X_l.shape[0])
    assert int(pred.shape[0]) == int(data_zero.X_l.shape[0])


def test_daso_fit_row_mismatch_raises(monkeypatch):
    base = _make_dataset()
    mismatch = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s[:-1],
    )
    monkeypatch.setattr(daso, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(daso, "ensure_1d_labels_torch", lambda y, name="y_l": y)
    spec = _make_base_spec(_DASONet())
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        DASOMethod(spec).fit(mismatch, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (_BadLogits1D(), "Model logits must be 2D"),
        (_BadBatchNet(), "Concatenated logits batch size does not match inputs"),
    ],
)
def test_daso_fit_use_cat_output_errors(model, match):
    data = _make_dataset()
    spec = _make_base_spec(model, use_cat=True)
    with pytest.raises(InductiveValidationError, match=match):
        DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_daso_fit_detach_target_no_grad_path():
    data = _make_dataset()
    spec = _make_base_spec(_DASONet(), use_cat=False, detach_target=True)
    DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    ("class_counts", "feat_dims", "match"),
    [
        ([2, 3, 2], [2, 2, 2], "Unlabeled logits shape mismatch"),
        ([3, 2, 2], [2, 2, 2], "Logits must agree on class dimension"),
        ([2, 2, 2], [2, 2, 3], "Unlabeled feature shape mismatch"),
    ],
)
def test_daso_fit_shape_mismatch_errors(class_counts, feat_dims, match):
    data = _make_dataset()
    spec = _make_base_spec(_VarShapeNet(class_counts, feat_dims), use_cat=False)
    with pytest.raises(InductiveValidationError, match=match):
        DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_daso_fit_logits_dim_error():
    data = _make_dataset()
    spec = _make_base_spec(_BadLogitsMap1D(), use_cat=False)
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_daso_fit_y_l_out_of_range():
    base = _make_dataset()
    data = InductiveDataset(
        X_l=base.X_l,
        y_l=torch.full_like(base.y_l, 2),
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    spec = _make_base_spec(_DASONet(), use_cat=False)
    with pytest.raises(InductiveValidationError, match="y_l labels must be within"):
        DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_daso_fit_prototype_dim_mismatch():
    base = _make_dataset()

    def _bad_feat(_x):
        return torch.zeros((int(_x.shape[0]), 3), device=_x.device)

    meta = {"forward_features_ema": _bad_feat}
    model = _DASONet()
    bundle = TorchModelBundle(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), meta=meta
    )
    spec = DASOSpec(
        model_bundle=bundle,
        batch_size=4,
        max_epochs=1,
        pretrain_steps=0,
        dist_update_period=1,
        queue_size=1,
        use_ema=True,
        dist_aware=True,
        p_cutoff=0.0,
    )
    with pytest.raises(InductiveValidationError, match="Prototype feature dimension mismatch"):
        DASOMethod(spec).fit(base, device=DeviceSpec(device="cpu"), seed=0)


def test_daso_fit_mask_empty_unsup_loss(monkeypatch):
    data = _make_dataset()

    def _empty_indices(*_args, **kwargs):
        device = kwargs.get("device")
        return [torch.tensor([], dtype=torch.long, device=device)]

    monkeypatch.setattr(daso, "cycle_batch_indices", _empty_indices)
    spec = _make_base_spec(_DASONet(), use_cat=False)
    DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_daso_fit_dist_accum_skipped(monkeypatch):
    data = _make_dataset()
    monkeypatch.setattr(torch, "zeros_like", lambda _t: None)
    spec = _make_base_spec(_DASONet(), use_cat=False)
    DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_daso_predict_proba_keeps_eval_state():
    bundle = _make_bundle(_DASONet())
    bundle.model.eval()
    method = DASOMethod()
    method._bundle = bundle
    method._backend = "torch"
    proba = method.predict_proba(torch.zeros((2, 2)))
    assert proba.shape == (2, 2)
    assert bundle.model.training is False


@pytest.mark.parametrize(
    "data,match",
    [
        (None, "data must not be None"),
        (make_numpy_dataset(), "requires torch tensors"),
    ],
)
def test_daso_fit_data_errors(data, match):
    spec = DASOSpec(model_bundle=_make_bundle(_DASONet()))
    with pytest.raises(InductiveValidationError, match=match):
        DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_daso_fit_dataset_validation_errors():
    base = _make_dataset()
    bundle = _make_bundle(_DASONet())

    no_u = InductiveDataset(X_l=base.X_l, y_l=base.y_l, X_u_w=None, X_u_s=None)
    with pytest.raises(InductiveValidationError, match="requires X_u_w and X_u_s"):
        DASOMethod(DASOSpec(model_bundle=bundle)).fit(no_u, device=DeviceSpec(device="cpu"))

    empty_l = InductiveDataset(
        X_l=base.X_l[:0],
        y_l=base.y_l[:0],
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="y_l must be non-empty"):
        DASOMethod(DASOSpec(model_bundle=bundle)).fit(empty_l, device=DeviceSpec(device="cpu"))

    empty_u = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w[:0],
        X_u_s=base.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        DASOMethod(DASOSpec(model_bundle=bundle)).fit(empty_u, device=DeviceSpec(device="cpu"))

    mismatch = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s[:-1],
    )
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        DASOMethod(DASOSpec(model_bundle=bundle)).fit(mismatch, device=DeviceSpec(device="cpu"))

    bad_dtype = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l.to(dtype=torch.int32),
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="y_l must be int64"):
        DASOMethod(DASOSpec(model_bundle=bundle)).fit(bad_dtype, device=DeviceSpec(device="cpu"))


def test_daso_fit_empty_x_l_raises(monkeypatch):
    base = _make_dataset()
    empty_l = InductiveDataset(
        X_l=base.X_l[:0],
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    monkeypatch.setattr(daso, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(daso, "ensure_1d_labels_torch", lambda y, name="y_l": y)
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        DASOMethod(DASOSpec(model_bundle=_make_bundle(_DASONet()))).fit(
            empty_l, device=DeviceSpec(device="cpu")
        )


def test_daso_fit_empty_unlabeled_raises(monkeypatch):
    base = _make_dataset()
    empty_u = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w[:0],
        X_u_s=base.X_u_s[:0],
    )
    monkeypatch.setattr(daso, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(daso, "ensure_1d_labels_torch", lambda y, name="y_l": y)
    with pytest.raises(InductiveValidationError, match="X_u_w and X_u_s must be non-empty"):
        DASOMethod(DASOSpec(model_bundle=_make_bundle(_DASONet()))).fit(
            empty_u, device=DeviceSpec(device="cpu")
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model_bundle": None}, "model_bundle must be provided"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"max_epochs": 0}, "max_epochs must be >= 1"),
        ({"lambda_u": -1.0}, "lambda_u must be >= 0"),
        ({"lambda_align": -1.0}, "lambda_align must be >= 0"),
        ({"p_cutoff": -0.1}, "p_cutoff must be in"),
        ({"t_proto": 0.0}, "t_proto must be > 0"),
        ({"t_dist": 0.0}, "t_dist must be > 0"),
        ({"queue_size": 0}, "queue_size must be >= 1"),
        ({"pretrain_steps": -1}, "pretrain_steps must be >= 0"),
        ({"dist_update_period": 0}, "dist_update_period must be >= 1"),
        ({"ema_decay": 2.0}, "ema_decay must be in"),
        ({"interp_alpha": 2.0}, "interp_alpha must be in"),
    ],
)
def test_daso_fit_spec_validation_errors(overrides, match):
    data = _make_dataset()
    base = DASOSpec(model_bundle=_make_bundle(_DASONet()))
    spec = DASOSpec(**{**base.__dict__, **overrides})
    with pytest.raises(InductiveValidationError, match=match):
        DASOMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_daso_predict_errors():
    method = DASOMethod()
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(torch.zeros((2, 2)))

    method._bundle = _make_bundle(_DASONet())
    with pytest.raises(InductiveValidationError, match="requires torch tensors"):
        method.predict_proba(np.zeros((2, 2)))

    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="requires torch.Tensor"):
        method.predict_proba(np.zeros((2, 2)))

    method._bundle = _make_bundle(_BadLogits1D())
    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method.predict_proba(torch.zeros((2, 2)))


def test_daso_predict_proba_dict_inputs():
    class _DictNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["x"]
            return self.fc(x)

    method = DASOMethod()
    method._bundle = _make_bundle(_DictNet())
    method._backend = "torch"
    X = {"x": torch.zeros((2, 2), dtype=torch.float32)}
    proba = method.predict_proba(X)
    assert proba.shape[0] == 2


def test_daso_predict_proba_empty_dict():
    class _DictNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["x"]
            return self.fc(x)

    method = DASOMethod()
    method._bundle = _make_bundle(_DictNet())
    method._backend = "torch"
    X = {"x": torch.zeros((0, 2), dtype=torch.float32)}
    proba = method.predict_proba(X)
    assert proba.shape[0] == 0
