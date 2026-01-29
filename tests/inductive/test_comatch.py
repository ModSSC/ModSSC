from __future__ import annotations

import sys

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

import modssc.inductive.methods.comatch as comatch
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.comatch import CoMatchMethod, CoMatchSpec
from modssc.inductive.types import DeviceSpec, InductiveDataset

from .conftest import make_numpy_dataset, make_torch_ssl_dataset


class _CoMatchNet(torch.nn.Module):
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


class _BadCatLogits1DNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.zeros((int(x.shape[0]),), device=x.device)
        feat = torch.zeros((int(x.shape[0]), 2), device=x.device)
        return {"logits": logits, "feat": feat}


class _BadCatBatchNet(torch.nn.Module):
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


class _BadLogitsMap1D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.zeros((int(x.shape[0]),), device=x.device)
        feat = torch.zeros((int(x.shape[0]), 2), device=x.device)
        return {"logits": logits, "feat": feat}


class _BadFeatMap1D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.zeros((int(x.shape[0]), 2), device=x.device)
        feat = torch.zeros((int(x.shape[0]),), device=x.device)
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


def _make_base_spec(model: torch.nn.Module, **overrides) -> CoMatchSpec:
    base = CoMatchSpec(
        model_bundle=_make_bundle(model),
        batch_size=4,
        max_epochs=1,
        queue_size=0,
        da_len=0,
        dist_align=False,
        min_queue_fill=0,
        p_cutoff=0.0,
        contrast_p_cutoff=0.0,
    )
    return CoMatchSpec(**{**base.__dict__, **overrides})


def _make_bundle(model: torch.nn.Module) -> TorchModelBundle:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TorchModelBundle(model=model, optimizer=optimizer)


def _make_comatch_dataset(*, with_views: bool = True) -> InductiveDataset:
    base = make_torch_ssl_dataset()
    views = None
    if with_views:
        views = {"X_u_s0": base.X_u_s, "X_u_s1": base.X_u_s + 0.01}
    return InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views=views,
    )


def test_comatch_extract_logits_and_features():
    x = torch.randn(2, 2)
    model = _CoMatchNet()
    logits, feat = comatch._extract_logits_and_features(model(x))
    assert logits.shape == (2, 2)
    assert feat.shape == (2, 2)

    tup_logits, tup_feat = comatch._extract_logits_and_features(_TupleNet()(x))
    assert tup_logits.shape == (2, 2)
    assert tup_feat.shape == (2, 2)

    with pytest.raises(InductiveValidationError, match="keys 'logits' and 'feat'"):
        comatch._extract_logits_and_features({"logits": x})
    with pytest.raises(InductiveValidationError, match="must be torch.Tensor"):
        comatch._extract_logits_and_features({"logits": "bad", "feat": x})
    with pytest.raises(InductiveValidationError, match="must be torch.Tensor"):
        comatch._extract_logits_and_features((x, "bad"))
    with pytest.raises(InductiveValidationError, match="requires model outputs"):
        comatch._extract_logits_and_features(x)
    with pytest.raises(InductiveValidationError, match="mapping or tuple"):
        comatch._extract_logits_and_features(123)


def test_comatch_helper_losses_and_graph():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    probs = comatch._compute_prob(logits, temperature=0.5)
    assert probs.shape == (2, 2)
    with pytest.raises(InductiveValidationError, match="temperature must be > 0"):
        comatch._compute_prob(logits, temperature=0.0)

    feats = torch.randn(2, 2)
    normed = comatch._l2_normalize(feats)
    assert normed.shape == feats.shape

    q = comatch._build_pseudo_graph(probs, threshold=0.0)
    q_thresh = comatch._build_pseudo_graph(probs, threshold=0.9)
    assert q.shape == probs.shape
    assert q_thresh.shape == probs.shape

    loss = comatch._contrastive_loss(normed, normed, q, temperature=0.5)
    assert float(loss) >= 0.0

    one_hot = comatch._one_hot(torch.tensor([0, 1]), n_classes=2)
    assert one_hot.shape == (2, 2)


def test_comatch_memory_and_da_helpers():
    method = CoMatchMethod(
        CoMatchSpec(queue_size=3, min_queue_fill=2, smoothing_alpha=0.5, da_len=2)
    )
    feats = torch.randn(4, 2)
    probs = torch.softmax(torch.randn(4, 2), dim=1)

    method._init_memory(feat_dim=2, n_classes=2, device=feats.device)
    method._update_memory(feats[:0], probs[:0])
    method._update_memory(feats, probs)
    method._update_memory(feats[:2], probs[:2])

    method._queue_count = 0
    assert torch.allclose(method._memory_smooth(probs[:2], feats[:2]), probs[:2])
    method._queue_count = 1
    assert torch.allclose(method._memory_smooth(probs[:2], feats[:2]), probs[:2])
    method.spec = CoMatchSpec(queue_size=3, min_queue_fill=0, smoothing_alpha=1.0, da_len=2)
    assert torch.allclose(method._memory_smooth(probs[:2], feats[:2]), probs[:2])

    method.spec = CoMatchSpec(queue_size=3, min_queue_fill=0, smoothing_alpha=0.5, da_len=2)
    method._queue_count = 2
    smoothed = method._memory_smooth(probs[:2], feats[:2])
    assert smoothed.shape == probs[:2].shape

    method_no_da = CoMatchMethod(CoMatchSpec(dist_align=False))
    aligned = method_no_da._dist_align(probs[:2], probs_l=probs[:2])
    assert torch.allclose(aligned, probs[:2])

    method_da_len = CoMatchMethod(CoMatchSpec(dist_align=True, da_len=0))
    aligned = method_da_len._dist_align(probs[:2], probs_l=probs[:2])
    assert torch.allclose(aligned, probs[:2])

    method_dist = CoMatchMethod(CoMatchSpec(dist_align=True, dist_uniform=False, da_len=2))
    with pytest.raises(InductiveValidationError, match="requires labeled probabilities"):
        method_dist._dist_align(probs[:2], probs_l=None)
    aligned = method_dist._dist_align(probs[:2], probs_l=probs[:2])
    assert aligned.shape == probs[:2].shape

    method_no_queue = CoMatchMethod(CoMatchSpec(queue_size=0, da_len=0))
    method_no_queue._init_memory(feat_dim=2, n_classes=2, device=feats.device)
    assert method_no_queue._queue_feats is None
    method_no_queue._init_da(n_classes=2, device=feats.device)
    assert method_no_queue._da_queue is None

    method_wrap = CoMatchMethod(CoMatchSpec(queue_size=3, min_queue_fill=0, da_len=0))
    method_wrap._init_memory(feat_dim=2, n_classes=2, device=feats.device)
    method_wrap._queue_ptr = 2
    method_wrap._queue_count = 2
    method_wrap._update_memory(feats[:2], probs[:2])
    assert int(method_wrap._queue_ptr) == 1


def test_comatch_fit_and_predict_variants():
    data = _make_comatch_dataset()
    bundle = _make_bundle(_CoMatchNet())
    spec = CoMatchSpec(
        model_bundle=bundle,
        batch_size=2,
        max_epochs=1,
        use_cat=True,
        detach_target=True,
        hard_label=True,
        dist_align=True,
        dist_uniform=True,
        queue_size=4,
        min_queue_fill=0,
        da_len=2,
        p_cutoff=0.0,
        contrast_p_cutoff=0.0,
    )
    method = CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    pred = method.predict(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    assert int(pred.shape[0]) == int(data.X_l.shape[0])

    bundle2 = _make_bundle(_CoMatchNet())
    spec2 = CoMatchSpec(
        model_bundle=bundle2,
        batch_size=2,
        max_epochs=1,
        use_cat=False,
        detach_target=False,
        hard_label=False,
        dist_align=True,
        dist_uniform=False,
        queue_size=0,
        da_len=2,
        p_cutoff=1.0,
        contrast_p_cutoff=0.5,
    )
    CoMatchMethod(spec2).fit(data, device=DeviceSpec(device="cpu"), seed=1)


def test_comatch_fit_uses_view_fallbacks():
    base = _make_comatch_dataset()
    data = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=None,
        views={"X_u_s0": base.X_u_s, "X_u_s2": base.X_u_s + 0.01},
    )
    spec = _make_base_spec(_CoMatchNet(), use_cat=False)
    method = CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=2)
    assert method._bundle is not None


def test_comatch_fit_uses_view_fallbacks_second_key():
    base = _make_comatch_dataset()
    data = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=None,
        views={"X_u_s_0": base.X_u_s, "X_u_s_2": base.X_u_s + 0.02},
    )
    spec = _make_base_spec(_CoMatchNet(), use_cat=False)
    CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=3)


def test_comatch_fit_missing_strong_view_in_views():
    base = _make_comatch_dataset()
    data = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=None,
        views={"other": base.X_u_s},
    )
    spec = _make_base_spec(_CoMatchNet(), use_cat=False)
    with pytest.raises(InductiveValidationError, match="requires X_u_s"):
        CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_fit_missing_second_strong_without_views():
    base = _make_comatch_dataset(with_views=False)
    data = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views=None,
    )
    spec = _make_base_spec(_CoMatchNet(), use_cat=False)
    with pytest.raises(InductiveValidationError, match="requires a second strong"):
        CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (_BadCatLogits1DNet(), "Model logits/feat must be 2D tensors"),
        (_BadCatBatchNet(), "Concatenated logits/feat batch size does not match inputs"),
    ],
)
def test_comatch_fit_use_cat_output_errors(model, match):
    data = _make_comatch_dataset()
    spec = _make_base_spec(model, use_cat=True)
    with pytest.raises(InductiveValidationError, match=match):
        CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_fit_logits_dim_error():
    data = _make_comatch_dataset()
    spec = _make_base_spec(_BadLogitsMap1D(), use_cat=False)
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_fit_feat_dim_error():
    data = _make_comatch_dataset()
    spec = _make_base_spec(_BadFeatMap1D(), use_cat=False)
    with pytest.raises(InductiveValidationError, match="Model feats must be 2D"):
        CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    ("class_counts", "feat_dims", "match"),
    [
        ([2, 2, 2, 3], [2, 2, 2, 2], "Unlabeled logits shape mismatch"),
        ([3, 2, 2, 2], [2, 2, 2, 2], "Logits must agree on class dimension"),
        ([2, 2, 2, 2], [2, 2, 3, 2], "Feature dims must match across unlabeled views"),
        ([2, 2, 2, 2], [3, 2, 2, 2], "Feature dims must match for labeled/unlabeled"),
    ],
)
def test_comatch_fit_shape_mismatch_errors(class_counts, feat_dims, match):
    data = _make_comatch_dataset()
    spec = _make_base_spec(_VarShapeNet(class_counts, feat_dims), use_cat=False)
    with pytest.raises(InductiveValidationError, match=match):
        CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_fit_y_l_out_of_range():
    base = _make_comatch_dataset()
    data = InductiveDataset(
        X_l=base.X_l,
        y_l=torch.full_like(base.y_l, 2),
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views=base.views,
    )
    spec = _make_base_spec(_CoMatchNet(), use_cat=False)
    with pytest.raises(InductiveValidationError, match="y_l labels must be within"):
        CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_fit_mask_empty_unsup_loss(monkeypatch):
    data = _make_comatch_dataset()

    def _empty_indices(*_args, **kwargs):
        device = kwargs.get("device")
        return [torch.tensor([], dtype=torch.long, device=device)]

    monkeypatch.setattr(comatch, "cycle_batch_indices", _empty_indices)
    spec = _make_base_spec(_CoMatchNet(), use_cat=False)
    CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_predict_proba_keeps_eval_state():
    bundle = _make_bundle(_CoMatchNet())
    bundle.model.eval()
    method = CoMatchMethod()
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
def test_comatch_fit_data_errors(data, match):
    spec = CoMatchSpec(model_bundle=_make_bundle(_CoMatchNet()))
    method = CoMatchMethod(spec)
    with pytest.raises(InductiveValidationError, match=match):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_fit_dataset_validation_errors():
    base = _make_comatch_dataset()
    bundle = _make_bundle(_CoMatchNet())

    no_u_w = InductiveDataset(X_l=base.X_l, y_l=base.y_l, X_u_w=None, X_u_s=base.X_u_s)
    with pytest.raises(InductiveValidationError, match="requires X_u_w"):
        CoMatchMethod(CoMatchSpec(model_bundle=bundle)).fit(no_u_w, device=DeviceSpec(device="cpu"))

    no_s0 = InductiveDataset(X_l=base.X_l, y_l=base.y_l, X_u_w=base.X_u_w, X_u_s=None)
    with pytest.raises(InductiveValidationError, match="requires X_u_s"):
        CoMatchMethod(CoMatchSpec(model_bundle=bundle)).fit(no_s0, device=DeviceSpec(device="cpu"))

    no_s1 = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views={"X_u_s0": base.X_u_s},
    )
    with pytest.raises(InductiveValidationError, match="requires a second strong"):
        CoMatchMethod(CoMatchSpec(model_bundle=bundle)).fit(no_s1, device=DeviceSpec(device="cpu"))

    empty_l = InductiveDataset(
        X_l=base.X_l[:0],
        y_l=base.y_l[:0],
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views=base.views,
    )
    with pytest.raises(InductiveValidationError, match="y_l must be non-empty"):
        CoMatchMethod(CoMatchSpec(model_bundle=bundle)).fit(
            empty_l, device=DeviceSpec(device="cpu")
        )

    empty_u = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w[:0],
        X_u_s=base.X_u_s,
        views=base.views,
    )
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        CoMatchMethod(CoMatchSpec(model_bundle=bundle)).fit(
            empty_u, device=DeviceSpec(device="cpu")
        )

    empty_s = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s[:0],
        views=base.views,
    )
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        CoMatchMethod(CoMatchSpec(model_bundle=bundle)).fit(
            empty_s, device=DeviceSpec(device="cpu")
        )

    mismatch = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views={"X_u_s0": base.X_u_s, "X_u_s1": base.X_u_s[:-1]},
    )
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        CoMatchMethod(CoMatchSpec(model_bundle=bundle)).fit(
            mismatch, device=DeviceSpec(device="cpu")
        )

    bad_dim = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views={"X_u_s0": base.X_u_s, "X_u_s1": torch.zeros((4, 3))},
    )
    with pytest.raises(InductiveValidationError, match="same feature size"):
        CoMatchMethod(CoMatchSpec(model_bundle=bundle)).fit(
            bad_dim, device=DeviceSpec(device="cpu")
        )

    bad_dtype = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l.to(dtype=torch.int32),
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views=base.views,
    )
    with pytest.raises(InductiveValidationError, match="y_l must be int64"):
        CoMatchMethod(CoMatchSpec(model_bundle=bundle)).fit(
            bad_dtype, device=DeviceSpec(device="cpu")
        )


def test_comatch_fit_empty_x_l_raises(monkeypatch):
    base = _make_comatch_dataset()
    empty_l = InductiveDataset(
        X_l=base.X_l[:0],
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views=base.views,
    )
    monkeypatch.setattr(comatch, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(comatch, "ensure_1d_labels_torch", lambda y, name="y_l": y)
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        CoMatchMethod(CoMatchSpec(model_bundle=_make_bundle(_CoMatchNet()))).fit(
            empty_l, device=DeviceSpec(device="cpu")
        )


def test_comatch_fit_empty_unlabeled_raises(monkeypatch):
    base = _make_comatch_dataset()
    empty_u = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w[:0],
        X_u_s=base.X_u_s,
        views=base.views,
    )
    monkeypatch.setattr(comatch, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(comatch, "ensure_1d_labels_torch", lambda y, name="y_l": y)
    with pytest.raises(InductiveValidationError, match="X_u_w must be non-empty"):
        CoMatchMethod(CoMatchSpec(model_bundle=_make_bundle(_CoMatchNet()))).fit(
            empty_u, device=DeviceSpec(device="cpu")
        )

    empty_s = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s[:0],
        views={"X_u_s0": base.X_u_s[:0], "X_u_s1": base.X_u_s[:0]},
    )
    with pytest.raises(InductiveValidationError, match="X_u_s0/X_u_s1 must be non-empty"):
        CoMatchMethod(CoMatchSpec(model_bundle=_make_bundle(_CoMatchNet()))).fit(
            empty_s, device=DeviceSpec(device="cpu")
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model_bundle": None}, "model_bundle must be provided"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"max_epochs": 0}, "max_epochs must be >= 1"),
        ({"lambda_u": -1.0}, "lambda_u must be >= 0"),
        ({"lambda_c": -1.0}, "lambda_c must be >= 0"),
        ({"p_cutoff": -0.1}, "p_cutoff must be in"),
        ({"temperature": 0.0}, "temperature must be > 0"),
        ({"contrast_p_cutoff": -0.1}, "contrast_p_cutoff must be in"),
        ({"smoothing_alpha": 1.5}, "smoothing_alpha must be in"),
        ({"queue_size": -1}, "queue_size must be >= 0"),
        ({"min_queue_fill": -1}, "min_queue_fill must be >= 0"),
        ({"queue_size": 1, "min_queue_fill": 2}, "min_queue_fill must be <= queue_size"),
        ({"da_len": -1}, "da_len must be >= 0"),
    ],
)
def test_comatch_fit_spec_validation_errors(overrides, match):
    data = _make_comatch_dataset()
    base = CoMatchSpec(model_bundle=_make_bundle(_CoMatchNet()))
    spec = CoMatchSpec(**{**base.__dict__, **overrides})
    with pytest.raises(InductiveValidationError, match=match):
        CoMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_predict_errors():
    method = CoMatchMethod()
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(torch.zeros((2, 2)))

    method._bundle = _make_bundle(_CoMatchNet())
    with pytest.raises(InductiveValidationError, match="requires torch tensors"):
        method.predict_proba(np.zeros((2, 2), dtype=np.float32))

    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="requires torch.Tensor"):
        method.predict_proba(np.zeros((2, 2), dtype=np.float32))

    method._bundle = _make_bundle(_BadLogits1D())
    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method.predict_proba(torch.zeros((2, 2)))


def test_comatch_predict_proba_dict_inputs():
    class _DictNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["x"]
            return self.fc(x)

    method = CoMatchMethod()
    method._bundle = _make_bundle(_DictNet())
    method._backend = "torch"
    X = {"x": torch.zeros((2, 2), dtype=torch.float32)}
    proba = method.predict_proba(X)
    assert proba.shape[0] == 2


def test_comatch_predict_proba_empty_dict():
    class _DictNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["x"]
            return self.fc(x)

    method = CoMatchMethod()
    method._bundle = _make_bundle(_DictNet())
    method._backend = "torch"
    X = {"x": torch.zeros((0, 2), dtype=torch.float32)}
    proba = method.predict_proba(X)
    assert proba.shape[0] == 0


def test_comatch_fit_len_zero_dict(monkeypatch):
    x_l = {"foo": torch.zeros((1, 2))}
    y_l = torch.tensor([0], dtype=torch.int64)
    x_u_w = {"x": torch.zeros((1, 2))}
    views = {
        "X_u_s0": {"x": torch.zeros((1, 2))},
        "X_u_s1": {"x": torch.zeros((1, 2))},
    }
    data = InductiveDataset(X_l=x_l, y_l=y_l, X_u_w=x_u_w, views=views)
    monkeypatch.setattr(comatch, "ensure_torch_data", lambda d, device: d)
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        CoMatchMethod().fit(data, device=DeviceSpec(device="cpu"), seed=0)


def _install_fake_tg_utils(monkeypatch, *, with_subgraph: bool):
    import types

    utils = types.ModuleType("torch_geometric.utils")
    if with_subgraph:

        def subgraph(idx, edge_index, relabel_nodes=True, num_nodes=None):
            return edge_index, None

        utils.subgraph = subgraph
    tg = types.ModuleType("torch_geometric")
    tg.utils = utils
    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.utils", utils)


def test_comatch_fit_slice_dict_with_subgraph(monkeypatch):
    x_l = {"x": torch.zeros((2, 2)), "edge_index": torch.tensor([[0], [1]])}
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u_w = {
        "x": torch.zeros((2, 2)),
        "edge_index": torch.tensor([[0], [1]]),
        "mask": torch.tensor([1, 0]),
        "meta": "keep",
    }
    views = {
        "X_u_s0": {
            "x": torch.zeros((2, 2)),
            "edge_index": torch.tensor([[0], [1]]),
            "mask": torch.tensor([0, 1]),
            "meta": "keep",
        },
        "X_u_s1": {
            "x": torch.zeros((2, 2)),
            "edge_index": torch.tensor([[0], [1]]),
            "mask": torch.tensor([1, 1]),
            "meta": "keep",
        },
    }
    data = InductiveDataset(X_l=x_l, y_l=y_l, X_u_w=x_u_w, views=views)
    monkeypatch.setattr(comatch, "ensure_torch_data", lambda d, device: d)
    monkeypatch.setattr(comatch, "cycle_batches", lambda *_args, **_kwargs: iter([(x_l["x"], y_l)]))
    monkeypatch.setattr(
        comatch, "cycle_batch_indices", lambda *_args, **_kwargs: iter([torch.tensor([0])])
    )
    monkeypatch.setattr(
        comatch,
        "_extract_logits_and_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    _install_fake_tg_utils(monkeypatch, with_subgraph=True)

    spec = _make_base_spec(_CoMatchNet())
    method = CoMatchMethod(spec)
    with pytest.raises(RuntimeError, match="stop"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_fit_slice_dict_without_subgraph(monkeypatch):
    x_l = {"x": torch.zeros((2, 2)), "edge_index": torch.tensor([[0], [1]])}
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u_w = {
        "x": torch.zeros((2, 2)),
        "edge_index": torch.tensor([[0], [1]]),
        "mask": torch.tensor([1, 0]),
        "meta": "keep",
    }
    views = {
        "X_u_s0": {
            "x": torch.zeros((2, 2)),
            "edge_index": torch.tensor([[0], [1]]),
            "mask": torch.tensor([0, 1]),
            "meta": "keep",
        },
        "X_u_s1": {
            "x": torch.zeros((2, 2)),
            "edge_index": torch.tensor([[0], [1]]),
            "mask": torch.tensor([1, 1]),
            "meta": "keep",
        },
    }
    data = InductiveDataset(X_l=x_l, y_l=y_l, X_u_w=x_u_w, views=views)
    monkeypatch.setattr(comatch, "ensure_torch_data", lambda d, device: d)
    monkeypatch.setattr(comatch, "cycle_batches", lambda *_args, **_kwargs: iter([(x_l["x"], y_l)]))
    monkeypatch.setattr(
        comatch, "cycle_batch_indices", lambda *_args, **_kwargs: iter([torch.tensor([0])])
    )
    monkeypatch.setattr(
        comatch,
        "_extract_logits_and_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    _install_fake_tg_utils(monkeypatch, with_subgraph=False)

    spec = _make_base_spec(_CoMatchNet())
    method = CoMatchMethod(spec)
    with pytest.raises(RuntimeError, match="stop"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_comatch_fit_slice_dict_missing_x_and_edge_index(monkeypatch):
    class _NoXDict(dict):
        @property
        def shape(self):
            return self["feat"].shape

    class _FeatModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = torch.nn.Linear(2, 2, bias=False)
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["feat"]
            feat = self.feat(x)
            logits = self.fc(feat)
            return {"logits": logits, "feat": feat}

    X_l = torch.zeros((2, 2))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    x_u_w = _NoXDict({"feat": torch.zeros((2, 2))})
    views = {
        "X_u_s0": _NoXDict({"feat": torch.zeros((2, 2))}),
        "X_u_s1": _NoXDict({"feat": torch.zeros((2, 2))}),
    }
    data = InductiveDataset(X_l=X_l, y_l=y_l, X_u_w=x_u_w, views=views)

    monkeypatch.setattr(comatch, "ensure_torch_data", lambda d, device: d)
    monkeypatch.setattr(comatch, "cycle_batches", lambda *_args, **_kwargs: iter([(X_l, y_l)]))
    monkeypatch.setattr(
        comatch, "cycle_batch_indices", lambda *_args, **_kwargs: iter([torch.tensor([0])])
    )
    monkeypatch.setattr(
        comatch,
        "_extract_logits_and_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )

    spec = _make_base_spec(_FeatModel(), use_cat=False)
    method = CoMatchMethod(spec)
    with pytest.raises(RuntimeError, match="stop"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
