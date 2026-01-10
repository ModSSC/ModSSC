from __future__ import annotations

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

import modssc.inductive.methods.trinet as trinet
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.trinet import TriNetMethod, TriNetSpec
from modssc.inductive.types import DeviceSpec, InductiveDataset

from .conftest import SimpleNet, make_numpy_dataset, make_torch_dataset


def _make_bundle(model: torch.nn.Module, *, meta: dict | None = None) -> TorchModelBundle:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TorchModelBundle(model=model, optimizer=optimizer, meta=meta)


def _make_shared_bundle() -> TorchModelBundle:
    model = SimpleNet(in_dim=2, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TorchModelBundle(model=model, optimizer=optimizer)


def _make_head_bundle(
    *, in_dim: int = 2, n_classes: int = 2, bias: torch.Tensor | None = None
) -> TorchModelBundle:
    model = torch.nn.Linear(in_dim, n_classes)
    if bias is not None:
        with torch.no_grad():
            model.weight.zero_()
            model.bias.copy_(bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TorchModelBundle(model=model, optimizer=optimizer)


class _BadMapping(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):
        return {"logits": self.fc(x)}


class _BadLogits1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((int(x.shape[0]),), device=x.device)


class _BadBatch(torch.nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.n_classes = int(n_classes)

    def forward(self, x):
        batch = max(0, int(x.shape[0]) - 1)
        return torch.zeros((batch, self.n_classes), device=x.device)


class _DropoutHead(torch.nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 2):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc = torch.nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


def test_trinet_forward_helpers():
    x = torch.randn(2, 2)

    model = torch.nn.Linear(2, 2)
    bundle = _make_bundle(model)
    out = trinet._forward_shared(bundle, x)
    assert out.shape == (2, 2)

    meta_bundle = _make_bundle(model, meta={"forward_features": lambda t: t + 1.0})
    out_meta = trinet._forward_shared(meta_bundle, x)
    assert torch.allclose(out_meta, x + 1.0)

    shared = _make_shared_bundle()
    out_feat = trinet._forward_shared(shared, x)
    assert out_feat.shape[0] == int(x.shape[0])

    bad = _make_bundle(_BadMapping())
    with pytest.raises(InductiveValidationError, match="shared_bundle.model must return"):
        trinet._forward_shared(bad, x)

    head_bundle = _make_bundle(torch.nn.Linear(2, 2), meta={"head": lambda t: t})
    logits = trinet._forward_head(head_bundle, x)
    assert logits.shape == x.shape

    logits2 = trinet._forward_head(_make_bundle(torch.nn.Linear(2, 2)), x)
    assert logits2.shape == (2, 2)


def test_trinet_forward_helpers_meta_without_callables():
    x = torch.randn(2, 2)
    model = torch.nn.Linear(2, 2)
    bundle = _make_bundle(model, meta=object())
    out = trinet._forward_shared(bundle, x)
    assert out.shape == (2, 2)

    head_bundle = _make_bundle(model, meta=object())
    logits = trinet._forward_head(head_bundle, x)
    assert logits.shape == (2, 2)


def test_trinet_math_helpers():
    labels = torch.tensor([0, 1], dtype=torch.int64)
    one_hot = trinet._one_hot(labels, n_classes=2)
    assert one_hot.dtype == torch.float32

    logits = torch.zeros((2, 2))
    loss = trinet._soft_cross_entropy(logits, one_hot)
    assert float(loss) >= 0.0

    gen = torch.Generator().manual_seed(0)
    smeared = trinet._output_smearing(labels, n_classes=2, std=0.0, generator=gen)
    assert torch.allclose(smeared, one_hot)

    gen = torch.Generator().manual_seed(1)
    smeared2 = trinet._output_smearing(labels, n_classes=2, std=0.1, generator=gen)
    assert smeared2.shape == (2, 2)
    assert torch.allclose(smeared2.sum(dim=1), torch.ones(2))

    X_u = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    pool_full = trinet._sample_pool(X_u, n_pool=4, generator=torch.Generator().manual_seed(2))
    assert pool_full.shape == (4, 2)

    pool_small = trinet._sample_pool(X_u, n_pool=2, generator=torch.Generator().manual_seed(3))
    assert pool_small.shape == (2, 2)

    X_meta = torch.empty((4, 2), device="meta")
    pool_meta = trinet._sample_pool(X_meta, n_pool=2, generator=torch.Generator().manual_seed(4))
    assert pool_meta.device.type == "meta"
    assert pool_meta.shape == (2, 2)


def test_trinet_dropout_filter_paths():
    shared = _make_shared_bundle()
    head_j = _make_bundle(_DropoutHead())
    head_h = _make_bundle(_DropoutHead())
    X = torch.randn(3, 2)
    labels = torch.zeros(3, dtype=torch.int64)

    mask = trinet._dropout_filter(
        X,
        labels,
        shared_bundle=shared,
        head_j=head_j,
        head_h=head_h,
        passes=2,
        drop_fraction=1.0,
        batch_size=2,
        freeze_bn=True,
    )
    assert mask.shape == (3,)

    empty_mask = trinet._dropout_filter(
        X[:0],
        labels[:0],
        shared_bundle=shared,
        head_j=head_j,
        head_h=head_h,
        passes=1,
        drop_fraction=0.5,
        batch_size=1,
        freeze_bn=False,
    )
    assert empty_mask.shape == (0,)

    bad_head = _make_bundle(_BadLogits1D())
    with pytest.raises(InductiveValidationError, match="Head logits must be 2D"):
        trinet._dropout_filter(
            X,
            labels,
            shared_bundle=shared,
            head_j=bad_head,
            head_h=head_h,
            passes=2,
            drop_fraction=1.0,
            batch_size=2,
            freeze_bn=False,
        )

    mismatch_head = _make_head_bundle(n_classes=3)
    with pytest.raises(InductiveValidationError, match="Head logits must have the same shape"):
        trinet._dropout_filter(
            X,
            labels,
            shared_bundle=shared,
            head_j=head_j,
            head_h=mismatch_head,
            passes=2,
            drop_fraction=1.0,
            batch_size=2,
            freeze_bn=False,
        )


def test_trinet_label_unlabeled_paths(monkeypatch):
    shared = _make_shared_bundle()
    bias = torch.tensor([5.0, -5.0])
    head_j = _make_head_bundle(bias=bias)
    head_h = _make_head_bundle(bias=bias)
    X_pool = torch.randn(4, 2)

    X_empty, y_empty = trinet._label_unlabeled(
        X_pool[:0],
        shared_bundle=shared,
        head_j=head_j,
        head_h=head_h,
        sigma_t=0.0,
        stability_passes=1,
        drop_fraction=1.0,
        batch_size=2,
        freeze_bn=False,
    )
    assert X_empty.shape[0] == 0
    assert y_empty.shape[0] == 0

    X_pl, y_pl = trinet._label_unlabeled(
        X_pool,
        shared_bundle=shared,
        head_j=head_j,
        head_h=head_h,
        sigma_t=0.0,
        stability_passes=2,
        drop_fraction=1.0,
        batch_size=2,
        freeze_bn=True,
    )
    assert X_pl.shape[0] > 0
    assert y_pl.shape[0] == X_pl.shape[0]

    X_none, y_none = trinet._label_unlabeled(
        X_pool,
        shared_bundle=shared,
        head_j=_make_head_bundle(),
        head_h=_make_head_bundle(),
        sigma_t=1.0,
        stability_passes=1,
        drop_fraction=1.0,
        batch_size=2,
        freeze_bn=False,
    )
    assert X_none.shape[0] == 0
    assert y_none.shape[0] == 0

    def _all_false(x, *_args, **_kwargs):
        return torch.zeros((int(x.shape[0]),), dtype=torch.bool, device=x.device)

    monkeypatch.setattr(trinet, "_dropout_filter", _all_false)
    X_pl2, y_pl2 = trinet._label_unlabeled(
        X_pool,
        shared_bundle=shared,
        head_j=head_j,
        head_h=head_h,
        sigma_t=0.0,
        stability_passes=2,
        drop_fraction=1.0,
        batch_size=2,
        freeze_bn=False,
    )
    assert X_pl2.shape[0] == 0
    assert y_pl2.shape[0] == 0

    bad_head = _make_bundle(_BadLogits1D())
    with pytest.raises(InductiveValidationError, match="Head logits must be 2D"):
        trinet._label_unlabeled(
            X_pool,
            shared_bundle=shared,
            head_j=bad_head,
            head_h=head_h,
            sigma_t=0.0,
            stability_passes=1,
            drop_fraction=1.0,
            batch_size=2,
            freeze_bn=False,
        )

    mismatch_head = _make_head_bundle(n_classes=3)
    with pytest.raises(InductiveValidationError, match="Head logits must have the same shape"):
        trinet._label_unlabeled(
            X_pool,
            shared_bundle=shared,
            head_j=head_j,
            head_h=mismatch_head,
            sigma_t=0.0,
            stability_passes=1,
            drop_fraction=1.0,
            batch_size=2,
            freeze_bn=False,
        )


def test_trinet_train_head_paths():
    shared = _make_shared_bundle()
    head = _make_head_bundle()
    X = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    targets = trinet._one_hot(labels, n_classes=2)

    trinet._train_head(
        X,
        targets,
        shared_bundle=shared,
        head_bundle=head,
        update_shared=True,
        batch_size=2,
        epochs=1,
        seed=0,
    )
    trinet._train_head(
        X,
        targets,
        shared_bundle=shared,
        head_bundle=head,
        update_shared=False,
        batch_size=2,
        epochs=1,
        seed=1,
    )
    trinet._train_head(
        X[:0],
        targets[:0],
        shared_bundle=shared,
        head_bundle=head,
        update_shared=True,
        batch_size=2,
        epochs=1,
        seed=2,
    )

    bad_head = _make_bundle(_BadLogits1D())
    with pytest.raises(InductiveValidationError, match="Head logits must be 2D"):
        trinet._train_head(
            X,
            targets,
            shared_bundle=shared,
            head_bundle=bad_head,
            update_shared=True,
            batch_size=2,
            epochs=1,
            seed=3,
        )

    bad_batch = _make_bundle(_BadBatch())
    with pytest.raises(InductiveValidationError, match="Logits batch size does not match"):
        trinet._train_head(
            X,
            targets,
            shared_bundle=shared,
            head_bundle=bad_batch,
            update_shared=True,
            batch_size=2,
            epochs=1,
            seed=4,
        )


def test_trinet_validate_bundle_set():
    shared = _make_shared_bundle()
    head1 = _make_head_bundle()
    head2 = _make_head_bundle()
    head3 = _make_head_bundle()
    trinet._validate_bundle_set(
        shared_bundle=shared,
        head_bundles=(head1, head2, head3),
        device=torch.device("cpu"),
    )

    with pytest.raises(InductiveValidationError, match="shared_bundle must be provided"):
        trinet._validate_bundle_set(
            shared_bundle=None, head_bundles=(head1, head2, head3), device=torch.device("cpu")
        )

    with pytest.raises(InductiveValidationError, match="head_bundles must be provided"):
        trinet._validate_bundle_set(
            shared_bundle=shared, head_bundles=None, device=torch.device("cpu")
        )

    with pytest.raises(InductiveValidationError, match="exactly three bundles"):
        trinet._validate_bundle_set(
            shared_bundle=shared, head_bundles=(head1, head2), device=torch.device("cpu")
        )

    shared_model = torch.nn.Linear(2, 2)
    shared_bundle = _make_bundle(shared_model)
    shared_head = _make_bundle(shared_model)
    with pytest.raises(InductiveValidationError, match="shares parameters with shared_bundle"):
        trinet._validate_bundle_set(
            shared_bundle=shared_bundle,
            head_bundles=(shared_head, _make_head_bundle(), _make_head_bundle()),
            device=torch.device("cpu"),
        )

    head_model = torch.nn.Linear(2, 2)
    head_a = _make_bundle(head_model)
    head_b = _make_bundle(head_model)
    with pytest.raises(InductiveValidationError, match="shares parameters with head_bundles"):
        trinet._validate_bundle_set(
            shared_bundle=shared,
            head_bundles=(head_a, head_b, _make_head_bundle()),
            device=torch.device("cpu"),
        )


def _valid_spec(
    shared: TorchModelBundle, heads: tuple[TorchModelBundle, ...], **overrides
) -> TriNetSpec:
    kwargs = dict(
        shared_bundle=shared,
        head_bundles=heads,
        sigma0=0.0,
        sigma_os=0.0,
        sigma_decay=0.0,
        output_smearing_std=0.0,
        pool_base=1,
        max_rounds=1,
        fine_tune_interval=0,
        batch_size=2,
        init_epochs=1,
        train_epochs=1,
        des_passes=1,
        des_drop_fraction=0.0,
        labeling_stability_passes=1,
        freeze_bn=False,
    )
    kwargs.update(overrides)
    return TriNetSpec(**kwargs)


def test_trinet_fit_errors(monkeypatch):
    with pytest.raises(InductiveValidationError, match="data must not be None"):
        TriNetMethod().fit(None, device=DeviceSpec(device="cpu"))

    with pytest.raises(InductiveValidationError, match="requires torch tensors"):
        TriNetMethod().fit(make_numpy_dataset(), device=DeviceSpec(device="cpu"))

    data = make_torch_dataset()
    no_u = InductiveDataset(X_l=data.X_l, y_l=data.y_l, X_u=None)
    with pytest.raises(InductiveValidationError, match="requires X_u"):
        TriNetMethod().fit(no_u, device=DeviceSpec(device="cpu"))

    bad_dtype = InductiveDataset(
        X_l=data.X_l,
        y_l=data.y_l.to(dtype=torch.int32),
        X_u=data.X_u,
    )
    with pytest.raises(InductiveValidationError, match="y_l must be int64"):
        TriNetMethod().fit(bad_dtype, device=DeviceSpec(device="cpu"))

    empty_u = InductiveDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u[:0],
    )
    spec = _valid_spec(
        _make_shared_bundle(), (_make_head_bundle(), _make_head_bundle(), _make_head_bundle())
    )
    with pytest.raises(InductiveValidationError, match="X_u must be non-empty"):
        TriNetMethod(spec).fit(empty_u, device=DeviceSpec(device="cpu"))

    empty_x = InductiveDataset(
        X_l=data.X_l[:0],
        y_l=data.y_l[:0],
        X_u=data.X_u,
    )
    monkeypatch.setattr(trinet, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(trinet, "ensure_1d_labels_torch", lambda y, name="y_l": y)
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        TriNetMethod(spec).fit(empty_x, device=DeviceSpec(device="cpu"))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"pool_base": 0}, "pool_base must be >= 1"),
        ({"max_rounds": 0}, "max_rounds must be >= 1"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"init_epochs": 0}, "init_epochs must be >= 1"),
        ({"train_epochs": 0}, "train_epochs must be >= 1"),
        ({"des_passes": 0}, "des_passes must be >= 1"),
        ({"labeling_stability_passes": 0}, "labeling_stability_passes must be >= 1"),
        ({"output_smearing_std": -0.1}, "output_smearing_std must be >= 0"),
        ({"sigma0": 1.1}, "sigma0 must be in \\[0, 1\\]"),
        ({"sigma_os": -0.1}, "sigma_os must be >= 0"),
        ({"sigma_decay": -0.1}, "sigma_decay must be >= 0"),
        ({"des_drop_fraction": 1.5}, "des_drop_fraction must be in \\[0, 1\\]"),
    ],
)
def test_trinet_fit_spec_validation(overrides, match):
    data = make_torch_dataset()
    shared = _make_shared_bundle()
    heads = (_make_head_bundle(), _make_head_bundle(), _make_head_bundle())
    spec = _valid_spec(shared, heads, **overrides)
    with pytest.raises(InductiveValidationError, match=match):
        TriNetMethod(spec).fit(data, device=DeviceSpec(device="cpu"))


def test_trinet_fit_negative_labels():
    data = make_torch_dataset()
    y_neg = data.y_l.clone()
    y_neg[0] = -1
    neg_data = InductiveDataset(X_l=data.X_l, y_l=y_neg, X_u=data.X_u)
    shared = _make_shared_bundle()
    heads = (_make_head_bundle(), _make_head_bundle(), _make_head_bundle())
    spec = _valid_spec(shared, heads)
    with pytest.raises(InductiveValidationError, match="y_l labels must be non-negative"):
        TriNetMethod(spec).fit(neg_data, device=DeviceSpec(device="cpu"))


def test_trinet_fit_and_predict_with_fine_tune():
    data = make_torch_dataset()
    shared = _make_shared_bundle()
    bias = torch.tensor([5.0, -5.0])
    heads = (
        _make_head_bundle(bias=bias),
        _make_head_bundle(bias=bias),
        _make_head_bundle(bias=bias),
    )
    spec = TriNetSpec(
        shared_bundle=shared,
        head_bundles=heads,
        sigma0=0.0,
        sigma_os=0.0,
        sigma_decay=0.0,
        output_smearing_std=0.0,
        pool_base=1,
        max_rounds=2,
        fine_tune_interval=2,
        batch_size=2,
        init_epochs=1,
        train_epochs=1,
        des_passes=2,
        des_drop_fraction=1.0,
        labeling_stability_passes=2,
        freeze_bn=True,
    )
    method = TriNetMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    pred = method.predict(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    assert int(pred.shape[0]) == int(data.X_l.shape[0])


def test_trinet_fit_output_smearing_cuda_generator():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda"
    data = make_torch_dataset(device=device)
    shared = _make_shared_bundle()
    shared.model.to(device)
    heads = (_make_head_bundle(), _make_head_bundle(), _make_head_bundle())
    for head in heads:
        head.model.to(device)
    spec = _valid_spec(
        shared,
        heads,
        output_smearing_std=0.1,
        pool_base=2,
        max_rounds=1,
        fine_tune_interval=1,
    )
    TriNetMethod(spec).fit(data, device=DeviceSpec(device=device), seed=0)


def test_trinet_fit_sigma_t_after_flag():
    data = make_torch_dataset()
    shared = _make_shared_bundle()
    heads = (_make_head_bundle(), _make_head_bundle(), _make_head_bundle())
    spec = TriNetSpec(
        shared_bundle=shared,
        head_bundles=heads,
        sigma0=0.0,
        sigma_os=0.0,
        sigma_decay=0.0,
        output_smearing_std=0.0,
        pool_base=1,
        max_rounds=2,
        fine_tune_interval=0,
        batch_size=2,
        init_epochs=1,
        train_epochs=1,
        des_passes=1,
        des_drop_fraction=0.0,
        labeling_stability_passes=1,
        freeze_bn=False,
    )
    TriNetMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=1)


def test_trinet_fit_no_pseudo_labels():
    data = make_torch_dataset()
    shared = _make_shared_bundle()
    heads = (_make_head_bundle(), _make_head_bundle(), _make_head_bundle())
    spec = TriNetSpec(
        shared_bundle=shared,
        head_bundles=heads,
        sigma0=1.0,
        sigma_os=0.0,
        sigma_decay=0.0,
        output_smearing_std=0.0,
        pool_base=1,
        max_rounds=1,
        fine_tune_interval=0,
        batch_size=2,
        init_epochs=1,
        train_epochs=1,
        des_passes=1,
        des_drop_fraction=0.0,
        labeling_stability_passes=1,
        freeze_bn=False,
    )
    TriNetMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=2)


def test_trinet_fit_des_filter_drops_all(monkeypatch):
    data = make_torch_dataset()
    shared = _make_shared_bundle()
    bias = torch.tensor([5.0, -5.0])
    heads = (
        _make_head_bundle(bias=bias),
        _make_head_bundle(bias=bias),
        _make_head_bundle(bias=bias),
    )

    def _all_false(x, *_args, **_kwargs):
        return torch.zeros((int(x.shape[0]),), dtype=torch.bool, device=x.device)

    monkeypatch.setattr(trinet, "_dropout_filter", _all_false)

    spec = TriNetSpec(
        shared_bundle=shared,
        head_bundles=heads,
        sigma0=0.0,
        sigma_os=0.0,
        sigma_decay=0.0,
        output_smearing_std=0.0,
        pool_base=1,
        max_rounds=1,
        fine_tune_interval=0,
        batch_size=2,
        init_epochs=1,
        train_epochs=1,
        des_passes=2,
        des_drop_fraction=1.0,
        labeling_stability_passes=1,
        freeze_bn=False,
    )
    TriNetMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=3)


def test_trinet_predict_proba_eval_models():
    data = make_torch_dataset()
    shared = _make_shared_bundle()
    heads = (_make_head_bundle(), _make_head_bundle(), _make_head_bundle())
    spec = _valid_spec(shared, heads)
    method = TriNetMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)
    method._shared_bundle.model.eval()
    for bundle in method._head_bundles:
        bundle.model.eval()
    proba = method.predict_proba(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])


def test_trinet_predict_proba_errors(monkeypatch):
    with pytest.raises(RuntimeError, match="not fitted"):
        TriNetMethod().predict_proba(torch.randn(2, 2))

    shared = _make_shared_bundle()
    heads = (_make_head_bundle(), _make_head_bundle(), _make_head_bundle())
    method = TriNetMethod(TriNetSpec(shared_bundle=shared, head_bundles=heads))
    method._shared_bundle = shared
    method._head_bundles = heads
    with pytest.raises(InductiveValidationError, match="predict_proba requires torch tensors"):
        method.predict_proba(make_numpy_dataset().X_l)

    monkeypatch.setattr(trinet, "detect_backend", lambda _x: "torch")
    with pytest.raises(InductiveValidationError, match="predict_proba requires torch.Tensor"):
        method.predict_proba([[0.0, 0.0]])

    bad_head = _make_bundle(_BadLogits1D())
    method._head_bundles = (bad_head, heads[1], heads[2])
    with pytest.raises(InductiveValidationError, match="Head logits must be 2D"):
        method.predict_proba(torch.randn(2, 2))

    mismatch_heads = (_make_head_bundle(), _make_head_bundle(n_classes=3), _make_head_bundle())
    method._head_bundles = mismatch_heads
    with pytest.raises(InductiveValidationError, match="heads disagree on class count"):
        method.predict_proba(torch.randn(2, 2))
