from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

import modssc.inductive.methods.meta_pseudo_labels as mpl
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.meta_pseudo_labels import (
    MetaPseudoLabelsMethod,
    MetaPseudoLabelsSpec,
)
from modssc.inductive.types import DeviceSpec, InductiveDataset

from .conftest import make_numpy_dataset, make_torch_ssl_dataset


class _LinearNet(torch.nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, n_classes, bias=False)

    def forward(self, x):
        return self.fc(x)


class _BadLogits1D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((int(x.shape[0]),), device=x.device)


class _SequenceNet(torch.nn.Module):
    def __init__(self, outputs: list[tuple[int, bool]]) -> None:
        super().__init__()
        self.outputs = outputs
        self.call = 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        idx = min(self.call, len(self.outputs) - 1)
        self.call += 1
        n_classes, one_d = self.outputs[idx]
        if one_d:
            return self.dummy * torch.ones((int(x.shape[0]),), device=x.device)
        return self.dummy * torch.ones((int(x.shape[0]), int(n_classes)), device=x.device)


def _make_bundle(model: torch.nn.Module) -> TorchModelBundle:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TorchModelBundle(model=model, optimizer=optimizer)


def _make_dataset(*, views: dict | None = None) -> InductiveDataset:
    base = make_torch_ssl_dataset()
    return InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views=views,
    )


def test_meta_pseudo_labels_helper_losses():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    probs = torch.softmax(logits, dim=1)
    loss = mpl._soft_cross_entropy(probs, logits)
    assert loss.shape == (2,)

    with pytest.raises(InductiveValidationError, match="Target distribution shape mismatch"):
        mpl._soft_cross_entropy(probs[:, :1], logits)

    loss_smoothed = mpl._label_smoothed_ce(logits, torch.tensor([0, 1]), smoothing=0.1)
    assert float(loss_smoothed) >= 0.0
    loss_nosmooth = mpl._label_smoothed_ce(logits, torch.tensor([0, 1]), smoothing=0.0)
    assert float(loss_nosmooth) >= 0.0
    with pytest.raises(InductiveValidationError, match="label_smoothing must be in"):
        mpl._label_smoothed_ce(logits, torch.tensor([0, 1]), smoothing=1.0)

    ent = mpl._entropy_loss(logits)
    assert float(ent) >= 0.0


def test_meta_pseudo_labels_uda_loss_branches():
    spec = MetaPseudoLabelsSpec(uda_threshold=0.0, uda_temperature=0.5)
    method = MetaPseudoLabelsMethod(spec)
    logits = torch.randn(2, 2)
    uda_loss, mask = method._uda_loss(logits, logits)
    assert uda_loss.shape == ()
    assert int(mask.numel()) == 2

    spec_high = MetaPseudoLabelsSpec(uda_threshold=1.0, uda_temperature=0.5)
    method_high = MetaPseudoLabelsMethod(spec_high)
    uda_loss_zero, mask_zero = method_high._uda_loss(logits, logits)
    assert float(uda_loss_zero) == 0.0
    assert float(mask_zero.sum()) == 0.0

    spec_bad = MetaPseudoLabelsSpec(uda_temperature=0.0)
    method_bad = MetaPseudoLabelsMethod(spec_bad)
    with pytest.raises(InductiveValidationError, match="uda_temperature must be > 0"):
        method_bad._uda_loss(logits, logits)


def test_meta_pseudo_labels_check_models():
    model = _LinearNet()
    method = MetaPseudoLabelsMethod()
    with pytest.raises(InductiveValidationError, match="distinct models"):
        method._check_models(model, model)

    shared = torch.nn.Linear(2, 2, bias=False)
    teacher = torch.nn.Sequential(shared)
    student = torch.nn.Sequential(shared)
    with pytest.raises(InductiveValidationError, match="must not share parameters"):
        method._check_models(teacher, student)


def test_meta_pseudo_labels_fit_variants():
    data = _make_dataset(views={"X_l_s": torch.ones((4, 2))})
    teacher_bundle = _make_bundle(_LinearNet())
    student_bundle = _make_bundle(_LinearNet())
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=teacher_bundle,
        student_bundle=student_bundle,
        batch_size=2,
        max_epochs=1,
        uda_weight=1.0,
        mpl_weight=1.0,
        init_teacher_from_student=True,
        detach_target=True,
        uda_threshold=0.0,
    )
    method = MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    pred = method.predict(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    assert int(pred.shape[0]) == int(data.X_l.shape[0])

    teacher_bundle2 = _make_bundle(_LinearNet())
    student_bundle2 = _make_bundle(_LinearNet())
    spec2 = MetaPseudoLabelsSpec(
        teacher_bundle=teacher_bundle2,
        student_bundle=student_bundle2,
        batch_size=2,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
        detach_target=False,
        uda_threshold=1.0,
        label_smoothing=0.0,
    )
    MetaPseudoLabelsMethod(spec2).fit(data, device=DeviceSpec(device="cpu"), seed=1)


def test_meta_pseudo_labels_fit_views_fallback():
    data = _make_dataset(views={"X_l_strong": torch.ones((4, 2))})
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(_LinearNet()),
        batch_size=4,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
    )
    MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_views_no_match():
    data = _make_dataset(views={"other": torch.ones((4, 2))})
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(_LinearNet()),
        batch_size=4,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
    )
    MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "data,match",
    [
        (None, "data must not be None"),
        (make_numpy_dataset(), "requires torch tensors"),
    ],
)
def test_meta_pseudo_labels_fit_data_errors(data, match):
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(_LinearNet()),
    )
    with pytest.raises(InductiveValidationError, match=match):
        MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_dataset_validation_errors():
    base = _make_dataset()
    teacher_bundle = _make_bundle(_LinearNet())
    student_bundle = _make_bundle(_LinearNet())
    spec = MetaPseudoLabelsSpec(teacher_bundle=teacher_bundle, student_bundle=student_bundle)

    no_u = InductiveDataset(X_l=base.X_l, y_l=base.y_l, X_u_w=None, X_u_s=None)
    with pytest.raises(InductiveValidationError, match="requires X_u_w and X_u_s"):
        MetaPseudoLabelsMethod(spec).fit(no_u, device=DeviceSpec(device="cpu"))

    empty_l = InductiveDataset(
        X_l=base.X_l[:0],
        y_l=base.y_l[:0],
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="y_l must be non-empty"):
        MetaPseudoLabelsMethod(spec).fit(empty_l, device=DeviceSpec(device="cpu"))

    empty_u = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w[:0],
        X_u_s=base.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        MetaPseudoLabelsMethod(spec).fit(empty_u, device=DeviceSpec(device="cpu"))

    mismatch = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s[:-1],
    )
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        MetaPseudoLabelsMethod(spec).fit(mismatch, device=DeviceSpec(device="cpu"))

    bad_x_l_s = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views={"X_l_s": torch.zeros((4, 2, 1))},
    )
    with pytest.raises(InductiveValidationError, match="X_l_s must be 2D"):
        MetaPseudoLabelsMethod(spec).fit(bad_x_l_s, device=DeviceSpec(device="cpu"))

    bad_x_l_s_rows = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views={"X_l_s": torch.zeros((3, 2))},
    )
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        MetaPseudoLabelsMethod(spec).fit(bad_x_l_s_rows, device=DeviceSpec(device="cpu"))

    bad_x_l_s_dim = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
        views={"X_l_s": torch.zeros((4, 3))},
    )
    with pytest.raises(InductiveValidationError, match="same feature dimension"):
        MetaPseudoLabelsMethod(spec).fit(bad_x_l_s_dim, device=DeviceSpec(device="cpu"))

    bad_dtype = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l.to(dtype=torch.int32),
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    with pytest.raises(InductiveValidationError, match="y_l must be int64"):
        MetaPseudoLabelsMethod(spec).fit(bad_dtype, device=DeviceSpec(device="cpu"))


def test_meta_pseudo_labels_fit_row_mismatch_raises(monkeypatch):
    base = _make_dataset()
    mismatch = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s[:-1],
    )
    monkeypatch.setattr(mpl, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(mpl, "ensure_1d_labels_torch", lambda y, name="y_l": y)
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(_LinearNet()),
    )
    with pytest.raises(InductiveValidationError, match="same number of rows"):
        MetaPseudoLabelsMethod(spec).fit(mismatch, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_unlabeled_logits_shape_mismatch():
    data = _make_dataset()
    teacher = _SequenceNet([(2, False), (2, False), (3, False)])
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(teacher),
        student_bundle=_make_bundle(_LinearNet()),
        batch_size=4,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="Unlabeled logits shape mismatch"):
        MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_logit_class_mismatch():
    data = _make_dataset()
    teacher = _SequenceNet([(3, False), (2, False), (2, False)])
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(teacher),
        student_bundle=_make_bundle(_LinearNet()),
        batch_size=4,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="Logits must agree on class dimension"):
        MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_y_l_out_of_range():
    base = _make_dataset()
    data = InductiveDataset(
        X_l=base.X_l,
        y_l=torch.full_like(base.y_l, 2),
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(_LinearNet()),
        batch_size=4,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="y_l labels must be within"):
        MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_student_logits_dim_error():
    data = _make_dataset()
    student = _SequenceNet([(2, False), (0, True), (2, False)])
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(student),
        batch_size=4,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="Student logits must be 2D"):
        MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_student_class_mismatch():
    data = _make_dataset()
    student = _SequenceNet([(2, False), (3, False), (2, False)])
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(student),
        batch_size=4,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="same class count"):
        MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_student_logits_dim_error_after_update():
    data = _make_dataset()
    student = _SequenceNet([(2, False), (2, False), (0, True)])
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(student),
        batch_size=4,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="Student logits must be 2D"):
        MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_student_class_mismatch_after_update():
    data = _make_dataset()
    student = _SequenceNet([(2, False), (2, False), (3, False)])
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(student),
        batch_size=4,
        max_epochs=1,
        uda_weight=0.0,
        mpl_weight=0.0,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="same class count"):
        MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_empty_x_l_raises(monkeypatch):
    base = _make_dataset()
    empty_l = InductiveDataset(
        X_l=base.X_l[:0],
        y_l=base.y_l,
        X_u_w=base.X_u_w,
        X_u_s=base.X_u_s,
    )
    monkeypatch.setattr(mpl, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(mpl, "ensure_1d_labels_torch", lambda y, name="y_l": y)
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(_LinearNet()),
    )
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        MetaPseudoLabelsMethod(spec).fit(empty_l, device=DeviceSpec(device="cpu"))


def test_meta_pseudo_labels_fit_empty_unlabeled_raises(monkeypatch):
    base = _make_dataset()
    empty_u = InductiveDataset(
        X_l=base.X_l,
        y_l=base.y_l,
        X_u_w=base.X_u_w[:0],
        X_u_s=base.X_u_s[:0],
    )
    monkeypatch.setattr(mpl, "ensure_torch_data", lambda data, device: data)
    monkeypatch.setattr(mpl, "ensure_1d_labels_torch", lambda y, name="y_l": y)
    spec = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(_LinearNet()),
    )
    with pytest.raises(InductiveValidationError, match="Provided X_u is empty"):
        MetaPseudoLabelsMethod(spec).fit(empty_u, device=DeviceSpec(device="cpu"))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"teacher_bundle": None}, "teacher_bundle and student_bundle must be provided"),
        ({"student_bundle": None}, "teacher_bundle and student_bundle must be provided"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"max_epochs": 0}, "max_epochs must be >= 1"),
        ({"uda_weight": -1.0}, "uda_weight must be >= 0"),
        ({"uda_threshold": 1.5}, "uda_threshold must be in"),
        ({"label_smoothing": 1.0}, "label_smoothing must be in"),
        ({"mpl_ema": 1.0}, "mpl_ema must be in"),
        ({"mpl_weight": -1.0}, "mpl_weight must be >= 0"),
    ],
)
def test_meta_pseudo_labels_fit_spec_validation_errors(overrides, match):
    data = _make_dataset()
    base = MetaPseudoLabelsSpec(
        teacher_bundle=_make_bundle(_LinearNet()),
        student_bundle=_make_bundle(_LinearNet()),
    )
    spec = MetaPseudoLabelsSpec(**{**base.__dict__, **overrides})
    with pytest.raises(InductiveValidationError, match=match):
        MetaPseudoLabelsMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_fit_model_errors():
    data = _make_dataset()
    shared = torch.nn.Linear(2, 2, bias=False)
    teacher_bundle = _make_bundle(torch.nn.Sequential(shared))
    student_bundle = _make_bundle(torch.nn.Sequential(shared))
    spec_shared = MetaPseudoLabelsSpec(teacher_bundle=teacher_bundle, student_bundle=student_bundle)
    with pytest.raises(InductiveValidationError, match="must not share parameters"):
        MetaPseudoLabelsMethod(spec_shared).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    teacher_bad = _make_bundle(_BadLogits1D())
    student_ok = _make_bundle(_LinearNet())
    spec_bad = MetaPseudoLabelsSpec(
        teacher_bundle=teacher_bad,
        student_bundle=student_ok,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        MetaPseudoLabelsMethod(spec_bad).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    teacher_ok = _make_bundle(_LinearNet())
    student_bad = _make_bundle(_BadLogits1D())
    spec_bad_student = MetaPseudoLabelsSpec(
        teacher_bundle=teacher_ok,
        student_bundle=student_bad,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="Student logits must be 2D"):
        MetaPseudoLabelsMethod(spec_bad_student).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    teacher_three = _make_bundle(_LinearNet(n_classes=3))
    student_two = _make_bundle(_LinearNet(n_classes=2))
    spec_mismatch = MetaPseudoLabelsSpec(
        teacher_bundle=teacher_three,
        student_bundle=student_two,
        init_teacher_from_student=False,
    )
    with pytest.raises(InductiveValidationError, match="same class count"):
        MetaPseudoLabelsMethod(spec_mismatch).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_meta_pseudo_labels_predict_errors():
    method = MetaPseudoLabelsMethod()
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(torch.zeros((2, 2)))

    method._student_bundle = _make_bundle(_LinearNet())
    with pytest.raises(InductiveValidationError, match="requires torch tensors"):
        method.predict_proba(np.zeros((2, 2), dtype=np.float32))

    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="requires torch.Tensor"):
        method.predict_proba(np.zeros((2, 2), dtype=np.float32))

    method._student_bundle = _make_bundle(_BadLogits1D())
    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        method.predict_proba(torch.zeros((2, 2)))


def test_meta_pseudo_labels_predict_proba_keeps_eval_state():
    bundle = _make_bundle(_LinearNet())
    bundle.model.eval()
    method = MetaPseudoLabelsMethod()
    method._student_bundle = bundle
    method._backend = "torch"
    proba = method.predict_proba(torch.zeros((2, 2)))
    assert proba.shape == (2, 2)
    assert bundle.model.training is False
