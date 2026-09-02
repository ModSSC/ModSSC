from __future__ import annotations

import importlib
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.pseudo_label import (
    PseudoLabelMethod,
    PseudoLabelSpec,
    _build_lee2013_mlp,
    _lee2013_alpha,
    _lee2013_epoch_batches,
    _lee2013_flatten_input,
    _lee2013_joint_loss,
    _lee2013_learning_rate,
    _lee2013_momentum,
    _lee2013_sgd_step,
    _Lee2013Classifier,
)
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
    _assert_module_importable("modssc.inductive.methods.pseudo_label")


def test_iterative_threshold_mode_remains_the_default() -> None:
    spec = PseudoLabelSpec()

    assert spec.training_mode == "iterative_threshold"
    assert spec.max_iter == 10
    assert spec.confidence_threshold == pytest.approx(0.95)
    PseudoLabelMethod(spec)._validate_training_mode()


def test_joint_mlp_mode_defaults_encode_reported_plus_pl_hyperparameters() -> None:
    spec = PseudoLabelSpec(training_mode="joint_mlp", classifier_backend="torch")

    assert spec.paper_input_dim == 784
    assert spec.paper_hidden_units == 5000
    assert spec.paper_num_classes == 10
    assert spec.paper_labeled_batch_size == 32
    assert spec.paper_unlabeled_batch_size == 256
    assert spec.paper_hidden_dropout == pytest.approx(0.5)
    assert spec.paper_initial_learning_rate == pytest.approx(1.5)
    assert spec.paper_learning_rate_decay == pytest.approx(0.998)
    assert spec.paper_momentum_initial == pytest.approx(0.5)
    assert spec.paper_momentum_final == pytest.approx(0.99)
    assert spec.paper_momentum_ramp_epochs == 500
    assert spec.paper_alpha_final == pytest.approx(3.0)
    assert spec.paper_alpha_start_epoch == 100
    assert spec.paper_alpha_end_epoch == 600
    assert PseudoLabelMethod(spec)._validate_training_mode() == "joint_mlp"
    canonical = PseudoLabelSpec(
        training_mode="joint_mlp",
        classifier_backend="torch",
    )
    assert PseudoLabelMethod(canonical)._validate_training_mode() == canonical.training_mode


def test_lee2013_schedules_match_equations_12_13_and_16() -> None:
    assert _lee2013_alpha(99) == 0.0
    assert _lee2013_alpha(100) == 0.0
    assert _lee2013_alpha(350) == pytest.approx(1.5)
    assert _lee2013_alpha(599) == pytest.approx(3.0 * 499.0 / 500.0)
    assert _lee2013_alpha(600) == 3.0
    assert _lee2013_alpha(700) == 3.0

    assert _lee2013_learning_rate(0) == pytest.approx(1.5)
    assert _lee2013_learning_rate(10) == pytest.approx(1.5 * 0.998**10)
    assert _lee2013_momentum(0) == pytest.approx(0.5)
    assert _lee2013_momentum(250) == pytest.approx(0.745)
    assert _lee2013_momentum(500) == pytest.approx(0.99)
    assert _lee2013_momentum(700) == pytest.approx(0.99)


def test_lee2013_joint_loss_uses_hard_argmax_pseudo_labels() -> None:
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    logits_l = torch.tensor([[0.2, -0.3], [-0.7, 1.1]], requires_grad=True)
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    logits_u = torch.tensor([[-1.0, 0.4], [2.0, -0.2]], requires_grad=True)

    total, supervised, unsupervised, pseudo = _lee2013_joint_loss(
        logits_l,
        y_l,
        logits_u,
        alpha=0.75,
        n_classes=2,
    )

    expected_l = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    expected_u = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected_supervised = (
        functional.binary_cross_entropy_with_logits(logits_l, expected_l, reduction="none")
        .sum(dim=1)
        .mean()
    )
    expected_unsupervised = (
        functional.binary_cross_entropy_with_logits(logits_u, expected_u, reduction="none")
        .sum(dim=1)
        .mean()
    )

    torch.testing.assert_close(pseudo, torch.tensor([1, 0]))
    torch.testing.assert_close(supervised, expected_supervised)
    torch.testing.assert_close(unsupervised, expected_unsupervised)
    torch.testing.assert_close(total, expected_supervised + 0.75 * expected_unsupervised)


def test_lee2013_sgd_step_matches_momentum_equations() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    buffers = {}

    parameter.grad = torch.tensor([2.0])
    _lee2013_sgd_step(
        [parameter],
        buffers,
        learning_rate=1.5,
        momentum=0.5,
    )
    torch.testing.assert_close(parameter, torch.tensor([-0.5]))

    parameter.grad = torch.tensor([1.0])
    _lee2013_sgd_step(
        [parameter],
        buffers,
        learning_rate=1.5,
        momentum=0.5,
    )
    torch.testing.assert_close(parameter, torch.tensor([-2.0]))


def test_lee2013_sgd_step_skips_parameters_without_gradients() -> None:
    torch = pytest.importorskip("torch")
    untouched = torch.nn.Parameter(torch.tensor([7.0]))
    updated = torch.nn.Parameter(torch.tensor([1.0]))
    updated.grad = torch.tensor([2.0])

    _lee2013_sgd_step(
        [untouched, updated],
        {},
        learning_rate=1.5,
        momentum=0.5,
    )

    torch.testing.assert_close(untouched, torch.tensor([7.0]))
    torch.testing.assert_close(updated, torch.tensor([-0.5]))


def test_lee2013_batches_keep_the_paper_batch_size_and_are_deterministic() -> None:
    torch = pytest.importorskip("torch")

    def generate():
        return _lee2013_epoch_batches(
            torch=torch,
            n_samples=5,
            batch_size=3,
            n_steps=4,
            generator=torch.Generator(device="cpu").manual_seed(19),
            device=torch.device("cpu"),
        )

    first = generate()
    second = generate()

    assert [int(batch.numel()) for batch in first] == [3, 3, 3, 3]
    for first_batch, second_batch in zip(first, second, strict=True):
        torch.testing.assert_close(first_batch, second_batch, rtol=0.0, atol=0.0)


def test_lee2013_dropout_matches_equation_9_convention() -> None:
    torch = pytest.importorskip("torch")
    model = _build_lee2013_mlp(
        torch=torch,
        input_dim=4,
        hidden_units=3,
        n_classes=2,
        hidden_dropout=0.25,
        input_dropout=0.0,
    )
    values = torch.ones((256, 3))

    model.hidden_dropout.train()
    torch.manual_seed(4)
    training_values = model.hidden_dropout(values)
    assert set(training_values.unique().tolist()) == {0.0, 1.0}

    model.hidden_dropout.eval()
    torch.testing.assert_close(model.hidden_dropout(values), values * 0.75)


def test_lee2013_flatten_accepts_mapping_and_empty_batch() -> None:
    torch = pytest.importorskip("torch")

    flattened = _lee2013_flatten_input(
        {"x": torch.zeros((0, 1, 2, 2))},
        torch=torch,
        name="X",
        input_dim=4,
    )

    assert flattened.shape == (0, 4)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({}, "must be a torch.Tensor"),
        (None, "must be a torch.Tensor"),
        ("not-a-tensor", "must be a torch.Tensor"),
    ],
)
def test_lee2013_flatten_rejects_non_tensor_values(value, message: str) -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(InductiveValidationError, match=message):
        _lee2013_flatten_input(value, torch=torch, name="X", input_dim=4)


def test_lee2013_flatten_rejects_rank_one_and_nonfinite_values() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(InductiveValidationError, match=r"shape \(n, \.\.\.\)"):
        _lee2013_flatten_input(torch.zeros(4), torch=torch, name="X", input_dim=4)
    with pytest.raises(InductiveValidationError, match="only finite"):
        _lee2013_flatten_input(
            torch.tensor([[0.0, 0.0, float("nan"), 0.0]]),
            torch=torch,
            name="X",
            input_dim=4,
        )


def test_lee2013_classifier_restores_training_mode_and_predicts() -> None:
    torch = pytest.importorskip("torch")
    model = _build_lee2013_mlp(
        torch=torch,
        input_dim=4,
        hidden_units=3,
        n_classes=2,
        hidden_dropout=0.0,
        input_dropout=0.0,
    )
    model.train()
    classifier = _Lee2013Classifier(model, input_dim=4, n_classes=2)

    predicted = classifier.predict(torch.zeros((2, 4)))

    assert predicted.shape == (2,)
    assert model.training is True


def test_lee2013_classifier_rejects_input_on_another_device() -> None:
    torch = pytest.importorskip("torch")
    model = _build_lee2013_mlp(
        torch=torch,
        input_dim=4,
        hidden_units=3,
        n_classes=2,
        hidden_dropout=0.0,
        input_dropout=0.0,
    )
    classifier = _Lee2013Classifier(model, input_dim=4, n_classes=2)
    meta_input = torch.empty((1, 4), device="meta")

    with (
        patch(
            "modssc.inductive.methods.pseudo_label._lee2013_flatten_input",
            return_value=meta_input,
        ),
        pytest.raises(InductiveValidationError, match="fitted model device"),
    ):
        classifier.predict_scores(torch.zeros((1, 4)))


def _tiny_lee2013_dataset():
    torch = pytest.importorskip("torch")
    return InductiveDataset(
        X_l=torch.tensor(
            [
                [0.0, 0.0, 0.2, 0.1],
                [0.1, 0.2, 0.0, 0.0],
                [0.9, 0.8, 1.0, 0.9],
                [1.0, 0.9, 0.8, 1.0],
            ],
            dtype=torch.float32,
        ),
        y_l=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        X_u=torch.tensor(
            [
                [0.0, 0.1, 0.1, 0.0],
                [0.2, 0.0, 0.1, 0.2],
                [0.1, 0.2, 0.2, 0.1],
                [0.8, 0.9, 0.8, 1.0],
                [1.0, 0.8, 0.9, 0.9],
                [0.9, 1.0, 0.8, 0.8],
            ],
            dtype=torch.float32,
        ),
    )


def _tiny_lee2013_spec() -> PseudoLabelSpec:
    return PseudoLabelSpec(
        training_mode="joint_mlp",
        classifier_backend="torch",
        paper_input_dim=4,
        paper_hidden_units=8,
        paper_num_classes=2,
        paper_epochs=3,
        paper_labeled_batch_size=2,
        paper_unlabeled_batch_size=3,
        paper_hidden_dropout=0.25,
        paper_input_dropout=0.0,
        paper_initial_learning_rate=0.05,
        paper_learning_rate_decay=0.9,
        paper_momentum_initial=0.1,
        paper_momentum_final=0.2,
        paper_momentum_ramp_epochs=2,
        paper_alpha_final=1.0,
        paper_alpha_start_epoch=1,
        paper_alpha_end_epoch=2,
    )


def test_lee2013_profile_is_deterministic_and_reports_diagnostics() -> None:
    torch = pytest.importorskip("torch")
    data = _tiny_lee2013_dataset()

    first = PseudoLabelMethod(_tiny_lee2013_spec()).fit(
        data,
        device=DeviceSpec(device="cpu"),
        seed=27,
    )
    second = PseudoLabelMethod(_tiny_lee2013_spec()).fit(
        data,
        device=DeviceSpec(device="cpu"),
        seed=27,
    )

    first_scores = first.predict_proba(data.X_l)
    second_scores = second.predict_proba(data.X_l)
    torch.testing.assert_close(first_scores, second_scores, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first_scores.sum(dim=1),
        torch.ones((len(data.X_l),), dtype=torch.float32),
    )
    assert first.diagnostics_ == second.diagnostics_
    assert first.diagnostics_["training_mode"] == "joint_mlp"
    assert first.diagnostics_["dae_pretraining"] is False
    assert first.diagnostics_["dropout_convention"] == "lee2013_non_inverted"
    assert first.diagnostics_["alpha_history"] == [0.0, 0.0, 1.0]
    assert first.diagnostics_["schedule_epoch_first"] == 0
    assert first.diagnostics_["schedule_epoch_last"] == 2
    assert first.diagnostics_["alpha_reached_final"] is True
    assert first.diagnostics_["steps_per_epoch"] == 2
    assert first.diagnostics_["parameter_updates"] == 6
    assert first.diagnostics_["pseudo_label_updates"] == 6
    assert first.diagnostics_["pseudo_labels_assigned_total"] == 18
    assert sum(first.diagnostics_["pseudo_label_class_counts"]) == 18
    assert first.diagnostics_["final_pseudo_labels_assigned"] == 6
    assert sum(first.diagnostics_["final_pseudo_label_class_counts"]) == 6
    assert first.diagnostics_["confidence_threshold_applied"] is False
    assert first._clf.model.hidden.out_features == 8
    assert first._clf.model.hidden_dropout.p == pytest.approx(0.25)


def test_lee2013_profile_validates_backend_and_inputs() -> None:
    torch = pytest.importorskip("torch")
    data = _tiny_lee2013_dataset()

    wrong_backend = PseudoLabelMethod(
        PseudoLabelSpec(training_mode="joint_mlp", classifier_backend="numpy")
    )
    with pytest.raises(InductiveValidationError, match="classifier_backend='torch'"):
        wrong_backend.fit(data, device=DeviceSpec(device="cpu"))

    no_unlabeled = InductiveDataset(X_l=data.X_l, y_l=data.y_l, X_u=None)
    with pytest.raises(InductiveValidationError, match="non-empty unlabeled"):
        PseudoLabelMethod(_tiny_lee2013_spec()).fit(
            no_unlabeled,
            device=DeviceSpec(device="cpu"),
        )

    unscaled = InductiveDataset(X_l=data.X_l * 255.0, y_l=data.y_l, X_u=data.X_u)
    with pytest.raises(InductiveValidationError, match=r"scaled to \[0, 1\]"):
        PseudoLabelMethod(_tiny_lee2013_spec()).fit(
            unscaled,
            device=DeviceSpec(device="cpu"),
        )

    wrong_shape = InductiveDataset(
        X_l=torch.zeros((4, 5)),
        y_l=data.y_l,
        X_u=torch.zeros((6, 5)),
    )
    with pytest.raises(InductiveValidationError, match="flatten to 4"):
        PseudoLabelMethod(_tiny_lee2013_spec()).fit(
            wrong_shape,
            device=DeviceSpec(device="cpu"),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training_mode": "unknown"}, "training_mode must be"),
        ({"paper_input_dim": 0}, "paper_input_dim must be > 0"),
        ({"paper_num_classes": 1}, "paper_num_classes must be >= 2"),
        ({"paper_hidden_dropout": 1.0}, "paper_hidden_dropout must be in"),
        ({"paper_input_dropout": -0.1}, "paper_input_dropout must be in"),
        ({"paper_initial_learning_rate": 0.0}, "paper_initial_learning_rate must be > 0"),
        ({"paper_learning_rate_decay": 0.0}, "paper_learning_rate_decay must be in"),
        ({"paper_momentum_initial": -0.1}, "paper_momentum_initial must be in"),
        ({"paper_momentum_final": 1.0}, "paper_momentum_final must be in"),
        (
            {"paper_momentum_initial": 0.8, "paper_momentum_final": 0.7},
            "must not exceed",
        ),
        ({"paper_alpha_final": -0.1}, "paper_alpha_final must be >= 0"),
        ({"paper_alpha_start_epoch": -1}, "paper_alpha_start_epoch must be >= 0"),
        (
            {"paper_alpha_start_epoch": 2, "paper_alpha_end_epoch": 2},
            "must be greater",
        ),
    ],
)
def test_lee2013_profile_rejects_invalid_hyperparameters(
    overrides: dict[str, object],
    message: str,
) -> None:
    method = PseudoLabelMethod(replace(_tiny_lee2013_spec(), **overrides))

    with pytest.raises(InductiveValidationError, match=message):
        method._validate_training_mode()


def test_lee2013_profile_rejects_empty_labeled_data_and_out_of_range_labels() -> None:
    torch = pytest.importorskip("torch")
    data = _tiny_lee2013_dataset()
    empty_labeled = InductiveDataset(
        X_l=torch.empty((0, 4), dtype=torch.float32),
        y_l=torch.empty((0,), dtype=torch.int64),
        X_u=data.X_u,
    )
    bad_labels = InductiveDataset(
        X_l=data.X_l,
        y_l=torch.tensor([0, 0, 1, 2], dtype=torch.int64),
        X_u=data.X_u,
    )

    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        PseudoLabelMethod(_tiny_lee2013_spec()).fit(
            empty_labeled,
            device=DeviceSpec(device="cpu"),
        )
    with pytest.raises(InductiveValidationError, match="y_l values must be"):
        PseudoLabelMethod(_tiny_lee2013_spec()).fit(
            bad_labels,
            device=DeviceSpec(device="cpu"),
        )


def test_lee2013_profile_rejects_nonfinite_training_loss() -> None:
    torch = pytest.importorskip("torch")
    data = _tiny_lee2013_dataset()

    def nonfinite_loss(logits_l, y_l, logits_u, *, alpha, n_classes):
        del logits_l, y_l, alpha, n_classes
        total = torch.tensor(float("inf"), requires_grad=True)
        pseudo = torch.zeros((len(logits_u),), dtype=torch.int64)
        return total, total, total, pseudo

    with (
        patch(
            "modssc.inductive.methods.pseudo_label._lee2013_joint_loss",
            side_effect=nonfinite_loss,
        ),
        pytest.raises(InductiveValidationError, match="non-finite loss"),
    ):
        PseudoLabelMethod(replace(_tiny_lee2013_spec(), paper_epochs=1)).fit(
            data,
            device=DeviceSpec(device="cpu"),
        )


def test_lee2013_cuda_seed_path_is_exercised_without_requiring_a_gpu() -> None:
    torch = pytest.importorskip("torch")
    data = _tiny_lee2013_dataset()

    class FirstThreeCudaDeviceReads:
        def __init__(self, tensor):
            self.tensor = tensor
            self.reads = 0

        @property
        def shape(self):
            return self.tensor.shape

        @property
        def device(self):
            self.reads += 1
            if self.reads <= 3:
                return SimpleNamespace(type="cuda", index=0)
            return self.tensor.device

        def __getitem__(self, item):
            return self.tensor[item]

    X_l = FirstThreeCudaDeviceReads(data.X_l)

    def flatten(value, *, torch, name, input_dim):
        del value, torch, input_dim
        return X_l if name == "X_l" else data.X_u

    with (
        patch(
            "modssc.inductive.methods.pseudo_label._lee2013_flatten_input",
            side_effect=flatten,
        ),
        patch.object(torch.random, "fork_rng", return_value=nullcontext()),
        patch.object(torch.cuda, "manual_seed_all") as manual_seed_all,
    ):
        method = PseudoLabelMethod(replace(_tiny_lee2013_spec(), paper_epochs=1)).fit(
            data,
            device=DeviceSpec(device="cpu"),
            seed=31,
        )

    assert method.diagnostics_["epochs_completed"] == 1
    manual_seed_all.assert_any_call(31)
