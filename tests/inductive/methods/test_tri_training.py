from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.tri_training import (
    TriTrainingMethod,
    TriTrainingSpec,
    _cap_numpy_candidates,
    _cap_torch_candidates,
    _global_class_order,
    _measure_error,
    _numpy_labels_from_scores,
    _paper_update_decision,
    _subsample_positions,
    _torch_labels_from_scores,
)
from modssc.inductive.types import DeviceSpec


def test_tri_training_measure_error_uses_only_agreements() -> None:
    estimate = _measure_error(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 1]),
    )

    assert estimate.agreements == 3
    assert estimate.wrong_agreements == 2
    assert estimate.rate == pytest.approx(2.0 / 3.0)

    no_agreement = _measure_error(
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([0, 1]),
    )
    assert no_agreement.rate == 0.5

    with pytest.raises(InductiveValidationError, match="same one-dimensional shape"):
        _measure_error(np.array([0]), np.array([0, 1]), np.array([0]))


def test_tri_training_paper_rule_accepts_and_rejects() -> None:
    accepted = _paper_update_decision(
        error=0.1,
        previous_error=0.3,
        previous_size=4,
        candidate_size=8,
    )
    assert accepted.accepted is True
    assert accepted.subsample is False
    assert accepted.selected_size == 8
    assert accepted.reason == "accepted_full"

    rejected = _paper_update_decision(
        error=0.3,
        previous_error=0.5,
        previous_size=1,
        candidate_size=10,
    )
    assert rejected.accepted is False
    assert rejected.selected_size == 0
    assert rejected.reason == "noise_bound_not_improved"

    not_improved = _paper_update_decision(
        error=0.5,
        previous_error=0.5,
        previous_size=0,
        candidate_size=10,
    )
    assert not_improved.accepted is False
    assert not_improved.reason == "error_not_improved"

    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        _paper_update_decision(
            error=-0.1,
            previous_error=0.5,
            previous_size=0,
            candidate_size=1,
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        _paper_update_decision(
            error=0.1,
            previous_error=0.5,
            previous_size=-1,
            candidate_size=1,
        )


def test_tri_training_paper_rule_rejects_exact_noise_bound() -> None:
    decision = _paper_update_decision(
        error=0.01,
        previous_error=0.02,
        previous_size=7,
        candidate_size=14,
    )

    assert decision.accepted is False
    assert decision.reason == "noise_bound_not_improved"


def test_tri_training_paper_rule_subsamples_to_equation_10_size() -> None:
    decision = _paper_update_decision(
        error=0.4,
        previous_error=0.5,
        previous_size=5,
        candidate_size=10,
    )

    assert decision.accepted is True
    assert decision.subsample is True
    assert decision.selected_size == 6
    assert 0.4 * decision.selected_size < 0.5 * decision.previous_size


def test_tri_training_subsample_is_seed_deterministic() -> None:
    first = _subsample_positions(20, 6, rng=np.random.default_rng(123))
    second = _subsample_positions(20, 6, rng=np.random.default_rng(123))
    different = _subsample_positions(20, 6, rng=np.random.default_rng(124))

    np.testing.assert_array_equal(first, second)
    assert np.all(first[:-1] < first[1:])
    assert not np.array_equal(first, different)

    np.testing.assert_array_equal(
        _subsample_positions(4, 4, rng=np.random.default_rng(1)),
        np.arange(4),
    )
    with pytest.raises(ValueError, match="between zero and n_candidates"):
        _subsample_positions(4, 5, rng=np.random.default_rng(1))


def test_tri_training_score_labels_cover_default_fallbacks_and_validation() -> None:
    numpy_scores = np.array([[0.1, 0.9], [0.8, 0.2]])
    np.testing.assert_array_equal(
        _numpy_labels_from_scores(SimpleNamespace(), numpy_scores),
        np.array([1, 0]),
    )
    np.testing.assert_array_equal(
        _numpy_labels_from_scores(SimpleNamespace(classes_t_=np.array([4, 9])), numpy_scores),
        np.array([9, 4]),
    )
    with pytest.raises(InductiveValidationError, match="align with score columns"):
        _numpy_labels_from_scores(SimpleNamespace(classes_=np.array([0])), numpy_scores)

    torch_scores = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    torch.testing.assert_close(
        _torch_labels_from_scores(SimpleNamespace(), torch_scores),
        torch.tensor([1, 0]),
    )
    torch.testing.assert_close(
        _torch_labels_from_scores(SimpleNamespace(classes_t_=torch.tensor([4, 9])), torch_scores),
        torch.tensor([9, 4]),
    )
    torch.testing.assert_close(
        _torch_labels_from_scores(SimpleNamespace(classes_=np.array([4, 9])), torch_scores),
        torch.tensor([9, 4]),
    )
    with pytest.raises(InductiveValidationError, match="align with score columns"):
        _torch_labels_from_scores(SimpleNamespace(classes_t_=torch.tensor([0])), torch_scores)


def test_tri_training_candidate_caps_preserve_confidence_order_and_zero_limit() -> None:
    indices_np = np.array([0, 1, 2], dtype=np.int64)
    scores_j_np = np.array([[0.6, 0.4], [0.9, 0.1], [0.9, 0.1]])
    scores_k_np = np.array([[0.6, 0.4], [0.8, 0.2], [0.8, 0.2]])
    assert (
        _cap_numpy_candidates(
            indices_np,
            scores_j_np,
            scores_k_np,
            max_new_labels=None,
        )
        is indices_np
    )
    np.testing.assert_array_equal(
        _cap_numpy_candidates(
            indices_np,
            scores_j_np,
            scores_k_np,
            max_new_labels=2,
        ),
        np.array([1, 2]),
    )
    assert (
        _cap_numpy_candidates(
            indices_np,
            scores_j_np,
            scores_k_np,
            max_new_labels=0,
        ).size
        == 0
    )

    indices_t = torch.tensor([0, 1, 2])
    scores_j_t = torch.as_tensor(scores_j_np)
    scores_k_t = torch.as_tensor(scores_k_np)
    assert (
        _cap_torch_candidates(
            indices_t,
            scores_j_t,
            scores_k_t,
            max_new_labels=None,
        )
        is indices_t
    )
    torch.testing.assert_close(
        _cap_torch_candidates(
            indices_t,
            scores_j_t,
            scores_k_t,
            max_new_labels=2,
        ),
        torch.tensor([1, 2]),
    )
    assert (
        _cap_torch_candidates(
            indices_t,
            scores_j_t,
            scores_k_t,
            max_new_labels=0,
        ).numel()
        == 0
    )


def test_tri_training_global_class_order_ignores_missing_classifier_metadata() -> None:
    np.testing.assert_array_equal(
        _global_class_order(np.array([4, 9]), [SimpleNamespace()]),
        np.array([4, 9]),
    )


def test_tri_training_probability_requirement_is_a_noop_for_score_average() -> None:
    method = TriTrainingMethod(TriTrainingSpec(prediction_rule="score_average"))

    method._require_probability_ensemble([SimpleNamespace()])


def test_tri_training_standardized_profile_preserves_legacy_golden_output() -> None:
    """The default profile remains byte-for-byte compatible with the public API."""

    X_l = np.asarray(
        [
            [-2.0, -2.0],
            [-1.8, -1.7],
            [-1.5, -1.9],
            [1.5, 1.8],
            [1.8, 1.7],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    y_l = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    X_u = np.asarray(
        [
            [-1.7, -1.6],
            [-1.2, -1.4],
            [1.2, 1.3],
            [1.7, 1.5],
            [0.0, 0.1],
        ],
        dtype=np.float32,
    )
    X_test = np.asarray(
        [[-1.6, -1.5], [1.6, 1.4], [0.1, 0.2]],
        dtype=np.float32,
    )

    method = TriTrainingMethod().fit(
        SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u),
        device=DeviceSpec(device="cpu"),
        seed=7,
    )

    np.testing.assert_array_equal(method.predict(X_test), np.asarray([0, 1, 0]))
    np.testing.assert_array_equal(
        method.predict_proba(X_test),
        np.asarray(
            [
                [0.8666667, 0.13333336],
                [0.2, 0.8],
                [0.59999996, 0.4],
            ],
            dtype=np.float32,
        ),
    )
    assert method.diagnostics_ == {}


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_tri_training_standardized_rejects_empty_labeled_data(backend: str) -> None:
    if backend == "numpy":
        X_l = np.empty((0, 2), dtype=np.float32)
        y_l = np.empty((0,), dtype=np.int64)
        X_u = np.zeros((1, 2), dtype=np.float32)
    else:
        X_l = torch.empty((0, 2), dtype=torch.float32)
        y_l = torch.empty((0,), dtype=torch.int64)
        X_u = torch.zeros((1, 2), dtype=torch.float32)

    method = TriTrainingMethod(TriTrainingSpec(classifier_backend=backend))
    module = "modssc.inductive.methods.helpers.tri_training_standardized"
    ensure_data = f"{module}.ensure_{backend}_data"
    ensure_labels = f"{module}.ensure_1d_labels{'_torch' if backend == 'torch' else ''}"
    with (
        patch(ensure_data, side_effect=lambda data, **_kwargs: data),
        patch(ensure_labels, return_value=y_l),
        pytest.raises(InductiveValidationError, match="X_l must be non-empty"),
    ):
        method.fit(
            SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u),
            device=DeviceSpec(device="cpu"),
        )


def test_tri_training_standardized_torch_stops_when_a_pair_never_agrees() -> None:
    class _FixedTorchClassifier:
        def __init__(self, predicted_class: int) -> None:
            self.predicted_class = predicted_class

        def fit(self, X, y):
            del X, y
            return self

        def predict_proba(self, X):
            scores = torch.zeros((len(X), 2), dtype=torch.float32, device=X.device)
            scores[:, self.predicted_class] = 1.0
            return scores

    classifiers = [
        _FixedTorchClassifier(0),
        _FixedTorchClassifier(0),
        _FixedTorchClassifier(1),
    ]
    data = SimpleNamespace(
        X_l=torch.zeros((4, 2), dtype=torch.float32),
        y_l=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        X_u=torch.zeros((2, 2), dtype=torch.float32),
    )
    method = TriTrainingMethod(TriTrainingSpec(classifier_backend="torch", max_iter=1))

    with patch(
        "modssc.inductive.methods.helpers.tri_training_standardized.build_classifier",
        side_effect=classifiers,
    ):
        method.fit(data, device=DeviceSpec(device="cpu"))

    assert method._clfs == classifiers


def test_tri_training_profile_gate_rejects_paper_features_in_standardized_mode() -> None:
    with pytest.raises(InductiveValidationError, match="error_bound_subsample"):
        TriTrainingMethod(TriTrainingSpec(retain_initial_ensemble=True)).fit(
            SimpleNamespace(
                X_l=np.zeros((2, 1), dtype=np.float32),
                y_l=np.asarray([0, 1]),
                X_u=np.zeros((1, 1), dtype=np.float32),
            ),
            device=DeviceSpec(device="cpu"),
        )

    with pytest.raises(InductiveValidationError, match="training_mode"):
        TriTrainingMethod(TriTrainingSpec(training_mode="unknown")).fit(
            SimpleNamespace(
                X_l=np.zeros((2, 1), dtype=np.float32),
                y_l=np.asarray([0, 1]),
                X_u=np.zeros((1, 1), dtype=np.float32),
            ),
            device=DeviceSpec(device="cpu"),
        )


class _RecordingTriClassifier:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])
        self.fit_inputs: list[np.ndarray] = []

    def fit(self, X, y):
        self.fit_inputs.append(np.asarray(X).copy())
        return self

    def predict_proba(self, X):
        return np.tile(np.array([[0.9, 0.1]], dtype=np.float32), (len(X), 1))


def test_tri_training_fit_reports_subsample_diagnostics_and_is_deterministic() -> None:
    X_l = np.column_stack(
        [np.full((10,), -1.0, dtype=np.float32), np.zeros((10,), dtype=np.float32)]
    )
    y_l = np.array([0] * 6 + [1] * 4, dtype=np.int64)
    X_u = np.column_stack([np.arange(10, dtype=np.float32), np.ones((10,), dtype=np.float32)])
    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u)

    def run_once(seed: int):
        classifiers: list[_RecordingTriClassifier] = []

        def factory(_spec, *, seed):
            del seed
            classifier = _RecordingTriClassifier()
            classifiers.append(classifier)
            return classifier

        method = TriTrainingMethod(
            TriTrainingSpec(max_iter=3, training_mode="error_bound_subsample")
        )
        with patch("modssc.inductive.methods.tri_training.build_classifier", side_effect=factory):
            method.fit(data, device=DeviceSpec(device="cpu"), seed=seed)

        selected_ids = [classifier.fit_inputs[-1][10:, 0].copy() for classifier in classifiers]
        return method, selected_ids

    first, first_ids = run_once(17)
    second, second_ids = run_once(17)

    assert first.diagnostics_["n_iter"] == 2
    assert first.diagnostics_["changed_rounds"] == 1
    assert first.diagnostics_["converged"] is True
    assert first.diagnostics_["prediction_rule"] == "score_average"
    assert first.diagnostics_["updates_per_learner"] == [1, 1, 1]
    assert first.diagnostics_["pseudo_labels_selected_per_learner"] == [6, 6, 6]
    assert first.diagnostics_["subsample_events_per_learner"] == [1, 1, 1]
    assert [record["reason"] for record in first.diagnostics_["rounds"][0]["learners"]] == [
        "accepted_subsample",
        "accepted_subsample",
        "accepted_subsample",
    ]
    for selected_first, selected_second in zip(first_ids, second_ids, strict=True):
        np.testing.assert_array_equal(selected_first, selected_second)


def test_tri_training_numpy_zero_rounds_and_confidence_rejection() -> None:
    X_l = np.zeros((10, 2), dtype=np.float32)
    y_l = np.array([0] * 6 + [1] * 4, dtype=np.int64)
    X_u = np.zeros((10, 2), dtype=np.float32)
    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u)

    with patch(
        "modssc.inductive.methods.tri_training.build_classifier",
        side_effect=lambda _spec, *, seed: _RecordingTriClassifier(),
    ):
        zero_rounds = TriTrainingMethod(
            TriTrainingSpec(max_iter=0, training_mode="error_bound_subsample")
        ).fit(
            data,
            device=DeviceSpec(device="cpu"),
            seed=3,
        )
        rejected = TriTrainingMethod(
            TriTrainingSpec(
                max_iter=1,
                confidence_threshold=0.95,
                training_mode="error_bound_subsample",
            )
        ).fit(
            data,
            device=DeviceSpec(device="cpu"),
            seed=3,
        )
        accepted_full = TriTrainingMethod(
            TriTrainingSpec(max_iter=1, training_mode="error_bound_subsample")
        ).fit(
            SimpleNamespace(X_l=X_l, y_l=np.zeros_like(y_l), X_u=X_u),
            device=DeviceSpec(device="cpu"),
            seed=3,
        )

    assert zero_rounds.diagnostics_["n_iter"] == 0
    assert zero_rounds.diagnostics_["converged"] is False
    assert rejected.diagnostics_["n_iter"] == 1
    assert rejected.diagnostics_["converged"] is True
    assert rejected.diagnostics_["updates_per_learner"] == [0, 0, 0]
    assert [learner["reason"] for learner in rejected.diagnostics_["rounds"][0]["learners"]] == [
        "insufficient_candidates"
    ] * 3
    assert accepted_full.diagnostics_["updates_per_learner"] == [1, 1, 1]
    assert accepted_full.diagnostics_["subsample_events_per_learner"] == [0, 0, 0]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("invalid_input", ["missing_unlabeled", "empty_labeled"])
def test_tri_training_paper_validates_required_training_sets(
    backend: str,
    invalid_input: str,
) -> None:
    if backend == "numpy":
        X_l = np.zeros((2, 1), dtype=np.float32)
        y_l = np.array([0, 1], dtype=np.int64)
        X_u = np.zeros((1, 1), dtype=np.float32)
        if invalid_input == "empty_labeled":
            X_l = np.empty((0, 1), dtype=np.float32)
            y_l = np.empty((0,), dtype=np.int64)
    else:
        X_l = torch.zeros((2, 1), dtype=torch.float32)
        y_l = torch.tensor([0, 1], dtype=torch.int64)
        X_u = torch.zeros((1, 1), dtype=torch.float32)
        if invalid_input == "empty_labeled":
            X_l = torch.empty((0, 1), dtype=torch.float32)
            y_l = torch.empty((0,), dtype=torch.int64)
    if invalid_input == "missing_unlabeled":
        X_u = None

    expected = "requires X_u" if invalid_input == "missing_unlabeled" else "X_l must be non-empty"
    method = TriTrainingMethod(
        TriTrainingSpec(classifier_backend=backend, training_mode="error_bound_subsample")
    )
    module = "modssc.inductive.methods.tri_training"
    ensure_data = f"{module}.ensure_{backend}_data"
    ensure_labels = f"{module}.ensure_1d_labels{'_torch' if backend == 'torch' else ''}"
    with (
        patch(ensure_data, side_effect=lambda data, **_kwargs: data),
        patch(ensure_labels, return_value=y_l),
        pytest.raises(InductiveValidationError, match=expected),
    ):
        method.fit(
            SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u),
            device=DeviceSpec(device="cpu"),
        )


def test_tri_training_retains_round_zero_numpy_ensemble_on_request() -> None:
    data = SimpleNamespace(
        X_l=np.zeros((6, 2), dtype=np.float32),
        y_l=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
        X_u=np.zeros((2, 2), dtype=np.float32),
    )
    classifiers: list[_RecordingTriClassifier] = []

    def factory(_spec, *, seed):
        del seed
        classifier = _RecordingTriClassifier()
        classifiers.append(classifier)
        return classifier

    method = TriTrainingMethod(
        TriTrainingSpec(
            max_iter=0,
            training_mode="error_bound_subsample",
            retain_initial_ensemble=True,
        )
    )
    with patch("modssc.inductive.methods.tri_training.build_classifier", side_effect=factory):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=4)

    assert len(classifiers) == 6
    assert method.diagnostics_["initial_ensemble_retained"] is True
    np.testing.assert_allclose(
        method.predict_proba_initial(data.X_l), method.predict_proba(data.X_l)
    )
    assert method._clfs == classifiers[:3]

    without_initial = TriTrainingMethod()
    with pytest.raises(RuntimeError, match="retain_initial_ensemble=true"):
        without_initial.predict_proba_initial(data.X_l)
    with pytest.raises(RuntimeError, match="retain_initial_ensemble=true"):
        without_initial.predict_initial(data.X_l)


class _RecordingTorchTriClassifier:
    def __init__(self) -> None:
        self.classes_t_ = torch.tensor([0, 1])
        self.fit_sizes: list[int] = []

    def fit(self, X, y):
        del y
        self.fit_sizes.append(len(X))
        return self

    def predict_proba(self, X):
        return torch.tensor([0.9, 0.1], dtype=torch.float32, device=X.device).repeat(len(X), 1)


def test_tri_training_torch_threshold_and_paper_subsample_path() -> None:
    X_l = torch.zeros((10, 2), dtype=torch.float32)
    y_l = torch.tensor([0] * 6 + [1] * 4, dtype=torch.int64)
    X_u = torch.zeros((10, 2), dtype=torch.float32)
    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u)
    classifiers: list[_RecordingTorchTriClassifier] = []

    def factory(_spec, *, seed):
        del seed
        classifier = _RecordingTorchTriClassifier()
        classifiers.append(classifier)
        return classifier

    method = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="torch",
            max_iter=2,
            confidence_threshold=0.5,
            training_mode="error_bound_subsample",
        )
    )
    with patch("modssc.inductive.methods.tri_training.build_classifier", side_effect=factory):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=17)
        rejected = TriTrainingMethod(
            TriTrainingSpec(
                classifier_backend="torch",
                max_iter=1,
                confidence_threshold=0.95,
                training_mode="error_bound_subsample",
            )
        ).fit(data, device=DeviceSpec(device="cpu"), seed=17)

    assert method.diagnostics_["n_iter"] == 2
    assert method.diagnostics_["converged"] is True
    assert method.diagnostics_["updates_per_learner"] == [1, 1, 1]
    assert method.diagnostics_["pseudo_labels_selected_per_learner"] == [6, 6, 6]
    assert method.diagnostics_["subsample_events_per_learner"] == [1, 1, 1]
    assert [classifier.fit_sizes for classifier in classifiers[:3]] == [[10, 16]] * 3
    assert rejected.diagnostics_["updates_per_learner"] == [0, 0, 0]


def test_tri_training_torch_without_threshold_accepts_full_candidate_set() -> None:
    data = SimpleNamespace(
        X_l=torch.zeros((10, 2), dtype=torch.float32),
        y_l=torch.zeros((10,), dtype=torch.int64),
        X_u=torch.zeros((10, 2), dtype=torch.float32),
    )
    method = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="torch",
            max_iter=1,
            training_mode="error_bound_subsample",
        )
    )

    with patch(
        "modssc.inductive.methods.tri_training.build_classifier",
        side_effect=lambda _spec, *, seed: _RecordingTorchTriClassifier(),
    ):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=5)

    assert method.diagnostics_["updates_per_learner"] == [1, 1, 1]
    assert method.diagnostics_["subsample_events_per_learner"] == [0, 0, 0]
    assert [record["reason"] for record in method.diagnostics_["rounds"][0]["learners"]] == [
        "accepted_full"
    ] * 3


def test_tri_training_retains_round_zero_torch_ensemble_on_request() -> None:
    data = SimpleNamespace(
        X_l=torch.zeros((6, 2), dtype=torch.float32),
        y_l=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
        X_u=torch.zeros((2, 2), dtype=torch.float32),
    )
    classifiers: list[_RecordingTorchTriClassifier] = []

    def factory(_spec, *, seed):
        del seed
        classifier = _RecordingTorchTriClassifier()
        classifiers.append(classifier)
        return classifier

    method = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="torch",
            max_iter=0,
            training_mode="error_bound_subsample",
            retain_initial_ensemble=True,
        )
    )
    with patch("modssc.inductive.methods.tri_training.build_classifier", side_effect=factory):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=4)

    assert len(classifiers) == 6
    torch.testing.assert_close(
        method.predict_proba_initial(data.X_l), method.predict_proba(data.X_l)
    )


def test_tri_training_torch_backend():
    X_l = torch.zeros(10, 5)
    y_l = torch.zeros(10, dtype=torch.long)
    X_u = torch.zeros(10, 5)

    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u, X_u_w=None, X_u_s=None, views=None, meta=None)

    spec = TriTrainingSpec(classifier_backend="torch", max_iter=1, bootstrap_ratio=1.0)

    model = TriTrainingMethod(spec)

    mock_clf = MagicMock()
    mock_clf.fit.return_value = None
    mock_clf.predict_proba.return_value = torch.tensor([[0.5, 0.5]] * 10)
    mock_clf.predict.return_value = torch.zeros(10)
    mock_clf.classes_ = np.array([0, 1])
    del mock_clf.predict_scores

    with patch("modssc.inductive.methods.tri_training.build_classifier", return_value=mock_clf):
        model.fit(data, device=DeviceSpec(device="cpu"))

        probs = model.predict_proba(X_l)
        assert torch.is_tensor(probs)


class MutableClassesClassifier:
    def __init__(self, responses, scores):
        self._responses = responses
        self._counter = 0
        self.scores = scores

    @property
    def classes_(self):
        val = self._responses[self._counter]
        if self._counter < len(self._responses) - 1:
            self._counter += 1
        return np.array(val)

    def predict_proba(self, X):
        return self.scores


def test_tri_training_alignment_branch_coverage_numpy():
    """Test for lines 310 (numpy) branch coverage where class is not in global map."""
    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    model._backend = "numpy"

    # Clf 1: stable classes [0, 1]
    clf1 = MagicMock()
    clf1.classes_ = np.array([0, 1])
    clf1.predict_proba.return_value = np.zeros((10, 2))

    # Clf 2: Unstable classes.
    # Call 1 (collection): [0, 1, 2] -> global map has {0, 1, 2}
    # Call 2 (alignment): [0, 1, 999] -> 999 not in map -> hits "else" (implicit) branch
    clf2 = MutableClassesClassifier(responses=[[0, 1, 2], [0, 1, 999]], scores=np.zeros((10, 3)))

    model._clfs = [clf1, clf2]

    with patch(
        "modssc.inductive.methods.tri_training.predict_scores",
        side_effect=lambda clf, X, backend: clf.predict_proba(X),
    ):
        X = np.zeros((10, 5))
        # trigger alignment
        _ = model.predict_proba(X)


def test_tri_training_alignment_branch_coverage_torch():
    """Test for line 323 (torch) branch coverage where class is not in global map."""
    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="torch"))
    model._backend = "torch"

    # Clf 1: stable classes [0, 1]
    clf1 = MagicMock()
    clf1.classes_ = torch.tensor([0, 1])
    clf1.predict_proba.return_value = torch.zeros((10, 2))

    # Clf 2: Unstable classes.
    clf2 = MutableClassesClassifier(responses=[[0, 1, 2], [0, 1, 999]], scores=torch.zeros((10, 3)))

    model._clfs = [clf1, clf2]

    with patch(
        "modssc.inductive.methods.tri_training.predict_scores",
        side_effect=lambda clf, X, backend: clf.predict_proba(X),
    ):
        X = torch.zeros(10, 5)
        # trigger alignment
        _ = model.predict_proba(X)


def test_tri_training_valid_alignment():
    """Test that TriTraining correctly aligns scores when classifiers have different class counts."""
    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    model._backend = "numpy"

    clf1 = MagicMock()
    clf1.classes_ = np.array([0, 1])
    # Returns [0.8, 0.2] for class 0 and 1
    clf1.predict_proba.return_value = np.array([[0.8, 0.2]])

    clf2 = MagicMock()
    clf2.classes_ = np.array([0, 1, 2])
    # Returns [0.1, 0.1, 0.8] for class 0, 1, 2
    clf2.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])

    model._clfs = [clf1, clf2]

    with patch(
        "modssc.inductive.methods.tri_training.predict_scores",
        side_effect=lambda clf, X, backend: clf.predict_proba(X),
    ):
        probs = model.predict_proba(np.array([[0]]))

    assert probs.shape == (1, 3)
    # Expected alignment:
    # Clf1: [0.8, 0.2] -> [0.8, 0.2, 0.0]
    # Clf2: [0.1, 0.1, 0.8] -> [0.1, 0.1, 0.8]
    # Avg: [0.45, 0.15, 0.4]
    np.testing.assert_allclose(probs, [[0.45, 0.15, 0.4]])


def test_tri_training_global_class_order_drives_final_and_initial_labels() -> None:
    """The first bootstrap may omit a class and must not define ensemble labels."""

    class _FixedClassifier:
        def __init__(self, classes, scores) -> None:
            self.classes_ = np.asarray(classes)
            self._scores = np.asarray(scores, dtype=np.float32)

        def predict_proba(self, X):
            return np.repeat(self._scores, len(X), axis=0)

    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    model._backend = "numpy"
    model._clfs = [
        _FixedClassifier([1, 2], [[0.1, 0.9]]),
        _FixedClassifier([2, 0, 1], [[0.8, 0.1, 0.1]]),
        _FixedClassifier([0, 2], [[0.1, 0.9]]),
    ]
    model._initial_clfs = [
        _FixedClassifier([2, 1], [[0.1, 0.9]]),
        _FixedClassifier([1, 0, 2], [[0.1, 0.8, 0.1]]),
        _FixedClassifier([0, 2], [[0.8, 0.2]]),
    ]
    X = np.zeros((2, 1), dtype=np.float32)

    final_proba = model.predict_proba(X)
    initial_proba = model.predict_proba_initial(X)

    np.testing.assert_allclose(
        final_proba,
        np.repeat([[1.0 / 15.0, 1.0 / 15.0, 13.0 / 15.0]], 2, axis=0),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        initial_proba,
        np.repeat([[8.0 / 15.0, 1.0 / 3.0, 2.0 / 15.0]], 2, axis=0),
        atol=1e-7,
    )
    np.testing.assert_array_equal(model.predict(X), np.array([2, 2]))
    np.testing.assert_array_equal(model.predict_initial(X), np.array([0, 0]))
    assert [classifier.classes_.tolist() for classifier in model._clfs] == [
        [1, 2],
        [2, 0, 1],
        [0, 2],
    ]


def test_tri_training_global_order_fast_path_and_missing_class_metadata() -> None:
    class _WithoutClasses:
        def __init__(self, scores) -> None:
            self._scores = np.asarray(scores, dtype=np.float32)

        def predict_proba(self, X):
            return np.repeat(self._scores, len(X), axis=0)

    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    model._backend = "numpy"
    model.classes_ = np.array([4, 9])
    model._clfs = [
        _WithoutClasses([[0.2, 0.8]]),
        _WithoutClasses([[0.4, 0.6]]),
    ]
    X = np.zeros((2, 1), dtype=np.float32)

    np.testing.assert_allclose(model.predict_proba(X), [[0.3, 0.7], [0.3, 0.7]])
    np.testing.assert_array_equal(model.predict(X), np.array([9, 9]))

    for classifier in model._clfs:
        classifier.classes_ = np.array([4, 9])
    np.testing.assert_allclose(model.predict_proba(X), [[0.3, 0.7], [0.3, 0.7]])
    np.testing.assert_array_equal(model.predict(X), np.array([9, 9]))

    reversed_order = _WithoutClasses([[0.6, 0.4]])
    reversed_order.classes_ = np.array([9, 4])
    model._clfs = [_WithoutClasses([[0.2, 0.8]]), reversed_order]
    np.testing.assert_allclose(model.predict_proba(X), [[0.3, 0.7], [0.3, 0.7]])


def test_tri_training_paper_prediction_uses_hard_majority_not_soft_average() -> None:
    """Table I returns majority_vote(h1, h2, h3), not averaged confidence."""

    class _FixedClassifier:
        classes_ = np.array([0, 1])

        def __init__(self, scores) -> None:
            self._scores = np.asarray(scores, dtype=np.float32)

        def predict_proba(self, X):
            return np.repeat(self._scores, len(X), axis=0)

    classifiers = [
        _FixedClassifier([[0.51, 0.49]]),
        _FixedClassifier([[0.51, 0.49]]),
        _FixedClassifier([[0.0, 1.0]]),
    ]
    X = np.zeros((2, 1), dtype=np.float32)

    paper = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="numpy",
            training_mode="error_bound_subsample",
            prediction_rule="majority_vote",
        )
    )
    paper._backend = "numpy"
    paper._clfs = classifiers
    paper.classes_ = np.array([0, 1])

    standardized = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    standardized._backend = "numpy"
    standardized._clfs = classifiers
    standardized.classes_ = np.array([0, 1])

    np.testing.assert_allclose(
        paper.predict_proba(X),
        np.repeat([[2.0 / 3.0, 1.0 / 3.0]], len(X), axis=0),
    )
    np.testing.assert_array_equal(paper.predict(X), np.zeros((len(X),), dtype=np.int64))
    np.testing.assert_array_equal(
        standardized.predict(X),
        np.ones((len(X),), dtype=np.int64),
    )


def test_tri_training_soft_average_requires_and_uses_native_probabilities() -> None:
    X = np.zeros((1, 1), dtype=np.float32)

    class ScoreOnly:
        classes_ = np.asarray([0, 1])

        def predict_scores(self, _X):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        def predict(self, _X):
            return np.asarray([0])

    unsupported = TriTrainingMethod(
        TriTrainingSpec(training_mode="error_bound_subsample", prediction_rule="soft_average")
    )
    unsupported._clfs = [ScoreOnly(), ScoreOnly(), ScoreOnly()]
    unsupported._backend = "numpy"
    unsupported.classes_ = np.asarray([0, 1])
    with pytest.raises(InductiveValidationError, match="native class probabilities"):
        unsupported.predict_proba(X)

    class NativeProbability(ScoreOnly):
        def predict_proba(self, _X):
            return np.asarray([[0.2, 0.8]], dtype=np.float32)

    supported = TriTrainingMethod(
        TriTrainingSpec(training_mode="error_bound_subsample", prediction_rule="soft_average")
    )
    supported._clfs = [NativeProbability(), NativeProbability(), NativeProbability()]
    supported._backend = "numpy"
    supported.classes_ = np.asarray([0, 1])
    np.testing.assert_allclose(supported.predict_proba(X), [[0.2, 0.8]])


def test_tri_training_torch_soft_average_validates_probability_tensor_contract() -> None:
    class _ProbabilityClassifier:
        def __init__(self, probabilities) -> None:
            self._probabilities = probabilities

        def predict_proba(self, _X):
            return self._probabilities

    X = torch.zeros((1, 1), dtype=torch.float32)
    method = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="torch",
            training_mode="error_bound_subsample",
            prediction_rule="soft_average",
        )
    )
    method._backend = "torch"

    method._clfs = [_ProbabilityClassifier(np.array([[0.2, 0.8]], dtype=np.float32))]
    with pytest.raises(InductiveValidationError, match="must return a torch.Tensor"):
        method.predict_proba(X)

    method._clfs = [_ProbabilityClassifier(torch.empty((1, 2), device="meta"))]
    with pytest.raises(InductiveValidationError, match="different device"):
        method.predict_proba(X)

    method._clfs = [_ProbabilityClassifier(torch.tensor([0.2, 0.8]))]
    with pytest.raises(InductiveValidationError, match=r"shape \(n_samples, n_classes\)"):
        method.predict_proba(X)


def test_tri_training_paper_majority_vote_supports_torch_and_initial_ensemble() -> None:
    class _FixedClassifier:
        classes_t_ = torch.tensor([4, 9])

        def __init__(self, scores) -> None:
            self._scores = torch.as_tensor(scores, dtype=torch.float32)

        def predict_proba(self, X):
            return self._scores.repeat(len(X), 1)

    classifiers = [
        _FixedClassifier([[0.51, 0.49]]),
        _FixedClassifier([[0.51, 0.49]]),
        _FixedClassifier([[0.0, 1.0]]),
    ]
    X = torch.zeros((2, 1), dtype=torch.float32)
    method = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="torch",
            training_mode="error_bound_subsample",
            prediction_rule="majority_vote",
        )
    )
    method._backend = "torch"
    method._clfs = classifiers
    method._initial_clfs = classifiers
    method.classes_ = np.array([4, 9])
    method.initial_classes_ = np.array([4, 9])

    torch.testing.assert_close(
        method.predict_proba(X),
        torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]).repeat(len(X), 1),
    )
    torch.testing.assert_close(method.predict(X), torch.full((len(X),), 4))
    torch.testing.assert_close(method.predict_initial(X), torch.full((len(X),), 4))
    torch.testing.assert_close(
        method.predict_proba_initial(X),
        method.predict_proba(X),
    )


def test_tri_training_paper_majority_vote_validates_class_order_and_rule() -> None:
    class _FixedClassifier:
        def __init__(self, classes) -> None:
            self.classes_ = None if classes is None else np.asarray(classes)

        def predict_proba(self, X):
            return np.repeat([[0.9, 0.1]], len(X), axis=0)

    X = np.zeros((1, 1), dtype=np.float32)
    no_metadata = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="numpy",
            training_mode="error_bound_subsample",
            prediction_rule="majority_vote",
        )
    )
    no_metadata._backend = "numpy"
    no_metadata._clfs = [_FixedClassifier(None) for _ in range(3)]
    np.testing.assert_allclose(no_metadata.predict_proba(X), [[1.0, 0.0]])

    disagreeing = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="numpy",
            training_mode="error_bound_subsample",
            prediction_rule="majority_vote",
        )
    )
    disagreeing._backend = "numpy"
    disagreeing._clfs = [_FixedClassifier([0, 1]), _FixedClassifier([1, 2])]
    with pytest.raises(InductiveValidationError, match="global class order"):
        disagreeing.predict_proba(X)

    empty = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="numpy",
            training_mode="error_bound_subsample",
            prediction_rule="majority_vote",
        )
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        empty.predict_proba(X)

    invalid = TriTrainingMethod(
        TriTrainingSpec(
            classifier_backend="numpy",
            training_mode="error_bound_subsample",
            prediction_rule="invalid",  # type: ignore[arg-type]
        )
    )
    invalid._backend = "numpy"
    invalid._clfs = [_FixedClassifier([0, 1])]
    with pytest.raises(InductiveValidationError, match="prediction_rule"):
        invalid.predict_proba(X)
    with pytest.raises(InductiveValidationError, match="prediction_rule"):
        invalid.fit(
            SimpleNamespace(
                X_l=np.zeros((2, 1), dtype=np.float32),
                y_l=np.array([0, 1], dtype=np.int64),
                X_u=np.zeros((1, 1), dtype=np.float32),
            ),
            device=DeviceSpec(device="cpu"),
        )


def test_tri_training_rejects_ambiguous_or_out_of_order_class_metadata() -> None:
    class _NumpyClassifier:
        def __init__(self, scores, classes=None) -> None:
            self._scores = np.asarray(scores, dtype=np.float32)
            if classes is not None:
                self.classes_ = np.asarray(classes)

        def predict_proba(self, X):
            return np.repeat(self._scores, len(X), axis=0)

    X_np = np.zeros((1, 1), dtype=np.float32)
    missing_np = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    missing_np._backend = "numpy"
    missing_np.classes_ = np.array([0, 1, 2])
    missing_np._clfs = [_NumpyClassifier([[0.5, 0.5]])]
    with pytest.raises(InductiveValidationError, match="cannot be aligned"):
        missing_np.predict_proba(X_np)

    outside_np = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    outside_np._backend = "numpy"
    outside_np.classes_ = np.array([0, 1])
    outside_np._clfs = [_NumpyClassifier([[0.5, 0.5]], classes=[0, 2])]
    with pytest.raises(InductiveValidationError, match="outside the fitted global class order"):
        outside_np.predict_proba(X_np)

    invalid_order = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    invalid_order._backend = "numpy"
    invalid_order.classes_ = np.array([0, 0])
    invalid_order._clfs = [_NumpyClassifier([[0.5, 0.5]], classes=[0, 1])]
    with pytest.raises(InductiveValidationError, match="distinct class labels"):
        invalid_order.predict_proba(X_np)

    class _TorchClassifier:
        def __init__(self, scores, classes=None) -> None:
            self._scores = torch.as_tensor(scores, dtype=torch.float32)
            if classes is not None:
                self.classes_t_ = torch.as_tensor(classes)

        def predict_proba(self, X):
            return self._scores.to(X.device).repeat(len(X), 1)

    X_t = torch.zeros((1, 1), dtype=torch.float32)
    partial_t = TriTrainingMethod(TriTrainingSpec(classifier_backend="torch"))
    partial_t._backend = "torch"
    partial_t.classes_ = np.array([0, 1])
    partial_t._clfs = [
        _TorchClassifier([[0.2, 0.8]]),
        _TorchClassifier([[0.6, 0.4]], classes=[1, 0]),
    ]
    torch.testing.assert_close(partial_t.predict_proba(X_t), torch.tensor([[0.3, 0.7]]))

    missing_t = TriTrainingMethod(TriTrainingSpec(classifier_backend="torch"))
    missing_t._backend = "torch"
    missing_t.classes_ = np.array([0, 1, 2])
    missing_t._clfs = [_TorchClassifier([[0.5, 0.5]])]
    with pytest.raises(InductiveValidationError, match="cannot be aligned"):
        missing_t.predict_proba(X_t)

    outside_t = TriTrainingMethod(TriTrainingSpec(classifier_backend="torch"))
    outside_t._backend = "torch"
    outside_t.classes_ = np.array([0, 1])
    outside_t._clfs = [_TorchClassifier([[0.5, 0.5]], classes=[0, 2])]
    with pytest.raises(InductiveValidationError, match="outside the fitted global class order"):
        outside_t.predict_proba(X_t)


def test_tri_training_validation_error_shape_mismatch():
    """Test validation error when one classifier returned shape doesn't match its classes."""
    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    model._backend = "numpy"

    clf1 = MagicMock()
    clf1.classes_ = np.array([0, 1])
    # Incorrect shape: 3 columns for 2 classes
    clf1.predict_proba.return_value = np.zeros((10, 3))

    clf2 = MagicMock()
    clf2.classes_ = np.array([0, 1, 2])
    clf2.predict_proba.return_value = np.zeros((10, 3))

    model._clfs = [clf1, clf2]

    with (
        patch(
            "modssc.inductive.methods.tri_training.predict_scores",
            side_effect=lambda clf, X, backend: clf.predict_proba(X),
        ),
        pytest.raises(
            InductiveValidationError, match="TriTraining classifiers disagree on class counts"
        ),
    ):
        model.predict_proba(np.zeros((10, 5)))


from ._tri_training_coverage import *  # noqa: E402,F401,F403
