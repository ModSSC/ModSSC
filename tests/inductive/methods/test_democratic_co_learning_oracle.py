from __future__ import annotations

from collections import Counter
from inspect import Parameter, signature
from types import SimpleNamespace

import numpy as np
import pytest

from modssc.inductive.methods import democratic_co_learning as dcl
from modssc.inductive.types import DeviceSpec


def _figure1_round_oracle(
    predictions: list[list[int]],
    weights: list[float],
    lower_bounds: list[float],
    *,
    learner_index: int,
    current_labeled_count: int,
    current_error_estimate: float,
    n_classes: int,
) -> dict[str, object]:
    """Independent transcription of the Figure 1 selection equations."""
    n_learners = len(predictions)
    n_samples = len(predictions[0])
    majority_labels: list[int] = []
    eligible: list[bool] = []
    proposed: list[int] = []
    proposal_error = 0.0

    for sample_index in range(n_samples):
        sample_predictions = [row[sample_index] for row in predictions]
        counts = Counter(sample_predictions)
        majority = max(range(n_classes), key=lambda label: (counts[label], -label))
        majority_labels.append(majority)

        strict_majority = 2 * counts[majority] > n_learners
        confidence_by_label = {
            label: sum(
                weights[index]
                for index, prediction in enumerate(sample_predictions)
                if prediction == label
            )
            for label in range(n_classes)
        }
        minority_confidence = max(
            confidence for label, confidence in confidence_by_label.items() if label != majority
        )
        confidence_ok = confidence_by_label[majority] > minority_confidence
        is_eligible = strict_majority and confidence_ok
        eligible.append(is_eligible)

        if is_eligible and sample_predictions[learner_index] != majority:
            proposed.append(sample_index)
            supporters = [
                index
                for index, prediction in enumerate(sample_predictions)
                if prediction == majority
            ]
            mean_lower_bound = sum(lower_bounds[index] for index in supporters) / len(supporters)
            proposal_error += 1.0 - mean_lower_bound

    q = current_labeled_count * (1.0 - 2.0 * current_error_estimate / current_labeled_count) ** 2
    new_count = current_labeled_count + len(proposed)
    q_prime = new_count * (1.0 - 2.0 * (current_error_estimate + proposal_error) / new_count) ** 2
    return {
        "majority_labels": majority_labels,
        "eligible": eligible,
        "proposed": proposed,
        "proposal_error": proposal_error,
        "q": q,
        "q_prime": q_prime,
        "accept": q_prime > q,
    }


def _figure2_combine_oracle(
    predictions: list[list[int]],
    weights: list[float],
    *,
    n_classes: int,
    min_confidence: float,
) -> list[list[float]]:
    """Independent transcription of the Figure 2 Laplace-corrected vote."""
    n_samples = len(predictions[0])
    rows: list[list[float]] = []
    for sample_index in range(n_samples):
        scores: list[float] = []
        for label in range(n_classes):
            members = [
                weights[index]
                for index, row in enumerate(predictions)
                if row[sample_index] == label and weights[index] > min_confidence
            ]
            if not members:
                scores.append(0.0)
                continue
            group_size = len(members)
            mean_confidence = sum(members) / group_size
            scores.append(((group_size + 0.5) / (group_size + 1.0)) * mean_confidence)
        rows.append(scores)
    return rows


def test_democratic_co_learning_does_not_claim_an_official_implementation() -> None:
    assert dcl.DemocraticCoLearningMethod.info.official_code is None
    assert dcl.DemocraticCoLearningMethod.info.paper_pdf.endswith(
        "2004-Democratic colearning/21-2004-Democratic colearning.pdf"
    )


def test_democratic_training_mode_is_keyword_only_and_fails_closed() -> None:
    mode = signature(dcl.DemocraticCoLearningSpec).parameters["training_mode"]
    assert mode.kind is Parameter.KEYWORD_ONLY
    with pytest.raises(dcl.InductiveValidationError, match="training_mode must be one of"):
        dcl.DemocraticCoLearningMethod(dcl.DemocraticCoLearningSpec(training_mode="unknown")).fit(
            None, device=DeviceSpec(device="cpu")
        )


def test_democratic_standardized_weighted_vote_golden_differs_from_paper_rule() -> None:
    predictions = np.asarray([[1], [0], [0]], dtype=np.int64)
    weights = np.asarray([0.99, 0.30, 0.30], dtype=np.float64)

    standardized_label, standardized_ok = dcl._standardized_weighted_majority_numpy(
        predictions,
        weights,
        n_classes=2,
    )
    paper_label, paper_ok = dcl._weighted_majority_numpy(
        predictions,
        weights,
        n_classes=2,
    )

    assert standardized_label.tolist() == [1]
    assert standardized_ok.tolist() == [True]
    assert paper_label.tolist() == [0]
    assert paper_ok.tolist() == [False]

    torch = pytest.importorskip("torch")
    single_numpy, ok_single_numpy = dcl._standardized_weighted_majority_numpy(
        np.zeros((3, 2), dtype=np.int64),
        np.ones(3, dtype=np.float64),
        n_classes=1,
    )
    single_torch, ok_single_torch = dcl._standardized_weighted_majority_torch(
        torch.zeros((3, 2), dtype=torch.int64),
        torch.ones(3, dtype=torch.float32),
        n_classes=1,
    )
    assert single_numpy.tolist() == [0, 0]
    assert ok_single_numpy.tolist() == [True, True]
    assert single_torch.tolist() == [0, 0]
    assert ok_single_torch.tolist() == [True, True]


def test_democratic_standardized_final_vote_preserves_weak_learner_filter() -> None:
    torch = pytest.importorskip("torch")
    predictions = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    weights = np.asarray([0.9, 0.1], dtype=np.float64)
    expected = dcl._standardized_combine_scores_numpy(
        predictions,
        weights,
        n_classes=2,
        min_confidence=0.5,
    )
    actual = dcl._standardized_combine_scores_torch(
        torch.as_tensor(predictions),
        torch.as_tensor(weights),
        n_classes=2,
        min_confidence=0.5,
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=1e-12)
    assert expected.tolist() == [[0.675, 0.0], [0.0, 0.675]]


def test_democratic_legacy_mode_rejects_confidence_weighted_controls() -> None:
    method = dcl.DemocraticCoLearningMethod(dcl.DemocraticCoLearningSpec(diagnostic_trace=True))
    data = SimpleNamespace(
        X_l=np.asarray([[0.0], [1.0]], dtype=np.float32),
        y_l=np.asarray([0, 1], dtype=np.int64),
        X_u=None,
    )

    with pytest.raises(dcl.InductiveValidationError, match="confidence_weighted"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_democratic_figure1_helpers_match_independent_equation_oracle() -> None:
    predictions = [
        [1, 1, 2, 0],
        [0, 1, 2, 1],
        [0, 0, 2, 2],
    ]
    weights = [0.99, 0.30, 0.30]
    lower_bounds = [0.90, 0.80, 0.60]
    oracle = _figure1_round_oracle(
        predictions,
        weights,
        lower_bounds,
        learner_index=2,
        current_labeled_count=10,
        current_error_estimate=0.0,
        n_classes=3,
    )

    assert oracle["majority_labels"] == [0, 1, 2, 0]
    assert oracle["eligible"] == [False, True, True, False]
    assert oracle["proposed"] == [1]
    assert oracle["proposal_error"] == pytest.approx(0.15)
    assert oracle["q"] == pytest.approx(10.0)
    assert oracle["q_prime"] == pytest.approx(10.408181818181818)
    assert oracle["accept"] is True

    predictions_array = np.asarray(predictions, dtype=np.int64)
    weights_array = np.asarray(weights, dtype=np.float64)
    majority, eligible = dcl._weighted_majority_numpy(
        predictions_array,
        weights_array,
        n_classes=3,
    )
    proposed = np.asarray(oracle["proposed"], dtype=np.int64)
    implementation_error = dcl._proposal_error_numpy(
        preds_idx=predictions_array,
        majority_idx=majority,
        proposed_idx=proposed,
        lower_bounds=np.asarray(lower_bounds, dtype=np.float64),
    )

    assert majority.tolist() == oracle["majority_labels"]
    assert eligible.tolist() == oracle["eligible"]
    assert implementation_error == pytest.approx(oracle["proposal_error"])


def test_democratic_figure2_helper_matches_independent_equation_oracle() -> None:
    predictions = [
        [1, 1],
        [1, 0],
        [0, 0],
    ]
    weights = [0.90, 0.70, 0.40]
    expected = _figure2_combine_oracle(
        predictions,
        weights,
        n_classes=2,
        min_confidence=0.5,
    )

    np.testing.assert_allclose(
        np.asarray(expected),
        np.asarray(
            [
                [0.0, 2.0 / 3.0],
                [0.525, 0.675],
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    )
    actual = dcl._combine_scores_numpy(
        np.asarray(predictions, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        n_classes=2,
        min_confidence=0.5,
    )

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "weights",
    [
        np.asarray([0.4, 0.3, 0.2], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    ],
)
def test_democratic_figure2_all_filtered_fails_closed_for_each_backend(
    weights: np.ndarray,
) -> None:
    predictions = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.int64)
    with pytest.raises(dcl.InductiveValidationError, match="no learner above"):
        dcl._combine_scores_numpy(
            predictions,
            weights,
            n_classes=2,
            min_confidence=0.5,
        )

    torch = pytest.importorskip("torch")
    with pytest.raises(dcl.InductiveValidationError, match="no learner above"):
        dcl._combine_scores_torch(
            torch.as_tensor(predictions),
            torch.as_tensor(weights),
            n_classes=2,
            min_confidence=0.5,
        )


def test_democratic_figure2_rejects_zero_effective_weight_for_each_backend() -> None:
    predictions = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.int64)
    weights = np.zeros(3, dtype=np.float64)
    with pytest.raises(dcl.InductiveValidationError, match="no positive eligible weight"):
        dcl._combine_scores_numpy(
            predictions,
            weights,
            n_classes=2,
            min_confidence=-0.1,
        )

    torch = pytest.importorskip("torch")
    with pytest.raises(dcl.InductiveValidationError, match="no positive eligible weight"):
        dcl._combine_scores_torch(
            torch.as_tensor(predictions),
            torch.as_tensor(weights),
            n_classes=2,
            min_confidence=-0.1,
        )


@pytest.mark.parametrize(
    "weights,message",
    [
        (np.asarray([0.5, 0.4]), "one vote weight"),
        (np.asarray([0.5, np.nan, 0.4]), "finite and non-negative"),
        (np.asarray([0.5, -0.1, 0.4]), "finite and non-negative"),
    ],
)
def test_democratic_figure2_rejects_invalid_vote_weights(
    weights: np.ndarray,
    message: str,
) -> None:
    predictions = np.asarray([[1], [1], [0]], dtype=np.int64)
    with pytest.raises(dcl.InductiveValidationError, match=message):
        dcl._combine_scores_numpy(
            predictions,
            weights,
            n_classes=2,
            min_confidence=0.5,
        )
    torch = pytest.importorskip("torch")
    with pytest.raises(dcl.InductiveValidationError, match=message):
        dcl._combine_scores_torch(
            torch.as_tensor(predictions),
            torch.as_tensor(weights),
            n_classes=2,
            min_confidence=0.5,
        )


@pytest.mark.parametrize(
    "interval,expected",
    [
        ("wald", (0.19010248384771933, 0.8098975161522807)),
        ("wilson", (0.23659309051256405, 0.763406909487436)),
        ("clopper_pearson", (0.1870860284473987, 0.8129139715526013)),
    ],
)
def test_democratic_confidence_intervals_match_binomial_references(
    interval: str, expected: tuple[float, float]
) -> None:
    assert dcl._accuracy_confidence_interval(
        5,
        10,
        confidence_level=0.95,
        interval=interval,
    ) == pytest.approx(expected)


def test_democratic_clopper_pearson_handles_boundaries_without_scipy() -> None:
    assert dcl._accuracy_confidence_interval(
        0,
        3,
        confidence_level=0.95,
        interval="clopper_pearson",
    ) == pytest.approx((0.0, 0.7075982261787133))
    assert dcl._accuracy_confidence_interval(
        3,
        3,
        confidence_level=0.95,
        interval="clopper_pearson",
    ) == pytest.approx((0.29240177382128674, 1.0))

    assert (
        dcl._binomial_tail_probability(
            correct=1,
            total=3,
            probability=0.0,
            upper_tail=True,
        )
        == 0.0
    )
    assert (
        dcl._binomial_tail_probability(
            correct=1,
            total=3,
            probability=0.0,
            upper_tail=False,
        )
        == 1.0
    )
    assert (
        dcl._binomial_tail_probability(
            correct=2,
            total=3,
            probability=1.0,
            upper_tail=True,
        )
        == 1.0
    )
    assert (
        dcl._binomial_tail_probability(
            correct=2,
            total=3,
            probability=1.0,
            upper_tail=False,
        )
        == 0.0
    )


def test_democratic_confidence_interval_validation() -> None:
    with pytest.raises(dcl.InductiveValidationError, match=r"correct must be in \[0, total\]"):
        dcl._accuracy_confidence_interval(
            -1,
            3,
            confidence_level=0.95,
        )
    with pytest.raises(dcl.InductiveValidationError, match="confidence_interval"):
        dcl._accuracy_confidence_interval(
            1,
            3,
            confidence_level=0.95,
            interval="unknown",
        )


def test_democratic_stratified_folds_are_seeded_and_replayable() -> None:
    labels = np.repeat(np.array([0, 1, 2], dtype=np.int64), 6)
    first = dcl._stratified_kfold_indices(labels, n_splits=3, seed=17)
    replay = dcl._stratified_kfold_indices(labels, n_splits=3, seed=17)
    other = dcl._stratified_kfold_indices(labels, n_splits=3, seed=18)

    assert len(first) == 3
    assert all(
        np.array_equal(train, replay[index][0]) and np.array_equal(validation, replay[index][1])
        for index, (train, validation) in enumerate(first)
    )
    assert any(
        not np.array_equal(validation, other[index][1])
        for index, (_train, validation) in enumerate(first)
    )
    np.testing.assert_array_equal(
        np.sort(np.concatenate([validation for _train, validation in first])),
        np.arange(labels.size),
    )
    for _train, validation in first:
        assert np.bincount(labels[validation], minlength=3).tolist() == [2, 2, 2]


@pytest.mark.parametrize(
    "labels,n_splits,message",
    [
        (np.array([0, 0]), 1, "confidence_folds must be"),
        (np.array([0, 0]), 3, "cannot exceed"),
        (np.array([0, 0, 0, 1]), 2, "at least 2 labeled examples"),
    ],
)
def test_democratic_stratified_fold_validation(
    labels: np.ndarray, n_splits: int, message: str
) -> None:
    with pytest.raises(dcl.InductiveValidationError, match=message):
        dcl._stratified_kfold_indices(labels, n_splits=n_splits, seed=0)


def test_democratic_stratified_folds_allow_vote_seed_17_class_counts() -> None:
    labels = np.asarray([0] * 31 + [1] * 9)

    folds = dcl._stratified_kfold_indices(labels, n_splits=10, seed=0)

    assert len(folds) == 10
    validation = np.concatenate([fold_validation for _fold_train, fold_validation in folds])
    np.testing.assert_array_equal(np.sort(validation), np.arange(40))
    assert [len(fold_validation) for _fold_train, fold_validation in folds] == [4] * 10
    assert all(np.unique(labels[fold_train]).size == 2 for fold_train, _fold_validation in folds)


def test_democratic_stratified_folds_never_emit_an_empty_validation_fold() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    folds = dcl._stratified_kfold_indices(labels, n_splits=4, seed=3)

    assert [len(validation) for _train, validation in folds] == [1, 1, 1, 1]
    np.testing.assert_array_equal(
        np.sort(np.concatenate([validation for _train, validation in folds])),
        np.arange(4),
    )


class _FeatureRuleClassifier:
    def __init__(self, learner_index: int, *, torch_backend: bool = False) -> None:
        self.learner_index = learner_index
        self.torch_backend = torch_backend
        if torch_backend:
            torch = pytest.importorskip("torch")
            self.classes_t_ = torch.tensor([0, 1], dtype=torch.int64)
        else:
            self.classes_ = np.array([0, 1], dtype=np.int64)

    def fit(self, _x, _y):
        return self

    def predict(self, x):
        values = x[:, 0].to(dtype=self.classes_t_.dtype) if self.torch_backend else x[:, 0]
        predictions = values % 2
        return 1 - predictions if self.learner_index == 1 else predictions


def _control_specs(*, backend: str = "numpy") -> tuple[dcl.BaseClassifierSpec, ...]:
    return tuple(
        dcl.BaseClassifierSpec(
            classifier_id=classifier_id,
            classifier_backend=backend,
        )
        for classifier_id in ("nb", "j48", "knn3")
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_democratic_paper_regular_prediction_uses_paper_vote(backend: str) -> None:
    torch_backend = backend == "torch"
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_backend=backend,
            training_mode="confidence_weighted",
        )
    )
    method._clfs = [
        _FeatureRuleClassifier(index, torch_backend=torch_backend) for index in range(3)
    ]
    method._backend = backend
    method._weights = np.asarray([0.9, 0.8, 0.7], dtype=np.float64)
    method._classes = np.asarray([0, 1], dtype=np.int64)
    if torch_backend:
        torch = pytest.importorskip("torch")
        method._classes_t = torch.tensor([0, 1], dtype=torch.int64)
        X = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
        expected = (X[:, 0] % 2).to(dtype=torch.int64)
        assert torch.equal(method.predict_proba(X).argmax(dim=1), expected)
        return

    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    expected = (X[:, 0] % 2).astype(np.int64)
    np.testing.assert_array_equal(method.predict_proba(X).argmax(axis=1), expected)


def test_democratic_paper_regular_prediction_validates_fitted_state() -> None:
    paper_spec = dcl.DemocraticCoLearningSpec(training_mode="confidence_weighted")
    torch = pytest.importorskip("torch")

    mismatch = dcl.DemocraticCoLearningMethod(paper_spec)
    mismatch._clfs = [object()]
    mismatch._backend = ""
    with pytest.raises(dcl.InductiveValidationError, match="backend mismatch"):
        mismatch.predict_proba(torch.zeros((1, 1)))

    missing_weights = dcl.DemocraticCoLearningMethod(paper_spec)
    missing_weights._clfs = [object()]
    missing_weights._backend = "numpy"
    with pytest.raises(RuntimeError, match="missing weights"):
        missing_weights.predict_proba(np.zeros((1, 1), dtype=np.float32))

    missing_numpy_classes = dcl.DemocraticCoLearningMethod(paper_spec)
    missing_numpy_classes._clfs = [object()]
    missing_numpy_classes._backend = "numpy"
    missing_numpy_classes._weights = np.ones(1, dtype=np.float64)
    with pytest.raises(RuntimeError, match="missing classes"):
        missing_numpy_classes.predict_proba(np.zeros((1, 1), dtype=np.float32))

    missing_torch_classes = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_backend="torch",
            training_mode="confidence_weighted",
        )
    )
    missing_torch_classes._clfs = [object()]
    missing_torch_classes._backend = "torch"
    missing_torch_classes._weights = np.ones(1, dtype=np.float64)
    with pytest.raises(RuntimeError, match="missing classes"):
        missing_torch_classes.predict_proba(torch.zeros((1, 1)))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_democratic_paper_without_unlabeled_keeps_controls_disabled(
    monkeypatch,
    backend: str,
) -> None:
    torch_backend = backend == "torch"
    specs = _control_specs(backend=backend)
    learner_by_id = {spec.classifier_id: index for index, spec in enumerate(specs)}
    monkeypatch.setattr(
        dcl,
        "build_classifier",
        lambda spec, seed=0: _FeatureRuleClassifier(
            learner_by_id[spec.classifier_id],
            torch_backend=torch_backend,
        ),
    )
    if torch_backend:
        torch = pytest.importorskip("torch")
        X_l = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
        y_l = (X_l[:, 0] % 2).to(dtype=torch.int64)
    else:
        X_l = np.arange(4, dtype=np.float32).reshape(-1, 1)
        y_l = (X_l[:, 0] % 2).astype(np.int64)
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            training_mode="confidence_weighted",
        )
    )

    method.fit(
        SimpleNamespace(X_l=X_l, y_l=y_l, X_u=None),
        device=DeviceSpec(device="cpu"),
        seed=5,
    )

    assert method._initial_clfs == []
    assert method._initial_weights is None
    assert method.converged_ is True


def test_democratic_paper_fit_validates_data_and_labeled_rows(monkeypatch) -> None:
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(training_mode="confidence_weighted")
    )
    with pytest.raises(dcl.InductiveValidationError, match="data must not be None"):
        method.fit(None, device=DeviceSpec(device="cpu"), seed=0)
    monkeypatch.setattr(dcl, "ensure_numpy_data", lambda data: data)
    with pytest.raises(dcl.InductiveValidationError, match="X_l must be non-empty"):
        method.fit(
            SimpleNamespace(
                X_l=np.empty((0, 1), dtype=np.float32),
                y_l=np.asarray([0], dtype=np.int64),
                X_u=None,
            ),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    torch = pytest.importorskip("torch")
    torch_method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_backend="torch",
            training_mode="confidence_weighted",
        )
    )
    monkeypatch.setattr(dcl, "ensure_torch_data", lambda data, device: data)
    with pytest.raises(dcl.InductiveValidationError, match="X_l must be non-empty"):
        torch_method.fit(
            SimpleNamespace(
                X_l=torch.empty((0, 1)),
                y_l=torch.tensor([0], dtype=torch.int64),
                X_u=None,
            ),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )


def test_democratic_kfold_oof_numpy_and_torch_are_deterministic(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    numpy_seeds: list[int] = []

    def build_numpy(_spec, *, seed):
        numpy_seeds.append(seed)
        return _FeatureRuleClassifier(0)

    monkeypatch.setattr(dcl, "build_classifier", build_numpy)
    X_numpy = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y_numpy = (X_numpy[:, 0] % 2).astype(np.int64)
    spec_numpy = dcl.DemocraticCoLearningSpec(
        confidence_estimator="kfold_oof",
        confidence_folds=3,
        confidence_seed=17,
    )
    method_numpy = dcl.DemocraticCoLearningMethod(spec_numpy)
    learner_spec_numpy = dcl.BaseClassifierSpec()
    first_numpy = method_numpy._oof_predictions_numpy(
        learner_spec_numpy,
        X_numpy,
        y_numpy,
        learner_index=2,
    )
    second_numpy = method_numpy._oof_predictions_numpy(
        learner_spec_numpy,
        X_numpy,
        y_numpy,
        learner_index=2,
    )
    numpy_intervals = method_numpy._confidence_intervals_numpy(
        [object()],
        [learner_spec_numpy],
        X_numpy,
        y_numpy,
    )
    np.testing.assert_array_equal(first_numpy, y_numpy)
    np.testing.assert_array_equal(second_numpy, y_numpy)
    assert numpy_intervals == [(1.0, 1.0)]
    assert numpy_seeds == [23, 24, 25, 23, 24, 25, 17, 18, 19]

    torch_seeds: list[int] = []

    def build_torch(_spec, *, seed):
        torch_seeds.append(seed)
        return _FeatureRuleClassifier(0, torch_backend=True)

    monkeypatch.setattr(dcl, "build_classifier", build_torch)
    X_torch = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
    y_torch = (X_torch[:, 0] % 2).to(dtype=torch.int64)
    spec_torch = dcl.DemocraticCoLearningSpec(
        classifier_backend="torch",
        confidence_estimator="kfold_oof",
        confidence_folds=3,
        confidence_seed=17,
    )
    method_torch = dcl.DemocraticCoLearningMethod(spec_torch)
    learner_spec_torch = dcl.BaseClassifierSpec(classifier_backend="torch")
    first_torch = method_torch._oof_predictions_torch(
        learner_spec_torch,
        X_torch,
        y_torch,
        learner_index=1,
    )
    torch_intervals = method_torch._confidence_intervals_torch(
        [object()],
        [learner_spec_torch],
        X_torch,
        y_torch,
    )
    assert torch.equal(first_torch, y_torch)
    assert torch_intervals == [(1.0, 1.0)]
    assert torch_seeds == [20, 21, 22, 17, 18, 19]


def test_democratic_kfold_oof_uses_each_held_out_fold_without_leakage(
    monkeypatch,
) -> None:
    x = np.arange(40, dtype=np.float32).reshape(-1, 1)
    y = (x[:, 0].astype(np.int64) % 2).astype(np.int64)
    fitted: list[dict[str, object]] = []

    class _FoldSpy:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.train_rows: list[int] = []
            self.validation_rows: list[int] = []
            fitted.append({"seed": seed, "classifier": self})

        def fit(self, fold_x, _fold_y):
            self.train_rows = np.asarray(fold_x)[:, 0].astype(int).tolist()
            return self

        def predict(self, fold_x):
            rows = np.asarray(fold_x)[:, 0].astype(int)
            self.validation_rows = rows.tolist()
            return rows % 2

    monkeypatch.setattr(
        dcl,
        "build_classifier",
        lambda _spec, seed=0: _FoldSpy(seed),
    )
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            confidence_estimator="kfold_oof",
            confidence_folds=10,
            confidence_seed=7,
        )
    )

    predictions = method._oof_predictions_numpy(
        dcl.BaseClassifierSpec(),
        x,
        y,
        learner_index=2,
    )

    expected_folds = dcl._stratified_kfold_indices(y, n_splits=10, seed=7)
    assert np.array_equal(predictions, y)
    assert [record["seed"] for record in fitted] == list(range(27, 37))
    assert len(fitted) == len(expected_folds)
    observed_validation: list[int] = []
    for record, (train_indices, validation_indices) in zip(
        fitted,
        expected_folds,
        strict=True,
    ):
        classifier = record["classifier"]
        assert isinstance(classifier, _FoldSpy)
        assert classifier.train_rows == x[train_indices, 0].astype(int).tolist()
        assert classifier.validation_rows == x[validation_indices, 0].astype(int).tolist()
        assert set(classifier.train_rows).isdisjoint(classifier.validation_rows)
        observed_validation.extend(classifier.validation_rows)
    assert sorted(observed_validation) == list(range(40))


def test_democratic_v2_controls_and_protocol_diagnostics_numpy(monkeypatch) -> None:
    specs = _control_specs()
    learner_by_id = {spec.classifier_id: index for index, spec in enumerate(specs)}

    monkeypatch.setattr(
        dcl,
        "build_classifier",
        lambda spec, seed=0: _FeatureRuleClassifier(learner_by_id[spec.classifier_id]),
    )
    X_l = np.arange(4, dtype=np.float32).reshape(-1, 1)
    y_l = (X_l[:, 0] % 2).astype(np.int64)
    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=None)
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            confidence_interval="wilson",
            diagnostic_trace=True,
            training_mode="confidence_weighted",
        )
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=11)

    learner_one = method.predict_proba_control(X_l, control_mode="learner_1")
    combining = method.predict_proba_control(X_l, control_mode="combining_only")
    np.testing.assert_array_equal(learner_one.argmax(axis=1), 1 - y_l)
    np.testing.assert_array_equal(combining.argmax(axis=1), y_l)
    np.testing.assert_array_equal(method.predict_proba_initial(X_l), combining)
    assert method.diagnostics_["confidence_protocol"] == {
        "estimator": "training_accuracy",
        "interval": "wilson",
        "folds": 10,
        "seed": 0,
    }
    assert method.diagnostics_["control"] == {
        "mode": "dcl",
        "available_modes": [
            "learner_0",
            "learner_1",
            "learner_2",
            "combining_only",
        ],
        "learner_ids": ["nb", "j48", "knn3"],
    }
    assert method.diagnostics_["round_trace"] == []

    selected = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            diagnostic_trace=False,
            control_mode="learner_1",
            training_mode="confidence_weighted",
        )
    )
    selected.fit(data, device=DeviceSpec(device="cpu"), seed=11)
    np.testing.assert_array_equal(selected.predict_proba(X_l).argmax(axis=1), 1 - y_l)
    assert selected.diagnostics_["control"]["mode"] == "learner_1"


def test_democratic_v2_controls_torch(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    specs = _control_specs(backend="torch")
    learner_by_id = {spec.classifier_id: index for index, spec in enumerate(specs)}
    monkeypatch.setattr(
        dcl,
        "build_classifier",
        lambda spec, seed=0: _FeatureRuleClassifier(
            learner_by_id[spec.classifier_id],
            torch_backend=True,
        ),
    )
    X_l = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    y_l = (X_l[:, 0] % 2).to(dtype=torch.int64)
    X_u = torch.arange(10, 13, dtype=torch.float32).reshape(-1, 1)
    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u)
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            diagnostic_trace=True,
            control_mode="combining_only",
            training_mode="confidence_weighted",
        )
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=11)

    learner_one = method.predict_proba_control(X_l, control_mode="learner_1")
    combining = method.predict_proba_control(X_l, control_mode="combining_only")
    assert torch.equal(learner_one.argmax(dim=1), 1 - y_l)
    assert torch.equal(combining.argmax(dim=1), y_l)
    assert torch.equal(method.predict(X_u), (X_u[:, 0] % 2).to(dtype=torch.int64))
    assert method.n_iter_ == 0
    assert method.pseudo_labels_added_per_learner_ == (0, 0, 0)
    assert method.diagnostics_["round_trace"] == []


def test_democratic_control_mode_skips_pseudo_label_rounds(monkeypatch) -> None:
    specs = _control_specs()
    learner_by_id = {spec.classifier_id: index for index, spec in enumerate(specs)}
    built: list[_RoundClassifier] = []

    def build_control(spec, seed=0):
        del seed
        classifier = _RoundClassifier(learner_by_id[spec.classifier_id])
        built.append(classifier)
        return classifier

    monkeypatch.setattr(dcl, "build_classifier", build_control)
    data = SimpleNamespace(
        X_l=np.array([[0.0], [1.0]], dtype=np.float32),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u=np.array([[10.0], [11.0], [12.0]], dtype=np.float32),
    )
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            control_mode="combining_only",
            diagnostic_trace=True,
            training_mode="confidence_weighted",
        )
    )

    method.fit(data, device=DeviceSpec(device="cpu"), seed=7)

    assert len(built) == 3
    assert method.n_iter_ == 0
    assert method.changed_rounds_ == 0
    assert method.converged_ is True
    assert method.pseudo_labels_added_per_learner_ == (0, 0, 0)
    assert method.diagnostics_["round_trace"] == []
    assert method.predict(data.X_u).tolist() == [0, 0, 0]


def test_democratic_control_validation_and_unretained_error() -> None:
    method = dcl.DemocraticCoLearningMethod()
    with pytest.raises(dcl.InductiveValidationError, match="control_mode"):
        method.predict_proba_control(
            np.zeros((1, 1), dtype=np.float32),
            control_mode="dcl",
        )
    with pytest.raises(RuntimeError, match="controls were not retained"):
        method.predict_proba_control(
            np.zeros((1, 1), dtype=np.float32),
            control_mode="learner_0",
        )

    torch = pytest.importorskip("torch")
    mismatch = dcl.DemocraticCoLearningMethod()
    mismatch._initial_clfs = [object()]
    mismatch._initial_weights = np.ones((1,), dtype=np.float64)
    mismatch._backend = ""
    with pytest.raises(dcl.InductiveValidationError, match="backend mismatch"):
        mismatch.predict_proba_control(
            torch.zeros((1, 1)),
            control_mode="learner_0",
        )

    missing_numpy_classes = dcl.DemocraticCoLearningMethod()
    missing_numpy_classes._initial_clfs = [object()]
    missing_numpy_classes._initial_weights = np.ones((1,), dtype=np.float64)
    missing_numpy_classes._backend = "numpy"
    with pytest.raises(RuntimeError, match="missing classes"):
        missing_numpy_classes.predict_proba_control(
            np.zeros((1, 1), dtype=np.float32),
            control_mode="learner_0",
        )

    missing_torch_classes = dcl.DemocraticCoLearningMethod()
    missing_torch_classes._initial_clfs = [object()]
    missing_torch_classes._initial_weights = np.ones((1,), dtype=np.float64)
    missing_torch_classes._backend = "torch"
    with pytest.raises(RuntimeError, match="missing classes"):
        missing_torch_classes.predict_proba_control(
            torch.zeros((1, 1)),
            control_mode="learner_0",
        )


@pytest.mark.parametrize(
    "spec,n_learners,message",
    [
        (
            dcl.DemocraticCoLearningSpec(confidence_estimator="unknown"),
            3,
            "confidence_estimator",
        ),
        (
            dcl.DemocraticCoLearningSpec(confidence_interval="unknown"),
            3,
            "confidence_interval",
        ),
        (
            dcl.DemocraticCoLearningSpec(confidence_folds=1),
            3,
            "confidence_folds",
        ),
        (
            dcl.DemocraticCoLearningSpec(control_mode="unknown"),
            3,
            "control_mode",
        ),
        (
            dcl.DemocraticCoLearningSpec(control_mode="learner_2"),
            2,
            "requires learner index",
        ),
    ],
)
def test_democratic_v2_spec_validation(
    spec: dcl.DemocraticCoLearningSpec,
    n_learners: int,
    message: str,
) -> None:
    with pytest.raises(dcl.InductiveValidationError, match=message):
        dcl._validate_v2_spec(spec, n_learners=n_learners)


class _RoundClassifier:
    def __init__(
        self,
        learner_index: int,
        *,
        torch_backend: bool = False,
    ) -> None:
        self.learner_index = learner_index
        self.torch_backend = torch_backend
        self._train_y = None
        if torch_backend:
            torch = pytest.importorskip("torch")
            self.classes_t_ = torch.tensor([0, 1], dtype=torch.int64)
        else:
            self.classes_ = np.array([0, 1], dtype=np.int64)

    def fit(self, _x, y):
        self._train_y = y.clone() if self.torch_backend else np.asarray(y).copy()
        return self

    def predict(self, x):
        is_unlabeled_pool = bool((x[:, 0] >= 10).all())
        if is_unlabeled_pool:
            if self.torch_backend:
                torch = pytest.importorskip("torch")
                return torch.full(
                    (int(x.shape[0]),),
                    int(self.learner_index == 2),
                    dtype=torch.int64,
                    device=x.device,
                )
            return np.full(
                (int(x.shape[0]),),
                int(self.learner_index == 2),
                dtype=np.int64,
            )
        if self._train_y is not None and int(x.shape[0]) == int(self._train_y.shape[0]):
            return self._train_y.clone() if self.torch_backend else self._train_y.copy()
        values = x[:, 0]
        return (
            (values % 2).to(dtype=self.classes_t_.dtype)
            if self.torch_backend
            else (values % 2).astype(np.int64)
        )


def _cache_probe_data(backend: str) -> SimpleNamespace:
    labels = np.tile(np.array([0, 1], dtype=np.int64), 20)
    labeled = labels.astype(np.float32).reshape(-1, 1)
    unlabeled = np.array([[10.0], [11.0], [12.0]], dtype=np.float32)
    if backend == "numpy":
        return SimpleNamespace(X_l=labeled, y_l=labels, X_u=unlabeled)

    torch = pytest.importorskip("torch")
    return SimpleNamespace(
        X_l=torch.as_tensor(labeled),
        y_l=torch.as_tensor(labels),
        X_u=torch.as_tensor(unlabeled),
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_democratic_paper_round_without_diagnostic_trace(
    monkeypatch,
    backend: str,
) -> None:
    torch_backend = backend == "torch"
    specs = _control_specs(backend=backend)
    learner_by_id = {spec.classifier_id: index for index, spec in enumerate(specs)}
    monkeypatch.setattr(
        dcl,
        "build_classifier",
        lambda spec, seed=0: _RoundClassifier(
            learner_by_id[spec.classifier_id],
            torch_backend=torch_backend,
        ),
    )
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            max_iter=1,
            training_mode="confidence_weighted",
        )
    )

    method.fit(
        _cache_probe_data(backend),
        device=DeviceSpec(device="cpu"),
        seed=7,
    )

    assert method.n_iter_ == 1
    assert method.round_trace_ == []
    assert "round_trace" not in method.diagnostics_


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_democratic_kfold_interval_cache_tracks_only_changed_learner(
    monkeypatch,
    backend: str,
) -> None:
    torch_backend = backend == "torch"
    specs = _control_specs(backend=backend)
    learner_by_id = {spec.classifier_id: index for index, spec in enumerate(specs)}
    monkeypatch.setattr(
        dcl,
        "build_classifier",
        lambda spec, seed=0: _RoundClassifier(
            learner_by_id[spec.classifier_id],
            torch_backend=torch_backend,
        ),
    )
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            confidence_estimator="kfold_oof",
            confidence_folds=2,
            diagnostic_trace=True,
            max_iter=2,
            training_mode="confidence_weighted",
        )
    )
    calls: list[tuple[int, int, int, tuple[str, ...]]] = []
    fixed_intervals = [(0.8, 1.0), (0.7, 0.9), (0.6, 0.8)]

    def confidence_intervals(
        clfs,
        learner_specs,
        _x,
        y,
        *,
        learner_index_offset=0,
    ):
        calls.append(
            (
                len(clfs),
                learner_index_offset,
                int(y.shape[0]),
                tuple(spec.classifier_id for spec in learner_specs),
            )
        )
        return fixed_intervals if len(clfs) == 3 else [(1.0, 1.0)]

    monkeypatch.setattr(
        method,
        f"_confidence_intervals_{backend}",
        confidence_intervals,
    )

    method.fit(
        _cache_probe_data(backend),
        device=DeviceSpec(device="cpu"),
        seed=7,
    )

    assert calls == [
        (3, 0, 40, ("nb", "j48", "knn3")),
        (1, 0, 40, ("nb",)),
        (1, 1, 40, ("j48",)),
        (1, 2, 40, ("knn3",)),
        (1, 2, 43, ("knn3",)),
    ]
    assert method.n_iter_ == 2
    assert method.pseudo_labels_added_per_learner_ == (0, 0, 3)
    np.testing.assert_allclose(method._weights, [0.9, 0.8, 0.7])
    np.testing.assert_allclose(method._initial_weights, [0.9, 0.8, 0.7])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_democratic_training_accuracy_does_not_use_kfold_interval_cache(
    monkeypatch,
    backend: str,
) -> None:
    torch_backend = backend == "torch"
    specs = _control_specs(backend=backend)
    learner_by_id = {spec.classifier_id: index for index, spec in enumerate(specs)}
    monkeypatch.setattr(
        dcl,
        "build_classifier",
        lambda spec, seed=0: _RoundClassifier(
            learner_by_id[spec.classifier_id],
            torch_backend=torch_backend,
        ),
    )
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            confidence_estimator="training_accuracy",
            diagnostic_trace=True,
            max_iter=2,
            training_mode="confidence_weighted",
        )
    )
    calls: list[tuple[int, int, int]] = []

    def confidence_intervals(
        clfs,
        _learner_specs,
        _x,
        y,
        *,
        learner_index_offset=0,
    ):
        calls.append((len(clfs), learner_index_offset, int(y.shape[0])))
        return [(1.0, 1.0) for _ in clfs]

    monkeypatch.setattr(
        method,
        f"_confidence_intervals_{backend}",
        confidence_intervals,
    )

    method.fit(
        _cache_probe_data(backend),
        device=DeviceSpec(device="cpu"),
        seed=7,
    )

    assert calls == [
        (3, 0, 40),
        (1, 0, 40),
        (1, 1, 40),
        (1, 2, 40),
        (3, 0, 40),
        (1, 0, 40),
        (1, 1, 40),
        (1, 2, 43),
        (3, 0, 40),
        (3, 0, 40),
    ]
    assert method.n_iter_ == 2
    assert method.pseudo_labels_added_per_learner_ == (0, 0, 3)
    np.testing.assert_allclose(method._weights, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(method._initial_weights, [1.0, 1.0, 1.0])


def _assert_round_trace(trace: list[dict[str, object]]) -> None:
    assert len(trace) == 1
    round_zero = trace[0]
    assert round_zero["round"] == 1
    assert round_zero["majority_eligible_count"] == 3
    learners = round_zero["learners"]
    assert isinstance(learners, list)
    assert [learner["proposal_count"] for learner in learners] == [0, 0, 3]
    assert [learner["accepted"] for learner in learners] == [False, False, True]
    assert [learner["added_count"] for learner in learners] == [0, 0, 3]
    assert [learner["training_size_before"] for learner in learners] == [2, 2, 2]
    assert [learner["training_size_after"] for learner in learners] == [2, 2, 5]
    assert [learner["disagreement_count"] for learner in learners] == [0, 0, 3]
    assert all(learner["original_interval"] == {"lower": 1.0, "upper": 1.0} for learner in learners)
    assert all(learner["evolving_interval"] == {"lower": 1.0, "upper": 1.0} for learner in learners)
    assert all(learner["error_estimate_before"] == 0.0 for learner in learners)
    assert all(learner["proposal_error"] == 0.0 for learner in learners)
    assert all(learner["error_estimate_after"] == 0.0 for learner in learners)
    assert [learner["q"] for learner in learners] == [2.0, 2.0, 2.0]
    assert [learner["q_prime"] for learner in learners] == [2.0, 2.0, 5.0]


def test_democratic_round_trace_records_numpy_decisions(monkeypatch) -> None:
    specs = _control_specs()
    learner_by_id = {spec.classifier_id: index for index, spec in enumerate(specs)}
    monkeypatch.setattr(
        dcl,
        "build_classifier",
        lambda spec, seed=0: _RoundClassifier(learner_by_id[spec.classifier_id]),
    )
    data = SimpleNamespace(
        X_l=np.array([[0.0], [1.0]], dtype=np.float32),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u=np.array([[10.0], [11.0], [12.0]], dtype=np.float32),
    )
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            max_iter=1,
            diagnostic_trace=True,
            training_mode="confidence_weighted",
        )
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    _assert_round_trace(method.diagnostics_["round_trace"])


def test_democratic_round_trace_records_torch_decisions(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    specs = _control_specs(backend="torch")
    learner_by_id = {spec.classifier_id: index for index, spec in enumerate(specs)}
    monkeypatch.setattr(
        dcl,
        "build_classifier",
        lambda spec, seed=0: _RoundClassifier(
            learner_by_id[spec.classifier_id],
            torch_backend=True,
        ),
    )
    data = SimpleNamespace(
        X_l=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        y_l=torch.tensor([0, 1], dtype=torch.int64),
        X_u=torch.tensor([[10.0], [11.0], [12.0]], dtype=torch.float32),
    )
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(
            classifier_specs=specs,
            max_iter=1,
            diagnostic_trace=True,
            training_mode="confidence_weighted",
        )
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    _assert_round_trace(method.diagnostics_["round_trace"])
