from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import modssc.inductive.methods.co_training as ct
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.types import DeviceSpec, InductiveDataset

from ..conftest import DummyDataset


class _RankingClassifier:
    def __init__(self, *, view: int, overlap: bool = False) -> None:
        self.view = int(view)
        self.overlap = bool(overlap)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.fit_history: list[tuple[np.ndarray, np.ndarray]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.fit_history.append((np.asarray(X).copy(), np.asarray(y).copy()))
        self.classes_ = np.unique(y)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X)[:, 0]
        if self.view == 2:
            values = values - 100.0
        order = np.argsort(values, kind="stable")
        scores = np.full((values.size, 2), 0.49, dtype=np.float64)
        scores[:, 0] = 0.51
        if values.size == 1:
            scores[order[0]] = [0.01, 0.99]
            return scores

        if self.view == 1 or self.overlap:
            negative_position = order[0]
            positive_position = order[-1]
        else:
            negative_position = order[min(1, values.size - 1)]
            positive_position = order[max(values.size - 2, 0)]
        scores[negative_position] = [0.99, 0.01]
        scores[positive_position] = [0.01, 0.99]
        return scores


class _FixedScoresClassifier:
    def __init__(self, scores: Any, *, classes: Any = (0, 1)) -> None:
        self.scores = scores
        self.classes_ = None if classes is None else np.asarray(classes)

    def predict_scores(self, _X: Any) -> Any:
        return self.scores


class _LogProbabilityModel:
    def __init__(self, log_scores: np.ndarray) -> None:
        self.log_scores = np.asarray(log_scores, dtype=np.float64)

    def predict_log_proba(self, _X: Any) -> np.ndarray:
        return self.log_scores


class _CravenModel:
    def __init__(self) -> None:
        self.class_log_prior_ = np.log(np.array([0.75, 0.25], dtype=np.float64))
        self.feature_log_prob_ = np.log(np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64))


class _CravenClassifier:
    def __init__(self) -> None:
        self._model = _CravenModel()


class _NigamModel:
    def __init__(self) -> None:
        self.class_count_ = np.array([1.0, 1.0], dtype=np.float64)
        self.class_log_prior_ = np.log(np.array([0.5, 0.5], dtype=np.float64))


class _NigamRankingClassifier(_RankingClassifier):
    def __init__(self, *, view: int, overlap: bool = False) -> None:
        super().__init__(view=view, overlap=overlap)
        self._model = _NigamModel()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        super().fit(X, y)
        self._model.class_count_ = np.asarray(
            [np.count_nonzero(y == label) for label in self.classes_],
            dtype=np.float64,
        )
        self._model.class_log_prior_ = np.log(
            self._model.class_count_ / self._model.class_count_.sum()
        )


class _TorchScoresClassifier:
    def __init__(self, scores: torch.Tensor, *, classes: torch.Tensor) -> None:
        self.scores = scores
        self.classes_t_ = classes

    def predict_scores(self, _X: Any) -> torch.Tensor:
        return self.scores


def _paper_data(n_unlabeled: int = 12) -> DummyDataset:
    X1_l = np.array([[-4.0], [-3.0], [-2.0], [-1.0]], dtype=np.float64)
    X2_l = X1_l + 100.0
    y_l = np.array([0, 0, 1, 1], dtype=np.int64)
    X1_u = np.arange(n_unlabeled, dtype=np.float64).reshape(-1, 1)
    X2_u = X1_u + 100.0
    return DummyDataset(
        X_l=X1_l,
        y_l=y_l,
        views={
            "page": {"X_l": X1_l, "X_u": X1_u},
            "links": {"X_l": X2_l, "X_u": X2_u},
        },
    )


def _nigam_spec(**overrides: Any) -> ct.CoTrainingSpec:
    params: dict[str, Any] = {
        "classifier_id": "multinomial_nb",
        "classifier_backend": "sklearn",
        "classifier_params": {"alpha": 1.0, "fit_prior": True},
        "protocol": "shared_pool_exhaustive_multiset",
        "p": 1,
        "n": 3,
        "u": 75,
        "k": 0,
        "positive_label": 1,
        "negative_label": 0,
        "dynamic_feature_selection": "none",
        "feature_selection_max_features": None,
        "selection_score": "posterior_probability",
        "view_keys": ("page", "links"),
    }
    params.update(overrides)
    return ct.CoTrainingSpec(**params)


def _blum_spec(**overrides: Any) -> ct.CoTrainingSpec:
    params: dict[str, Any] = {
        "protocol": "fixed_pool_binary",
    }
    params.update(overrides)
    return ct.CoTrainingSpec(**params)


def _blum_v2_spec(**overrides: Any) -> ct.CoTrainingSpec:
    params: dict[str, Any] = {
        "protocol": "fixed_pool_binary_feature_selection",
    }
    params.update(overrides)
    return ct.CoTrainingSpec(**params)


def test_evaluation_reference_splits_are_protocol_owned() -> None:
    assert ct.CoTrainingMethod().evaluation_reference_splits == ()
    assert ct.CoTrainingMethod(_nigam_spec()).evaluation_reference_splits == (
        "train_labeled",
        "train",
    )


def _nigam_data(
    *,
    n_unlabeled: int = 776,
    labeled_counts: tuple[int, int] = (9, 3),
) -> DummyDataset:
    negative_count, positive_count = labeled_counts
    n_labeled = negative_count + positive_count
    X1_l = -np.arange(1, n_labeled + 1, dtype=np.float64).reshape(-1, 1)
    X2_l = X1_l + 1000.0
    y_l = np.concatenate(
        [
            np.zeros(negative_count, dtype=np.int64),
            np.ones(positive_count, dtype=np.int64),
        ]
    )
    X1_u = np.arange(n_unlabeled, dtype=np.float64).reshape(-1, 1)
    X2_u = X1_u + 100.0
    return DummyDataset(
        X_l=X1_l,
        y_l=y_l,
        views={
            "page": {"X_l": X1_l, "X_u": X1_u},
            "links": {"X_l": X2_l, "X_u": X2_u},
        },
    )


def _install_ranking_classifiers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    overlap: bool = False,
) -> tuple[_RankingClassifier, _RankingClassifier]:
    classifiers = (
        _RankingClassifier(view=1, overlap=overlap),
        _RankingClassifier(view=2, overlap=overlap),
    )
    pending = iter(classifiers)
    monkeypatch.setattr(ct, "build_classifier", lambda _spec, *, seed: next(pending))
    return classifiers


def _fit_paper(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed: int = 7,
    overlap: bool = False,
    k: int = 2,
    u: int = 4,
    n: int = 1,
    n_unlabeled: int = 12,
) -> tuple[ct.CoTrainingMethod, tuple[_RankingClassifier, _RankingClassifier]]:
    classifiers = _install_ranking_classifiers(monkeypatch, overlap=overlap)
    method = ct.CoTrainingMethod(
        _blum_spec(
            p=1,
            n=n,
            u=u,
            k=k,
            view_keys=("page", "links"),
        )
    )
    method.fit(_paper_data(n_unlabeled), device=DeviceSpec(device="cpu"), seed=seed)
    return method, classifiers


def test_blum_mitchell_multiround_oracle_grows_shared_l_and_replenishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, (view1, view2) = _fit_paper(monkeypatch)
    permutation = np.random.default_rng(7).permutation(12)

    assert method.initial_pool_indices_ == tuple(permutation[:4].tolist())
    assert method.n_iter_ == 2
    assert len(method.round_trace_) == 2
    assert method.round_trace_[0]["pool_indices_before"] == permutation[:4].tolist()
    assert method.round_trace_[0]["replenished_indices"] == permutation[4:8].tolist()
    assert method.round_trace_[0]["pool_indices_after"] == permutation[4:8].tolist()
    assert method.round_trace_[1]["replenished_indices"] == permutation[8:12].tolist()
    assert method.round_trace_[1]["pool_indices_after"] == permutation[8:12].tolist()

    for trace in method.round_trace_:
        selected1 = trace["selected_by_view1"]
        selected2 = trace["selected_by_view2"]
        assert [entry["label"] for entry in selected1] == [1, 0]
        assert [entry["label"] for entry in selected2] == [1, 0]
        assert trace["overlap_indices"] == []
        assert trace["round_status"] == "completed"
        assert trace["overlap_policy"] == "ordered_multiset_view1_then_view2"
        assert len(trace["multiset_additions"]) == 4
        assert trace["requested_replenishment_count"] == 4
        assert trace["pool_size_before"] == trace["pool_size_after"] == 4
        assert len(set(trace["pool_indices_after"])) == 4

    shared_indices = [
        entry["unlabeled_index"]
        for trace in method.round_trace_
        for entry in trace["multiset_additions"]
    ]
    shared_labels = [
        entry["label"] for trace in method.round_trace_ for entry in trace["multiset_additions"]
    ]
    np.testing.assert_array_equal(view1.fit_history[-1][0][-8:, 0], shared_indices)
    np.testing.assert_array_equal(view2.fit_history[-1][0][-8:, 0] - 100.0, shared_indices)
    np.testing.assert_array_equal(view1.fit_history[-1][1][-8:], shared_labels)
    np.testing.assert_array_equal(view2.fit_history[-1][1][-8:], shared_labels)
    assert method.diagnostics_["shared_labeled_multiset"] is True
    assert method.diagnostics_["selection_score_space"] == "log_probability"
    assert method.diagnostics_["combination_score_space"] == "summed_log_probability"
    assert method.diagnostics_["probability_underflow_safe"] is True
    assert method.diagnostics_["pseudo_labels_added_to_shared_l"] == 8
    assert method.diagnostics_["pseudo_labels_received_by_view1"] == 8
    assert method.diagnostics_["pseudo_labels_received_by_view2"] == 8
    assert method.diagnostics_["final_labeled_size"] == 12
    assert method.diagnostics_["remaining_unlabeled_count"] == 4
    assert method.diagnostics_["round_trace"] == method.round_trace_
    assert "test_metrics_used_for_protocol_selection" not in method.diagnostics_


def test_blum_mitchell_overlap_is_removed_once_but_retained_in_training_multiset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, classifiers = _fit_paper(monkeypatch, overlap=True, k=1)
    trace = method.round_trace_[0]

    assert len(trace["overlap_indices"]) == 2
    assert trace["conflicting_overlap_indices"] == []
    assert len(trace["multiset_additions"]) == 4
    assert [entry["source_view"] for entry in trace["multiset_additions"]] == [
        "view1",
        "view1",
        "view2",
        "view2",
    ]
    assert len(trace["removed_indices"]) == 2
    assert len(trace["replenished_indices"]) == 4
    assert trace["pool_size_after"] == 6
    assert len(set(trace["pool_indices_after"])) == 6
    assert trace["pool_growth"] == 2
    assert classifiers[0].fit_history[-1][0].shape[0] == 8
    assert classifiers[1].fit_history[-1][0].shape[0] == 8
    np.testing.assert_array_equal(
        classifiers[0].fit_history[-1][1], classifiers[1].fit_history[-1][1]
    )
    assert method.diagnostics_["pseudo_labels_added_to_shared_l"] == 4
    assert method.diagnostics_["unique_pseudo_labeled_examples"] == 2
    assert method.diagnostics_["same_label_overlap_count"] == 2
    assert method.diagnostics_["conflicting_overlap_count"] == 0
    assert method.diagnostics_["remaining_unlabeled_count"] == 10


def test_blum_mitchell_paper_quotas_append_eight_proposals_to_shared_multiset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, classifiers = _fit_paper(
        monkeypatch,
        overlap=True,
        k=1,
        u=8,
        n=3,
        n_unlabeled=20,
    )
    trace = method.round_trace_[0]

    assert [entry["label"] for entry in trace["selected_by_view1"]] == [1, 0, 0, 0]
    assert [entry["label"] for entry in trace["selected_by_view2"]] == [1, 0, 0, 0]
    assert len(trace["multiset_additions"]) == 8
    assert len(trace["removed_indices"]) == 4
    assert len(trace["replenished_indices"]) == 8
    assert trace["pool_growth"] == 4
    assert trace["pool_size_after"] == 12
    assert method.diagnostics_["pseudo_labels_added_to_shared_l"] == 8
    assert method.diagnostics_["unique_pseudo_labeled_examples"] == 4
    assert method.diagnostics_["final_labeled_size"] == 12
    assert classifiers[0].fit_history[-1][0].shape[0] == 12
    assert classifiers[1].fit_history[-1][0].shape[0] == 12


def test_blum_mitchell_exhausted_pool_stops_before_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, classifiers = _fit_paper(monkeypatch, overlap=True, k=3, u=2, n_unlabeled=2)

    assert method.n_iter_ == 1
    assert method.round_trace_[0]["replenished_indices"] == []
    assert method.round_trace_[0]["pool_indices_after"] == []
    # round 2 fits before observing the empty pool, then the final paper fit is retained.
    assert [len(classifier.fit_history) for classifier in classifiers] == [3, 3]


@pytest.mark.parametrize("selected_side", [1, 2])
def test_blum_mitchell_supports_a_single_view_proposal(
    monkeypatch: pytest.MonkeyPatch,
    selected_side: int,
) -> None:
    classifiers = _install_ranking_classifiers(monkeypatch)
    selected = (
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0.9], dtype=np.float64),
    )
    empty = (
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.float64),
    )
    proposals = iter((selected, empty) if selected_side == 1 else (empty, selected))
    monkeypatch.setattr(ct, "_select_binary_quota_numpy", lambda *_args, **_kwargs: next(proposals))

    method = ct.CoTrainingMethod(_blum_spec(p=1, n=1, u=2, k=1)).fit(
        _paper_data(6), device=DeviceSpec(device="cpu"), seed=0
    )

    assert len(method.round_trace_[0][f"selected_by_view{selected_side}"]) == 1
    assert len(method.round_trace_[0][f"selected_by_view{3 - selected_side}"]) == 0
    assert classifiers[0].fit_history[-1][0].shape[0] == 5
    assert classifiers[1].fit_history[-1][0].shape[0] == 5
    np.testing.assert_array_equal(
        classifiers[0].fit_history[-1][1], classifiers[1].fit_history[-1][1]
    )


def test_blum_mitchell_keeps_conflicting_overlap_in_ordered_multiset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifiers = _install_ranking_classifiers(monkeypatch)
    proposals = iter(
        (
            (
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([0.9]),
            ),
            (
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([0.8]),
            ),
        )
    )
    monkeypatch.setattr(ct, "_select_binary_quota_numpy", lambda *_args, **_kwargs: next(proposals))
    method = ct.CoTrainingMethod(
        _blum_spec(
            p=1,
            n=1,
            u=2,
            k=1,
            view_keys=("page", "links"),
        )
    ).fit(_paper_data(6), device=DeviceSpec(device="cpu"), seed=0)

    trace = method.round_trace_[0]
    assert trace["overlap_indices"] == trace["conflicting_overlap_indices"]
    assert len(trace["conflicting_overlap_indices"]) == 1
    assert trace["round_status"] == "completed"
    assert trace["overlap_policy"] == "ordered_multiset_view1_then_view2"
    assert [entry["label"] for entry in trace["multiset_additions"]] == [1, 0]
    assert [entry["source_view"] for entry in trace["multiset_additions"]] == [
        "view1",
        "view2",
    ]
    assert len(trace["removed_indices"]) == 1
    assert len(trace["replenished_indices"]) == 4
    assert trace["pool_growth"] == 3
    assert method.diagnostics_["conflicting_overlap_count"] == 1
    assert method.diagnostics_["pseudo_labels_added_to_shared_l"] == 2
    assert method.diagnostics_["unique_pseudo_labeled_examples"] == 1
    for classifier, offset in zip(classifiers, (0.0, 100.0), strict=True):
        np.testing.assert_array_equal(classifier.fit_history[-1][1][-2:], [1, 0])
        np.testing.assert_array_equal(
            classifier.fit_history[-1][0][-2:, 0] - offset,
            [trace["overlap_indices"][0], trace["overlap_indices"][0]],
        )


def test_blum_mitchell_pool_is_seed_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    first, _ = _fit_paper(monkeypatch, seed=23, k=0)
    second, _ = _fit_paper(monkeypatch, seed=23, k=0)
    third, _ = _fit_paper(monkeypatch, seed=24, k=0)

    assert first.initial_pool_indices_ == second.initial_pool_indices_
    assert first.initial_pool_indices_ != third.initial_pool_indices_
    assert first.round_trace_ == []
    assert first.diagnostics_["n_iter"] == 0


def test_nigam_ghani_views_select_independently_from_the_same_pre_round_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifiers = (
        _NigamRankingClassifier(view=1),
        _NigamRankingClassifier(view=2),
    )
    pending = iter(classifiers)
    monkeypatch.setattr(ct, "build_classifier", lambda _spec, *, seed: next(pending))

    method = ct.CoTrainingMethod(_nigam_spec()).fit(
        _nigam_data(),
        device=DeviceSpec(device="cpu"),
        seed=19,
    )

    assert method.n_iter_ == len(method.round_trace_)
    assert [len(classifier.fit_history) for classifier in classifiers] == [
        method.n_iter_ + 1,
        method.n_iter_ + 1,
    ]
    additions = [entry for trace in method.round_trace_ for entry in trace["multiset_additions"]]
    promoted = [index for trace in method.round_trace_ for index in trace["removed_indices"]]
    assert len(promoted) == len(set(promoted)) == 776
    assert set(promoted) == set(range(776))
    for trace in method.round_trace_:
        pool_before = set(trace["pool_indices_before"])
        selected1 = trace["selected_by_view1"]
        selected2 = trace["selected_by_view2"]
        selected_indices1 = [entry["unlabeled_index"] for entry in selected1]
        selected_indices2 = [entry["unlabeled_index"] for entry in selected2]
        multiset_indices = [entry["unlabeled_index"] for entry in trace["multiset_additions"]]
        overlap = set(selected_indices1).intersection(selected_indices2)
        conflicts = {
            first["unlabeled_index"]
            for first in selected1
            for second in selected2
            if first["unlabeled_index"] == second["unlabeled_index"]
            and first["label"] != second["label"]
        }

        assert set(selected_indices1).issubset(pool_before)
        assert set(selected_indices2).issubset(pool_before)
        assert trace["overlap_policy"] == "ordered_multiset_view1_then_view2"
        assert multiset_indices == selected_indices1 + selected_indices2
        assert [entry["source_view"] for entry in trace["multiset_additions"]] == [
            *(["view1"] * len(selected1)),
            *(["view2"] * len(selected2)),
        ]
        assert set(trace["overlap_indices"]) == overlap
        assert set(trace["conflicting_overlap_indices"]) == conflicts
        assert trace["proposal_count_view1"] == len(selected1)
        assert trace["proposal_count_view2"] == len(selected2)
        assert trace["multiset_addition_count"] == len(multiset_indices)
        assert trace["unique_removed_count"] == len(trace["removed_indices"])
        assert trace["duplicate_multiset_addition_count"] == len(multiset_indices) - len(
            trace["removed_indices"]
        )
        assert trace["same_label_overlap_count"] == len(overlap - conflicts)
        assert trace["conflicting_overlap_count"] == len(conflicts)

    first_trace = method.round_trace_[0]
    assert first_trace["proposal_count_view1"] == first_trace["proposal_count_view2"] == 4
    assert first_trace["multiset_addition_count"] == 8
    assert first_trace["unique_removed_count"] == 6
    assert first_trace["same_label_overlap_count"] == 2

    diagnostics = method.diagnostics_
    assert diagnostics["initial_labeled_size"] == 12
    assert diagnostics["initial_unlabeled_count"] == 776
    assert diagnostics["initial_class_counts"] == {"0": 9, "1": 3}
    assert diagnostics["unique_pseudo_labeled_examples"] == 776
    assert diagnostics["pseudo_labels_added_to_shared_l"] == len(additions)
    assert diagnostics["final_labeled_size"] == 12 + len(additions)
    assert diagnostics["remaining_unlabeled_count"] == 0
    assert diagnostics["remaining_unlabeled_indices"] == []
    assert diagnostics["termination"] == "unlabeled_exhausted"
    assert diagnostics["addition_policy"] == "ordered_multiset_view1_then_view2"
    assert diagnostics["views_select_from_same_pre_round_pool"] is True
    assert diagnostics["pseudo_label_proposals_view1"] == sum(
        trace["proposal_count_view1"] for trace in method.round_trace_
    )
    assert diagnostics["pseudo_label_proposals_view2"] == sum(
        trace["proposal_count_view2"] for trace in method.round_trace_
    )
    assert diagnostics["overlap_count"] == sum(
        trace["same_label_overlap_count"] + trace["conflicting_overlap_count"]
        for trace in method.round_trace_
    )
    assert diagnostics["duplicate_multiset_additions"] == len(additions) - 776
    assert diagnostics["overlap_count"] == diagnostics["duplicate_multiset_additions"]
    assert diagnostics["word_likelihood_smoothing"] == "add_one"
    assert diagnostics["class_prior_smoothing"] == "add_one"
    assert diagnostics["dynamic_feature_selection"] == "none"
    assert diagnostics["test_metrics_used_for_protocol_selection"] is False


def test_nigam_ghani_retains_conflicts_and_handles_a_one_example_final_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifiers = (
        _NigamRankingClassifier(view=1, overlap=True),
        _NigamRankingClassifier(view=2, overlap=True),
    )
    pending = iter(classifiers)
    monkeypatch.setattr(ct, "build_classifier", lambda _spec, *, seed: next(pending))
    original_selection = ct._select_binary_quota_numpy
    call_count = 0

    def select_with_first_round_conflict(
        scores: np.ndarray,
        classes: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            indices = np.array([0, 1, 2, 3], dtype=np.int64)
        elif call_count == 2:
            indices = np.array([4, 0, 5, 6], dtype=np.int64)
        else:
            return original_selection(scores, classes, **kwargs)
        labels = np.array([1, 0, 0, 0], dtype=np.int64)
        columns = {label: column for column, label in enumerate(classes.tolist())}
        confidences = np.asarray(
            [
                scores[index, columns[label]]
                for index, label in zip(indices.tolist(), labels.tolist(), strict=True)
            ],
            dtype=np.float64,
        )
        return indices, labels, confidences

    monkeypatch.setattr(ct, "_select_binary_quota_numpy", select_with_first_round_conflict)
    method = ct.CoTrainingMethod(_nigam_spec()).fit(
        _nigam_data(),
        device=DeviceSpec(device="cpu"),
        seed=19,
    )

    first_trace = method.round_trace_[0]
    assert first_trace["multiset_addition_count"] == 8
    assert first_trace["unique_removed_count"] == 7
    assert first_trace["duplicate_multiset_addition_count"] == 1
    assert first_trace["same_label_overlap_count"] == 0
    assert first_trace["conflicting_overlap_count"] == 1
    assert first_trace["overlap_indices"] == first_trace["conflicting_overlap_indices"]
    conflict_index = first_trace["conflicting_overlap_indices"][0]
    assert [
        entry["label"]
        for entry in first_trace["multiset_additions"]
        if entry["unlabeled_index"] == conflict_index
    ] == [1, 0]

    last_trace = method.round_trace_[-1]
    assert last_trace["pool_size_before"] == 1
    assert last_trace["proposal_count_view1"] == last_trace["proposal_count_view2"] == 1
    assert last_trace["multiset_addition_count"] == 2
    assert last_trace["unique_removed_count"] == 1
    assert last_trace["duplicate_multiset_addition_count"] == 1
    assert last_trace["same_label_overlap_count"] == 1
    assert last_trace["conflicting_overlap_count"] == 0
    assert last_trace["pool_indices_after"] == []

    diagnostics = method.diagnostics_
    assert diagnostics["unique_pseudo_labeled_examples"] == 776
    assert diagnostics["pseudo_labels_added_to_shared_l"] == 1546
    assert diagnostics["final_labeled_size"] == 1558
    assert diagnostics["same_label_overlap_count"] == 769
    assert diagnostics["conflicting_overlap_count"] == 1
    assert diagnostics["overlap_count"] == 770
    assert diagnostics["duplicate_multiset_additions"] == 770
    assert diagnostics["remaining_unlabeled_count"] == 0
    assert diagnostics["remaining_unlabeled_indices"] == []


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_nigam_data(n_unlabeled=775), "exactly 776 unlabeled"),
        (_nigam_data(labeled_counts=(8, 3)), "exactly 12 labeled"),
        (_nigam_data(labeled_counts=(8, 4)), "exactly 9 negative and 3 positive"),
    ],
)
def test_nigam_ghani_rejects_nonhistorical_labeled_and_unlabeled_sets(
    monkeypatch: pytest.MonkeyPatch,
    data: DummyDataset,
    message: str,
) -> None:
    classifiers = iter((_NigamRankingClassifier(view=1), _NigamRankingClassifier(view=2)))
    monkeypatch.setattr(ct, "build_classifier", lambda _spec, *, seed: next(classifiers))

    with pytest.raises(InductiveValidationError, match=message):
        ct.CoTrainingMethod(_nigam_spec()).fit(
            data,
            device=DeviceSpec(device="cpu"),
            seed=0,
        )


def test_nigam_ghani_add_one_prior_helper_replaces_empirical_prior() -> None:
    classifier = _NigamRankingClassifier(view=1)
    X = np.arange(12, dtype=np.float64).reshape(-1, 1)
    y = np.array([0] * 9 + [1] * 3, dtype=np.int64)

    ct._fit_add_one_multinomial_nb(classifier, X, y)

    np.testing.assert_allclose(
        np.exp(classifier._model.class_log_prior_),
        np.array([10.0 / 14.0, 4.0 / 14.0]),
    )
    assert len(classifier.fit_history) == 1


def test_nigam_ghani_add_one_prior_helper_validates_fitted_metadata() -> None:
    class MissingMetadataClassifier:
        def fit(self, _X: np.ndarray, _y: np.ndarray) -> None:
            return None

    with pytest.raises(InductiveValidationError, match="fitted class counts"):
        ct._fit_add_one_multinomial_nb(
            MissingMetadataClassifier(),
            np.zeros((2, 1)),
            np.array([0, 1]),
        )

    classifier = _NigamRankingClassifier(view=1)
    classifier.fit = lambda _X, _y: None  # type: ignore[method-assign]
    classifier._model.class_count_ = np.array([2.0, 0.0])
    with pytest.raises(InductiveValidationError, match="two non-empty aligned"):
        ct._fit_add_one_multinomial_nb(
            classifier,
            np.zeros((2, 1)),
            np.array([0, 1]),
        )


def test_nigam_ghani_uses_normalized_product_for_final_prediction() -> None:
    X = np.zeros((2, 1), dtype=np.float64)
    method = ct.CoTrainingMethod(_nigam_spec())
    method._clf1 = _FixedScoresClassifier([[0.8, 0.2], [2.0, 1.0]])
    method._clf2 = _FixedScoresClassifier([[0.5, 0.5], [1.0, 4.0]])
    method._backend = "numpy"

    np.testing.assert_allclose(
        method._predict_scores_pair(X, X),
        np.array([[0.8, 0.2], [1.0 / 3.0, 2.0 / 3.0]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("views", "message"),
    [
        (None, "require two prediction views"),
        (
            {"page": {"X": np.zeros((2, 1), dtype=np.float64)}},
            "Missing required view 'links'",
        ),
        (
            {
                "page": {"X": np.zeros((2, 1), dtype=np.float64)},
                "links": {"X": np.zeros((1, 1), dtype=np.float64)},
            },
            "must have the same row count",
        ),
    ],
)
def test_nigam_ghani_concatenated_prediction_views_validate_contract(
    views: dict[str, Any] | None,
    message: str,
) -> None:
    data = DummyDataset(
        X_l=np.zeros((2, 1), dtype=np.float64),
        y_l=np.array([0, 1], dtype=np.int64),
        views=views,
    )

    with pytest.raises(InductiveValidationError, match=message):
        ct._concatenate_prediction_views(data, ("page", "links"))


def _configured_named_prediction_method(spec: ct.CoTrainingSpec) -> ct.CoTrainingMethod:
    method = ct.CoTrainingMethod(spec)
    method._clf1 = _FixedScoresClassifier([[0.8, 0.2], [0.2, 0.8]])
    method._clf2 = _FixedScoresClassifier([[0.6, 0.4], [0.3, 0.7]])
    method._view_keys = ("page", "links")
    method._backend = "numpy"
    return method


def _named_prediction_data(*, meta: dict[str, Any] | None = None) -> InductiveDataset:
    page = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    links = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    return InductiveDataset(
        X_l=page,
        y_l=np.array([0, 1], dtype=np.int64),
        views={"page": {"X": page}, "links": {"X": links}},
        meta=meta,
    )


def test_predict_named_proba_requires_fitted_view_keys() -> None:
    with pytest.raises(RuntimeError, match="missing view keys"):
        ct.CoTrainingMethod().predict_named_proba(None)


def test_generic_predict_named_proba_returns_only_view_probabilities() -> None:
    named = _configured_named_prediction_method(_blum_spec()).predict_named_proba(
        _named_prediction_data()
    )

    assert set(named) == {"page", "links"}
    np.testing.assert_allclose(named["page"], [[0.8, 0.2], [0.2, 0.8]])
    np.testing.assert_allclose(named["links"], [[0.6, 0.4], [0.3, 0.7]])


def test_nigam_ghani_named_predictions_require_reference_split_mapping() -> None:
    method = _configured_named_prediction_method(_nigam_spec())

    with pytest.raises(InductiveValidationError, match="require evaluation reference splits"):
        method.predict_named_proba(_named_prediction_data())


@pytest.mark.parametrize("missing_split", ["train_labeled", "train"])
def test_nigam_ghani_named_predictions_require_each_reference_split(
    monkeypatch: pytest.MonkeyPatch,
    missing_split: str,
) -> None:
    monkeypatch.setattr(
        ct,
        "build_classifier",
        lambda _spec, *, seed: _NigamRankingClassifier(view=1),
    )
    references = {
        "train_labeled": _named_prediction_data(),
        "train": _named_prediction_data(),
    }
    references.pop(missing_split)
    data = _named_prediction_data(meta={"evaluation_reference_splits": references})
    method = _configured_named_prediction_method(_nigam_spec())

    with pytest.raises(
        InductiveValidationError,
        match=rf"require {missing_split!r}",
    ):
        method.predict_named_proba(data)


def test_nigam_ghani_named_predictions_include_supervised_controls() -> None:
    method = ct.CoTrainingMethod(_nigam_spec())
    method._clf1 = _FixedScoresClassifier(
        [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3]],
    )
    method._clf2 = _FixedScoresClassifier(
        [[0.6, 0.4], [0.3, 0.7], [0.4, 0.6]],
    )
    method._view_keys = ("page", "links")
    method._backend = "numpy"

    def reference(size: int) -> InductiveDataset:
        labels = np.arange(size, dtype=np.int64) % 2
        page = np.column_stack((labels + 1, 2 - labels)).astype(np.float64)
        links = np.column_stack((2 - labels, labels + 1)).astype(np.float64)
        return InductiveDataset(
            X_l=page,
            y_l=labels,
            views={"page": {"X": page}, "links": {"X": links}},
        )

    current_page = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
    current_links = np.array([[2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    data = InductiveDataset(
        X_l=current_page,
        y_l=np.array([0, 1, 0]),
        views={"page": {"X": current_page}, "links": {"X": current_links}},
        meta={
            "evaluation_reference_splits": {
                "train_labeled": reference(12),
                "train": reference(788),
            }
        },
    )

    named = method.predict_named_proba(data)

    assert set(named) == {"page", "links", "nb12", "nb788"}
    assert all(np.asarray(scores).shape == (3, 2) for scores in named.values())
    assert method.diagnostics_["supervised_controls"] == {
        "nb12_training_size": 12,
        "nb788_training_size": 788,
        "feature_space": "concatenated_namespaced_views",
        "class_prior_smoothing": "add_one",
        "test_metrics_used_for_protocol_selection": False,
    }


def test_nigam_ghani_evaluation_controls_remain_test_only() -> None:
    method = _configured_named_prediction_method(_nigam_spec())
    data = _named_prediction_data(meta={"evaluation_split": "validation"})

    named = method.predict_evaluation_outputs(data)

    assert set(named) == {"page", "links"}


def test_blum_mitchell_multiplicative_combination_and_legacy_average() -> None:
    X = np.zeros((2, 1), dtype=np.float64)
    first = np.array([[0.8, 0.2], [2.0, 1.0]], dtype=np.float64)
    second = np.array([[0.5, 0.5], [1.0, 4.0]], dtype=np.float64)

    paper = ct.CoTrainingMethod(_blum_spec())
    paper._clf1 = _FixedScoresClassifier(first)
    paper._clf2 = _FixedScoresClassifier(second)
    paper._backend = "numpy"
    np.testing.assert_allclose(
        paper._predict_scores_pair(X, X),
        np.array([[0.8, 0.2], [1.0 / 3.0, 2.0 / 3.0]], dtype=np.float32),
    )

    legacy = ct.CoTrainingMethod(ct.CoTrainingSpec())
    legacy._clf1 = _FixedScoresClassifier(first)
    legacy._clf2 = _FixedScoresClassifier(second)
    legacy._backend = "numpy"
    np.testing.assert_allclose(legacy._predict_scores_pair(X, X), (first + second) / 2.0)


def test_predict_view_proba_uses_requested_numpy_classifier_and_normalizes() -> None:
    method = ct.CoTrainingMethod(_blum_spec())
    method._clf1 = _FixedScoresClassifier([[8.0, 2.0], [1.0, 3.0]])
    method._clf2 = _FixedScoresClassifier([[1.0, 4.0], [9.0, 1.0]])
    method._view_keys = ("page", "links")
    method._backend = "numpy"
    data = DummyDataset(
        X_l=np.zeros((2, 1)),
        y_l=np.array([0, 1]),
        views={
            "page": {"X": np.zeros((2, 1))},
            "links": np.zeros((2, 1)),
        },
    )

    np.testing.assert_allclose(
        method.predict_view_proba(data, "page"),
        [[0.8, 0.2], [0.25, 0.75]],
    )
    np.testing.assert_allclose(
        method.predict_view_proba(data, "links"),
        [[0.2, 0.8], [0.9, 0.1]],
    )

    method._clf1 = _FixedScoresClassifier([[3.0, 1.0], [1.0, 3.0]], classes=[0, 1])
    method._clf2 = _FixedScoresClassifier([[1.0, 1.0], [1.0, 1.0]], classes=None)
    np.testing.assert_allclose(
        method.predict_view_proba(data, "page"),
        [[0.75, 0.25], [0.25, 0.75]],
    )

    method._clf1 = _FixedScoresClassifier([[1.0, 1.0], [1.0, 1.0]], classes=None)
    np.testing.assert_allclose(
        method.predict_view_proba(data, "page"),
        [[0.5, 0.5], [0.5, 0.5]],
    )


def test_predict_view_proba_preserves_torch_backend_and_class_order() -> None:
    method = ct.CoTrainingMethod(ct.CoTrainingSpec(classifier_backend="torch"))
    classes = torch.tensor([0, 1])
    method._clf1 = _TorchScoresClassifier(torch.tensor([[2.0, 1.0]]), classes=classes)
    method._clf2 = _TorchScoresClassifier(torch.tensor([[1.0, 2.0]]), classes=classes)
    method._view_keys = ("page", "links")
    method._backend = "torch"
    data = DummyDataset(
        X_l=torch.zeros((1, 1)),
        y_l=torch.tensor([0]),
        views={"page": {"X": torch.zeros((1, 1))}},
    )

    probabilities = method.predict_view_proba(data, "page")

    assert isinstance(probabilities, torch.Tensor)
    assert probabilities.device == data.X_l.device
    torch.testing.assert_close(probabilities, torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]))

    method._clf1 = _TorchScoresClassifier(torch.tensor([[3.0, 1.0]]), classes=classes)
    del method._clf1.classes_t_
    method._clf1.classes_ = np.array([0, 1])
    torch.testing.assert_close(
        method.predict_view_proba(data, "page"),
        torch.tensor([[0.75, 0.25]]),
    )


def test_predict_view_proba_validates_fitted_state_view_and_backend() -> None:
    data = DummyDataset(
        X_l=np.zeros((1, 1)),
        y_l=np.array([0]),
        views={"page": {"X": np.zeros((1, 1))}},
    )
    unfitted = ct.CoTrainingMethod()
    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.predict_view_proba(data, "page")

    method = ct.CoTrainingMethod()
    method._clf1 = _FixedScoresClassifier([[0.5, 0.5]])
    method._clf2 = _FixedScoresClassifier([[0.5, 0.5]])
    method._view_keys = ("page", "links")
    method._backend = "numpy"
    with pytest.raises(InductiveValidationError, match="view_key must be"):
        method.predict_view_proba(data, "unknown")
    with pytest.raises(InductiveValidationError, match="requires data.views"):
        method.predict_view_proba(None, "page")
    with pytest.raises(InductiveValidationError, match="requires data.views"):
        method.predict_view_proba(DummyDataset(X_l=data.X_l, y_l=data.y_l, views=None), "page")
    with pytest.raises(InductiveValidationError, match="Missing required view"):
        method.predict_view_proba(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, views={"links": data.views["page"]}),
            "page",
        )
    with pytest.raises(InductiveValidationError, match="backend mismatch"):
        method.predict_view_proba(
            DummyDataset(
                X_l=torch.zeros((1, 1)),
                y_l=torch.tensor([0]),
                views={"page": {"X": torch.zeros((1, 1))}},
            ),
            "page",
        )


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([[np.nan, 1.0]], "finite and non-negative"),
        ([[-1.0, 2.0]], "finite and non-negative"),
        ([[0.0, 0.0]], "positive mass"),
    ],
)
def test_predict_view_proba_rejects_invalid_numpy_probabilities(
    scores: list[list[float]],
    message: str,
) -> None:
    method = ct.CoTrainingMethod()
    method._clf1 = _FixedScoresClassifier(scores)
    method._clf2 = _FixedScoresClassifier([[0.5, 0.5]])
    method._view_keys = ("page", "links")
    method._backend = "numpy"
    data = DummyDataset(
        X_l=np.zeros((1, 1)),
        y_l=np.array([0]),
        views={"page": {"X": np.zeros((1, 1))}},
    )

    with pytest.raises(InductiveValidationError, match=message):
        method.predict_view_proba(data, "page")


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([[float("inf"), 1.0]], "finite and non-negative"),
        ([[-1.0, 2.0]], "finite and non-negative"),
        ([[0.0, 0.0]], "positive mass"),
    ],
)
def test_predict_view_proba_rejects_invalid_torch_probabilities(
    scores: list[list[float]],
    message: str,
) -> None:
    classes = torch.tensor([0, 1])
    method = ct.CoTrainingMethod(ct.CoTrainingSpec(classifier_backend="torch"))
    method._clf1 = _TorchScoresClassifier(torch.tensor(scores), classes=classes)
    method._clf2 = _TorchScoresClassifier(torch.tensor([[0.5, 0.5]]), classes=classes)
    method._view_keys = ("page", "links")
    method._backend = "torch"
    data = DummyDataset(
        X_l=torch.zeros((1, 1)),
        y_l=torch.tensor([0]),
        views={"page": {"X": torch.zeros((1, 1))}},
    )
    with pytest.raises(InductiveValidationError, match=message):
        method.predict_view_proba(data, "page")


def test_predict_view_proba_validates_classifier_class_metadata() -> None:
    method = ct.CoTrainingMethod()
    method._view_keys = ("page", "links")
    method._backend = "numpy"
    data = DummyDataset(
        X_l=np.zeros((1, 1)),
        y_l=np.array([0]),
        views={"page": {"X": np.zeros((1, 1))}},
    )

    method._clf1 = _FixedScoresClassifier([[0.5, 0.5]], classes=[[0, 1]])
    method._clf2 = _FixedScoresClassifier([[0.5, 0.5]])
    with pytest.raises(InductiveValidationError, match="one-dimensional"):
        method.predict_view_proba(data, "page")

    method._clf1 = _FixedScoresClassifier([[0.5, 0.5]], classes=[0])
    with pytest.raises(InductiveValidationError, match="do not align"):
        method.predict_view_proba(data, "page")

    method._clf1 = _FixedScoresClassifier([[0.5, 0.5]], classes=[0, 1])
    method._clf2 = _FixedScoresClassifier([[0.5, 0.5]], classes=[1, 0])
    with pytest.raises(InductiveValidationError, match="disagree on class labels"):
        method.predict_view_proba(data, "page")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"protocol": "unknown"}, "protocol must be"),
        ({"p": True}, "p must be an integer"),
        ({"p": -1}, "p and n must be"),
        ({"p": 0, "n": 0}, "At least one"),
        ({"u": 0}, "u must be"),
        ({"k": -1}, "k must be"),
        ({"p": 5, "u": 4}, "must not exceed"),
        ({"positive_label": 1}, "both be set"),
        ({"positive_label": 1, "negative_label": 1}, "must be distinct"),
        ({"confidence_threshold": 0.5}, "not part"),
    ],
)
def test_blum_mitchell_spec_validation(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(InductiveValidationError, match=message):
        ct._validate_protocol(_blum_spec(**overrides))


def test_nigam_ghani_spec_accepts_only_the_frozen_profile() -> None:
    ct._validate_protocol(_nigam_spec())


def test_co_training_protocols_accept_all_supported_generic_modes() -> None:
    ct._validate_protocol(ct.CoTrainingSpec())
    ct._validate_protocol(_blum_spec())
    ct._validate_protocol(
        _blum_v2_spec(
            classifier_id="multinomial_nb",
            classifier_backend="sklearn",
            dynamic_feature_selection="mutual_information_presence",
            feature_selection_max_features=2000,
            selection_score="craven_1998_normalized_nb",
        )
    )
    ct._validate_protocol(_nigam_spec())


def test_co_training_protocol_rejects_unknown_mode() -> None:
    with pytest.raises(InductiveValidationError, match="protocol must be one of"):
        ct._validate_protocol(ct.CoTrainingSpec(protocol="unknown"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"p": True}, "p must be an integer"),
        ({"n": np.float64(3.0)}, "n must be an integer"),
        ({"p": 2}, "freezes p=1"),
        ({"n": 2}, "freezes p=1"),
        ({"u": 74}, "freezes p=1"),
        ({"k": 1}, "freezes p=1"),
        ({"positive_label": 0, "negative_label": 1}, "requires positive_label=1"),
        ({"dynamic_feature_selection": "mutual_information_presence"}, "no feature selection"),
        ({"feature_selection_max_features": 2000}, "no feature selection"),
        ({"selection_score": "log_probability"}, "posterior_probability"),
        ({"classifier_id": "knn"}, "requires sklearn multinomial_nb"),
        ({"classifier_backend": "numpy"}, "requires sklearn multinomial_nb"),
        ({"classifier_params": {"alpha": 1.0}}, "freezes classifier_params"),
        (
            {"classifier_params": {"alpha": 1.0, "fit_prior": True, "extra": 1}},
            "freezes classifier_params",
        ),
        ({"classifier_params": {"alpha": True, "fit_prior": True}}, "types are validated"),
        ({"classifier_params": {"alpha": "1.0", "fit_prior": True}}, "types are validated"),
        ({"classifier_params": {"alpha": np.nan, "fit_prior": True}}, "types are validated"),
        ({"classifier_params": {"alpha": 0.5, "fit_prior": True}}, "types are validated"),
        ({"classifier_params": {"alpha": 1.0, "fit_prior": 1}}, "types are validated"),
        ({"classifier_params": {"alpha": 1.0, "fit_prior": False}}, "types are validated"),
    ],
)
def test_nigam_ghani_spec_rejects_protocol_drift(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(InductiveValidationError, match=message):
        ct._validate_protocol(_nigam_spec(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dynamic_feature_selection": "none"}, "mutual_information_presence"),
        ({"feature_selection_max_features": 1999}, "freezes"),
        ({"feature_selection_max_features": True}, "freezes"),
        ({"selection_score": "log_probability"}, "craven_1998"),
        ({"classifier_id": "knn"}, "multinomial_nb"),
    ],
)
def test_blum_mitchell_diagnostic_v2_freezes_scientific_core(
    overrides: dict[str, Any],
    message: str,
) -> None:
    params: dict[str, Any] = {
        "classifier_id": "multinomial_nb",
        "classifier_backend": "sklearn",
        "dynamic_feature_selection": "mutual_information_presence",
        "feature_selection_max_features": 2000,
        "selection_score": "craven_1998_normalized_nb",
    }
    params.update(overrides)
    with pytest.raises(InductiveValidationError, match=message):
        ct._validate_protocol(_blum_v2_spec(**params))


def test_blum_mitchell_v1_rejects_v2_switches_and_legacy_rejects_dynamic_selection() -> None:
    with pytest.raises(InductiveValidationError, match="v1.*immutable"):
        ct._validate_protocol(
            _blum_spec(
                dynamic_feature_selection="mutual_information_presence",
            )
        )
    with pytest.raises(InductiveValidationError, match="only available"):
        ct._validate_protocol(
            ct.CoTrainingSpec(dynamic_feature_selection="mutual_information_presence")
        )


def test_mutual_information_presence_feature_selection_oracle() -> None:
    X = np.array(
        [
            [2.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 3.0],
            [0.0, 1.0, 0.0, 2.0],
        ]
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    scores = ct._mutual_information_presence_scores_numpy(X, y)
    np.testing.assert_allclose(scores[[0, 3]], np.log(2.0))
    np.testing.assert_allclose(scores[[1, 2]], 0.0, atol=1e-15)
    selected, returned_scores = ct._select_mutual_information_features_numpy(
        X,
        y,
        max_features=2,
    )
    np.testing.assert_array_equal(selected, [0, 3])
    np.testing.assert_array_equal(returned_scores, scores)
    assert len(ct._feature_indices_sha256(selected)) == 64


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.zeros((2, 2, 1)), np.array([0, 1]), "2D"),
        (np.zeros((2, 2)), np.array([0]), "one label"),
        (np.zeros((0, 2)), np.array([], dtype=np.int64), "non-empty"),
        (np.array([[np.nan]]), np.array([0]), "finite non-negative"),
        (np.array([[-1.0]]), np.array([0]), "finite non-negative"),
    ],
)
def test_mutual_information_presence_validation(
    X: np.ndarray,
    y: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(InductiveValidationError, match=message):
        ct._mutual_information_presence_scores_numpy(X, y)


def test_mutual_information_presence_rejects_all_zero_view() -> None:
    with pytest.raises(InductiveValidationError, match="no observed feature"):
        ct._select_mutual_information_features_numpy(
            np.zeros((2, 3)),
            np.array([0, 1]),
            max_features=2,
        )


def test_craven_normalized_nb_score_matches_equation_one_and_empty_policy() -> None:
    classifier = _CravenClassifier()
    X = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float64)

    observed = ct._craven_normalized_nb_scores_numpy(classifier, X)

    expected_first = classifier._model.class_log_prior_ / 2.0 + np.log([0.8, 0.1])
    expected_second = (
        classifier._model.class_log_prior_ / 2.0
        + 0.5 * np.log(np.array([0.8, 0.1]) / 0.5)
        + 0.5 * np.log(np.array([0.2, 0.9]) / 0.5)
    )
    np.testing.assert_allclose(observed[0], expected_first)
    np.testing.assert_allclose(observed[1], expected_second)
    np.testing.assert_allclose(observed[2], classifier._model.class_log_prior_)


def test_craven_normalized_nb_score_handles_an_all_empty_batch() -> None:
    classifier = _CravenClassifier()

    observed = ct._craven_normalized_nb_scores_numpy(
        classifier,
        np.zeros((2, 2), dtype=np.float64),
    )

    np.testing.assert_allclose(
        observed,
        np.broadcast_to(classifier._model.class_log_prior_, (2, 2)),
    )


def test_craven_normalized_nb_score_validates_counts_and_model() -> None:
    with pytest.raises(InductiveValidationError, match="finite non-negative"):
        ct._craven_normalized_nb_scores_numpy(_CravenClassifier(), np.array([[-1.0, 0.0]]))
    with pytest.raises(InductiveValidationError, match="feature_log_prob"):
        ct._craven_normalized_nb_scores_numpy(object(), np.zeros((1, 2)))
    bad = _CravenClassifier()
    bad._model.feature_log_prob_ = np.zeros((2, 3))
    with pytest.raises(InductiveValidationError, match="do not align"):
        ct._craven_normalized_nb_scores_numpy(bad, np.zeros((1, 2)))


def test_blum_mitchell_diagnostic_v2_multiround_oracle_and_prediction_masks() -> None:
    X1_l = np.array(
        [[3, 0, 1, 0], [2, 0, 1, 0], [0, 3, 0, 1], [0, 2, 0, 1]],
        dtype=np.float64,
    )
    X1_u = np.array(
        [
            [3, 0, 1, 0],
            [0, 3, 0, 1],
            [2, 0, 0, 1],
            [0, 2, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.float64,
    )
    X2_l = X1_l[:, ::-1].copy()
    X2_u = X1_u[:, ::-1].copy()
    data = DummyDataset(
        X_l=X1_l,
        y_l=np.array([0, 0, 1, 1], dtype=np.int64),
        views={
            "page": {"X_l": X1_l, "X_u": X1_u},
            "links": {"X_l": X2_l, "X_u": X2_u},
        },
    )
    spec = _blum_v2_spec(
        classifier_id="multinomial_nb",
        classifier_backend="sklearn",
        view_keys=("page", "links"),
        p=1,
        n=1,
        u=4,
        k=1,
        dynamic_feature_selection="mutual_information_presence",
        feature_selection_max_features=2000,
        selection_score="craven_1998_normalized_nb",
    )

    method = ct.CoTrainingMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=4)

    assert method.n_iter_ == 1
    assert method.diagnostics_["protocol"] == "fixed_pool_binary_feature_selection"
    assert method.diagnostics_["selection_score_space"] == "craven_1998_normalized_nb"
    assert method.diagnostics_["dynamic_feature_selection"] == "mutual_information_presence"
    assert method.diagnostics_["test_metrics_used_for_protocol_selection"] is False
    assert method.diagnostics_["selection_diagnostics_scope"] == ("training_and_pseudo_labels_only")
    assert method.round_trace_[0]["selected_feature_count_view1"] == 4
    assert method.round_trace_[0]["selected_feature_count_view2"] == 4
    assert len(method.round_trace_[0]["selected_features_sha256_view1"]) == 64

    prediction = DummyDataset(
        X_l=X1_u,
        y_l=np.zeros(X1_u.shape[0], dtype=np.int64),
        views={"page": {"X": X1_u}, "links": {"X": X2_u}},
    )
    combined = method.predict_proba(prediction)
    page = method.predict_view_proba(prediction, "page")
    assert combined.shape == page.shape == (6, 2)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0)
    np.testing.assert_allclose(page.sum(axis=1), 1.0)


def test_blum_mitchell_diagnostic_v2_rejects_missing_prediction_masks() -> None:
    method = ct.CoTrainingMethod(
        _blum_v2_spec(
            classifier_id="multinomial_nb",
            classifier_backend="sklearn",
            dynamic_feature_selection="mutual_information_presence",
            feature_selection_max_features=2000,
            selection_score="craven_1998_normalized_nb",
        )
    )
    method._clf1 = _FixedScoresClassifier([[0.5, 0.5]])
    method._clf2 = _FixedScoresClassifier([[0.5, 0.5]])
    method._backend = "numpy"
    method._view_keys = ("page", "links")
    X = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(RuntimeError, match="feature-selection state is missing"):
        method._predict_scores_pair(X, X)

    data = DummyDataset(
        X_l=X,
        y_l=np.array([0]),
        views={"page": {"X": X}},
    )
    with pytest.raises(RuntimeError, match="feature-selection state is missing"):
        method.predict_view_proba(data, "page")


def test_blum_mitchell_binary_label_resolution_and_score_selection() -> None:
    spec = _blum_spec(positive_label=7, negative_label=3, p=1, n=1)
    classes = np.array([3, 7], dtype=np.int64)
    assert ct._resolve_binary_labels(spec, classes) == (3, 7)
    indices, labels, confidences = ct._select_binary_quota_numpy(
        np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]]),
        classes,
        positive_label=7,
        negative_label=3,
        p=1,
        n=1,
    )
    np.testing.assert_array_equal(indices, [2, 0])
    np.testing.assert_array_equal(labels, [7, 3])
    np.testing.assert_allclose(confidences, [0.9, 0.9])

    empty = ct._select_binary_quota_numpy(
        np.array([[0.9, 0.1]]),
        classes,
        positive_label=7,
        negative_label=3,
        p=0,
        n=0,
    )
    assert all(array.size == 0 for array in empty)


def test_blum_mitchell_selection_fills_exact_quotas_without_argmax_filter() -> None:
    classes = np.array([0, 1], dtype=np.int64)
    scores = np.tile(np.array([[0.6, 0.4]], dtype=np.float64), (5, 1))

    indices, labels, confidences = ct._select_binary_quota_numpy(
        scores,
        classes,
        positive_label=1,
        negative_label=0,
        p=1,
        n=3,
    )

    # Every row has argmax=negative. The paper nevertheless asks for one positive
    # and three negative labels; stable ties follow pool order and cannot overlap.
    np.testing.assert_array_equal(indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(labels, [1, 0, 0, 0])
    np.testing.assert_allclose(confidences, [0.4, 0.6, 0.6, 0.6])
    assert np.unique(indices).size == 4


@pytest.mark.parametrize(
    ("scores1", "scores2", "message"),
    [
        (np.ones((1, 2)), np.ones((2, 2)), "score shape"),
        (np.array([[np.nan, 1.0]]), np.ones((1, 2)), "finite"),
        (np.array([[-1.0, 2.0]]), np.ones((1, 2)), "non-negative"),
        (np.zeros((1, 2)), np.ones((1, 2)), "positive mass"),
        (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), "zero joint mass"),
    ],
)
def test_blum_mitchell_probability_product_errors(
    scores1: np.ndarray,
    scores2: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(InductiveValidationError, match=message):
        ct._normalized_probability_product_numpy(scores1, scores2)


def test_blum_mitchell_recovers_underflowed_probabilities_from_log_space() -> None:
    X = np.zeros((1, 1), dtype=np.float64)
    first = _FixedScoresClassifier(np.array([[1.0, 0.0]], dtype=np.float32))
    second = _FixedScoresClassifier(np.array([[0.0, 1.0]], dtype=np.float32))
    first._model = _LogProbabilityModel(np.array([[0.0, -1000.0]]))
    second._model = _LogProbabilityModel(np.array([[-1000.0, 0.0]]))
    method = ct.CoTrainingMethod(_blum_spec())
    method._clf1 = first
    method._clf2 = second
    method._backend = "numpy"

    np.testing.assert_allclose(method._predict_scores_pair(X, X), [[0.5, 0.5]])


@pytest.mark.parametrize(
    ("first_log", "second_log", "message"),
    [
        (np.zeros((1, 2)), None, "provided together"),
        (np.zeros((1, 3)), np.zeros((1, 2)), "agree with probability-score shape"),
        (np.array([[np.nan, 0.0]]), np.zeros((1, 2)), "finite or negative infinity"),
        (np.zeros((1, 2)), np.array([[np.nan, 0.0]]), "finite or negative infinity"),
        (np.array([[np.inf, 0.0]]), np.zeros((1, 2)), "finite or negative infinity"),
        (np.zeros((1, 2)), np.array([[np.inf, 0.0]]), "finite or negative infinity"),
    ],
)
def test_blum_mitchell_log_probability_contract(
    first_log: np.ndarray,
    second_log: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(InductiveValidationError, match=message):
        ct._normalized_probability_product_numpy(
            np.array([[0.5, 0.5]]),
            np.array([[0.5, 0.5]]),
            log_scores1=first_log,
            log_scores2=second_log,
        )


def test_blum_mitchell_paper_fit_rejects_nonpaper_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ranking_classifiers(monkeypatch)
    method = ct.CoTrainingMethod(_blum_spec(p=1, n=1, u=13, k=1))
    with pytest.raises(InductiveValidationError, match="at least u"):
        method.fit(_paper_data(), device=DeviceSpec(device="cpu"), seed=0)

    _install_ranking_classifiers(monkeypatch)
    X_l = torch.tensor([[0.0], [1.0]])
    X_u = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    torch_data = DummyDataset(
        X_l=X_l,
        y_l=torch.tensor([0, 1]),
        views={
            "a": {"X_l": X_l, "X_u": X_u},
            "b": {"X_l": X_l.clone(), "X_u": X_u.clone()},
        },
    )
    with pytest.raises(InductiveValidationError, match="requires numpy"):
        ct.CoTrainingMethod(
            _blum_spec(
                classifier_backend="torch",
                n=1,
                u=2,
                k=1,
            )
        ).fit(torch_data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    ("change_first", "message"),
    [(False, "disagree on class labels"), (True, "classes changed")],
)
def test_blum_mitchell_rejects_classifier_class_drift(
    monkeypatch: pytest.MonkeyPatch,
    change_first: bool,
    message: str,
) -> None:
    classifiers = _install_ranking_classifiers(monkeypatch)

    def change_classes_after_fit(classifier: _RankingClassifier) -> None:
        original_fit = classifier.fit

        def changed_fit(X: np.ndarray, y: np.ndarray) -> None:
            original_fit(X, y)
            classifier.classes_ = np.array([0, 2], dtype=np.int64)

        classifier.fit = changed_fit  # type: ignore[method-assign]

    change_classes_after_fit(classifiers[1])
    if change_first:
        change_classes_after_fit(classifiers[0])

    with pytest.raises(InductiveValidationError, match=message):
        ct.CoTrainingMethod(_blum_spec(p=1, n=1, u=2, k=1)).fit(
            _paper_data(6), device=DeviceSpec(device="cpu"), seed=0
        )


def test_blum_mitchell_class_helpers_and_selection_errors() -> None:
    fallback = ct._classifier_classes_numpy(
        _FixedScoresClassifier([[0.5, 0.5]], classes=None), np.array([2, 4])
    )
    np.testing.assert_array_equal(fallback, [2, 4])

    bad_classifier = _FixedScoresClassifier([[0.5, 0.5]])
    bad_classifier.classes_ = np.array([[0, 1]])
    with pytest.raises(InductiveValidationError, match="one-dimensional"):
        ct._classifier_classes_numpy(bad_classifier, np.array([0, 1]))
    with pytest.raises(InductiveValidationError, match="exactly two"):
        ct._resolve_binary_labels(ct.CoTrainingSpec(), np.array([0, 1, 2]))
    with pytest.raises(InductiveValidationError, match="must both occur"):
        ct._resolve_binary_labels(
            ct.CoTrainingSpec(positive_label=3, negative_label=0), np.array([0, 1])
        )
    with pytest.raises(InductiveValidationError, match="align"):
        ct._select_binary_quota_numpy(
            np.ones((2, 3)),
            np.array([0, 1]),
            positive_label=1,
            negative_label=0,
            p=1,
            n=1,
        )
    with pytest.raises(InductiveValidationError, match="finite"):
        ct._select_binary_quota_numpy(
            np.array([[np.inf, 0.0]]),
            np.array([0, 1]),
            positive_label=1,
            negative_label=0,
            p=1,
            n=1,
        )


def test_blum_mitchell_torch_probability_product_branch() -> None:
    X = torch.zeros((1, 1))
    method = ct.CoTrainingMethod(_blum_spec())
    method._clf1 = _FixedScoresClassifier(torch.tensor([[0.8, 0.2]]))
    method._clf2 = _FixedScoresClassifier(torch.tensor([[0.5, 0.5]]))
    method._backend = "torch"
    torch.testing.assert_close(method._predict_scores_pair(X, X), torch.tensor([[0.8, 0.2]]))


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        (torch.tensor([[float("nan"), 1.0]]), torch.ones((1, 2)), "finite"),
        (torch.tensor([[0.0, 0.0]]), torch.ones((1, 2)), "positive mass"),
        (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), "zero joint mass"),
    ],
)
def test_blum_mitchell_torch_probability_product_errors(
    first: torch.Tensor,
    second: torch.Tensor,
    message: str,
) -> None:
    X = torch.zeros((1, 1))
    method = ct.CoTrainingMethod(_blum_spec())
    method._clf1 = _FixedScoresClassifier(first)
    method._clf2 = _FixedScoresClassifier(second)
    method._backend = "torch"
    with pytest.raises(InductiveValidationError, match=message):
        method._predict_scores_pair(X, X)


def test_blum_mitchell_zero_rounds_still_fit_both_classifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, classifiers = _fit_paper(monkeypatch, k=0)
    assert method.n_iter_ == 0
    assert [len(classifier.fit_history) for classifier in classifiers] == [1, 1]
    assert method.diagnostics_["remaining_unlabeled_count"] == 12


def test_trace_selection_preserves_pool_positions() -> None:
    trace = ct._trace_selection(
        pool_indices=np.array([9, 3]),
        local_indices=np.array([1]),
        labels=np.array([7]),
        confidences=np.array([0.75]),
    )
    assert trace == [{"pool_position": 1, "unlabeled_index": 3, "label": 7, "confidence": 0.75}]
