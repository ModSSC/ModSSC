from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods import self_training as self_training_mod
from modssc.inductive.methods.self_training import (
    SelfTrainingMethod,
    SelfTrainingRoundTrace,
    SelfTrainingSpec,
    _apply_dynamic_labeled_minmax_numpy,
    _apply_dynamic_labeled_minmax_torch,
    _dynamic_labeled_minmax_parameters_numpy,
    _dynamic_labeled_minmax_parameters_torch,
    _normalize_group_ids_numpy,
    _normalize_group_ids_torch,
    _paper_pool_ids,
    _resolve_group_ids,
    _select_candidates_numpy,
    _select_li_zhou_2005_1nn_candidates_numpy,
    _validate_paper_selection_spec,
)
from modssc.inductive.types import DeviceSpec

from ..conftest import DummyDataset, make_numpy_dataset, make_torch_dataset

torch = pytest.importorskip("torch")


def _paper_spec(**overrides):
    params = {
        "classifier_params": {"k": 1, "metric": "euclidean"},
        "max_iter": 40,
        "confidence_threshold": None,
        "use_group_propagation": False,
        "selection_strategy": "li_zhou_2005_1nn_distance",
    }
    params.update(overrides)
    return SelfTrainingSpec(**params)


def _paper_oracle_numpy_dataset():
    return DummyDataset(
        X_l=np.array([[0.0], [10.0]], dtype=np.float32),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u=np.array([[0.1], [4.0], [9.9], [6.0]], dtype=np.float32),
    )


def test_group_id_normalization_numpy_and_torch():
    arr = np.array([1, 2], dtype=np.int64)
    out = _normalize_group_ids_numpy(arr, n_expected=2, name="group")
    assert out.shape == (2,)
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_numpy(np.array([[1, 2]]), n_expected=2, name="group")
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_numpy(np.array([1, 2, 3]), n_expected=2, name="group")

    t = torch.tensor([1, 2], dtype=torch.int64)
    out_t = _normalize_group_ids_torch(t, n_expected=2, name="group")
    assert out_t is t
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_torch([1, 2], n_expected=2, name="group")
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_torch(torch.tensor([[1, 2]]), n_expected=2, name="group")
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_torch(torch.tensor([1, 2, 3]), n_expected=2, name="group")
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_torch(torch.tensor([1.0, 2.0]), n_expected=2, name="group")


def test_resolve_group_ids_paths():
    assert (
        _resolve_group_ids(
            None,
            group_key=None,
            n_expected=2,
            backend="numpy",
            name="group",
            key_candidates=("group_u",),
        )
        is None
    )
    with pytest.raises(InductiveValidationError):
        _resolve_group_ids(
            ["bad"],
            group_key=None,
            n_expected=2,
            backend="numpy",
            name="group",
            key_candidates=("group_u",),
        )
    with pytest.raises(InductiveValidationError):
        _resolve_group_ids(
            {"other": np.array([1, 2])},
            group_key="group_u",
            n_expected=2,
            backend="numpy",
            name="group",
            key_candidates=("group_u",),
        )

    meta = {"group_u": np.array([1, 2])}
    out = _resolve_group_ids(
        meta,
        group_key="group_u",
        n_expected=2,
        backend="numpy",
        name="group",
        key_candidates=("group_u",),
    )
    assert np.array_equal(out, meta["group_u"])

    meta2 = {"group_u": np.array([[1, 2]]), "groups": np.array([0, 1])}
    out2 = _resolve_group_ids(
        meta2,
        group_key=None,
        n_expected=2,
        backend="numpy",
        name="group",
        key_candidates=("group_u", "groups"),
    )
    assert np.array_equal(out2, meta2["groups"])

    meta3 = {"group_u": np.array([[1, 2]])}
    out3 = _resolve_group_ids(
        meta3,
        group_key=None,
        n_expected=2,
        backend="numpy",
        name="group",
        key_candidates=("group_u",),
    )
    assert out3 is None

    meta_t = {"group_u": torch.tensor([1, 2], dtype=torch.int64)}
    out_t = _resolve_group_ids(
        meta_t,
        group_key="group_u",
        n_expected=2,
        backend="torch",
        name="group",
        key_candidates=("group_u",),
    )
    assert out_t is meta_t["group_u"]


def test_select_candidates_numpy_group_add_and_truncate():
    scores = np.array(
        [
            [0.2, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.6, 0.4],
            [0.95, 0.05],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )
    pred = scores.argmax(axis=1)
    group_u = np.array([1, 2, 2, 3, 4, 4])
    group_l = np.array([4])
    y_l = np.array([0], dtype=np.int64)
    idx, labels, direct_count, group_added = _select_candidates_numpy(
        scores,
        pred,
        threshold=0.8,
        max_new=2,
        use_group=True,
        group_u=group_u,
        group_l=group_l,
        y_l=y_l,
        group_min_count=2,
        group_min_fraction=0.6,
        group_conf_threshold=0.5,
    )
    assert direct_count == 3
    assert group_added == 1
    assert idx.size == 2
    assert labels.size == 2


def test_select_candidates_numpy_conf_thresholds_and_empty():
    scores = np.array(
        [
            [0.6, 0.4],
            [0.6, 0.4],
            [0.95, 0.05],
            [0.95, 0.05],
        ],
        dtype=np.float32,
    )
    pred = scores.argmax(axis=1)
    group_u = np.array([1, 1, 2, 2])
    idx, labels, direct_count, group_added = _select_candidates_numpy(
        scores,
        pred,
        threshold=0.5,
        max_new=None,
        use_group=True,
        group_u=group_u,
        group_l=None,
        y_l=None,
        group_min_count=2,
        group_min_fraction=0.5,
        group_conf_threshold=0.7,
    )
    assert direct_count == 4
    assert group_added == 0
    assert idx.size == 4
    assert labels.size == 4

    scores2 = np.array([[0.2, 0.8], [0.3, 0.7]], dtype=np.float32)
    pred2 = scores2.argmax(axis=1)
    idx2, labels2, direct_count2, group_added2 = _select_candidates_numpy(
        scores2,
        pred2,
        threshold=None,
        max_new=None,
        use_group=True,
        group_u=np.array([1, 1]),
        group_l=None,
        y_l=None,
        group_min_count=1,
        group_min_fraction=0.5,
        group_conf_threshold=None,
    )
    assert direct_count2 == 2
    assert group_added2 == 0
    assert idx2.size == 2
    assert labels2.size == 2

    idx3, labels3, _, _ = _select_candidates_numpy(
        scores2,
        pred2,
        threshold=1.1,
        max_new=None,
        use_group=False,
        group_u=None,
        group_l=None,
        y_l=None,
        group_min_count=1,
        group_min_fraction=0.5,
        group_conf_threshold=None,
    )
    assert idx3.size == 0
    assert labels3.size == 0


def test_select_candidates_numpy_empty_group_idx(monkeypatch):
    scores = np.array([[0.9, 0.1]], dtype=np.float32)
    pred = scores.argmax(axis=1)
    group_u = np.array([1])
    orig_unique = self_training_mod.np.unique

    def fake_unique(arr, *args, **kwargs):
        if arr is group_u:
            return np.array([999], dtype=arr.dtype)
        return orig_unique(arr, *args, **kwargs)

    monkeypatch.setattr(self_training_mod.np, "unique", fake_unique)
    idx, labels, direct_count, group_added = _select_candidates_numpy(
        scores,
        pred,
        threshold=0.5,
        max_new=None,
        use_group=True,
        group_u=group_u,
        group_l=None,
        y_l=None,
        group_min_count=1,
        group_min_fraction=0.5,
        group_conf_threshold=0.5,
    )
    assert direct_count == 1
    assert group_added == 0
    assert idx.size == 1
    assert labels.size == 1


def test_li_zhou_2005_distance_selection_oracle_and_assumptions():
    X_l = np.array([[0.0], [10.0]], dtype=np.float32)
    y_l = np.array([0, 1], dtype=np.int64)
    X_pool = np.array([[0.1], [4.0], [9.9], [6.0]], dtype=np.float32)
    pool_ids = np.array([30, 10, 20, 40], dtype=np.int64)

    margin_idx, margin_labels = _select_li_zhou_2005_1nn_candidates_numpy(
        X_l,
        y_l,
        X_pool,
        pool_ids,
        per_class_unspecified=1,
        distance_confidence_unspecified="margin",
    )
    assert margin_idx.tolist() == [0, 2]
    assert margin_labels.tolist() == [0, 1]

    ratio_idx, ratio_labels = _select_li_zhou_2005_1nn_candidates_numpy(
        X_l,
        y_l,
        X_pool,
        pool_ids,
        per_class_unspecified={0: 2, 1: 0},
        distance_confidence_unspecified="ratio",
    )
    assert ratio_idx.tolist() == [0, 1]
    assert ratio_labels.tolist() == [0, 0]

    no_class_one_idx, no_class_one_labels = _select_li_zhou_2005_1nn_candidates_numpy(
        X_l,
        y_l,
        np.array([[0.2]], dtype=np.float32),
        np.array([7], dtype=np.int64),
        per_class_unspecified=1,
        distance_confidence_unspecified="margin",
    )
    assert no_class_one_idx.tolist() == [0]
    assert no_class_one_labels.tolist() == [0]


def test_li_zhou_2005_nearest_neighbor_distance_confidence_is_distinct_from_margin():
    X_l = np.array([[0.0], [10.0]], dtype=np.float64)
    y_l = np.array([0, 1], dtype=np.int64)
    X_pool = np.array([[-100.0], [4.0], [9.9]], dtype=np.float64)
    pool_ids = np.array([5, 4, 3], dtype=np.int64)

    margin_idx, _ = _select_li_zhou_2005_1nn_candidates_numpy(
        X_l,
        y_l,
        X_pool,
        pool_ids,
        per_class_unspecified=1,
        distance_confidence_unspecified="margin",
    )
    nearest_idx, nearest_labels = _select_li_zhou_2005_1nn_candidates_numpy(
        X_l,
        y_l,
        X_pool,
        pool_ids,
        per_class_unspecified=1,
        distance_confidence_unspecified="nearest_neighbor_distance",
    )

    assert margin_idx.tolist() == [0, 2]
    assert nearest_idx.tolist() == [1, 2]
    assert nearest_labels.tolist() == [0, 1]


def test_dynamic_labeled_minmax_numpy_and_torch_ignore_constant_features():
    X_l = np.array([[0.0, 5.0], [10.0, 5.0]], dtype=np.float32)
    X_query = np.array([[5.0, 999.0], [20.0, -999.0]], dtype=np.float32)
    numpy_parameters = _dynamic_labeled_minmax_parameters_numpy(X_l)
    numpy_scaled = _apply_dynamic_labeled_minmax_numpy(X_query, numpy_parameters)
    assert np.array_equal(numpy_scaled, np.array([[0.5, 0.0], [2.0, 0.0]]))

    X_l_t = torch.tensor(X_l)
    X_query_t = torch.tensor(X_query)
    torch_parameters = _dynamic_labeled_minmax_parameters_torch(X_l_t)
    torch_scaled = _apply_dynamic_labeled_minmax_torch(X_query_t, torch_parameters)
    assert torch.equal(torch_scaled, torch.tensor([[0.5, 0.0], [2.0, 0.0]]))


@pytest.mark.parametrize(
    "fit_values",
    [np.empty((0, 2)), np.array([1.0, 2.0])],
)
def test_dynamic_labeled_minmax_numpy_rejects_invalid_fit_matrix(fit_values):
    with pytest.raises(InductiveValidationError, match="non-empty 2D"):
        _dynamic_labeled_minmax_parameters_numpy(fit_values)


def test_dynamic_labeled_minmax_rejects_incompatible_transform_matrix():
    numpy_parameters = _dynamic_labeled_minmax_parameters_numpy(np.array([[0.0, 1.0], [1.0, 2.0]]))
    with pytest.raises(InductiveValidationError, match="fitted feature width"):
        _apply_dynamic_labeled_minmax_numpy(np.array([[1.0]]), numpy_parameters)

    torch_parameters = _dynamic_labeled_minmax_parameters_torch(
        torch.tensor([[0.0, 1.0], [1.0, 2.0]])
    )
    with pytest.raises(InductiveValidationError, match="fitted feature width"):
        _apply_dynamic_labeled_minmax_torch(torch.tensor([[1.0]]), torch_parameters)
    with pytest.raises(InductiveValidationError, match="non-empty 2D"):
        _dynamic_labeled_minmax_parameters_torch(torch.empty((0, 2)))


@pytest.mark.parametrize(
    ("X_l", "y_l", "X_pool", "pool_ids", "confidence", "message"),
    [
        (
            np.array([0.0, 1.0]),
            np.array([0, 1]),
            np.array([[0.2]]),
            np.array([0]),
            "margin",
            "2D feature matrices",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([0, 1]),
            np.array([[0.2, 0.3]]),
            np.array([0]),
            "margin",
            "same width",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([0, 1]),
            np.array([[0.2]]),
            np.array([0, 1]),
            "margin",
            "pool_ids",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([0, 0]),
            np.array([[0.2]]),
            np.array([0]),
            "margin",
            "at least two",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([0, 1]),
            np.array([[0.2]]),
            np.array([0]),
            "unknown",
            "must be 'margin', 'ratio', or 'nearest_neighbor_distance'",
        ),
    ],
)
def test_li_zhou_2005_distance_selection_errors(X_l, y_l, X_pool, pool_ids, confidence, message):
    with pytest.raises(InductiveValidationError, match=message):
        _select_li_zhou_2005_1nn_candidates_numpy(
            X_l,
            y_l,
            X_pool,
            pool_ids,
            per_class_unspecified=1,
            distance_confidence_unspecified=confidence,
        )


def test_li_zhou_2005_distance_selection_empty_pool():
    idx, labels = _select_li_zhou_2005_1nn_candidates_numpy(
        np.array([[0.0], [1.0]]),
        np.array([0, 1]),
        np.empty((0, 1)),
        np.empty((0,), dtype=np.int64),
        per_class_unspecified=1,
        distance_confidence_unspecified="margin",
    )
    assert idx.size == 0
    assert labels.size == 0


def test_li_zhou_2005_seeded_pool_is_retained_and_replenished():
    remaining = np.arange(6, dtype=np.int64)
    rng = np.random.default_rng(19)
    initial = _paper_pool_ids(
        remaining,
        previous_pool_ids=None,
        pool_size=3,
        rng=rng,
    )
    reduced = remaining[remaining != initial[0]]
    replenished = _paper_pool_ids(
        reduced,
        previous_pool_ids=initial,
        pool_size=3,
        rng=rng,
    )
    assert set(initial[1:]).issubset(set(replenished))
    assert len(set(replenished)) == 3

    replay_rng = np.random.default_rng(19)
    replay_initial = _paper_pool_ids(
        remaining,
        previous_pool_ids=None,
        pool_size=3,
        rng=replay_rng,
    )
    replay_replenished = _paper_pool_ids(
        reduced,
        previous_pool_ids=replay_initial,
        pool_size=3,
        rng=replay_rng,
    )
    assert np.array_equal(initial, replay_initial)
    assert np.array_equal(replenished, replay_replenished)

    retained = _paper_pool_ids(
        remaining,
        previous_pool_ids=initial,
        pool_size=2,
        rng=rng,
    )
    assert retained.tolist() == initial[:2].tolist()
    assert np.array_equal(
        _paper_pool_ids(
            remaining,
            previous_pool_ids=initial,
            pool_size=None,
            rng=rng,
        ),
        remaining,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"classifier_id": "logistic_regression"}, "classifier_id='knn'"),
        ({"classifier_params": {}}, "classifier_params.k=1"),
        ({"classifier_params": {"k": 1.5}}, "classifier_params.k=1"),
        ({"classifier_params": {"k": 1, "metric": "cosine"}}, "Euclidean"),
        ({"confidence_threshold": 0.5}, "confidence_threshold=None"),
        ({"max_new_labels": 2}, "max_new_labels must be None"),
        ({"use_group_propagation": True}, "incompatible with group propagation"),
        ({"paper_pool_size_unspecified": 0}, "paper_pool_size_unspecified"),
        ({"paper_pool_size_unspecified": True}, "paper_pool_size_unspecified"),
        ({"paper_pool_size_unspecified": 1.5}, "paper_pool_size_unspecified"),
        ({"paper_candidates_per_class_unspecified": {}}, "must not be empty"),
        ({"paper_candidates_per_class_unspecified": -1}, "non-negative integers"),
        ({"paper_candidates_per_class_unspecified": True}, "non-negative integers"),
        ({"paper_candidates_per_class_unspecified": 1.5}, "non-negative integers"),
        ({"paper_candidates_per_class_unspecified": {0: 0}}, "positive quota"),
        ({"paper_distance_confidence_unspecified": "unknown"}, "nearest_neighbor_distance"),
        ({"paper_feature_scaling_unspecified": "unknown"}, "dynamic_labeled_minmax"),
    ],
)
def test_li_zhou_2005_spec_validation_errors(overrides, message):
    with pytest.raises(InductiveValidationError, match=message):
        _validate_paper_selection_spec(_paper_spec(**overrides))


def test_li_zhou_2005_spec_accepts_explicit_positive_pool_size():
    _validate_paper_selection_spec(_paper_spec(paper_pool_size_unspecified=75))


def test_li_zhou_2005_confirmation_v2_numpy_and_torch_are_replayable():
    spec = _paper_spec(
        paper_pool_size_unspecified=3,
        paper_distance_confidence_unspecified="nearest_neighbor_distance",
        paper_feature_scaling_unspecified="dynamic_labeled_minmax",
    )
    numpy_data = _paper_oracle_numpy_dataset()
    numpy_method = SelfTrainingMethod(spec)
    numpy_method.fit(numpy_data, device=DeviceSpec(device="cpu"), seed=19)

    replay = SelfTrainingMethod(spec)
    replay.fit(numpy_data, device=DeviceSpec(device="cpu"), seed=19)
    assert replay.round_trace_ == numpy_method.round_trace_
    assert np.array_equal(replay.predict(numpy_data.X_l), numpy_method.predict(numpy_data.X_l))
    assert numpy_method.diagnostics_["selection_parameters"] == {
        "selection_strategy": "li_zhou_2005_1nn_distance",
        "classifier_id": "knn",
        "classifier_k": 1,
        "classifier_metric": "euclidean",
        "max_iter": 40,
        "confidence_threshold": None,
        "max_new_labels": None,
        "min_new_labels": 1,
        "use_group_propagation": False,
        "paper_pool_size_unspecified": 3,
        "paper_candidates_per_class_unspecified": 1,
        "paper_distance_confidence_unspecified": "nearest_neighbor_distance",
        "paper_feature_scaling_unspecified": "dynamic_labeled_minmax",
    }

    torch_data = DummyDataset(
        X_l=torch.tensor(numpy_data.X_l),
        y_l=torch.tensor(numpy_data.y_l, dtype=torch.int64),
        X_u=torch.tensor(numpy_data.X_u),
    )
    torch_method = SelfTrainingMethod(
        _paper_spec(
            classifier_backend="torch",
            paper_pool_size_unspecified=3,
            paper_distance_confidence_unspecified="nearest_neighbor_distance",
            paper_feature_scaling_unspecified="dynamic_labeled_minmax",
        )
    )
    torch_method.fit(torch_data, device=DeviceSpec(device="cpu"), seed=19)
    assert torch_method.round_trace_ == numpy_method.round_trace_
    assert torch.equal(
        torch_method.predict(torch_data.X_l),
        torch.tensor(numpy_method.predict(numpy_data.X_l)),
    )


def test_li_zhou_2005_numpy_full_round_trace_is_exact_and_immutable():
    method = SelfTrainingMethod(_paper_spec())
    method.fit(_paper_oracle_numpy_dataset(), device=DeviceSpec(device="cpu"), seed=3)

    assert method.round_trace_ == (
        SelfTrainingRoundTrace(
            iteration=0,
            labeled_before=2,
            unlabeled_before=4,
            pool_indices=(0, 1, 2, 3),
            candidate_indices=(0, 2),
            candidate_labels=(0, 1),
            accepted_indices=(0, 2),
            accepted_labels=(0, 1),
            labeled_after=4,
            remaining_unlabeled=2,
        ),
        SelfTrainingRoundTrace(
            iteration=1,
            labeled_before=4,
            unlabeled_before=2,
            pool_indices=(1, 3),
            candidate_indices=(1, 3),
            candidate_labels=(0, 1),
            accepted_indices=(1, 3),
            accepted_labels=(0, 1),
            labeled_after=6,
            remaining_unlabeled=0,
        ),
    )
    with pytest.raises(FrozenInstanceError):
        method.round_trace_[0].remaining_unlabeled = 99
    with pytest.raises(AttributeError):
        method.round_trace_ = ()

    assert method.diagnostics_ == {
        "protocol": "li_zhou_2005_1nn_distance",
        "seed": 3,
        "n_iter": 2,
        "initial_labeled_size": 2,
        "initial_unlabeled_count": 4,
        "final_labeled_size": 6,
        "remaining_unlabeled_count": 0,
        "pseudo_labels_added": 4,
        "selection_parameters": {
            "selection_strategy": "li_zhou_2005_1nn_distance",
            "classifier_id": "knn",
            "classifier_k": 1,
            "classifier_metric": "euclidean",
            "max_iter": 40,
            "confidence_threshold": None,
            "max_new_labels": None,
            "min_new_labels": 1,
            "use_group_propagation": False,
            "paper_pool_size_unspecified": None,
            "paper_candidates_per_class_unspecified": 1,
            "paper_distance_confidence_unspecified": "margin",
        },
        "round_trace": [
            {
                "iteration": 0,
                "labeled_before": 2,
                "unlabeled_before": 4,
                "pool_indices": [0, 1, 2, 3],
                "candidate_indices": [0, 2],
                "candidate_labels": [0, 1],
                "accepted_indices": [0, 2],
                "accepted_labels": [0, 1],
                "labeled_after": 4,
                "remaining_unlabeled": 2,
            },
            {
                "iteration": 1,
                "labeled_before": 4,
                "unlabeled_before": 2,
                "pool_indices": [1, 3],
                "candidate_indices": [1, 3],
                "candidate_labels": [0, 1],
                "accepted_indices": [1, 3],
                "accepted_labels": [0, 1],
                "labeled_after": 6,
                "remaining_unlabeled": 0,
            },
        ],
    }
    assert json.loads(json.dumps(method.diagnostics_)) == method.diagnostics_

    replay = SelfTrainingMethod(_paper_spec())
    replay.fit(_paper_oracle_numpy_dataset(), device=DeviceSpec(device="cpu"), seed=3)
    assert replay.round_trace_ == method.round_trace_


def test_li_zhou_2005_diagnostics_reset_on_every_fit():
    method = SelfTrainingMethod(_paper_spec())
    method.fit(_paper_oracle_numpy_dataset(), device=DeviceSpec(device="cpu"), seed=3)
    method.diagnostics_["stale"] = True

    no_unlabeled = DummyDataset(
        X_l=np.array([[0.0], [10.0]], dtype=np.float32),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u=np.empty((0, 1), dtype=np.float32),
    )
    method.fit(no_unlabeled, device=DeviceSpec(device="cpu"), seed=7)

    assert method.round_trace_ == ()
    assert "stale" not in method.diagnostics_
    assert method.diagnostics_["seed"] == 7
    assert method.diagnostics_["n_iter"] == 0
    assert method.diagnostics_["initial_labeled_size"] == 2
    assert method.diagnostics_["initial_unlabeled_count"] == 0
    assert method.diagnostics_["final_labeled_size"] == 2
    assert method.diagnostics_["remaining_unlabeled_count"] == 0
    assert method.diagnostics_["pseudo_labels_added"] == 0
    assert method.diagnostics_["round_trace"] == []

    method.diagnostics_["stale"] = True
    one_class = DummyDataset(
        X_l=np.array([[0.0], [1.0]], dtype=np.float32),
        y_l=np.array([0, 0], dtype=np.int64),
        X_u=np.array([[0.5]], dtype=np.float32),
    )
    with pytest.raises(InductiveValidationError, match="at least two labeled classes"):
        method.fit(one_class, device=DeviceSpec(device="cpu"), seed=9)
    assert method.diagnostics_ == {}
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict(np.array([[0.0]], dtype=np.float32))


def test_li_zhou_2005_diagnostics_serialize_mapping_quotas():
    method = SelfTrainingMethod(_paper_spec(paper_candidates_per_class_unspecified={0: 1, 1: 1}))
    method.fit(_paper_oracle_numpy_dataset(), device=DeviceSpec(device="cpu"), seed=3)

    assert method.diagnostics_["selection_parameters"][
        "paper_candidates_per_class_unspecified"
    ] == {"0": 1, "1": 1}
    json.dumps(method.diagnostics_)


def test_li_zhou_2005_rejected_candidates_are_traced_not_added():
    method = SelfTrainingMethod(_paper_spec(min_new_labels=3))
    method.fit(_paper_oracle_numpy_dataset(), device=DeviceSpec(device="cpu"), seed=0)
    assert len(method.round_trace_) == 1
    trace = method.round_trace_[0]
    assert trace.candidate_indices == (0, 2)
    assert trace.accepted_indices == ()
    assert trace.labeled_after == 2
    assert trace.remaining_unlabeled == 4


def test_self_training_errors_and_predict_mismatch():
    data = make_numpy_dataset()
    with pytest.raises(InductiveValidationError, match="Unknown selection_strategy"):
        SelfTrainingMethod(SelfTrainingSpec(selection_strategy="unknown")).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError, match="only valid"):
        SelfTrainingMethod(
            SelfTrainingSpec(paper_feature_scaling_unspecified="dynamic_labeled_minmax")
        ).fit(data, device=DeviceSpec(device="cpu"), seed=0)
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec(group_min_count=0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec(group_min_fraction=1.1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec(use_group_propagation=True)).fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=data.X_u, meta={}),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    method = SelfTrainingMethod()
    with pytest.raises(RuntimeError):
        method.predict_proba(np.zeros((1, 2)))
    with pytest.raises(RuntimeError):
        method.predict(np.zeros((1, 2)))

    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    method._backend = ""
    with pytest.raises(InductiveValidationError):
        method.predict_proba(torch.tensor([[0.0, 1.0]]))
    with pytest.raises(InductiveValidationError):
        method.predict(torch.tensor([[0.0, 1.0]]))


def test_self_training_numpy_group_and_breaks():
    data = make_numpy_dataset(n_l=4, n_u=2)
    ds = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        meta={"group_u": np.array([0, 1], dtype=np.int64)},
    )
    spec = SelfTrainingSpec(max_iter=2, confidence_threshold=0.0, min_new_labels=1)
    method = SelfTrainingMethod(spec)
    method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    assert proba.shape[0] == data.X_l.shape[0]
    assert method.diagnostics_ == {}


def test_self_training_numpy_min_new_labels_break():
    data = make_numpy_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(
        max_iter=1,
        confidence_threshold=0.0,
        min_new_labels=2,
        use_group_propagation=False,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_numpy_no_unlabeled_and_skip_loop():
    data = make_numpy_dataset(n_l=4, n_u=0)
    method = SelfTrainingMethod(SelfTrainingSpec())
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    data_loop = make_numpy_dataset(n_l=4, n_u=2)
    spec = SelfTrainingSpec(max_iter=0, use_group_propagation=False)
    method2 = SelfTrainingMethod(spec)
    method2.fit(data_loop, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_numpy_no_group_update_branch():
    data = make_numpy_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(
        max_iter=1,
        confidence_threshold=0.0,
        min_new_labels=1,
        use_group_propagation=False,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    pred = method.predict(data.X_l)
    assert pred.shape[0] == data.X_l.shape[0]


def test_self_training_numpy_empty_xl_error(monkeypatch):
    def fake_ensure_numpy_data(_data):
        return SimpleNamespace(
            X_l=np.empty((0, 2), dtype=np.float32),
            y_l=np.array([0], dtype=np.int64),
            X_u=np.array([[0.1, 0.2]], dtype=np.float32),
            meta=None,
        )

    monkeypatch.setattr(self_training_mod, "ensure_numpy_data", fake_ensure_numpy_data)
    data = DummyDataset(
        X_l=np.array([[0.0, 1.0]], dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=np.array([[0.1, 0.2]], dtype=np.float32),
    )
    method = SelfTrainingMethod()
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_early_return_and_predict_proba():
    X_l = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    ds = DummyDataset(X_l=X_l, y_l=y_l, X_u=None)
    method = SelfTrainingMethod(SelfTrainingSpec(classifier_backend="torch"))
    method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(X_l)
    assert int(proba.shape[0]) == int(X_l.shape[0])


def test_li_zhou_2005_torch_matches_numpy_round_trace():
    numpy_data = _paper_oracle_numpy_dataset()
    torch_data = DummyDataset(
        X_l=torch.tensor(numpy_data.X_l),
        y_l=torch.tensor(numpy_data.y_l, dtype=torch.int64),
        X_u=torch.tensor(numpy_data.X_u),
    )
    method = SelfTrainingMethod(_paper_spec(classifier_backend="torch"))
    method.fit(torch_data, device=DeviceSpec(device="cpu"), seed=3)

    numpy_method = SelfTrainingMethod(_paper_spec())
    numpy_method.fit(numpy_data, device=DeviceSpec(device="cpu"), seed=3)
    assert method.round_trace_ == numpy_method.round_trace_
    assert method.diagnostics_ == numpy_method.diagnostics_


def test_self_training_torch_group_flow_and_breaks():
    data = make_torch_dataset(n_l=4, n_u=2)
    ds = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        meta={"group_u": torch.tensor([0, 1], dtype=torch.int64)},
    )
    spec = SelfTrainingSpec(
        classifier_backend="torch",
        max_iter=2,
        confidence_threshold=0.0,
        min_new_labels=1,
    )
    method = SelfTrainingMethod(spec)
    method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_min_new_labels_break():
    data = make_torch_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(
        classifier_backend="torch",
        max_iter=1,
        confidence_threshold=0.0,
        min_new_labels=2,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_group_missing_error():
    data = make_torch_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(classifier_backend="torch", use_group_propagation=True)
    method = SelfTrainingMethod(spec)
    with pytest.raises(InductiveValidationError):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_skip_loop_and_no_group_update():
    data = make_torch_dataset(n_l=4, n_u=2)
    spec = SelfTrainingSpec(
        classifier_backend="torch",
        max_iter=0,
        use_group_propagation=False,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_no_group_update_branch():
    data = make_torch_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(
        classifier_backend="torch",
        max_iter=1,
        confidence_threshold=0.0,
        min_new_labels=1,
        use_group_propagation=False,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_empty_xl_error(monkeypatch):
    def fake_ensure_torch_data(_data, *, device):
        return SimpleNamespace(
            X_l=torch.zeros((0, 2)),
            y_l=torch.tensor([0], dtype=torch.int64),
            X_u=torch.tensor([[0.1, 0.2]]),
            meta=None,
        )

    monkeypatch.setattr(self_training_mod, "ensure_torch_data", fake_ensure_torch_data)
    data = DummyDataset(
        X_l=torch.tensor([[0.0, 1.0]]),
        y_l=torch.tensor([0], dtype=torch.int64),
        X_u=torch.tensor([[0.1, 0.2]]),
    )
    method = SelfTrainingMethod(SelfTrainingSpec(classifier_backend="torch"))
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
