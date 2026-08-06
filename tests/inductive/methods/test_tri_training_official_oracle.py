from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from modssc.inductive.methods.tri_training import (
    TriTrainingMethod,
    _measure_error,
    _paper_update_decision,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tri_training_official_oracle.json"


def _load_oracle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class _FixedClassifier:
    def __init__(self, probabilities: list[list[float]]) -> None:
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self._probabilities = np.asarray(probabilities, dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert len(X) == len(self._probabilities)
        return self._probabilities.copy()


def test_tri_training_matches_pinned_official_transition_oracle() -> None:
    oracle = _load_oracle()
    assert oracle["provenance"]["archive_sha256"] == (
        "f23cda982f521cca607e3fdc50a9f2bc4b0fe5352e7d856083c7c564af66f9e8"
    )
    assert TriTrainingMethod.info.official_code == oracle["provenance"]["archive_url"]

    error_case = oracle["error_estimate"]
    estimate = _measure_error(
        np.asarray(error_case["pred_j"]),
        np.asarray(error_case["pred_k"]),
        np.asarray(error_case["y_true"]),
    )
    assert estimate.agreements == error_case["agreements"]
    assert estimate.wrong_agreements == error_case["wrong_agreements"]
    assert estimate.rate == pytest.approx(error_case["rate"])

    for case in oracle["update_cases"]:
        decision = _paper_update_decision(
            error=case["error"],
            previous_error=case["previous_error"],
            previous_size=case["previous_size"],
            candidate_size=case["candidate_size"],
        )
        assert decision.previous_size == case["effective_previous_size"], case["id"]
        assert decision.selected_size == case["selected_size"], case["id"]
        assert decision.accepted is case["accepted"], case["id"]
        assert decision.subsample is case["subsample"], case["id"]


def test_tri_training_paper_profile_uses_official_probability_aggregation() -> None:
    oracle = _load_oracle()["prediction"]
    method = TriTrainingMethod()
    method._backend = "numpy"
    method._clfs = [
        _FixedClassifier(probabilities) for probabilities in oracle["classifier_probabilities"]
    ]
    method.classes_ = np.array([0, 1], dtype=np.int64)

    X = np.zeros((1, 1), dtype=np.float64)
    probabilities = method.predict_proba(X)
    prediction = method.predict(X)

    np.testing.assert_allclose(
        probabilities,
        np.asarray(oracle["official_probability_average"]),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(prediction, np.asarray(oracle["official_class"]))
    assert oracle["official_class"] != oracle["paper_hard_vote_class"]
