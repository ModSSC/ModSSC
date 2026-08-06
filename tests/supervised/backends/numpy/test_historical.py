from __future__ import annotations

import numpy as np
import pytest

from modssc.supervised.api import classifier_info, create_classifier
from modssc.supervised.backends.numpy.c45 import NumpyC45Classifier, _entropy
from modssc.supervised.backends.numpy.naive_bayes import NumpyNaiveBayesClassifier
from modssc.supervised.backends.numpy.tabular import TabularEncoder, TabularFeature, _is_missing
from modssc.supervised.errors import SupervisedValidationError

VOTE_SCHEMA = (
    {"type": "nominal", "values": ("n", "y")},
    {"type": "nominal", "values": ("n", "y")},
)


def test_historical_numpy_backends_are_registered_and_dependency_free() -> None:
    tree = classifier_info("decision_tree")["backends"]["numpy"]
    bayes = classifier_info("gaussian_nb")["backends"]["numpy"]
    assert tree["required_extra"] is None
    assert bayes["required_extra"] is None
    assert isinstance(create_classifier("decision_tree", backend="numpy"), NumpyC45Classifier)
    assert isinstance(create_classifier("gaussian_nb", backend="numpy"), NumpyNaiveBayesClassifier)


def test_tabular_encoder_declared_schema_missing_values_and_transform_validation() -> None:
    encoder = TabularEncoder(
        feature_schema=("numeric", {"kind": "nominal", "values": ("n", "y")}),
        missing_values=("?", -999),
        classifier_name="test",
    )
    encoded = encoder.fit_transform([[1, "n"], ["?", "y"], [-999, None]])
    assert encoder.features_ == (
        TabularFeature("numeric"),
        TabularFeature("nominal", ("n", "y")),
    )
    assert np.array_equal(encoded[0], [1.0, 0.0])
    assert np.isnan(encoded[1, 0])
    assert np.isnan(encoded[2]).all()
    assert np.array_equal(encoder.transform([[2.5, "y"]]), [[2.5, 1.0]])

    with pytest.raises(SupervisedValidationError, match="expected 2 features"):
        encoder.transform([[1.0]])
    with pytest.raises(SupervisedValidationError, match="requires numeric"):
        encoder.transform([["bad", "n"]])
    with pytest.raises(SupervisedValidationError, match="finite numeric"):
        encoder.transform([[np.inf, "n"]])
    with pytest.raises(SupervisedValidationError, match="received nominal value"):
        encoder.transform([[1.0, "maybe"]])


def test_tabular_encoder_infers_numeric_and_nominal_features() -> None:
    encoder = TabularEncoder(
        feature_schema=None,
        missing_values=("?",),
        classifier_name="inferred",
    )
    encoded = encoder.fit_transform([["1", "z"], ["2", "a"], ["?", "z"]])
    assert encoder.features_ == (
        TabularFeature("numeric"),
        TabularFeature("nominal", ("a", "z")),
    )
    assert np.array_equal(encoded[:2], [[1.0, 1.0], [2.0, 0.0]])
    assert np.isnan(encoded[2, 0])

    empty = TabularEncoder(
        feature_schema=None,
        missing_values=("?",),
        classifier_name="empty",
    )
    assert empty.fit_transform([["?"], [None]]).shape == (2, 1)

    nonfinite = TabularEncoder(
        feature_schema=None,
        missing_values=("?",),
        classifier_name="nonfinite",
    )
    assert nonfinite.fit_transform([[np.inf], ["token"]]).shape == (2, 1)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (("numeric",), "exactly 2 entries"),
        ((42, "numeric"), "must be a string or mapping"),
        (({"type": "numeric", "values": ("x",)}, "numeric"), "cannot define values"),
        (({"type": "date"}, "numeric"), "type must be"),
        (({"type": "nominal", "values": "xy"}, "numeric"), "requires a values sequence"),
        (({"type": "nominal", "values": ()}, "numeric"), "has invalid values"),
        (({"type": "nominal", "values": ("x", "x")}, "numeric"), "has invalid values"),
        (({"type": "nominal", "values": ("?", "x")}, "numeric"), "has invalid values"),
    ],
)
def test_tabular_encoder_rejects_invalid_schemas(schema: object, message: str) -> None:
    encoder = TabularEncoder(
        feature_schema=schema,  # type: ignore[arg-type]
        missing_values=("?",),
        classifier_name="test",
    )
    with pytest.raises(SupervisedValidationError, match=message):
        encoder.fit_transform([[1, 2]])


def test_tabular_encoder_requires_fit_and_nonempty_input() -> None:
    encoder = TabularEncoder(feature_schema=None, missing_values=("?",), classifier_name="test")
    with pytest.raises(RuntimeError, match="not fitted"):
        encoder.transform([[1]])
    with pytest.raises(SupervisedValidationError, match="X must be non-empty"):
        encoder.fit_transform(np.empty((0, 2)))


def test_missing_value_detection_is_defensive() -> None:
    class Ambiguous:
        def __eq__(self, _other: object) -> bool:
            raise ValueError("ambiguous")

    assert _is_missing(None, ("?",))
    assert _is_missing(float("nan"), ("?",))
    assert _is_missing("?", (None, "?"))
    assert not _is_missing(Ambiguous(), ("?",))


def test_numpy_historical_naive_bayes_nominal_missing_and_string_labels() -> None:
    X = np.asarray(
        [["n", "n"], ["n", "y"], ["y", "n"], ["y", "y"], ["?", "y"]],
        dtype=object,
    )
    y = np.asarray(["dem", "dem", "rep", "rep", "dem"])
    classifier = NumpyNaiveBayesClassifier(
        feature_schema=VOTE_SCHEMA,
        missing_values=("?",),
        alpha=1.0,
    )
    result = classifier.fit(X, y)
    probabilities = classifier.predict_proba(X)
    assert result.n_samples == 5
    assert probabilities.shape == (5, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(classifier.predict_scores(X), probabilities)
    assert np.array_equal(classifier.predict(X), y)
    assert classifier.supports_proba


def test_numpy_historical_naive_bayes_numeric_fallbacks_and_uniform_prior() -> None:
    X = np.asarray([[0.0, "?"], [0.2, "?"], [10.0, "?"], ["?", "?"]], dtype=object)
    y = np.asarray([0, 0, 1, 2])
    classifier = NumpyNaiveBayesClassifier(
        feature_schema=("numeric", "numeric"),
        fit_prior=False,
        missing_values=("?",),
    )
    classifier.fit(X, y)
    assert np.allclose(np.exp(classifier.class_log_prior_), np.full(3, 1.0 / 3.0))
    assert classifier._numeric_mean[0][2] == pytest.approx(3.4)
    assert np.isfinite(classifier._numeric_var[1]).all()
    missing_probabilities = classifier.predict_proba([["?", "?"]])
    assert np.allclose(missing_probabilities, [[1 / 3, 1 / 3, 1 / 3]])
    assert classifier.predict_proba([[0.1, "?"]]).shape == (1, 3)

    all_missing_nominal = NumpyNaiveBayesClassifier(
        feature_schema=({"type": "nominal", "values": ("n", "y")},),
        missing_values=("?",),
    )
    all_missing_nominal.fit([["?"], ["?"]], [0, 1])
    assert np.allclose(all_missing_nominal.predict_proba([["?"]]), [[0.5, 0.5]])


@pytest.mark.parametrize(
    ("params", "message"),
    [({"alpha": 0.0}, "alpha must be"), ({"var_smoothing": 0.0}, "var_smoothing")],
)
def test_numpy_historical_naive_bayes_validates_parameters(
    params: dict[str, float], message: str
) -> None:
    with pytest.raises(SupervisedValidationError, match=message):
        NumpyNaiveBayesClassifier(**params).fit([[0.0]], [0])


def test_numpy_historical_naive_bayes_validates_fit_state_and_sizes() -> None:
    classifier = NumpyNaiveBayesClassifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        classifier.predict_proba([[0.0]])
    with pytest.raises(SupervisedValidationError, match="incompatible sizes"):
        classifier.fit([[0.0], [1.0]], [0])


def test_entropy_and_numpy_c45_numeric_gain_ratio_probabilities() -> None:
    assert _entropy(np.asarray([0.0, 0.0])) == 0.0
    assert _entropy(np.asarray([2.0, 2.0])) == pytest.approx(1.0)
    X = np.asarray([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2], [np.nan]])
    y = np.asarray([0, 0, 0, 1, 1, 1, 0])
    classifier = NumpyC45Classifier(min_num_obj=1, probability_smoothing=0.5)
    result = classifier.fit(X, y)
    assert result.n_features == 1
    assert classifier.tree_ is not None
    assert classifier.tree_.threshold == pytest.approx(0.6)
    assert np.array_equal(classifier.predict([[0.05], [1.05]]), [0, 1])
    probabilities = classifier.predict_proba([[np.nan]])
    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(classifier.predict_scores([[0.05]]), classifier.predict_proba([[0.05]]))
    assert classifier.supports_proba


def test_numpy_c45_nominal_multiway_and_unseen_declared_branch() -> None:
    schema = ({"type": "nominal", "values": ("a", "b", "c")},)
    classifier = NumpyC45Classifier(feature_schema=schema, min_num_obj=1)
    classifier.fit([["a"], ["a"], ["b"], ["b"], ["?"]], [0, 0, 1, 1, 0])
    assert classifier.tree_ is not None
    assert classifier.tree_.threshold is None
    assert classifier.tree_.branches == (0.0, 1.0)
    assert np.array_equal(classifier.predict([["a"], ["b"]]), [0, 1])
    unseen = classifier.predict_proba([["c"]])
    missing = classifier.predict_proba([["?"]])
    assert np.allclose(unseen, missing)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"min_num_obj": 0}, "min_num_obj"),
        ({"max_depth": -1}, "max_depth"),
        ({"min_gain": -1.0}, "min_gain"),
        ({"probability_smoothing": -1.0}, "probability_smoothing"),
        ({"unpruned": False}, "unpruned historical tree"),
        ({"binary_splits": True}, "multi-way nominal splits"),
    ],
)
def test_numpy_c45_validates_parameters(params: dict[str, object], message: str) -> None:
    with pytest.raises(SupervisedValidationError, match=message):
        NumpyC45Classifier(**params).fit([[0], [1]], [0, 1])


def test_numpy_c45_validates_fit_state_and_sizes() -> None:
    classifier = NumpyC45Classifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        classifier.predict_proba([[0]])
    with pytest.raises(RuntimeError, match="not fitted"):
        classifier._candidate_split(np.asarray([0]), np.asarray([1.0]), feature_index=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        classifier._build_tree(
            np.asarray([0]), np.asarray([1.0]), depth=0, nominal_available=frozenset()
        )
    with pytest.raises(SupervisedValidationError, match="incompatible sizes"):
        classifier.fit([[0], [1]], [0])


def test_numpy_c45_leaf_stopping_rules_and_no_valid_split() -> None:
    pure = NumpyC45Classifier(min_num_obj=1).fit([[0], [1]], ["a", "a"])
    assert pure.n_classes == 1

    depth_limited = NumpyC45Classifier(max_depth=0, min_num_obj=1)
    depth_limited.fit([[0], [1]], [0, 1])
    assert depth_limited.tree_ is not None and depth_limited.tree_.feature is None

    too_small = NumpyC45Classifier(min_num_obj=2)
    too_small.fit([[0], [1]], [0, 1])
    assert too_small.tree_ is not None and too_small.tree_.feature is None

    no_gain = NumpyC45Classifier(min_num_obj=1)
    no_gain.fit([[0], [0], [0], [0]], [0, 1, 0, 1])
    assert no_gain.tree_ is not None and no_gain.tree_.feature is None

    one_nominal_value = NumpyC45Classifier(
        feature_schema=({"type": "nominal", "values": ("a", "b")},), min_num_obj=1
    )
    one_nominal_value.fit([["a"], ["a"]], [0, 1])
    assert one_nominal_value.tree_ is not None and one_nominal_value.tree_.feature is None


def test_numpy_c45_internal_candidate_filters_are_deterministic() -> None:
    classifier = NumpyC45Classifier(min_num_obj=1, min_gain=2.0)
    classifier.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert classifier.tree_ is not None and classifier.tree_.feature is None

    classifier.min_gain = 0.0
    assert (
        classifier._best_split(np.arange(4), np.ones(4), nominal_available=frozenset()) is not None
    )
    assert (
        classifier._candidate_split(np.arange(4), np.asarray([0.5, 0.0, 0.0, 0.5]), feature_index=0)
        == []
    )
    classifier._y = np.asarray([0, 1, 1, 1])
    classifier.min_num_obj = 2
    assert classifier._candidate_split(np.arange(4), np.ones(4), feature_index=0) == []

    assert classifier._leaf_probabilities(np.zeros(2)).tolist() == [0.5, 0.5]


def test_numpy_c45_declared_nominal_feature_can_be_removed_from_candidates() -> None:
    classifier = NumpyC45Classifier(
        feature_schema=({"type": "nominal", "values": ("a", "b")},), min_num_obj=1
    )
    classifier.fit([["a"], ["a"], ["b"], ["b"]], [0, 0, 1, 1])
    assert classifier._best_split(np.arange(4), np.ones(4), nominal_available=frozenset()) is None
