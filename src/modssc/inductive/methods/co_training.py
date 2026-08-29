from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import numpy as np

from modssc.capabilities import MethodCapabilities
from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.utils import (
    BaseClassifierSpec,
    build_classifier,
    detect_backend,
    ensure_1d_labels,
    ensure_1d_labels_torch,
    ensure_classifier_backend,
    ensure_cpu_device,
    flatten_if_numpy,
    predict_scores,
    select_top_per_class,
    select_top_per_class_torch,
    unwrap_torch_x,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec
from modssc.runtime.contracts import MethodExecutionContract
from modssc.runtime.method_contracts import (
    fallback_method_execution_contract,
    with_inductive_input_roles,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoTrainingSpec(BaseClassifierSpec):
    view_keys: tuple[str, str] | None = None
    max_iter: int = 20
    k_per_class: int = 1
    confidence_threshold: float | None = None
    protocol: str = field(default="legacy", kw_only=True)
    p: int = field(default=1, kw_only=True)
    n: int = field(default=3, kw_only=True)
    u: int = field(default=75, kw_only=True)
    k: int = field(default=30, kw_only=True)
    positive_label: int | None = field(default=None, kw_only=True)
    negative_label: int | None = field(default=None, kw_only=True)
    dynamic_feature_selection: str = field(default="none", kw_only=True)
    feature_selection_max_features: int | None = field(default=None, kw_only=True)
    selection_score: str = field(default="log_probability", kw_only=True)


def _effective_view_keys(
    spec: CoTrainingSpec,
    views: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    raw_keys = spec.view_keys
    if raw_keys is None:
        raw_keys = tuple(sorted(views))[:2] if views is not None else ("view_a", "view_b")
    keys = tuple(str(key) for key in raw_keys)
    if len(keys) != 2:
        raise ValueError("CoTraining view_keys must contain exactly two names")
    return keys[0], keys[1]


_FIXED_POOL_PROTOCOL = "fixed_pool_binary"
_FEATURE_SELECTED_POOL_PROTOCOL = "fixed_pool_binary_feature_selection"
_SHARED_POOL_MULTISET_PROTOCOL = "shared_pool_exhaustive_multiset"
_SUPPORTED_PROTOCOLS = frozenset(
    {
        "legacy",
        _FIXED_POOL_PROTOCOL,
        _FEATURE_SELECTED_POOL_PROTOCOL,
        _SHARED_POOL_MULTISET_PROTOCOL,
    }
)
_OVERLAP_POLICY = "ordered_multiset_view1_then_view2"


def _is_fixed_pool_protocol(protocol: str) -> bool:
    return protocol in {_FIXED_POOL_PROTOCOL, _FEATURE_SELECTED_POOL_PROTOCOL}


def _is_explicit_pool_protocol(protocol: str) -> bool:
    return _is_fixed_pool_protocol(protocol) or protocol == _SHARED_POOL_MULTISET_PROTOCOL


def _validate_protocol(spec: CoTrainingSpec) -> None:
    if spec.protocol not in _SUPPORTED_PROTOCOLS:
        raise InductiveValidationError(
            f"protocol must be one of {sorted(_SUPPORTED_PROTOCOLS)!r}; got {spec.protocol!r}."
        )
    if spec.protocol == "legacy":
        if spec.dynamic_feature_selection != "none":
            raise InductiveValidationError(
                "dynamic_feature_selection is only available for a paper Co-Training protocol."
            )
        return

    for name in ("p", "n", "u", "k"):
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise InductiveValidationError(f"{name} must be an integer in paper protocol.")
    if int(spec.p) < 0 or int(spec.n) < 0:
        raise InductiveValidationError("p and n must be >= 0 in paper protocol.")
    if int(spec.p) + int(spec.n) == 0:
        raise InductiveValidationError("At least one of p or n must be > 0 in paper protocol.")
    if int(spec.u) <= 0:
        raise InductiveValidationError("u must be >= 1 in paper protocol.")
    if int(spec.k) < 0:
        raise InductiveValidationError("k must be >= 0 in paper protocol.")
    if int(spec.p) > int(spec.u) or int(spec.n) > int(spec.u):
        raise InductiveValidationError("p and n must not exceed pool size u.")
    if (spec.positive_label is None) != (spec.negative_label is None):
        raise InductiveValidationError(
            "positive_label and negative_label must either both be set or both be omitted."
        )
    if spec.positive_label is not None and spec.positive_label == spec.negative_label:
        raise InductiveValidationError("positive_label and negative_label must be distinct.")
    if spec.confidence_threshold is not None:
        raise InductiveValidationError(
            "confidence_threshold is not part of a supported paper Co-Training protocol."
        )
    if spec.protocol == _SHARED_POOL_MULTISET_PROTOCOL:
        if (int(spec.p), int(spec.n), int(spec.u), int(spec.k)) != (1, 3, 75, 0):
            raise InductiveValidationError(
                "The Nigam-Ghani protocol freezes p=1, n=3, u=75, k=0; k=0 denotes "
                "unlabeled-set exhaustion rather than a fixed round limit."
            )
        if spec.positive_label != 1 or spec.negative_label != 0:
            raise InductiveValidationError(
                "The Nigam-Ghani WebKB protocol requires positive_label=1 and negative_label=0."
            )
        if (
            spec.dynamic_feature_selection != "none"
            or spec.feature_selection_max_features is not None
            or spec.selection_score != "posterior_probability"
        ):
            raise InductiveValidationError(
                "The Nigam-Ghani protocol requires no feature selection and "
                "selection_score='posterior_probability'."
            )
        if spec.classifier_id != "multinomial_nb" or spec.classifier_backend != "sklearn":
            raise InductiveValidationError(
                "The Nigam-Ghani protocol requires sklearn multinomial_nb."
            )
        classifier_params = dict(spec.classifier_params)
        if set(classifier_params) != {"alpha", "fit_prior"}:
            raise InductiveValidationError(
                "The Nigam-Ghani protocol freezes classifier_params to "
                "{'alpha': 1.0, 'fit_prior': True}; its add-one class prior is applied "
                "explicitly after every fit."
            )
        alpha = classifier_params["alpha"]
        fit_prior = classifier_params["fit_prior"]
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(float(alpha))
            or float(alpha) != 1.0
            or not isinstance(fit_prior, (bool, np.bool_))
            or not bool(fit_prior)
        ):
            raise InductiveValidationError(
                "The Nigam-Ghani protocol freezes classifier_params to "
                "{'alpha': 1.0, 'fit_prior': True}; parameter types are validated "
                "strictly."
            )
        return
    if spec.protocol == _FIXED_POOL_PROTOCOL:
        if (
            spec.dynamic_feature_selection != "none"
            or spec.feature_selection_max_features is not None
            or spec.selection_score != "log_probability"
        ):
            raise InductiveValidationError(
                "The v1 Blum-Mitchell protocol is immutable; use the diagnostic-v2 protocol "
                "for feature selection or length-normalized scores."
            )
        return

    if spec.dynamic_feature_selection != "mutual_information_presence":
        raise InductiveValidationError(
            "The diagnostic-v2 protocol requires "
            "dynamic_feature_selection='mutual_information_presence'."
        )
    if (
        isinstance(spec.feature_selection_max_features, bool)
        or not isinstance(spec.feature_selection_max_features, (int, np.integer))
        or int(spec.feature_selection_max_features) != 2000
    ):
        raise InductiveValidationError(
            "The diagnostic-v2 protocol freezes feature_selection_max_features=2000."
        )
    if spec.selection_score != "craven_1998_normalized_nb":
        raise InductiveValidationError(
            "The diagnostic-v2 protocol requires selection_score='craven_1998_normalized_nb'."
        )
    if spec.classifier_id != "multinomial_nb" or spec.classifier_backend != "sklearn":
        raise InductiveValidationError(
            "The diagnostic-v2 protocol requires sklearn multinomial_nb so the historical "
            "selection score can be audited from fitted word probabilities."
        )


def _mutual_information_presence_scores_numpy(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Return empirical MI(feature presence; class) for every input column.

    Historical text feature selection is interpreted as document occurrence,
    rather than token multiplicity.  The implementation is deterministic and
    deliberately has no estimator jitter or dependency on a random seed.
    """

    features = np.asarray(X)
    labels = np.asarray(y)
    if features.ndim != 2:
        raise InductiveValidationError("Mutual-information feature selection requires 2D views.")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise InductiveValidationError(
            "Mutual-information feature selection requires one label per training row."
        )
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise InductiveValidationError(
            "Mutual-information feature selection requires non-empty training data."
        )
    if not np.all(np.isfinite(features)) or np.any(features < 0.0):
        raise InductiveValidationError(
            "Mutual-information feature selection requires finite non-negative feature counts."
        )

    present = features > 0
    n_samples = float(features.shape[0])
    present_count = present.sum(axis=0, dtype=np.float64)
    scores = np.zeros((features.shape[1],), dtype=np.float64)

    for label in np.unique(labels):
        class_rows = labels == label
        class_count = float(np.count_nonzero(class_rows))
        joint_present = present[class_rows].sum(axis=0, dtype=np.float64)
        joint_absent = class_count - joint_present
        for joint, feature_count in (
            (joint_present, present_count),
            (joint_absent, n_samples - present_count),
        ):
            valid = joint > 0.0
            if np.any(valid):
                scores[valid] += (joint[valid] / n_samples) * np.log(
                    (joint[valid] * n_samples) / (class_count * feature_count[valid])
                )
    return scores


def _select_mutual_information_features_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _mutual_information_presence_scores_numpy(X, y)
    observed = np.flatnonzero(np.any(np.asarray(X) > 0, axis=0))
    if observed.size == 0:
        raise InductiveValidationError(
            "Mutual-information feature selection found no observed feature."
        )
    count = min(int(max_features), int(observed.size))
    ranked = observed[np.argsort(-scores[observed], kind="stable")[:count]]
    # Preserve the original vocabulary column order; ranking order has no
    # semantic meaning for Multinomial NB and needlessly changes fingerprints.
    selected = np.sort(ranked).astype(np.int64, copy=False)
    return selected, scores


def _feature_indices_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _craven_normalized_nb_scores_numpy(
    classifier: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Compute the document-length-normalized NB score from Craven et al.

    For a non-empty document ``d`` and class ``c`` this implements

    ``log P(c) / n + sum_w P(w|d) log(P(w|c) / P(w|d))``.

    Empty documents contain no word evidence and deterministically receive the
    fitted log prior.  The score is used only to rank pool candidates; final
    predictions retain the paper's product of posterior probabilities.
    """

    counts = np.asarray(X, dtype=np.float64)
    if counts.ndim != 2 or not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise InductiveValidationError(
            "Length-normalized Naive Bayes selection requires finite non-negative counts."
        )
    model = getattr(classifier, "_model", None)
    feature_log_prob = getattr(model, "feature_log_prob_", None)
    class_log_prior = getattr(model, "class_log_prior_", None)
    if feature_log_prob is None or class_log_prior is None:
        raise InductiveValidationError(
            "The Craven normalized score requires fitted multinomial Naive Bayes "
            "feature_log_prob_ and class_log_prior_."
        )
    word_log_probability = np.asarray(feature_log_prob, dtype=np.float64)
    log_prior = np.asarray(class_log_prior, dtype=np.float64)
    if (
        word_log_probability.ndim != 2
        or log_prior.ndim != 1
        or word_log_probability.shape[0] != log_prior.size
        or word_log_probability.shape[1] != counts.shape[1]
        or not np.all(np.isfinite(word_log_probability))
        or not np.all(np.isfinite(log_prior))
    ):
        raise InductiveValidationError(
            "Fitted multinomial Naive Bayes parameters do not align with the text view."
        )

    result = np.broadcast_to(log_prior, (counts.shape[0], log_prior.size)).copy()
    document_lengths = counts.sum(axis=1)
    nonempty = document_lengths > 0.0
    if np.any(nonempty):
        word_distribution = counts[nonempty] / document_lengths[nonempty, None]
        log_word_distribution = np.zeros_like(word_distribution)
        np.log(
            word_distribution,
            out=log_word_distribution,
            where=word_distribution > 0.0,
        )
        entropy = -np.sum(word_distribution * log_word_distribution, axis=1)
        mean_class_log_likelihood = word_distribution @ word_log_probability.T
        result[nonempty] = (
            log_prior[None, :] / document_lengths[nonempty, None]
            + mean_class_log_likelihood
            + entropy[:, None]
        )
    return result


def _classifier_classes_numpy(classifier: Any, y_fallback: np.ndarray) -> np.ndarray:
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        classes = np.unique(y_fallback)
    classes_array = np.asarray(classes)
    if classes_array.ndim != 1:
        raise InductiveValidationError("Base classifier classes_ must be one-dimensional.")
    return classes_array


def _resolve_binary_labels(
    spec: CoTrainingSpec,
    classes: np.ndarray,
) -> tuple[Any, Any]:
    if classes.size != 2:
        raise InductiveValidationError(
            "The paper Co-Training protocol requires exactly two labeled classes."
        )
    if spec.positive_label is None:
        negative_label, positive_label = classes[0], classes[1]
    else:
        positive_label = spec.positive_label
        negative_label = spec.negative_label
    if positive_label not in classes or negative_label not in classes:
        raise InductiveValidationError(
            "positive_label and negative_label must both occur in the labeled data."
        )
    return negative_label, positive_label


def _select_binary_quota_numpy(
    scores: np.ndarray,
    classes: np.ndarray,
    *,
    positive_label: Any,
    negative_label: Any,
    p: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores_array = np.asarray(scores)
    if scores_array.ndim != 2 or scores_array.shape[1] != classes.size:
        raise InductiveValidationError(
            "Paper-protocol classifier scores must align with its two classes."
        )
    if not np.all(np.isfinite(scores_array)):
        raise InductiveValidationError("Paper-protocol classifier scores must be finite.")

    class_to_column = {label: index for index, label in enumerate(classes.tolist())}
    positive_column = class_to_column[positive_label]
    negative_column = class_to_column[negative_label]
    available = np.ones((scores_array.shape[0],), dtype=bool)

    selected_indices: list[np.ndarray] = []
    selected_labels: list[np.ndarray] = []
    selected_confidences: list[np.ndarray] = []
    for label, column, quota in (
        (positive_label, positive_column, int(p)),
        (negative_label, negative_column, int(n)),
    ):
        candidates = np.flatnonzero(available)
        if quota == 0 or candidates.size == 0:
            continue
        confidences = scores_array[candidates, column]
        # Stable sorting makes confidence ties follow the deterministic pool order.
        order = np.argsort(-confidences, kind="stable")[:quota]
        chosen = candidates[order]
        available[chosen] = False
        selected_indices.append(chosen.astype(np.int64, copy=False))
        selected_labels.append(np.full(chosen.size, label, dtype=classes.dtype))
        selected_confidences.append(scores_array[chosen, column].astype(np.float64, copy=False))

    if not selected_indices:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=classes.dtype),
            np.empty((0,), dtype=np.float64),
        )
    return (
        np.concatenate(selected_indices),
        np.concatenate(selected_labels),
        np.concatenate(selected_confidences),
    )


def _normalized_probability_product_numpy(
    scores1: np.ndarray,
    scores2: np.ndarray,
    *,
    log_scores1: np.ndarray | None = None,
    log_scores2: np.ndarray | None = None,
) -> np.ndarray:
    first = np.asarray(scores1, dtype=np.float64)
    second = np.asarray(scores2, dtype=np.float64)
    if first.shape != second.shape:
        raise InductiveValidationError("CoTraining classifiers must agree on score shape.")
    if (
        first.ndim != 2
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
        or np.any(first < 0.0)
        or np.any(second < 0.0)
    ):
        raise InductiveValidationError(
            "The paper protocol requires finite, non-negative probability scores."
        )
    first_sum = first.sum(axis=1, keepdims=True)
    second_sum = second.sum(axis=1, keepdims=True)
    if np.any(first_sum <= 0.0) or np.any(second_sum <= 0.0):
        raise InductiveValidationError("Each classifier probability row must have positive mass.")

    if (log_scores1 is None) != (log_scores2 is None):
        raise InductiveValidationError(
            "Both classifier log-probability arrays must be provided together."
        )
    if log_scores1 is None:
        first_probability = first / first_sum
        second_probability = second / second_sum
        first_log = np.full_like(first_probability, -np.inf)
        second_log = np.full_like(second_probability, -np.inf)
        np.log(first_probability, out=first_log, where=first_probability > 0.0)
        np.log(second_probability, out=second_log, where=second_probability > 0.0)
    else:
        first_log = np.asarray(log_scores1, dtype=np.float64)
        second_log = np.asarray(log_scores2, dtype=np.float64)
        if first_log.shape != first.shape or second_log.shape != second.shape:
            raise InductiveValidationError(
                "Classifier log probabilities must agree with probability-score shape."
            )
        if (
            np.any(np.isnan(first_log))
            or np.any(np.isnan(second_log))
            or np.any(np.isposinf(first_log))
            or np.any(np.isposinf(second_log))
        ):
            raise InductiveValidationError(
                "Classifier log probabilities must be finite or negative infinity."
            )

    joint_log = first_log + second_log
    finite_joint = np.isfinite(joint_log)
    if np.any(~np.any(finite_joint, axis=1)):
        raise InductiveValidationError(
            "Multiplicative CoTraining probabilities have zero joint mass."
        )
    row_max = np.max(joint_log, axis=1, keepdims=True)
    shifted = np.zeros_like(joint_log)
    np.exp(joint_log - row_max, out=shifted, where=finite_joint)
    return (shifted / shifted.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)


def _classifier_log_probabilities_numpy(
    classifier: Any,
    X: np.ndarray,
    probability_scores: np.ndarray,
) -> np.ndarray:
    """Recover stable log probabilities when a sklearn classifier exposes them."""

    model = getattr(classifier, "_model", None)
    predict_log_proba = getattr(model, "predict_log_proba", None)
    if callable(predict_log_proba):
        return np.asarray(predict_log_proba(X), dtype=np.float64)

    probabilities = np.asarray(probability_scores, dtype=np.float64)
    row_sum = probabilities.sum(axis=1, keepdims=True)
    normalized = probabilities / row_sum
    result = np.full_like(normalized, -np.inf)
    np.log(normalized, out=result, where=normalized > 0.0)
    return result


def _fit_add_one_multinomial_nb(
    classifier: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Fit add-one multinomial NB, including Nigam--Ghani's class prior.

    Scikit-learn's ``MultinomialNB(alpha=1, fit_prior=True)`` implements the
    paper's add-one word likelihood but uses an empirical class prior.  Equations
    (1)--(3) of Nigam and Ghani (2000) add one pseudo-count to the class prior as
    well, so the fitted prior is replaced after every round.
    """

    classifier.fit(X, y)
    model = getattr(classifier, "_model", None)
    class_count = getattr(model, "class_count_", None)
    class_log_prior = getattr(model, "class_log_prior_", None)
    classes = getattr(classifier, "classes_", None)
    if class_count is None or class_log_prior is None or classes is None:
        raise InductiveValidationError(
            "Nigam-Ghani multinomial NB requires fitted class counts, priors, and classes."
        )
    counts = np.asarray(class_count, dtype=np.float64)
    labels = np.asarray(classes)
    if (
        counts.ndim != 1
        or labels.ndim != 1
        or counts.size != labels.size
        or counts.size != 2
        or not np.all(np.isfinite(counts))
        or np.any(counts <= 0.0)
        or not np.isclose(float(counts.sum()), float(np.asarray(y).size))
    ):
        raise InductiveValidationError(
            "Nigam-Ghani multinomial NB requires two non-empty aligned class counts."
        )
    smoothed_prior = (counts + 1.0) / (float(counts.sum()) + float(counts.size))
    model.class_log_prior_ = np.log(smoothed_prior)


def _fit_explicit_pool_classifier(
    classifier: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    protocol: str,
) -> None:
    if protocol == _SHARED_POOL_MULTISET_PROTOCOL:
        _fit_add_one_multinomial_nb(classifier, X, y)
    else:
        classifier.fit(X, y)


def _classifier_prediction_classes(classifier: Any, *, backend: str) -> np.ndarray | None:
    """Return a classifier's score-column labels as a CPU array when exposed."""
    attribute_order = ("classes_t_", "classes_") if backend == "torch" else ("classes_",)
    classes: Any | None = None
    for attribute in attribute_order:
        classes = getattr(classifier, attribute, None)
        if classes is not None:
            break
    if classes is None:
        return None
    detach = getattr(classes, "detach", None)
    if callable(detach):
        classes = detach().cpu().numpy()
    classes_array = np.asarray(classes)
    if classes_array.ndim != 1:
        raise InductiveValidationError("Base classifier classes must be one-dimensional.")
    return classes_array


def _normalize_view_probabilities(scores: Any, *, backend: str) -> Any:
    """Validate and normalize one classifier's probability-like scores."""
    if backend == "numpy":
        probabilities = np.asarray(scores)
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise InductiveValidationError(
                "CoTraining view probabilities must be finite and non-negative."
            )
        row_sum = probabilities.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0.0):
            raise InductiveValidationError(
                "Each CoTraining view probability row must have positive mass."
            )
        return (probabilities / row_sum).astype(np.float32, copy=False)

    torch = optional_import("torch", extra="inductive-torch")
    if not bool(torch.all(torch.isfinite(scores)).item()) or bool(torch.any(scores < 0).item()):
        raise InductiveValidationError(
            "CoTraining view probabilities must be finite and non-negative."
        )
    row_sum = scores.sum(dim=1, keepdim=True)
    if bool(torch.any(row_sum <= 0).item()):
        raise InductiveValidationError(
            "Each CoTraining view probability row must have positive mass."
        )
    return scores / row_sum


def _trace_selection(
    *,
    pool_indices: np.ndarray,
    local_indices: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "pool_position": int(local_index),
            "unlabeled_index": int(pool_indices[local_index]),
            "label": int(label),
            "confidence": float(confidence),
        }
        for local_index, label, confidence in zip(
            local_indices.tolist(), labels.tolist(), confidences.tolist(), strict=True
        )
    ]


def _ordered_multiset_additions(
    *,
    pool_indices: np.ndarray,
    global1: np.ndarray,
    labels1: np.ndarray,
    global2: np.ndarray,
    labels2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the literal ordered additions described in the paper pseudocode.

    Each learner contributes its own labeled proposals to the single shared ``L``.
    Consequently an example proposed by both learners occurs twice in the training
    multiset, including when the proposed labels disagree.  The corresponding
    source example is nevertheless removed from ``U'`` only once.
    """

    additions = np.concatenate([global1, global2]).astype(np.int64, copy=False)
    label_dtype = np.result_type(labels1.dtype, labels2.dtype)
    addition_labels = np.concatenate(
        [labels1.astype(label_dtype, copy=False), labels2.astype(label_dtype, copy=False)]
    )

    proposed1 = {
        int(index): label for index, label in zip(global1.tolist(), labels1.tolist(), strict=True)
    }
    proposed2 = {
        int(index): label for index, label in zip(global2.tolist(), labels2.tolist(), strict=True)
    }
    overlap_set = set(proposed1).intersection(proposed2)
    conflict_set = {index for index in overlap_set if proposed1[index] != proposed2[index]}
    overlap = np.asarray(
        [int(index) for index in pool_indices.tolist() if int(index) in overlap_set],
        dtype=np.int64,
    )
    conflicts = np.asarray(
        [int(index) for index in pool_indices.tolist() if int(index) in conflict_set],
        dtype=np.int64,
    )
    selected_set = set(int(index) for index in additions.tolist())
    removed = np.asarray(
        [int(index) for index in pool_indices.tolist() if int(index) in selected_set],
        dtype=np.int64,
    )
    return additions, addition_labels, removed, overlap, conflicts


def _view_payload_numpy(value: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, Mapping):
        if "X_l" not in value or "X_u" not in value:
            raise InductiveValidationError(f"views[{name!r}] must contain keys 'X_l' and 'X_u'.")
        X_l = value["X_l"]
        X_u = value["X_u"]
    elif isinstance(value, tuple) and len(value) == 2:
        X_l, X_u = value
    else:
        raise InductiveValidationError(
            f"views[{name!r}] must be a mapping with X_l/X_u or a tuple (X_l, X_u)."
        )

    if not isinstance(X_l, np.ndarray) or not isinstance(X_u, np.ndarray):
        raise InductiveValidationError(
            f"views[{name!r}] X_l/X_u must be numpy arrays. Use preprocess core.to_numpy."
        )
    if X_l.ndim < 2 or X_u.ndim < 2:
        raise InductiveValidationError(f"views[{name!r}] X_l/X_u must be at least 2D arrays.")
    return X_l, X_u


def _view_predict_payload_numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, Mapping):
        if "X" in value:
            X = value["X"]
        elif "X_u" in value:
            X = value["X_u"]
        elif "X_l" in value:
            X = value["X_l"]
        else:
            raise InductiveValidationError(
                f"views[{name!r}] must contain key 'X', 'X_u', or 'X_l' for prediction."
            )
    else:
        X = value
    if not isinstance(X, np.ndarray):
        raise InductiveValidationError(f"views[{name!r}] must be a numpy array for prediction.")
    if X.ndim < 2:
        raise InductiveValidationError(f"views[{name!r}] must be at least 2D for prediction.")
    return X


def _concatenate_prediction_views(data: Any, view_keys: tuple[str, str]) -> np.ndarray:
    """Return the two declared prediction views in one stable feature space."""

    views = getattr(data, "views", None)
    if not isinstance(views, Mapping):
        raise InductiveValidationError(
            "Nigam-Ghani supervised controls require two prediction views."
        )
    matrices: list[np.ndarray] = []
    row_count: int | None = None
    for view_key in view_keys:
        view = views.get(view_key)
        if view is None:
            raise InductiveValidationError(
                f"Missing required view {view_key!r} for supervised controls."
            )
        matrix = flatten_if_numpy(_view_predict_payload_numpy(view, name=view_key))
        if row_count is None:
            row_count = int(matrix.shape[0])
        elif int(matrix.shape[0]) != row_count:
            raise InductiveValidationError(
                "Nigam-Ghani supervised-control views must have the same row count."
            )
        matrices.append(matrix)
    return np.concatenate(matrices, axis=1)


def _is_valid_torch(obj: Any, torch: Any) -> bool:
    return isinstance(obj, torch.Tensor) or (isinstance(obj, dict) and "x" in obj)


_get_torch_tensor = unwrap_torch_x


def _get_torch_len(obj: Any) -> int:
    if isinstance(obj, dict) and "x" in obj:
        return int(obj["x"].shape[0])
    return int(obj.shape[0])


def _get_torch_device(obj: Any) -> Any:
    if isinstance(obj, dict) and "x" in obj:
        return obj["x"].device
    return obj.device


def _same_device(a: Any, b: Any) -> bool:
    return (a == b) or (
        getattr(a, "type", None) == getattr(b, "type", None)
        and (getattr(a, "index", None) is None or getattr(b, "index", None) is None)
    )


def _view_payload_torch(value: Any, *, name: str):
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(value, Mapping):
        if "X_l" not in value or "X_u" not in value:
            raise InductiveValidationError(f"views[{name!r}] must contain keys 'X_l' and 'X_u'.")
        X_l = value["X_l"]
        X_u = value["X_u"]
    elif isinstance(value, tuple) and len(value) == 2:
        X_l, X_u = value
    else:
        raise InductiveValidationError(
            f"views[{name!r}] must be a mapping with X_l/X_u or a tuple (X_l, X_u)."
        )

    if not _is_valid_torch(X_l, torch) or not _is_valid_torch(X_u, torch):
        raise InductiveValidationError(
            f"views[{name!r}] X_l/X_u must be torch tensors. Use preprocess core.to_torch."
        )

    tl = _get_torch_tensor(X_l)
    tu = _get_torch_tensor(X_u)

    if tl.ndim < 2 or tu.ndim < 2:
        raise InductiveValidationError(f"views[{name!r}] X_l/X_u must be at least 2D tensors.")
    if tl.device != tu.device:
        raise InductiveValidationError(f"views[{name!r}] X_l/X_u must share device.")
    return X_l, X_u


def _view_predict_payload_torch(value: Any, *, name: str):
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(value, Mapping):
        if "X" in value:
            X = value["X"]
        elif "X_u" in value:
            X = value["X_u"]
        elif "X_l" in value:
            X = value["X_l"]
        else:
            raise InductiveValidationError(
                f"views[{name!r}] must contain key 'X', 'X_u', or 'X_l' for prediction."
            )
    else:
        X = value
    if not _is_valid_torch(X, torch):
        raise InductiveValidationError(f"views[{name!r}] must be a torch tensor for prediction.")

    Xt = _get_torch_tensor(X)
    if Xt.ndim < 2:
        raise InductiveValidationError(f"views[{name!r}] must be at least 2D for prediction.")
    return X


def _index_torch(obj: Any, idx: Any) -> Any:
    if isinstance(obj, dict) and "x" in obj:
        torch = optional_import("torch", extra="inductive-torch")
        res = obj.copy()
        res["x"] = obj["x"][idx]

        if "edge_index" in obj:
            try:
                from torch_geometric.utils import subgraph
            except ImportError as exc:
                raise InductiveValidationError(
                    "PyG is required to slice edge_index for graph inputs. "
                    "Install torch_geometric or remove graph preprocessing."
                ) from exc
            else:
                edge_index = obj["edge_index"]
                num_nodes = obj["x"].shape[0]

                subset = idx
                if isinstance(subset, slice):
                    subset = torch.arange(num_nodes, device=obj["x"].device)[subset]

                new_ei, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
                res["edge_index"] = new_ei

        return res
    return obj[idx]


def _cat_torch(objs: list[Any], dim: int = 0) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(objs[0], dict) and "x" in objs[0]:
        res = objs[0].copy()
        res["x"] = torch.cat([o["x"] for o in objs], dim=dim)

        if "edge_index" in objs[0]:
            edge_indices = []
            offset = 0
            for o in objs:
                ei = o["edge_index"]
                edge_indices.append(ei + offset)
                offset += o["x"].shape[0]
            res["edge_index"] = torch.cat(edge_indices, dim=1)

        return res
    return torch.cat(objs, dim=dim)


class CoTrainingMethod(InductiveMethod):
    """Co-training with two views (CPU/GPU)."""

    info = MethodInfo(
        method_id="co_training",
        name="Co-Training",
        year=1998,
        family="classic",
        supports_gpu=True,
        paper_title="Combining Labeled and Unlabeled Data with Co-Training",
        paper_pdf="https://www.cs.cmu.edu/~avrim/Papers/co-training.pdf",
        official_code=None,
        requires_views=True,
        prediction_input="dataset",
        capabilities=MethodCapabilities(
            regime="inductive",
            requires_unlabeled=True,
            min_views=2,
            required_classifier_outputs=frozenset({"scores"}),
        ),
    )

    @classmethod
    def execution_contract(
        cls,
        spec: CoTrainingSpec,
        capabilities: MethodCapabilities,
        model_binding: Any | None = None,
    ) -> MethodExecutionContract:
        contract = fallback_method_execution_contract(cls, capabilities, model_binding)
        keys = _effective_view_keys(spec)
        first_l = f"fit.views.{keys[0]}.X_l"
        first_u = f"fit.views.{keys[0]}.X_u"
        second_l = f"fit.views.{keys[1]}.X_l"
        second_u = f"fit.views.{keys[1]}.X_u"
        contract = with_inductive_input_roles(
            contract,
            feature_roles=("fit.X_l", first_l, first_u, second_l, second_u),
            row_groups=(
                (first_l, "fit.y_l"),
                (second_l, "fit.y_l"),
                (first_u, second_u),
            ),
        )
        return replace(
            contract,
            inputs=tuple(
                replace(requirement, consumption="alignment_only")
                if requirement.role == "fit.X_l"
                else requirement
                for requirement in contract.inputs
            ),
        )

    def __init__(self, spec: CoTrainingSpec | None = None) -> None:
        self.spec = spec or CoTrainingSpec()
        self._clf1: Any | None = None
        self._clf2: Any | None = None
        self._view_keys: tuple[str, str] | None = None
        self._backend: str | None = None
        self.round_trace_: list[dict[str, Any]] = []
        self.n_iter_: int = 0
        self.initial_pool_indices_: tuple[int, ...] = ()
        self.diagnostics_: dict[str, Any] = {}
        self._feature_indices1: np.ndarray | None = None
        self._feature_indices2: np.ndarray | None = None

    @property
    def evaluation_reference_splits(self) -> tuple[str, ...]:
        """Reference data needed only by the Nigam--Ghani control outputs."""

        if self.spec.protocol == _SHARED_POOL_MULTISET_PROTOCOL:
            return ("train_labeled", "train")
        return ()

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> CoTrainingMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        if data is None:
            raise InductiveValidationError("data must not be None.")
        if data.views is None:
            raise InductiveValidationError("CoTraining requires data.views with two views.")
        _validate_protocol(self.spec)
        self.round_trace_ = []
        self.n_iter_ = 0
        self.initial_pool_indices_ = ()
        self.diagnostics_ = {}
        self._feature_indices1 = None
        self._feature_indices2 = None

        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)
        if backend == "numpy":
            ensure_cpu_device(device)
            if not isinstance(data.X_l, np.ndarray):
                raise InductiveValidationError(
                    "X_l must be a numpy array. Use preprocess core.to_numpy."
                )
            if not isinstance(data.y_l, np.ndarray):
                raise InductiveValidationError(
                    "y_l must be a numpy array. Use preprocess labels.to_numpy."
                )
            y_l = ensure_1d_labels(data.y_l, name="y_l")
        else:
            torch = optional_import("torch", extra="inductive-torch")
            if not _is_valid_torch(data.X_l, torch):
                raise InductiveValidationError(
                    "X_l must be a torch tensor. Use preprocess core.to_torch."
                )
            if not isinstance(data.y_l, torch.Tensor):
                raise InductiveValidationError(
                    "y_l must be a torch tensor. Use preprocess labels.to_torch."
                )
            y_l = ensure_1d_labels_torch(data.y_l, name="y_l")

        try:
            keys = _effective_view_keys(self.spec, data.views)
        except ValueError as exc:
            raise InductiveValidationError("CoTraining requires exactly two view keys.") from exc

        if backend == "numpy":
            v1_l, v1_u = _view_payload_numpy(data.views[keys[0]], name=keys[0])
            v2_l, v2_u = _view_payload_numpy(data.views[keys[1]], name=keys[1])
            # Ensure flattening for standard classifiers
            v1_l = flatten_if_numpy(v1_l)
            v1_u = flatten_if_numpy(v1_u)
            v2_l = flatten_if_numpy(v2_l)
            v2_u = flatten_if_numpy(v2_u)
        else:
            v1_l, v1_u = _view_payload_torch(data.views[keys[0]], name=keys[0])
            v2_l, v2_u = _view_payload_torch(data.views[keys[1]], name=keys[1])

        if backend == "torch":
            d1 = _get_torch_device(v1_l)
            d2 = _get_torch_device(v2_l)
            if d1 != d2:
                raise InductiveValidationError("views must be on the same device.")
            if not _same_device(y_l.device, d1):
                try:
                    y_l = y_l.to(d1)
                except Exception as exc:
                    raise InductiveValidationError(
                        "y_l must be on the same device as the view tensors."
                    ) from exc
            if not _same_device(y_l.device, d1) or not _same_device(y_l.device, d2):
                raise InductiveValidationError(
                    "y_l must be on the same device as the view tensors."
                )

        l1 = v1_l.shape[0] if backend == "numpy" else _get_torch_len(v1_l)
        l2 = v2_l.shape[0] if backend == "numpy" else _get_torch_len(v2_l)
        u1 = v1_u.shape[0] if backend == "numpy" else _get_torch_len(v1_u)
        u2 = v2_u.shape[0] if backend == "numpy" else _get_torch_len(v2_u)

        if l1 != y_l.shape[0] or l2 != y_l.shape[0]:
            raise InductiveValidationError("View X_l must align with y_l length.")
        if u1 != u2:
            raise InductiveValidationError("View X_u must have the same number of rows.")
        logger.info(
            "Co-training sizes: n_labeled=%s n_unlabeled=%s",
            int(l1),
            int(u1),
        )

        clf1 = build_classifier(self.spec, seed=seed)
        clf2 = build_classifier(self.spec, seed=seed)

        X1_l = v1_l
        X2_l = v2_l
        y1_l = y_l
        y2_l = y_l

        X1_u = v1_u
        X2_u = v2_u

        if _is_explicit_pool_protocol(self.spec.protocol):
            if backend != "numpy":
                raise InductiveValidationError(
                    "The paper Co-Training protocol requires numpy views and a CPU classifier."
                )
            self._fit_explicit_pool_numpy(
                clf1=clf1,
                clf2=clf2,
                X1_l=X1_l,
                X2_l=X2_l,
                y_l=np.asarray(y_l),
                X1_u=X1_u,
                X2_u=X2_u,
                seed=int(seed),
            )
            self._clf1 = clf1
            self._clf2 = clf2
            self._view_keys = keys
            self._backend = backend
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        iter_count = 0
        while iter_count < int(self.spec.max_iter):
            clf1.fit(X1_l, y1_l)
            clf2.fit(X2_l, y2_l)

            u1_len = X1_u.shape[0] if backend == "numpy" else _get_torch_len(X1_u)
            if u1_len == 0:
                break

            scores1 = predict_scores(clf1, X1_u, backend=backend)
            scores2 = predict_scores(clf2, X2_u, backend=backend)

            if backend == "numpy":
                idx1 = select_top_per_class(
                    scores1,
                    k_per_class=int(self.spec.k_per_class),
                    threshold=self.spec.confidence_threshold,
                )
                idx2 = select_top_per_class(
                    scores2,
                    k_per_class=int(self.spec.k_per_class),
                    threshold=self.spec.confidence_threshold,
                )
            else:
                idx1 = select_top_per_class_torch(
                    scores1,
                    k_per_class=int(self.spec.k_per_class),
                    threshold=self.spec.confidence_threshold,
                )
                idx2 = select_top_per_class_torch(
                    scores2,
                    k_per_class=int(self.spec.k_per_class),
                    threshold=self.spec.confidence_threshold,
                )

            sel1 = int(idx1.numel()) if backend == "torch" else int(idx1.size)
            sel2 = int(idx2.numel()) if backend == "torch" else int(idx2.size)
            if sel1 == 0 and sel2 == 0:
                logger.debug("Co-training iter=%s no new labels; stopping.", iter_count)
                break

            if backend == "numpy":
                if idx1.size:
                    y_from_1 = np.asarray(clf1.predict(X1_u[idx1]))
                    X2_l = np.concatenate([X2_l, X2_u[idx1]], axis=0)
                    y2_l = np.concatenate([y2_l, y_from_1], axis=0)
                if idx2.size:
                    y_from_2 = np.asarray(clf2.predict(X2_u[idx2]))
                    X1_l = np.concatenate([X1_l, X1_u[idx2]], axis=0)
                    y1_l = np.concatenate([y1_l, y_from_2], axis=0)

                mask = np.ones((X1_u.shape[0],), dtype=bool)
                if idx1.size:
                    mask[idx1] = False
                if idx2.size:
                    mask[idx2] = False
                X1_u = X1_u[mask]
                X2_u = X2_u[mask]
            else:
                if idx1.numel():
                    y_from_1 = clf1.predict(_index_torch(X1_u, idx1))
                    X2_l = _cat_torch([X2_l, _index_torch(X2_u, idx1)], dim=0)
                    y2_l = torch.cat([y2_l, y_from_1], dim=0)
                if idx2.numel():
                    y_from_2 = clf2.predict(_index_torch(X2_u, idx2))
                    X1_l = _cat_torch([X1_l, _index_torch(X1_u, idx2)], dim=0)
                    y1_l = torch.cat([y1_l, y_from_2], dim=0)

                d_u = _get_torch_device(X1_u)
                l_u = _get_torch_len(X1_u)
                mask = torch.ones((l_u,), dtype=torch.bool, device=d_u)
                if idx1.numel():
                    mask[idx1] = False
                if idx2.numel():
                    mask[idx2] = False
                X1_u = _index_torch(X1_u, mask)
                X2_u = _index_torch(X2_u, mask)

            logger.debug(
                "Co-training iter=%s selected_view1=%s selected_view2=%s remaining=%s",
                iter_count,
                sel1,
                sel2,
                _get_torch_len(X1_u),
            )
            iter_count += 1

        clf1.fit(X1_l, y1_l)
        clf2.fit(X2_l, y2_l)

        self._clf1 = clf1
        self._clf2 = clf2
        self._view_keys = keys
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def _fit_explicit_pool_numpy(
        self,
        *,
        clf1: Any,
        clf2: Any,
        X1_l: np.ndarray,
        X2_l: np.ndarray,
        y_l: np.ndarray,
        X1_u: np.ndarray,
        X2_u: np.ndarray,
        seed: int,
    ) -> None:
        n_unlabeled = int(X1_u.shape[0])
        if n_unlabeled < int(self.spec.u):
            raise InductiveValidationError(
                "The paper protocol requires at least u unlabeled examples for its initial pool."
            )

        initial_classes = np.unique(y_l)
        negative_label, positive_label = _resolve_binary_labels(self.spec, initial_classes)
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(n_unlabeled).astype(np.int64, copy=False)
        pool_indices = shuffled_indices[: int(self.spec.u)].copy()
        reservoir_indices = shuffled_indices[int(self.spec.u) :]
        reservoir_cursor = 0
        self.initial_pool_indices_ = tuple(int(index) for index in pool_indices.tolist())

        X1_train = np.asarray(X1_l)
        X2_train = np.asarray(X2_l)
        y_train = np.asarray(y_l)
        promoted_from_unlabeled: set[int] = set()
        pseudo_labels_added_to_shared_l = 0
        pseudo_label_proposals_view1 = 0
        pseudo_label_proposals_view2 = 0
        same_label_overlap_count = 0
        conflicting_overlap_count = 0
        replenishment_size = 2 * int(self.spec.p) + 2 * int(self.spec.n)
        use_v2 = self.spec.protocol == _FEATURE_SELECTED_POOL_PROTOCOL
        use_nigam = self.spec.protocol == _SHARED_POOL_MULTISET_PROTOCOL

        if use_nigam:
            if n_unlabeled != 776:
                raise InductiveValidationError(
                    "The Nigam-Ghani WebKB protocol requires exactly 776 unlabeled examples."
                )
            if y_train.size != 12:
                raise InductiveValidationError(
                    "The Nigam-Ghani WebKB protocol requires exactly 12 labeled examples."
                )
            initial_negative_count = int(np.count_nonzero(y_train == negative_label))
            initial_positive_count = int(np.count_nonzero(y_train == positive_label))
            if (initial_negative_count, initial_positive_count) != (9, 3):
                raise InductiveValidationError(
                    "The Nigam-Ghani WebKB protocol requires exactly 9 negative and 3 positive "
                    "initial labels."
                )

        round_limit = n_unlabeled if use_nigam else int(self.spec.k)
        for round_index in range(round_limit):
            if use_nigam and pool_indices.size == 0:
                break
            feature_trace: dict[str, Any] = {}
            if use_v2:
                feature_indices1, feature_scores1 = _select_mutual_information_features_numpy(
                    X1_train,
                    y_train,
                    max_features=int(self.spec.feature_selection_max_features or 0),
                )
                feature_indices2, feature_scores2 = _select_mutual_information_features_numpy(
                    X2_train,
                    y_train,
                    max_features=int(self.spec.feature_selection_max_features or 0),
                )
                fit_X1 = X1_train[:, feature_indices1]
                fit_X2 = X2_train[:, feature_indices2]
                feature_trace = {
                    "feature_selection": "mutual_information_presence",
                    "selected_feature_count_view1": int(feature_indices1.size),
                    "selected_feature_count_view2": int(feature_indices2.size),
                    "selected_features_sha256_view1": _feature_indices_sha256(feature_indices1),
                    "selected_features_sha256_view2": _feature_indices_sha256(feature_indices2),
                    "maximum_mutual_information_view1": float(
                        np.max(feature_scores1[feature_indices1])
                    ),
                    "maximum_mutual_information_view2": float(
                        np.max(feature_scores2[feature_indices2])
                    ),
                }
            else:
                feature_indices1 = None
                feature_indices2 = None
                fit_X1 = X1_train
                fit_X2 = X2_train
            _fit_explicit_pool_classifier(
                clf1,
                fit_X1,
                y_train,
                protocol=self.spec.protocol,
            )
            _fit_explicit_pool_classifier(
                clf2,
                fit_X2,
                y_train,
                protocol=self.spec.protocol,
            )
            if pool_indices.size == 0:
                break

            pool_X1 = X1_u[pool_indices]
            pool_X2 = X2_u[pool_indices]
            if feature_indices1 is not None and feature_indices2 is not None:
                pool_X1 = pool_X1[:, feature_indices1]
                pool_X2 = pool_X2[:, feature_indices2]
            scores1 = predict_scores(clf1, pool_X1, backend="numpy")
            scores2 = predict_scores(clf2, pool_X2, backend="numpy")
            if use_v2:
                selection_scores1 = _craven_normalized_nb_scores_numpy(
                    clf1,
                    pool_X1,
                )
                selection_scores2 = _craven_normalized_nb_scores_numpy(
                    clf2,
                    pool_X2,
                )
            else:
                # Ranking in log space is strictly order-equivalent to ranking by
                # the normalized posterior from Nigam--Ghani Equation (3), while
                # avoiding float32 underflow for long web documents.
                selection_scores1 = _classifier_log_probabilities_numpy(
                    clf1,
                    pool_X1,
                    scores1,
                )
                selection_scores2 = _classifier_log_probabilities_numpy(
                    clf2,
                    pool_X2,
                    scores2,
                )
            classes1 = _classifier_classes_numpy(clf1, y_train)
            classes2 = _classifier_classes_numpy(clf2, y_train)
            if not np.array_equal(classes1, classes2):
                raise InductiveValidationError("CoTraining classifiers disagree on class labels.")
            resolved_negative, resolved_positive = _resolve_binary_labels(self.spec, classes1)
            if resolved_negative != negative_label or resolved_positive != positive_label:
                raise InductiveValidationError(
                    "CoTraining classifier classes changed during the paper protocol."
                )

            idx1, labels1, confidences1 = _select_binary_quota_numpy(
                selection_scores1,
                classes1,
                positive_label=positive_label,
                negative_label=negative_label,
                p=int(self.spec.p),
                n=int(self.spec.n),
            )
            idx2, labels2, confidences2 = _select_binary_quota_numpy(
                selection_scores2,
                classes2,
                positive_label=positive_label,
                negative_label=negative_label,
                p=int(self.spec.p),
                n=int(self.spec.n),
            )
            global1 = pool_indices[idx1]
            global2 = pool_indices[idx2]
            additions, addition_labels, removed_indices, overlap, conflicts = (
                _ordered_multiset_additions(
                    pool_indices=pool_indices,
                    global1=global1,
                    labels1=labels1,
                    global2=global2,
                    labels2=labels2,
                )
            )
            if use_nigam:
                # Ranking is performed stably in log space, but the paper trace
                # exposes the posterior probabilities from Equation (3).
                confidences1 = np.exp(confidences1)
                confidences2 = np.exp(confidences2)
            overlap_policy = _OVERLAP_POLICY

            duplicate_promotions = promoted_from_unlabeled.intersection(removed_indices.tolist())
            if duplicate_promotions:  # pragma: no cover
                # The pool/reservoir partition and union removal make this an internal invariant.
                raise RuntimeError("Paper-protocol U' attempted to promote an example twice.")

            size_before = int(y_train.shape[0])
            selected_trace1 = _trace_selection(
                pool_indices=pool_indices,
                local_indices=idx1,
                labels=labels1,
                confidences=confidences1,
            )
            selected_trace2 = _trace_selection(
                pool_indices=pool_indices,
                local_indices=idx2,
                labels=labels2,
                confidences=confidences2,
            )
            X1_train = np.concatenate([X1_train, X1_u[additions]], axis=0)
            X2_train = np.concatenate([X2_train, X2_u[additions]], axis=0)
            y_train = np.concatenate([y_train, addition_labels], axis=0)
            promoted_from_unlabeled.update(int(index) for index in removed_indices.tolist())
            pseudo_labels_added_to_shared_l += int(additions.size)
            pseudo_label_proposals_view1 += int(global1.size)
            pseudo_label_proposals_view2 += int(global2.size)
            conflicting_overlap_count += int(conflicts.size)
            same_label_overlap_count += int(overlap.size - conflicts.size)

            keep = ~np.isin(pool_indices, removed_indices, assume_unique=True)
            remaining_pool = pool_indices[keep]
            available = int(reservoir_indices.size) - reservoir_cursor
            replenish_count = min(replenishment_size, max(available, 0))
            replenished = reservoir_indices[
                reservoir_cursor : reservoir_cursor + replenish_count
            ].copy()
            reservoir_cursor += replenish_count
            pool_after = np.concatenate([remaining_pool, replenished])
            if np.unique(pool_after).size != pool_after.size:  # pragma: no cover
                # Both slices originate from one permutation and are therefore disjoint.
                raise RuntimeError("Paper-protocol pool contains duplicate unlabeled indices.")

            nigam_round_diagnostics: dict[str, int] = {}
            if use_nigam:
                nigam_round_diagnostics = {
                    "proposal_count_view1": int(global1.size),
                    "proposal_count_view2": int(global2.size),
                    "multiset_addition_count": int(additions.size),
                    "unique_removed_count": int(removed_indices.size),
                    "duplicate_multiset_addition_count": int(additions.size - removed_indices.size),
                    "same_label_overlap_count": int(overlap.size - conflicts.size),
                    "conflicting_overlap_count": int(conflicts.size),
                }
            self.round_trace_.append(
                {
                    "round": round_index + 1,
                    "round_status": "completed",
                    "overlap_policy": overlap_policy,
                    "pool_indices_before": [int(index) for index in pool_indices.tolist()],
                    "pool_size_before": int(pool_indices.size),
                    "selected_by_view1": selected_trace1,
                    "selected_by_view2": selected_trace2,
                    "overlap_indices": [int(index) for index in overlap.tolist()],
                    "conflicting_overlap_indices": [int(index) for index in conflicts.tolist()],
                    **nigam_round_diagnostics,
                    "multiset_additions": [
                        {
                            "proposal_order": int(position),
                            "source_view": ("view1" if position < int(global1.size) else "view2"),
                            "unlabeled_index": int(index),
                            "label": int(label),
                        }
                        for position, (index, label) in enumerate(
                            zip(additions.tolist(), addition_labels.tolist(), strict=True)
                        )
                    ],
                    "removed_indices": [int(index) for index in removed_indices.tolist()],
                    "requested_replenishment_count": replenishment_size,
                    "replenished_indices": [int(index) for index in replenished.tolist()],
                    "pool_indices_after": [int(index) for index in pool_after.tolist()],
                    "pool_size_after": int(pool_after.size),
                    "pool_growth": int(pool_after.size) - int(pool_indices.size),
                    "reservoir_remaining": int(reservoir_indices.size) - reservoir_cursor,
                    "training_size_view1_before": size_before,
                    "training_size_view1_after": int(y_train.shape[0]),
                    "training_size_view2_before": size_before,
                    "training_size_view2_after": int(y_train.shape[0]),
                    **feature_trace,
                }
            )
            pool_indices = pool_after

        if use_v2:
            self._feature_indices1, final_feature_scores1 = (
                _select_mutual_information_features_numpy(
                    X1_train,
                    y_train,
                    max_features=int(self.spec.feature_selection_max_features or 0),
                )
            )
            self._feature_indices2, final_feature_scores2 = (
                _select_mutual_information_features_numpy(
                    X2_train,
                    y_train,
                    max_features=int(self.spec.feature_selection_max_features or 0),
                )
            )
            _fit_explicit_pool_classifier(
                clf1,
                X1_train[:, self._feature_indices1],
                y_train,
                protocol=self.spec.protocol,
            )
            _fit_explicit_pool_classifier(
                clf2,
                X2_train[:, self._feature_indices2],
                y_train,
                protocol=self.spec.protocol,
            )
        else:
            final_feature_scores1 = None
            final_feature_scores2 = None
            _fit_explicit_pool_classifier(
                clf1,
                X1_train,
                y_train,
                protocol=self.spec.protocol,
            )
            _fit_explicit_pool_classifier(
                clf2,
                X2_train,
                y_train,
                protocol=self.spec.protocol,
            )
        self.n_iter_ = len(self.round_trace_)
        remaining_indices = np.concatenate([pool_indices, reservoir_indices[reservoir_cursor:]])
        if use_nigam and (  # pragma: no cover - enforced by exact quotas and source partition
            promoted_from_unlabeled != set(range(n_unlabeled))
            or remaining_indices.size != 0
            or int(y_train.shape[0]) != 12 + pseudo_labels_added_to_shared_l
        ):
            raise RuntimeError(
                "Nigam-Ghani exhaustion invariants failed: every one of the 776 unlabeled "
                "indices must be promoted exactly once from U, no source data may remain, "
                "and the shared labeled multiset must retain every ordered proposal."
            )
        overlap_count = same_label_overlap_count + conflicting_overlap_count
        duplicate_multiset_additions = pseudo_labels_added_to_shared_l - len(
            promoted_from_unlabeled
        )
        overlap_policy = _OVERLAP_POLICY
        self.diagnostics_ = {
            "protocol": self.spec.protocol,
            "seed": int(seed),
            "p": int(self.spec.p),
            "n": int(self.spec.n),
            "u": int(self.spec.u),
            "k": int(self.spec.k),
            "negative_label": int(negative_label),
            "positive_label": int(positive_label),
            "initial_pool_indices": list(self.initial_pool_indices_),
            "n_iter": int(self.n_iter_),
            "shared_labeled_multiset": True,
            "overlap_policy": overlap_policy,
            "selection_score_space": self.spec.selection_score,
            "combination_score_space": "summed_log_probability",
            "probability_underflow_safe": True,
            "same_label_overlap_count": int(same_label_overlap_count),
            "conflicting_overlap_count": int(conflicting_overlap_count),
            "unique_pseudo_labeled_examples": len(promoted_from_unlabeled),
            "pseudo_labels_added_to_shared_l": int(pseudo_labels_added_to_shared_l),
            "pseudo_labels_received_by_view1": int(pseudo_labels_added_to_shared_l),
            "pseudo_labels_received_by_view2": int(pseudo_labels_added_to_shared_l),
            "final_labeled_size": int(y_train.shape[0]),
            "remaining_unlabeled_count": int(remaining_indices.size),
            "remaining_unlabeled_indices": [int(index) for index in remaining_indices.tolist()],
            "round_trace": list(self.round_trace_),
        }
        if use_nigam:
            self.diagnostics_.update(
                {
                    "initial_labeled_size": 12,
                    "initial_unlabeled_count": 776,
                    "initial_class_counts": {
                        str(int(negative_label)): int(initial_negative_count),
                        str(int(positive_label)): int(initial_positive_count),
                    },
                    "termination": "unlabeled_exhausted",
                    "addition_policy": _OVERLAP_POLICY,
                    "views_select_from_same_pre_round_pool": True,
                    "overlap_count": int(overlap_count),
                    "duplicate_multiset_additions": int(duplicate_multiset_additions),
                    "pseudo_label_proposals_view1": int(pseudo_label_proposals_view1),
                    "pseudo_label_proposals_view2": int(pseudo_label_proposals_view2),
                    "paper_confidence": "posterior_probability",
                    "ranking_space": "log_posterior_probability",
                    "word_likelihood_smoothing": "add_one",
                    "class_prior_smoothing": "add_one",
                    "dynamic_feature_selection": "none",
                    "selection_diagnostics_scope": "training_and_pseudo_labels_only",
                    "test_metrics_used_for_protocol_selection": False,
                }
            )
        if use_v2:
            if self._feature_indices1 is None or self._feature_indices2 is None:  # pragma: no cover
                raise RuntimeError("Diagnostic-v2 feature-selection state is missing.")
            self.diagnostics_.update(
                {
                    "dynamic_feature_selection": "mutual_information_presence",
                    "feature_selection_max_features": int(
                        self.spec.feature_selection_max_features or 0
                    ),
                    "final_feature_count_view1": int(self._feature_indices1.size),
                    "final_feature_count_view2": int(self._feature_indices2.size),
                    "final_features_sha256_view1": _feature_indices_sha256(self._feature_indices1),
                    "final_features_sha256_view2": _feature_indices_sha256(self._feature_indices2),
                    "final_maximum_mutual_information_view1": float(
                        np.max(final_feature_scores1[self._feature_indices1])
                    ),
                    "final_maximum_mutual_information_view2": float(
                        np.max(final_feature_scores2[self._feature_indices2])
                    ),
                    "selection_diagnostics_scope": "training_and_pseudo_labels_only",
                    "test_metrics_used_for_protocol_selection": False,
                }
            )

    def _predict_scores_pair(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self._clf1 is None or self._clf2 is None:
            raise RuntimeError("CoTrainingMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X1)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict input backend mismatch.")
        prediction_X1 = X1
        prediction_X2 = X2
        if self.spec.protocol == _FEATURE_SELECTED_POOL_PROTOCOL:
            if self._feature_indices1 is None or self._feature_indices2 is None:
                raise RuntimeError("Diagnostic-v2 feature-selection state is missing.")
            prediction_X1 = X1[:, self._feature_indices1]
            prediction_X2 = X2[:, self._feature_indices2]
        s1 = predict_scores(self._clf1, prediction_X1, backend=backend)
        s2 = predict_scores(self._clf2, prediction_X2, backend=backend)
        if s1.shape[1] != s2.shape[1]:
            raise InductiveValidationError("CoTraining classifiers must agree on class count.")
        c1 = getattr(self._clf1, "classes_", None)
        c2 = getattr(self._clf2, "classes_", None)
        if c1 is not None and c2 is not None and not np.array_equal(c1, c2):
            raise InductiveValidationError("CoTraining classifiers disagree on class labels.")
        if _is_explicit_pool_protocol(self.spec.protocol):
            if backend == "numpy":
                log_s1 = _classifier_log_probabilities_numpy(self._clf1, prediction_X1, s1)
                log_s2 = _classifier_log_probabilities_numpy(self._clf2, prediction_X2, s2)
                return _normalized_probability_product_numpy(
                    s1,
                    s2,
                    log_scores1=log_s1,
                    log_scores2=log_s2,
                )
            torch = optional_import("torch", extra="inductive-torch")
            if (
                not bool(torch.all(torch.isfinite(s1)).item())
                or not bool(torch.all(torch.isfinite(s2)).item())
                or bool(torch.any(s1 < 0).item())
                or bool(torch.any(s2 < 0).item())
            ):
                raise InductiveValidationError(
                    "The paper protocol requires finite, non-negative probability scores."
                )
            first_sum = s1.sum(dim=1, keepdim=True)
            second_sum = s2.sum(dim=1, keepdim=True)
            if bool(torch.any(first_sum <= 0).item()) or bool(torch.any(second_sum <= 0).item()):
                raise InductiveValidationError(
                    "Each classifier probability row must have positive mass."
                )
            product = (s1 / first_sum) * (s2 / second_sum)
            product_sum = product.sum(dim=1, keepdim=True)
            if bool(torch.any(product_sum <= 0).item()):
                raise InductiveValidationError(
                    "Multiplicative CoTraining probabilities have zero joint mass."
                )
            return product / product_sum
        return (s1 + s2) / 2.0

    def predict_proba(self, data: Any) -> np.ndarray:
        if data is None or data.views is None:
            raise InductiveValidationError("CoTraining requires data.views at prediction time.")
        if self._view_keys is None:
            raise RuntimeError("CoTrainingMethod missing view keys; fit() was not called.")
        v1 = data.views.get(self._view_keys[0])
        v2 = data.views.get(self._view_keys[1])
        if v1 is None or v2 is None:
            raise InductiveValidationError("Missing required views for prediction.")
        backend = self._backend or detect_backend(data.X_l)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")
        if backend == "numpy":
            X1 = _view_predict_payload_numpy(v1, name=self._view_keys[0])
            X2 = _view_predict_payload_numpy(v2, name=self._view_keys[1])
            X1 = flatten_if_numpy(X1)
            X2 = flatten_if_numpy(X2)
        else:
            X1 = _view_predict_payload_torch(v1, name=self._view_keys[0])
            X2 = _view_predict_payload_torch(v2, name=self._view_keys[1])
        scores = self._predict_scores_pair(X1, X2)
        if backend == "numpy":
            row_sum = scores.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)
        torch = optional_import("torch", extra="inductive-torch")
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def predict_view_proba(self, data: Any, view_key: str) -> Any:
        """Predict with one fitted view while preserving its score-column class order."""
        if self._clf1 is None or self._clf2 is None or self._view_keys is None:
            raise RuntimeError("CoTrainingMethod is not fitted yet. Call fit() first.")
        if view_key not in self._view_keys:
            raise InductiveValidationError(
                f"view_key must be one of {self._view_keys!r}; got {view_key!r}."
            )
        if data is None or data.views is None:
            raise InductiveValidationError("CoTraining requires data.views at prediction time.")
        view = data.views.get(view_key)
        if view is None:
            raise InductiveValidationError(f"Missing required view {view_key!r} for prediction.")

        backend = self._backend
        if backend is None:  # pragma: no cover - guarded by the fitted-state invariant above
            raise RuntimeError("CoTrainingMethod is missing its fitted backend.")
        if detect_backend(data.X_l) != backend:
            raise InductiveValidationError("predict_view_proba input backend mismatch.")

        view_index = self._view_keys.index(view_key)
        classifier = (self._clf1, self._clf2)[view_index]
        other_classifier = (self._clf2, self._clf1)[view_index]
        if backend == "numpy":
            X = flatten_if_numpy(_view_predict_payload_numpy(view, name=view_key))
            if self.spec.protocol == _FEATURE_SELECTED_POOL_PROTOCOL:
                feature_indices = (self._feature_indices1, self._feature_indices2)[view_index]
                if feature_indices is None:
                    raise RuntimeError("Diagnostic-v2 feature-selection state is missing.")
                X = X[:, feature_indices]
        else:
            X = _view_predict_payload_torch(view, name=view_key)
        scores = predict_scores(classifier, X, backend=backend)

        classes = _classifier_prediction_classes(classifier, backend=backend)
        other_classes = _classifier_prediction_classes(other_classifier, backend=backend)
        if classes is not None and int(classes.shape[0]) != int(scores.shape[1]):
            raise InductiveValidationError(
                "CoTraining classifier classes do not align with its probability columns."
            )
        if (
            classes is not None
            and other_classes is not None
            and not np.array_equal(classes, other_classes)
        ):
            raise InductiveValidationError("CoTraining classifiers disagree on class labels.")
        return _normalize_view_probabilities(scores, backend=backend)

    def predict_named_proba(self, data: Any) -> dict[str, Any]:
        """Return per-view predictions and any protocol-owned supervised controls."""
        if self._view_keys is None:
            raise RuntimeError("CoTrainingMethod missing view keys; fit() was not called.")
        named = {view_key: self.predict_view_proba(data, view_key) for view_key in self._view_keys}
        if self.spec.protocol != _SHARED_POOL_MULTISET_PROTOCOL:
            return named

        meta = data.meta if isinstance(getattr(data, "meta", None), Mapping) else {}
        references = meta.get("evaluation_reference_splits")
        if not isinstance(references, Mapping):
            raise InductiveValidationError(
                "Nigam-Ghani supervised controls require evaluation reference splits."
            )
        test_matrix = _concatenate_prediction_views(data, self._view_keys)
        control_sizes: dict[str, int] = {}
        for reference_split, output_name in (
            ("train_labeled", "nb12"),
            ("train", "nb788"),
        ):
            reference = references.get(reference_split)
            if reference is None:
                raise InductiveValidationError(
                    f"Nigam-Ghani supervised controls require {reference_split!r}."
                )
            train_matrix = _concatenate_prediction_views(reference, self._view_keys)
            train_labels = ensure_1d_labels(reference.y_l, name=f"{reference_split}.y_l")
            classifier = build_classifier(
                BaseClassifierSpec(
                    classifier_id="multinomial_nb",
                    classifier_backend="sklearn",
                    classifier_params={"alpha": 1.0, "fit_prior": True},
                ),
                seed=0,
            )
            _fit_add_one_multinomial_nb(classifier, train_matrix, train_labels)
            named[output_name] = predict_scores(classifier, test_matrix, backend="numpy")
            control_sizes[output_name] = int(train_labels.size)

        self.diagnostics_["supervised_controls"] = {
            "nb12_training_size": control_sizes["nb12"],
            "nb788_training_size": control_sizes["nb788"],
            "feature_space": "concatenated_namespaced_views",
            "class_prior_smoothing": "add_one",
            "test_metrics_used_for_protocol_selection": False,
        }
        return named

    def predict_evaluation_outputs(self, data: Any) -> dict[str, Any]:
        """Expose protocol diagnostics through the generic evaluation API."""

        meta = data.meta if isinstance(getattr(data, "meta", None), Mapping) else {}
        split = meta.get("evaluation_split")
        if self.spec.protocol == _SHARED_POOL_MULTISET_PROTOCOL and split not in {
            None,
            "test",
        }:
            if self._view_keys is None:
                raise RuntimeError("CoTrainingMethod missing view keys; fit() was not called.")
            return {
                view_key: self.predict_view_proba(data, view_key) for view_key in self._view_keys
            }
        return self.predict_named_proba(data)

    def predict(self, data: Any) -> np.ndarray:
        if self._clf1 is None:
            raise RuntimeError("CoTrainingMethod is not fitted yet. Call fit() first.")
        scores = self.predict_proba(data)
        backend = self._backend or detect_backend(data.X_l)
        if backend == "numpy":
            idx = scores.argmax(axis=1)
            classes = getattr(self._clf1, "classes_", None)
            if classes is None:
                return idx
            return np.asarray(classes)[idx]
        idx = scores.argmax(dim=1)
        classes_t = getattr(self._clf1, "classes_t_", None)
        if classes_t is None:
            return idx
        return classes_t[idx]
