from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist
from time import perf_counter
from typing import Any

import numpy as np

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    concat_data,
    get_torch_device,
    get_torch_len,
    slice_data,
)
from modssc.inductive.methods.utils import (
    BaseClassifierSpec,
    build_classifier,
    detect_backend,
    ensure_1d_labels,
    ensure_1d_labels_torch,
    ensure_classifier_backend,
    ensure_cpu_device,
    ensure_numpy_data,
    ensure_torch_data,
    predict_in_batches,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)

_TRAINING_MODES = frozenset({"legacy", "confidence_weighted"})


def _z_value(confidence_level: float) -> float:
    if not (0.0 < float(confidence_level) < 1.0):
        raise InductiveValidationError("confidence_level must be in (0, 1).")
    return NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0)


def _accuracy_confidence_interval(
    correct: int,
    total: int,
    *,
    confidence_level: float,
    interval: str = "wald",
) -> tuple[float, float]:
    if total <= 0:
        raise InductiveValidationError("Cannot compute confidence interval with total=0.")
    if not 0 <= int(correct) <= int(total):
        raise InductiveValidationError("correct must be in [0, total].")
    p_hat = float(correct) / float(total)
    z = _z_value(confidence_level)
    if interval == "wilson":
        z_squared = z * z
        denominator = 1.0 + z_squared / float(total)
        center = (p_hat + z_squared / (2.0 * float(total))) / denominator
        radius = (
            z
            * math.sqrt(
                p_hat * (1.0 - p_hat) / float(total) + z_squared / (4.0 * float(total) ** 2)
            )
            / denominator
        )
        return max(0.0, center - radius), min(1.0, center + radius)
    if interval == "clopper_pearson":
        alpha = 1.0 - float(confidence_level)
        lo = (
            0.0
            if int(correct) == 0
            else _bisect_binomial_tail(
                correct=int(correct),
                total=int(total),
                target=alpha / 2.0,
                upper_tail=True,
            )
        )
        hi = (
            1.0
            if int(correct) == int(total)
            else _bisect_binomial_tail(
                correct=int(correct),
                total=int(total),
                target=alpha / 2.0,
                upper_tail=False,
            )
        )
        return lo, hi
    if interval != "wald":
        raise InductiveValidationError(
            "confidence_interval must be one of {'wald', 'wilson', 'clopper_pearson'}."
        )
    se = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / float(total))
    lo = max(0.0, p_hat - z * se)
    hi = min(1.0, p_hat + z * se)
    return lo, hi


def _binomial_tail_probability(
    *, correct: int, total: int, probability: float, upper_tail: bool
) -> float:
    """Evaluate a binomial tail without SciPy.

    The direct log-PMF sum is intentionally used here: the DCL paper protocol
    has small labeled sets, and this keeps the exact interval available without
    introducing a dependency that is absent from the core installation.
    """

    p = float(probability)
    if p <= 0.0:
        return float((correct <= 0) if upper_tail else (correct >= 0))
    if p >= 1.0:
        return float((correct <= total) if upper_tail else (correct >= total))
    start, stop = (correct, total + 1) if upper_tail else (0, correct + 1)
    log_p = math.log(p)
    log_one_minus_p = math.log1p(-p)
    terms = []
    for successes in range(start, stop):
        log_probability = (
            math.lgamma(total + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(total - successes + 1)
            + successes * log_p
            + (total - successes) * log_one_minus_p
        )
        terms.append(math.exp(log_probability))
    return min(1.0, max(0.0, math.fsum(terms)))


def _bisect_binomial_tail(*, correct: int, total: int, target: float, upper_tail: bool) -> float:
    lo = 0.0
    hi = 1.0
    for _ in range(64):
        midpoint = (lo + hi) / 2.0
        value = _binomial_tail_probability(
            correct=correct,
            total=total,
            probability=midpoint,
            upper_tail=upper_tail,
        )
        if upper_tail:
            if value < target:
                lo = midpoint
            else:
                hi = midpoint
        elif value > target:
            lo = midpoint
        else:
            hi = midpoint
    return (lo + hi) / 2.0


def _confidence_interval_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    confidence_level: float,
    interval: str = "wald",
) -> tuple[float, float]:
    correct = int(np.sum(y_true == y_pred))
    return _accuracy_confidence_interval(
        correct,
        int(y_true.shape[0]),
        confidence_level=confidence_level,
        interval=interval,
    )


def _confidence_interval_torch(
    y_true: Any,
    y_pred: Any,
    *,
    confidence_level: float,
    interval: str = "wald",
):
    correct = int((y_true == y_pred).sum().item())
    return _accuracy_confidence_interval(
        correct,
        int(y_true.numel()),
        confidence_level=confidence_level,
        interval=interval,
    )


def _stratified_kfold_indices(
    y: np.ndarray, *, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(y).reshape(-1)
    if int(n_splits) < 2:
        raise InductiveValidationError("confidence_folds must be >= 2.")
    if labels.size < int(n_splits):
        raise InductiveValidationError(
            "confidence_folds cannot exceed the number of labeled examples."
        )
    _classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    if counts.size == 0 or min(counts.tolist()) < 2:
        raise InductiveValidationError(
            "Each class must contain at least 2 labeled examples for kfold_oof."
        )

    rng = np.random.default_rng(int(seed))
    validation_parts: list[list[np.ndarray]] = [[] for _ in range(int(n_splits))]
    fold_offset = 0
    for class_index in range(int(counts.size)):
        class_indices = np.flatnonzero(inverse == class_index)
        class_indices = class_indices[rng.permutation(class_indices.size)]
        for position, sample_index in enumerate(class_indices):
            fold_index = (fold_offset + position) % int(n_splits)
            validation_parts[fold_index].append(np.asarray([sample_index], dtype=np.int64))
        fold_offset = (fold_offset + int(class_indices.size)) % int(n_splits)

    all_indices = np.arange(labels.size, dtype=np.int64)
    folds = []
    for parts in validation_parts:
        validation = np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
        in_validation = np.zeros((labels.size,), dtype=bool)
        in_validation[validation] = True
        folds.append((all_indices[~in_validation], validation))
    return folds


def _resolve_classifier_specs(spec: DemocraticCoLearningSpec) -> list[BaseClassifierSpec]:
    if spec.classifier_specs is not None:
        specs: list[BaseClassifierSpec] = []
        for index, item in enumerate(spec.classifier_specs):
            if isinstance(item, BaseClassifierSpec):
                specs.append(item)
                continue
            if isinstance(item, Mapping):
                try:
                    specs.append(BaseClassifierSpec(**dict(item)))
                except TypeError as exc:
                    raise InductiveValidationError(
                        f"classifier_specs[{index}] is invalid: {exc}"
                    ) from exc
                continue
            raise InductiveValidationError(
                "classifier_specs entries must be BaseClassifierSpec instances or mappings."
            )
        if len(specs) < 3:
            raise InductiveValidationError("DemocraticCoLearning requires at least 3 learners.")
        return specs
    if int(spec.n_learners) < 3:
        raise InductiveValidationError("n_learners must be >= 3.")
    return [
        BaseClassifierSpec(
            classifier_id=spec.classifier_id,
            classifier_backend=spec.classifier_backend,
            classifier_params=spec.classifier_params,
        )
        for _ in range(int(spec.n_learners))
    ]


def _resolve_classes_numpy(clfs: list[Any], y_l: np.ndarray) -> np.ndarray:
    classes = None
    for clf in clfs:
        c = getattr(clf, "classes_", None)
        if c is None:
            continue
        c = np.asarray(c)
        if classes is None:
            classes = c
        elif not np.array_equal(classes, c):
            raise InductiveValidationError(
                "DemocraticCoLearning classifiers disagree on class labels."
            )
    if classes is None:
        classes = np.unique(y_l)
    return np.asarray(classes)


def _resolve_classes_torch(clfs: list[Any], y_l: Any):
    torch = optional_import("torch", extra="inductive-torch")
    classes_t = torch.unique(y_l, sorted=True)
    classes_np = None
    for clf in clfs:
        c_t = getattr(clf, "classes_t_", None)
        if c_t is not None and not torch.equal(c_t.to(classes_t.device), classes_t):
            raise InductiveValidationError(
                "DemocraticCoLearning classifiers disagree on class labels."
            )
        c_np = getattr(clf, "classes_", None)
        if c_np is not None:
            c_np = np.asarray(c_np)
            if classes_np is None:
                classes_np = c_np
            elif not np.array_equal(classes_np, c_np):
                raise InductiveValidationError(
                    "DemocraticCoLearning classifiers disagree on class labels."
                )
    if classes_np is not None and not np.array_equal(classes_np, classes_t.detach().cpu().numpy()):
        raise InductiveValidationError("DemocraticCoLearning classifiers disagree on class labels.")
    return classes_t


def _encode_predictions_numpy(preds: list[np.ndarray], classes: np.ndarray) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(classes.tolist())}
    idx_all = []
    for pred in preds:
        pred = np.asarray(pred).reshape(-1)
        idx = np.vectorize(mapping.get, otypes=[int])(pred)
        idx_all.append(idx)
    return np.stack(idx_all, axis=0)


def _encode_predictions_torch(preds: list[Any], classes_t: Any):
    torch = optional_import("torch", extra="inductive-torch")
    idx_all = []
    for pred in preds:
        if pred.ndim != 1:
            pred = pred.reshape(-1)
        if pred.dtype != classes_t.dtype:
            pred = pred.to(classes_t.dtype)
        idx_all.append(torch.searchsorted(classes_t, pred))
    return torch.stack(idx_all, dim=0)


def _standardized_weighted_majority_numpy(
    preds_idx: np.ndarray,
    weights: np.ndarray,
    *,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Historical ModSSC weighted-score vote used by ``standardized``."""
    n_learners, n_samples = preds_idx.shape
    scores = np.zeros((n_samples, n_classes), dtype=np.float64)
    row_idx = np.arange(n_samples)
    for i in range(n_learners):
        scores[row_idx, preds_idx[i]] += float(weights[i])
    majority_idx = scores.argmax(axis=1)
    if n_classes <= 1:
        return majority_idx, np.ones((n_samples,), dtype=bool)
    max_vals = scores.max(axis=1)
    second_vals = np.partition(scores, -2, axis=1)[:, -2]
    return majority_idx, max_vals > second_vals


def _standardized_weighted_majority_torch(
    preds_idx: Any,
    weights: Any,
    *,
    n_classes: int,
):
    """Torch equivalent of the historical ModSSC weighted-score vote."""
    torch = optional_import("torch", extra="inductive-torch")
    n_learners, n_samples = preds_idx.shape
    one_hot = torch.nn.functional.one_hot(preds_idx, num_classes=int(n_classes)).to(
        dtype=weights.dtype
    )
    scores = (one_hot * weights.view(n_learners, 1, 1)).sum(dim=0)
    majority_idx = scores.argmax(dim=1)
    if int(n_classes) <= 1:
        majority_ok = torch.ones((n_samples,), dtype=torch.bool, device=preds_idx.device)
        return majority_idx, majority_ok
    top2 = torch.topk(scores, k=2, dim=1).values
    majority_ok = top2[:, 0] > top2[:, 1]
    return majority_idx, majority_ok


def _weighted_majority_numpy(
    preds_idx: np.ndarray, weights: np.ndarray, *, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the vote majority and whether it satisfies the paper's confidence rule."""
    n_learners, n_samples = preds_idx.shape
    counts = np.zeros((n_samples, n_classes), dtype=np.int64)
    confidence_sums = np.zeros((n_samples, n_classes), dtype=np.float64)
    row_idx = np.arange(n_samples)
    for i in range(n_learners):
        counts[row_idx, preds_idx[i]] += 1
        confidence_sums[row_idx, preds_idx[i]] += float(weights[i])

    majority_idx = counts.argmax(axis=1)
    if n_classes <= 1:
        return majority_idx, np.ones((n_samples,), dtype=bool)

    majority_counts = counts[row_idx, majority_idx]
    strict_majority = 2 * majority_counts > n_learners

    minority_mask = counts > 0
    minority_mask[row_idx, majority_idx] = False
    minority_sums = np.where(minority_mask, confidence_sums, -np.inf)
    majority_sums = confidence_sums[row_idx, majority_idx]
    confidence_ok = majority_sums > minority_sums.max(axis=1, initial=-np.inf)
    return majority_idx, strict_majority & confidence_ok


def _weighted_majority_torch(preds_idx: Any, weights: Any, *, n_classes: int):
    """Torch equivalent of :func:`_weighted_majority_numpy`."""
    torch = optional_import("torch", extra="inductive-torch")
    n_learners, n_samples = preds_idx.shape
    one_hot = torch.nn.functional.one_hot(preds_idx, num_classes=int(n_classes))
    counts = one_hot.sum(dim=0)
    confidence_sums = (one_hot.to(dtype=weights.dtype) * weights.view(n_learners, 1, 1)).sum(dim=0)

    majority_idx = counts.argmax(dim=1)
    if int(n_classes) <= 1:
        majority_ok = torch.ones((n_samples,), dtype=torch.bool, device=preds_idx.device)
        return majority_idx, majority_ok

    row_idx = torch.arange(n_samples, device=preds_idx.device)
    majority_counts = counts[row_idx, majority_idx]
    strict_majority = 2 * majority_counts > n_learners

    minority_mask = counts > 0
    minority_mask[row_idx, majority_idx] = False
    minority_sums = confidence_sums.masked_fill(~minority_mask, -torch.inf)
    majority_sums = confidence_sums[row_idx, majority_idx]
    confidence_ok = majority_sums > minority_sums.max(dim=1).values
    majority_ok = strict_majority & confidence_ok
    return majority_idx, majority_ok


def _proposal_error_numpy(
    *,
    preds_idx: np.ndarray,
    majority_idx: np.ndarray,
    proposed_idx: np.ndarray,
    lower_bounds: np.ndarray,
) -> float:
    """Equation from Figure 1: expected errors in a proposed label set.

    For every proposed example, ``d`` is the number of classifiers in its
    majority group and the estimated correctness is the mean of those
    classifiers' lower 95%-confidence bounds.  Summing the complementary
    probabilities handles majority groups that differ across examples.
    """

    proposed = np.asarray(proposed_idx, dtype=np.int64).reshape(-1)
    if proposed.size == 0:
        return 0.0
    voters = preds_idx[:, proposed] == majority_idx[proposed][None, :]
    denominator = voters.sum(axis=0)
    if np.any(denominator <= 0):  # pragma: no cover - argmax always has a voter
        raise InductiveValidationError("A proposed label has no majority voter.")
    confidence = (voters * lower_bounds[:, None]).sum(axis=0) / denominator
    return float(np.sum(1.0 - confidence))


def _proposal_error_torch(
    *,
    preds_idx: Any,
    majority_idx: Any,
    proposed_idx: Any,
    lower_bounds: Any,
) -> float:
    torch = optional_import("torch", extra="inductive-torch")
    if int(proposed_idx.numel()) == 0:
        return 0.0
    voters = preds_idx[:, proposed_idx] == majority_idx[proposed_idx].view(1, -1)
    denominator = voters.sum(dim=0)
    if bool(torch.any(denominator <= 0)):  # pragma: no cover - argmax always has a voter
        raise InductiveValidationError("A proposed label has no majority voter.")
    confidence = (voters.to(lower_bounds.dtype) * lower_bounds.view(-1, 1)).sum(
        dim=0
    ) / denominator.to(lower_bounds.dtype)
    return float(torch.sum(1.0 - confidence).item())


def _combine_scores_numpy(
    preds_idx: np.ndarray, weights: np.ndarray, *, n_classes: int, min_confidence: float
) -> np.ndarray:
    n_learners, n_samples = preds_idx.shape
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if int(weights.size) != int(n_learners):
        raise InductiveValidationError("DCL requires one vote weight per learner.")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise InductiveValidationError("DCL vote weights must be finite and non-negative.")
    eligible = weights > float(min_confidence)
    if not np.any(eligible):
        raise InductiveValidationError("DCL paper vote has no learner above min_confidence.")
    effective_weights = weights * eligible
    if not np.any(effective_weights > 0.0):
        raise InductiveValidationError("DCL paper vote has no positive eligible weight.")
    scores = np.zeros((n_samples, n_classes), dtype=np.float64)
    counts = np.zeros((n_samples, n_classes), dtype=np.float64)
    row_idx = np.arange(n_samples)
    for i in range(n_learners):
        if not eligible[i]:
            continue
        scores[row_idx, preds_idx[i]] += float(effective_weights[i])
        counts[row_idx, preds_idx[i]] += 1.0
    avg = scores / np.maximum(counts, 1.0)
    corr = (counts + 0.5) / (counts + 1.0)
    out = avg * corr
    out[counts == 0] = 0.0
    return out


def _combine_scores_torch(preds_idx: Any, weights: Any, *, n_classes: int, min_confidence: float):
    torch = optional_import("torch", extra="inductive-torch")
    n_learners, n_samples = preds_idx.shape
    weights = weights.reshape(-1)
    if int(weights.numel()) != int(n_learners):
        raise InductiveValidationError("DCL requires one vote weight per learner.")
    if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0.0)):
        raise InductiveValidationError("DCL vote weights must be finite and non-negative.")
    eligible = weights > float(min_confidence)
    if not bool(eligible.any()):
        raise InductiveValidationError("DCL paper vote has no learner above min_confidence.")
    weights_eff = weights * eligible.to(weights.dtype)
    if not bool(torch.any(weights_eff > 0.0)):
        raise InductiveValidationError("DCL paper vote has no positive eligible weight.")
    one_hot = torch.nn.functional.one_hot(preds_idx, num_classes=int(n_classes)).to(
        dtype=weights.dtype
    )
    scores = (one_hot * weights_eff.view(n_learners, 1, 1)).sum(dim=0)
    counts = (one_hot * eligible.to(weights.dtype).view(n_learners, 1, 1)).sum(dim=0)
    avg = scores / torch.where(counts == 0, torch.ones_like(counts), counts)
    corr = (counts + 0.5) / (counts + 1.0)
    out = avg * corr
    out = torch.where(counts == 0, torch.zeros_like(out), out)
    return out


def _standardized_combine_scores_numpy(
    preds_idx: np.ndarray,
    weights: np.ndarray,
    *,
    n_classes: int,
    min_confidence: float,
) -> np.ndarray:
    """Historical ModSSC final vote, kept byte-for-byte in semantics."""
    n_learners, n_samples = preds_idx.shape
    eligible = weights > float(min_confidence)
    if not np.any(eligible):
        eligible = np.ones_like(eligible, dtype=bool)
    scores = np.zeros((n_samples, n_classes), dtype=np.float64)
    counts = np.zeros((n_samples, n_classes), dtype=np.float64)
    row_idx = np.arange(n_samples)
    for i in range(n_learners):
        if not eligible[i]:
            continue
        scores[row_idx, preds_idx[i]] += float(weights[i])
        counts[row_idx, preds_idx[i]] += 1.0
    avg = scores / np.maximum(counts, 1.0)
    corr = (counts + 0.5) / (counts + 1.0)
    out = avg * corr
    out[counts == 0] = 0.0
    return out


def _standardized_combine_scores_torch(
    preds_idx: Any,
    weights: Any,
    *,
    n_classes: int,
    min_confidence: float,
):
    """Torch equivalent of the historical ModSSC final vote."""
    torch = optional_import("torch", extra="inductive-torch")
    n_learners, _n_samples = preds_idx.shape
    eligible = weights > float(min_confidence)
    if not bool(eligible.any()):
        eligible = torch.ones_like(eligible, dtype=torch.bool)
    one_hot = torch.nn.functional.one_hot(preds_idx, num_classes=int(n_classes)).to(
        dtype=weights.dtype
    )
    weights_eff = weights * eligible.to(weights.dtype)
    scores = (one_hot * weights_eff.view(n_learners, 1, 1)).sum(dim=0)
    counts = (one_hot * eligible.to(weights.dtype).view(n_learners, 1, 1)).sum(dim=0)
    avg = scores / torch.where(counts == 0, torch.ones_like(counts), counts)
    corr = (counts + 0.5) / (counts + 1.0)
    out = avg * corr
    return torch.where(counts == 0, torch.zeros_like(out), out)


@dataclass(frozen=True)
class DemocraticCoLearningSpec(BaseClassifierSpec):
    max_iter: int = 20
    confidence_level: float = 0.95
    min_confidence: float = 0.5
    n_learners: int = 3
    classifier_specs: tuple[BaseClassifierSpec, ...] | None = None
    confidence_estimator: str = field(default="training_accuracy", kw_only=True)
    confidence_interval: str = field(default="wald", kw_only=True)
    confidence_folds: int = field(default=10, kw_only=True)
    confidence_seed: int = field(default=0, kw_only=True)
    diagnostic_trace: bool = field(default=False, kw_only=True)
    control_mode: str = field(default="dcl", kw_only=True)
    training_mode: str = field(default="legacy", kw_only=True)
    require_convergence: bool = field(default=False, kw_only=True)
    min_pseudo_labels_added: int | None = field(default=None, kw_only=True)


_CONFIDENCE_ESTIMATORS = frozenset({"training_accuracy", "kfold_oof"})
_CONFIDENCE_INTERVALS = frozenset({"wald", "wilson", "clopper_pearson"})
_CONTROL_MODES = frozenset({"dcl", "learner_0", "learner_1", "learner_2", "combining_only"})


def _validate_v2_spec(spec: DemocraticCoLearningSpec, *, n_learners: int) -> None:
    if spec.confidence_estimator not in _CONFIDENCE_ESTIMATORS:
        raise InductiveValidationError(
            "confidence_estimator must be one of {'training_accuracy', 'kfold_oof'}."
        )
    if spec.confidence_interval not in _CONFIDENCE_INTERVALS:
        raise InductiveValidationError(
            "confidence_interval must be one of {'wald', 'wilson', 'clopper_pearson'}."
        )
    if int(spec.confidence_folds) < 2:
        raise InductiveValidationError("confidence_folds must be >= 2.")
    if spec.control_mode not in _CONTROL_MODES:
        raise InductiveValidationError(
            "control_mode must be one of "
            "{'dcl', 'learner_0', 'learner_1', 'learner_2', 'combining_only'}."
        )
    if spec.control_mode.startswith("learner_"):
        learner_index = int(spec.control_mode.rsplit("_", maxsplit=1)[1])
        if learner_index >= int(n_learners):
            raise InductiveValidationError(
                f"control_mode={spec.control_mode!r} requires learner index {learner_index}."
            )


class DemocraticCoLearningMethod(InductiveMethod):
    """Democratic co-learning with multiple learners (CPU/GPU)."""

    info = MethodInfo(
        method_id="democratic_co_learning",
        name="Democratic Co-Learning",
        year=2004,
        family="classic",
        supports_gpu=True,
        paper_title="Democratic Co-Learning",
        paper_pdf="docs/article_code/inductive/2004-Democratic colearning/21-2004-Democratic colearning.pdf",
        official_code=None,
    )

    def __init__(self, spec: DemocraticCoLearningSpec | None = None) -> None:
        self.spec = spec or DemocraticCoLearningSpec()
        self._clfs: list[Any] = []
        self._backend: str | None = None
        self._weights: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._classes_t: Any | None = None
        self._initial_clfs: list[Any] = []
        self._initial_weights: np.ndarray | None = None
        self._resolved_specs: list[BaseClassifierSpec] = []
        self._fit_seed: int = 0
        self.round_trace_: list[dict[str, Any]] = []
        self.n_iter_: int = 0
        self.changed_rounds_: int = 0
        self.converged_: bool = False
        self.pseudo_labels_added_per_learner_: tuple[int, ...] = ()
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> DemocraticCoLearningMethod:
        if self.spec.training_mode not in _TRAINING_MODES:
            raise InductiveValidationError(
                f"training_mode must be one of {sorted(_TRAINING_MODES)!r}."
            )
        if self.spec.training_mode == "legacy":
            return self._fit_standardized(data, device=device, seed=seed)
        return self._fit_confidence_weighted(data, device=device, seed=seed)

    def _fit_standardized(
        self,
        data: Any,
        *,
        device: DeviceSpec,
        seed: int = 0,
    ) -> DemocraticCoLearningMethod:
        """Run the historical ModSSC algorithm without paper-v2 semantics."""
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        if data is None:
            raise InductiveValidationError("data must not be None.")
        if (
            self.spec.confidence_estimator != "training_accuracy"
            or self.spec.confidence_interval != "wald"
            or int(self.spec.confidence_folds) != 10
            or int(self.spec.confidence_seed) != 0
            or self.spec.diagnostic_trace
            or self.spec.control_mode != "dcl"
        ):
            raise InductiveValidationError(
                "DCL confidence diagnostics and controls require "
                "training_mode='confidence_weighted'."
            )

        backend = detect_backend(data.X_l)
        logger.debug("backend=%s", backend)
        specs = _resolve_classifier_specs(self.spec)
        self._resolved_specs = list(specs)
        self._fit_seed = int(seed)
        self._initial_clfs = []
        self._initial_weights = None
        self.round_trace_ = []
        self.n_iter_ = 0
        self.changed_rounds_ = 0
        self.converged_ = False
        self.pseudo_labels_added_per_learner_ = tuple(0 for _ in specs)
        self.diagnostics_ = {}
        for learner_spec in specs:
            ensure_classifier_backend(learner_spec, backend=backend)

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")
            X_l = np.asarray(ds.X_l)
            X_u = np.asarray(ds.X_u) if ds.X_u is not None else None
            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            clfs = [
                build_classifier(learner_spec, seed=seed + i)
                for i, learner_spec in enumerate(specs)
            ]
            if X_u is None or X_u.size == 0:
                for clf in clfs:
                    clf.fit(X_l, y_l)
                self._standardized_finalize_numpy(clfs, X_l, y_l)
                self.converged_ = True
                self._log_fit_diagnostics()
                logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
                return self

            n_u = int(X_u.shape[0])
            X_l_i = [X_l for _ in clfs]
            y_l_i = [y_l for _ in clfs]
            e_i = [0.0 for _ in clfs]
            added_mask = [np.zeros((n_u,), dtype=bool) for _ in clfs]

            iter_count = 0
            while iter_count < int(self.spec.max_iter):
                for i, clf in enumerate(clfs):
                    clf.fit(X_l_i[i], y_l_i[i])

                weights = self._standardized_weights_numpy(clfs, X_l, y_l)
                classes = _resolve_classes_numpy(clfs, y_l)
                preds = [clf.predict(X_u) for clf in clfs]
                preds_idx = _encode_predictions_numpy(preds, classes)
                majority_idx, majority_ok = _standardized_weighted_majority_numpy(
                    preds_idx,
                    weights,
                    n_classes=int(classes.size),
                )
                majority_labels = classes[majority_idx]

                idx_per = []
                for i in range(len(clfs)):
                    mask = majority_ok & (preds_idx[i] != majority_idx) & (~added_mask[i])
                    idx_per.append(np.where(mask)[0])

                lower_bounds = []
                for i, clf in enumerate(clfs):
                    pred_l = np.asarray(clf.predict(X_l_i[i]))
                    lo, _hi = _confidence_interval_numpy(
                        y_l_i[i],
                        pred_l,
                        confidence_level=float(self.spec.confidence_level),
                    )
                    lower_bounds.append(lo)
                avg_lower = float(np.mean(lower_bounds)) if lower_bounds else 0.0
                avg_lower = min(max(avg_lower, 0.0), 1.0)

                changed = False
                for i, idx in enumerate(idx_per):
                    if idx.size == 0:
                        continue
                    n_i = int(y_l_i[i].shape[0])
                    q_i = float(n_i) * (1.0 - 2.0 * (e_i[i] / float(n_i))) ** 2
                    e_prime = (1.0 - avg_lower) * float(idx.size)
                    n_new = n_i + int(idx.size)
                    q_prime = float(n_new) * (1.0 - 2.0 * ((e_i[i] + e_prime) / float(n_new))) ** 2
                    if q_prime > q_i:
                        X_l_i[i] = np.concatenate([X_l_i[i], X_u[idx]], axis=0)
                        y_l_i[i] = np.concatenate([y_l_i[i], majority_labels[idx]], axis=0)
                        added_mask[i][idx] = True
                        e_i[i] += e_prime
                        changed = True

                logger.debug("Democratic co-learning iter=%s changed=%s", iter_count, changed)
                if not changed:
                    self.converged_ = True
                    break
                iter_count += 1

            for i, clf in enumerate(clfs):
                clf.fit(X_l_i[i], y_l_i[i])

            self.n_iter_ = iter_count
            self.changed_rounds_ = iter_count
            self.pseudo_labels_added_per_learner_ = tuple(int(mask.sum()) for mask in added_mask)
            self._standardized_finalize_numpy(clfs, X_l, y_l)
            self._log_fit_diagnostics()
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")
        X_l = ds.X_l
        X_u = ds.X_u
        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")

        clfs = [
            build_classifier(learner_spec, seed=seed + i) for i, learner_spec in enumerate(specs)
        ]
        if X_u is None or int(get_torch_len(X_u)) == 0:
            for clf in clfs:
                clf.fit(X_l, y_l)
            self._standardized_finalize_torch(clfs, X_l, y_l)
            self.converged_ = True
            self._log_fit_diagnostics()
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        n_u = int(get_torch_len(X_u))
        X_l_i = [X_l for _ in clfs]
        y_l_i = [y_l for _ in clfs]
        e_i = [0.0 for _ in clfs]
        added_mask = [
            torch.zeros((n_u,), dtype=torch.bool, device=get_torch_device(X_l)) for _ in clfs
        ]

        iter_count = 0
        while iter_count < int(self.spec.max_iter):
            for i, clf in enumerate(clfs):
                clf.fit(X_l_i[i], y_l_i[i])

            weights = self._standardized_weights_torch(clfs, X_l, y_l)
            classes_t = _resolve_classes_torch(clfs, y_l)
            preds = [clf.predict(X_u) for clf in clfs]
            preds_idx = _encode_predictions_torch(preds, classes_t)
            majority_idx, majority_ok = _standardized_weighted_majority_torch(
                preds_idx,
                torch.tensor(weights, device=get_torch_device(X_l), dtype=torch.float32),
                n_classes=int(classes_t.numel()),
            )
            majority_labels = classes_t[majority_idx]

            idx_per = []
            for i in range(len(clfs)):
                mask = majority_ok & (preds_idx[i] != majority_idx) & (~added_mask[i])
                idx_per.append(mask.nonzero(as_tuple=False).reshape(-1))

            lower_bounds = []
            for i, clf in enumerate(clfs):
                pred_l = clf.predict(X_l_i[i])
                lo, _hi = _confidence_interval_torch(
                    y_l_i[i],
                    pred_l,
                    confidence_level=float(self.spec.confidence_level),
                )
                lower_bounds.append(lo)
            avg_lower = float(np.mean(lower_bounds)) if lower_bounds else 0.0
            avg_lower = min(max(avg_lower, 0.0), 1.0)

            changed = False
            for i, idx in enumerate(idx_per):
                if int(idx.numel()) == 0:
                    continue
                n_i = int(y_l_i[i].shape[0])
                q_i = float(n_i) * (1.0 - 2.0 * (e_i[i] / float(n_i))) ** 2
                e_prime = (1.0 - avg_lower) * float(int(idx.numel()))
                n_new = n_i + int(idx.numel())
                q_prime = float(n_new) * (1.0 - 2.0 * ((e_i[i] + e_prime) / float(n_new))) ** 2
                if q_prime > q_i:
                    X_l_i[i] = concat_data([X_l_i[i], slice_data(X_u, idx)])
                    y_l_i[i] = torch.cat([y_l_i[i], majority_labels[idx]], dim=0)
                    added_mask[i][idx] = True
                    e_i[i] += e_prime
                    changed = True

            logger.debug("Democratic co-learning iter=%s changed=%s", iter_count, changed)
            if not changed:
                self.converged_ = True
                break
            iter_count += 1

        for i, clf in enumerate(clfs):
            clf.fit(X_l_i[i], y_l_i[i])

        self.n_iter_ = iter_count
        self.changed_rounds_ = iter_count
        self.pseudo_labels_added_per_learner_ = tuple(int(mask.sum().item()) for mask in added_mask)
        self._standardized_finalize_torch(clfs, X_l, y_l)
        self._log_fit_diagnostics()
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def _fit_confidence_weighted(
        self,
        data: Any,
        *,
        device: DeviceSpec,
        seed: int = 0,
    ) -> DemocraticCoLearningMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        logger.debug("backend=%s", backend)
        specs = _resolve_classifier_specs(self.spec)
        _validate_v2_spec(self.spec, n_learners=len(specs))
        self._resolved_specs = list(specs)
        self._fit_seed = int(seed)
        self._initial_clfs = []
        self._initial_weights = None
        self.round_trace_ = []
        self.n_iter_ = 0
        self.changed_rounds_ = 0
        self.converged_ = False
        self.pseudo_labels_added_per_learner_ = tuple(0 for _ in specs)
        self.diagnostics_ = {}
        for spec in specs:
            ensure_classifier_backend(spec, backend=backend)

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")
            X_l = np.asarray(ds.X_l)
            X_u = np.asarray(ds.X_u) if ds.X_u is not None else None

            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            clfs = [build_classifier(spec, seed=seed + i) for i, spec in enumerate(specs)]
            if self.spec.control_mode != "dcl":
                for clf in clfs:
                    clf.fit(X_l, y_l)
                control_intervals = self._confidence_intervals_numpy(
                    clfs,
                    specs,
                    X_l,
                    y_l,
                )
                self._initial_clfs = clfs
                self._initial_weights = np.asarray(
                    [(lo + hi) / 2.0 for lo, hi in control_intervals],
                    dtype=np.float64,
                )
                self._finalize_numpy(
                    clfs,
                    X_l,
                    y_l,
                    intervals=control_intervals,
                )
                self.converged_ = True
                self._log_fit_diagnostics()
                logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
                return self
            fixed_original_intervals = (
                self._confidence_intervals_numpy(clfs, specs, X_l, y_l)
                if self.spec.confidence_estimator == "kfold_oof"
                else None
            )
            if X_u is None or X_u.size == 0:
                for clf in clfs:
                    clf.fit(X_l, y_l)
                self._finalize_numpy(
                    clfs,
                    X_l,
                    y_l,
                    intervals=fixed_original_intervals,
                )
                self._fit_initial_controls_numpy(
                    specs,
                    X_l,
                    y_l,
                    intervals=fixed_original_intervals,
                )
                self.converged_ = True
                self._log_fit_diagnostics()
                logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
                return self

            n_u = int(X_u.shape[0])
            X_l_i = [X_l for _ in clfs]
            y_l_i = [y_l for _ in clfs]
            e_i = [0.0 for _ in range(len(clfs))]
            added_mask = [np.zeros((n_u,), dtype=bool) for _ in range(len(clfs))]
            evolving_interval_cache: list[tuple[float, float] | None] = [None for _ in clfs]

            while self.n_iter_ < int(self.spec.max_iter):
                for i, clf in enumerate(clfs):
                    clf.fit(X_l_i[i], y_l_i[i])

                original_intervals = (
                    fixed_original_intervals
                    if fixed_original_intervals is not None
                    else self._confidence_intervals_numpy(
                        clfs,
                        specs,
                        X_l,
                        y_l,
                    )
                )
                weights = np.asarray(
                    [(lo + hi) / 2.0 for lo, hi in original_intervals],
                    dtype=np.float64,
                )
                classes = _resolve_classes_numpy(clfs, y_l)
                preds = [predict_in_batches(clf, X_u, backend="numpy") for clf in clfs]
                preds_idx = _encode_predictions_numpy(preds, classes)
                majority_idx, majority_ok = _weighted_majority_numpy(
                    preds_idx, weights, n_classes=int(classes.size)
                )
                majority_labels = classes[majority_idx]

                idx_per = []
                for i in range(len(clfs)):
                    mask = majority_ok & (preds_idx[i] != majority_idx) & (~added_mask[i])
                    idx_per.append(np.where(mask)[0])

                if self.spec.confidence_estimator == "kfold_oof":
                    for i, clf in enumerate(clfs):
                        if evolving_interval_cache[i] is None:
                            evolving_interval_cache[i] = self._confidence_intervals_numpy(
                                [clf],
                                [specs[i]],
                                np.asarray(X_l_i[i]),
                                np.asarray(y_l_i[i]),
                                learner_index_offset=i,
                            )[0]
                    evolving_intervals = [
                        interval for interval in evolving_interval_cache if interval is not None
                    ]
                else:
                    evolving_intervals = [
                        self._confidence_intervals_numpy(
                            [clf],
                            [specs[i]],
                            np.asarray(X_l_i[i]),
                            np.asarray(y_l_i[i]),
                            learner_index_offset=i,
                        )[0]
                        for i, clf in enumerate(clfs)
                    ]
                lower_bounds = [lo for lo, _hi in evolving_intervals]
                lower_bounds_array = np.clip(np.asarray(lower_bounds, dtype=np.float64), 0.0, 1.0)

                changed = False
                round_learners: list[dict[str, Any]] = []
                for i, idx in enumerate(idx_per):
                    n_i = int(y_l_i[i].shape[0])
                    q_i = float(n_i) * (1.0 - 2.0 * (e_i[i] / float(n_i))) ** 2
                    e_prime = (
                        _proposal_error_numpy(
                            preds_idx=preds_idx,
                            majority_idx=majority_idx,
                            proposed_idx=idx,
                            lower_bounds=lower_bounds_array,
                        )
                        if idx.size
                        else 0.0
                    )
                    n_new = n_i + int(idx.size)
                    q_prime = float(n_new) * (1.0 - 2.0 * ((e_i[i] + e_prime) / float(n_new))) ** 2
                    accepted = bool(idx.size and q_prime > q_i)
                    error_before = float(e_i[i])
                    if accepted:
                        X_l_i[i] = np.concatenate([X_l_i[i], X_u[idx]], axis=0)
                        y_l_i[i] = np.concatenate([y_l_i[i], majority_labels[idx]], axis=0)
                        added_mask[i][idx] = True
                        e_i[i] += e_prime
                        if self.spec.confidence_estimator == "kfold_oof":
                            evolving_interval_cache[i] = None
                        changed = True
                    if self.spec.diagnostic_trace:
                        round_learners.append(
                            {
                                "learner_index": i,
                                "classifier_id": specs[i].classifier_id,
                                "original_interval": {
                                    "lower": float(original_intervals[i][0]),
                                    "upper": float(original_intervals[i][1]),
                                },
                                "weight": float(weights[i]),
                                "evolving_interval": {
                                    "lower": float(evolving_intervals[i][0]),
                                    "upper": float(evolving_intervals[i][1]),
                                },
                                "training_size_before": n_i,
                                "training_size_after": int(y_l_i[i].shape[0]),
                                "disagreement_count": int(np.sum(preds_idx[i] != majority_idx)),
                                "proposal_count": int(idx.size),
                                "error_estimate_before": error_before,
                                "proposal_error": float(e_prime),
                                "error_estimate_after": float(e_i[i]),
                                "q": float(q_i),
                                "q_prime": float(q_prime),
                                "accepted": accepted,
                                "added_count": int(idx.size) if accepted else 0,
                            }
                        )

                self.n_iter_ += 1
                if changed:
                    self.changed_rounds_ += 1
                if self.spec.diagnostic_trace:
                    self.round_trace_.append(
                        {
                            "round": int(self.n_iter_),
                            "majority_eligible_count": int(np.sum(majority_ok)),
                            "learners": round_learners,
                        }
                    )
                logger.debug("Democratic co-learning iter=%s changed=%s", self.n_iter_, changed)
                if not changed:
                    self.converged_ = True
                    break

            for i, clf in enumerate(clfs):
                clf.fit(X_l_i[i], y_l_i[i])

            self.pseudo_labels_added_per_learner_ = tuple(int(mask.sum()) for mask in added_mask)
            self._finalize_numpy(
                clfs,
                X_l,
                y_l,
                intervals=fixed_original_intervals,
            )
            self._fit_initial_controls_numpy(
                specs,
                X_l,
                y_l,
                intervals=fixed_original_intervals,
            )
            self._log_fit_diagnostics()
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")

        X_l = ds.X_l
        X_u = ds.X_u
        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")

        clfs = [build_classifier(spec, seed=seed + i) for i, spec in enumerate(specs)]
        if self.spec.control_mode != "dcl":
            for clf in clfs:
                clf.fit(X_l, y_l)
            control_intervals = self._confidence_intervals_torch(
                clfs,
                specs,
                X_l,
                y_l,
            )
            self._initial_clfs = clfs
            self._initial_weights = np.asarray(
                [(lo + hi) / 2.0 for lo, hi in control_intervals],
                dtype=np.float64,
            )
            self._finalize_torch(
                clfs,
                X_l,
                y_l,
                intervals=control_intervals,
            )
            self.converged_ = True
            self._log_fit_diagnostics()
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self
        fixed_original_intervals = (
            self._confidence_intervals_torch(clfs, specs, X_l, y_l)
            if self.spec.confidence_estimator == "kfold_oof"
            else None
        )
        if X_u is None or int(get_torch_len(X_u)) == 0:
            for clf in clfs:
                clf.fit(X_l, y_l)
            self._finalize_torch(
                clfs,
                X_l,
                y_l,
                intervals=fixed_original_intervals,
            )
            self._fit_initial_controls_torch(
                specs,
                X_l,
                y_l,
                intervals=fixed_original_intervals,
            )
            self.converged_ = True
            self._log_fit_diagnostics()
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        n_u = int(get_torch_len(X_u))
        X_l_i = [X_l for _ in clfs]
        y_l_i = [y_l for _ in clfs]
        e_i = [0.0 for _ in range(len(clfs))]
        added_mask = [
            torch.zeros((n_u,), dtype=torch.bool, device=get_torch_device(X_l)) for _ in clfs
        ]
        evolving_interval_cache: list[tuple[float, float] | None] = [None for _ in clfs]

        while self.n_iter_ < int(self.spec.max_iter):
            for i, clf in enumerate(clfs):
                clf.fit(X_l_i[i], y_l_i[i])

            original_intervals = (
                fixed_original_intervals
                if fixed_original_intervals is not None
                else self._confidence_intervals_torch(
                    clfs,
                    specs,
                    X_l,
                    y_l,
                )
            )
            weights = np.asarray(
                [(lo + hi) / 2.0 for lo, hi in original_intervals],
                dtype=np.float64,
            )
            classes_t = _resolve_classes_torch(clfs, y_l)
            preds = [predict_in_batches(clf, X_u, backend="torch") for clf in clfs]
            preds_idx = _encode_predictions_torch(preds, classes_t)
            majority_idx, majority_ok = _weighted_majority_torch(
                preds_idx,
                torch.tensor(weights, device=get_torch_device(X_l), dtype=torch.float32),
                n_classes=int(classes_t.numel()),
            )
            majority_labels = classes_t[majority_idx]

            idx_per = []
            for i in range(len(clfs)):
                mask = majority_ok & (preds_idx[i] != majority_idx) & (~added_mask[i])
                idx_per.append(mask.nonzero(as_tuple=False).reshape(-1))

            if self.spec.confidence_estimator == "kfold_oof":
                for i, clf in enumerate(clfs):
                    if evolving_interval_cache[i] is None:
                        evolving_interval_cache[i] = self._confidence_intervals_torch(
                            [clf],
                            [specs[i]],
                            X_l_i[i],
                            y_l_i[i],
                            learner_index_offset=i,
                        )[0]
                evolving_intervals = [
                    interval for interval in evolving_interval_cache if interval is not None
                ]
            else:
                evolving_intervals = [
                    self._confidence_intervals_torch(
                        [clf],
                        [specs[i]],
                        X_l_i[i],
                        y_l_i[i],
                        learner_index_offset=i,
                    )[0]
                    for i, clf in enumerate(clfs)
                ]
            lower_bounds = [lo for lo, _hi in evolving_intervals]
            lower_bounds_t = torch.as_tensor(
                np.clip(np.asarray(lower_bounds, dtype=np.float64), 0.0, 1.0),
                dtype=torch.float32,
                device=get_torch_device(X_l),
            )

            changed = False
            round_learners = []
            for i, idx in enumerate(idx_per):
                n_i = int(y_l_i[i].shape[0])
                q_i = float(n_i) * (1.0 - 2.0 * (e_i[i] / float(n_i))) ** 2
                e_prime = (
                    _proposal_error_torch(
                        preds_idx=preds_idx,
                        majority_idx=majority_idx,
                        proposed_idx=idx,
                        lower_bounds=lower_bounds_t,
                    )
                    if int(idx.numel())
                    else 0.0
                )
                n_new = n_i + int(idx.numel())
                q_prime = float(n_new) * (1.0 - 2.0 * ((e_i[i] + e_prime) / float(n_new))) ** 2
                accepted = bool(int(idx.numel()) and q_prime > q_i)
                error_before = float(e_i[i])
                if accepted:
                    X_l_i[i] = concat_data([X_l_i[i], slice_data(X_u, idx)])
                    y_l_i[i] = torch.cat([y_l_i[i], majority_labels[idx]], dim=0)
                    added_mask[i][idx] = True
                    e_i[i] += e_prime
                    if self.spec.confidence_estimator == "kfold_oof":
                        evolving_interval_cache[i] = None
                    changed = True
                if self.spec.diagnostic_trace:
                    round_learners.append(
                        {
                            "learner_index": i,
                            "classifier_id": specs[i].classifier_id,
                            "original_interval": {
                                "lower": float(original_intervals[i][0]),
                                "upper": float(original_intervals[i][1]),
                            },
                            "weight": float(weights[i]),
                            "evolving_interval": {
                                "lower": float(evolving_intervals[i][0]),
                                "upper": float(evolving_intervals[i][1]),
                            },
                            "training_size_before": n_i,
                            "training_size_after": int(y_l_i[i].shape[0]),
                            "disagreement_count": int(
                                torch.sum(preds_idx[i] != majority_idx).item()
                            ),
                            "proposal_count": int(idx.numel()),
                            "error_estimate_before": error_before,
                            "proposal_error": float(e_prime),
                            "error_estimate_after": float(e_i[i]),
                            "q": float(q_i),
                            "q_prime": float(q_prime),
                            "accepted": accepted,
                            "added_count": int(idx.numel()) if accepted else 0,
                        }
                    )

            self.n_iter_ += 1
            if changed:
                self.changed_rounds_ += 1
            if self.spec.diagnostic_trace:
                self.round_trace_.append(
                    {
                        "round": int(self.n_iter_),
                        "majority_eligible_count": int(majority_ok.sum().item()),
                        "learners": round_learners,
                    }
                )
            logger.debug("Democratic co-learning iter=%s changed=%s", self.n_iter_, changed)
            if not changed:
                self.converged_ = True
                break

        for i, clf in enumerate(clfs):
            clf.fit(X_l_i[i], y_l_i[i])

        self.pseudo_labels_added_per_learner_ = tuple(int(mask.sum().item()) for mask in added_mask)
        self._finalize_torch(
            clfs,
            X_l,
            y_l,
            intervals=fixed_original_intervals,
        )
        self._fit_initial_controls_torch(
            specs,
            X_l,
            y_l,
            intervals=fixed_original_intervals,
        )
        self._log_fit_diagnostics()
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def _log_fit_diagnostics(self) -> None:
        self.diagnostics_ = {
            "n_iter": int(self.n_iter_),
            "changed_rounds": int(self.changed_rounds_),
            "converged": bool(self.converged_),
            "pseudo_labels_added_per_learner": list(self.pseudo_labels_added_per_learner_),
            "pseudo_labels_added_total": int(sum(self.pseudo_labels_added_per_learner_)),
        }
        if self._v2_diagnostics_enabled():
            self.diagnostics_.update(
                {
                    "confidence_protocol": {
                        "estimator": self.spec.confidence_estimator,
                        "interval": self.spec.confidence_interval,
                        "folds": int(self.spec.confidence_folds),
                        "seed": int(self.spec.confidence_seed),
                    },
                    "control": {
                        "mode": self.spec.control_mode,
                        "available_modes": [
                            "learner_0",
                            "learner_1",
                            "learner_2",
                            "combining_only",
                        ],
                        "learner_ids": [
                            learner_spec.classifier_id for learner_spec in self._resolved_specs
                        ],
                    },
                    "round_trace": list(self.round_trace_),
                }
            )
        logger.info(
            "Democratic co-learning diagnostics "
            "pseudo_labels_added_per_learner=%s changed_rounds=%s converged=%s",
            self.pseudo_labels_added_per_learner_,
            self.changed_rounds_,
            self.converged_,
        )

    def _v2_diagnostics_enabled(self) -> bool:
        return bool(self.spec.diagnostic_trace) or self.spec.control_mode != "dcl"

    def _confidence_intervals_numpy(
        self,
        clfs: list[Any],
        specs: list[BaseClassifierSpec],
        X_l: np.ndarray,
        y_l: np.ndarray,
        *,
        learner_index_offset: int = 0,
    ) -> list[tuple[float, float]]:
        intervals = []
        for local_index, (clf, learner_spec) in enumerate(zip(clfs, specs, strict=True)):
            learner_index = int(learner_index_offset) + local_index
            if self.spec.confidence_estimator == "training_accuracy":
                pred = np.asarray(predict_in_batches(clf, X_l, backend="numpy"))
            else:
                pred = self._oof_predictions_numpy(
                    learner_spec,
                    X_l,
                    y_l,
                    learner_index=learner_index,
                )
            lo, hi = _confidence_interval_numpy(
                np.asarray(y_l),
                pred,
                confidence_level=float(self.spec.confidence_level),
                interval=self.spec.confidence_interval,
            )
            intervals.append((lo, hi))
        return intervals

    def _confidence_intervals_torch(
        self,
        clfs: list[Any],
        specs: list[BaseClassifierSpec],
        X_l: Any,
        y_l: Any,
        *,
        learner_index_offset: int = 0,
    ) -> list[tuple[float, float]]:
        intervals = []
        for local_index, (clf, learner_spec) in enumerate(zip(clfs, specs, strict=True)):
            learner_index = int(learner_index_offset) + local_index
            if self.spec.confidence_estimator == "training_accuracy":
                pred = predict_in_batches(clf, X_l, backend="torch")
            else:
                pred = self._oof_predictions_torch(
                    learner_spec,
                    X_l,
                    y_l,
                    learner_index=learner_index,
                )
            lo, hi = _confidence_interval_torch(
                y_l,
                pred,
                confidence_level=float(self.spec.confidence_level),
                interval=self.spec.confidence_interval,
            )
            intervals.append((lo, hi))
        return intervals

    def _oof_predictions_numpy(
        self,
        learner_spec: BaseClassifierSpec,
        X_l: np.ndarray,
        y_l: np.ndarray,
        *,
        learner_index: int,
    ) -> np.ndarray:
        folds = _stratified_kfold_indices(
            np.asarray(y_l),
            n_splits=int(self.spec.confidence_folds),
            seed=int(self.spec.confidence_seed),
        )
        predictions = np.empty_like(np.asarray(y_l))
        seed_base = int(self.spec.confidence_seed) + learner_index * len(folds)
        for fold_index, (train_idx, validation_idx) in enumerate(folds):
            clf = build_classifier(learner_spec, seed=seed_base + fold_index)
            clf.fit(X_l[train_idx], y_l[train_idx])
            predictions[validation_idx] = np.asarray(
                predict_in_batches(clf, X_l[validation_idx], backend="numpy")
            )
        return predictions

    def _oof_predictions_torch(
        self,
        learner_spec: BaseClassifierSpec,
        X_l: Any,
        y_l: Any,
        *,
        learner_index: int,
    ):
        torch = optional_import("torch", extra="inductive-torch")
        folds = _stratified_kfold_indices(
            y_l.detach().cpu().numpy(),
            n_splits=int(self.spec.confidence_folds),
            seed=int(self.spec.confidence_seed),
        )
        predictions = torch.empty_like(y_l)
        seed_base = int(self.spec.confidence_seed) + learner_index * len(folds)
        for fold_index, (train_idx, validation_idx) in enumerate(folds):
            train_t = torch.as_tensor(
                train_idx,
                dtype=torch.int64,
                device=get_torch_device(X_l),
            )
            validation_t = torch.as_tensor(
                validation_idx,
                dtype=torch.int64,
                device=get_torch_device(X_l),
            )
            clf = build_classifier(learner_spec, seed=seed_base + fold_index)
            clf.fit(slice_data(X_l, train_t), y_l[train_t])
            predictions[validation_t] = predict_in_batches(
                clf,
                slice_data(X_l, validation_t),
                backend="torch",
            )
        return predictions

    def _standardized_weights_numpy(
        self,
        clfs: list[Any],
        X_l: Any,
        y_l: Any,
    ) -> np.ndarray:
        weights = []
        for clf in clfs:
            pred = np.asarray(clf.predict(X_l))
            lo, hi = _confidence_interval_numpy(
                np.asarray(y_l),
                pred,
                confidence_level=float(self.spec.confidence_level),
            )
            weights.append((lo + hi) / 2.0)
        return np.asarray(weights, dtype=np.float64)

    def _standardized_weights_torch(
        self,
        clfs: list[Any],
        X_l: Any,
        y_l: Any,
    ) -> np.ndarray:
        weights = []
        for clf in clfs:
            pred = clf.predict(X_l)
            lo, hi = _confidence_interval_torch(
                y_l,
                pred,
                confidence_level=float(self.spec.confidence_level),
            )
            weights.append((lo + hi) / 2.0)
        return np.asarray(weights, dtype=np.float64)

    def _standardized_finalize_numpy(
        self,
        clfs: list[Any],
        X_l: np.ndarray,
        y_l: np.ndarray,
    ) -> None:
        self._weights = self._standardized_weights_numpy(clfs, X_l, y_l)
        self._classes = _resolve_classes_numpy(clfs, y_l)
        self._classes_t = None
        self._clfs = clfs
        self._backend = "numpy"

    def _standardized_finalize_torch(
        self,
        clfs: list[Any],
        X_l: Any,
        y_l: Any,
    ) -> None:
        self._weights = self._standardized_weights_torch(clfs, X_l, y_l)
        self._classes_t = _resolve_classes_torch(clfs, y_l)
        self._classes = self._classes_t.detach().cpu().numpy()
        self._clfs = clfs
        self._backend = "torch"

    def _weights_from_labeled_numpy(
        self,
        clfs: list[Any],
        X_l: Any,
        y_l: Any,
        *,
        specs: list[BaseClassifierSpec] | None = None,
    ) -> np.ndarray:
        resolved_specs = specs if specs is not None else self._resolved_specs
        intervals = self._confidence_intervals_numpy(
            clfs,
            resolved_specs,
            np.asarray(X_l),
            np.asarray(y_l),
        )
        return np.asarray([(lo + hi) / 2.0 for lo, hi in intervals], dtype=np.float64)

    def _weights_from_labeled_torch(
        self,
        clfs: list[Any],
        X_l: Any,
        y_l: Any,
        *,
        specs: list[BaseClassifierSpec] | None = None,
    ) -> np.ndarray:
        resolved_specs = specs if specs is not None else self._resolved_specs
        intervals = self._confidence_intervals_torch(
            clfs,
            resolved_specs,
            X_l,
            y_l,
        )
        return np.asarray([(lo + hi) / 2.0 for lo, hi in intervals], dtype=np.float64)

    def _fit_initial_controls_numpy(
        self,
        specs: list[BaseClassifierSpec],
        X_l: np.ndarray,
        y_l: np.ndarray,
        *,
        intervals: list[tuple[float, float]] | None = None,
    ) -> None:
        if not self._v2_diagnostics_enabled():
            return
        self._initial_clfs = [
            build_classifier(spec, seed=self._fit_seed + i) for i, spec in enumerate(specs)
        ]
        for clf in self._initial_clfs:
            clf.fit(X_l, y_l)
        self._initial_weights = np.asarray(
            (
                [(lo + hi) / 2.0 for lo, hi in intervals]
                if intervals is not None
                else self._weights_from_labeled_numpy(
                    self._initial_clfs,
                    X_l,
                    y_l,
                    specs=specs,
                )
            ),
            dtype=np.float64,
        )

    def _fit_initial_controls_torch(
        self,
        specs: list[BaseClassifierSpec],
        X_l: Any,
        y_l: Any,
        *,
        intervals: list[tuple[float, float]] | None = None,
    ) -> None:
        if not self._v2_diagnostics_enabled():
            return
        self._initial_clfs = [
            build_classifier(spec, seed=self._fit_seed + i) for i, spec in enumerate(specs)
        ]
        for clf in self._initial_clfs:
            clf.fit(X_l, y_l)
        self._initial_weights = np.asarray(
            (
                [(lo + hi) / 2.0 for lo, hi in intervals]
                if intervals is not None
                else self._weights_from_labeled_torch(
                    self._initial_clfs,
                    X_l,
                    y_l,
                    specs=specs,
                )
            ),
            dtype=np.float64,
        )

    def _finalize_numpy(
        self,
        clfs: list[Any],
        X_l: np.ndarray,
        y_l: np.ndarray,
        *,
        intervals: list[tuple[float, float]] | None = None,
    ) -> None:
        self._weights = np.asarray(
            (
                [(lo + hi) / 2.0 for lo, hi in intervals]
                if intervals is not None
                else self._weights_from_labeled_numpy(clfs, X_l, y_l)
            ),
            dtype=np.float64,
        )
        self._classes = _resolve_classes_numpy(clfs, y_l)
        self._classes_t = None
        self._clfs = clfs
        self._backend = "numpy"

    def _finalize_torch(
        self,
        clfs: list[Any],
        X_l: Any,
        y_l: Any,
        *,
        intervals: list[tuple[float, float]] | None = None,
    ) -> None:
        self._weights = np.asarray(
            (
                [(lo + hi) / 2.0 for lo, hi in intervals]
                if intervals is not None
                else self._weights_from_labeled_torch(clfs, X_l, y_l)
            ),
            dtype=np.float64,
        )
        self._classes_t = _resolve_classes_torch(clfs, y_l)
        self._classes = self._classes_t.detach().cpu().numpy()
        self._clfs = clfs
        self._backend = "torch"

    def predict_proba_initial(self, X: Any):
        """Return the Figure 2 ``Combining Only`` control trained on original L."""
        return self.predict_proba_control(X, control_mode="combining_only")

    def predict_proba_control(self, X: Any, *, control_mode: str):
        """Predict with an original-L learner or the original-L ensemble."""
        if control_mode not in _CONTROL_MODES - {"dcl"}:
            raise InductiveValidationError(
                "control_mode must be one of "
                "{'learner_0', 'learner_1', 'learner_2', 'combining_only'}."
            )
        if not self._initial_clfs or self._initial_weights is None:
            raise RuntimeError(
                "DCL controls were not retained; set diagnostic_trace=true or "
                "select a non-'dcl' control_mode before fit()."
            )
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")

        if backend == "numpy":
            if self._classes is None:
                raise RuntimeError(
                    "DemocraticCoLearningMethod missing classes; fit() was not called."
                )
            predictions = [
                predict_in_batches(clf, X, backend="numpy") for clf in self._initial_clfs
            ]
            predictions_idx = _encode_predictions_numpy(predictions, self._classes)
            if control_mode.startswith("learner_"):
                learner_index = int(control_mode.rsplit("_", maxsplit=1)[1])
                return np.eye(
                    int(self._classes.size),
                    dtype=np.float32,
                )[predictions_idx[learner_index]]
            scores = _combine_scores_numpy(
                predictions_idx,
                np.asarray(self._initial_weights, dtype=np.float64),
                n_classes=int(self._classes.size),
                min_confidence=float(self.spec.min_confidence),
            )
            row_sum = scores.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)

        torch = optional_import("torch", extra="inductive-torch")
        if self._classes_t is None:
            raise RuntimeError("DemocraticCoLearningMethod missing classes; fit() was not called.")
        predictions = [predict_in_batches(clf, X, backend="torch") for clf in self._initial_clfs]
        predictions_idx = _encode_predictions_torch(predictions, self._classes_t)
        if control_mode.startswith("learner_"):
            learner_index = int(control_mode.rsplit("_", maxsplit=1)[1])
            return torch.nn.functional.one_hot(
                predictions_idx[learner_index],
                num_classes=int(self._classes_t.numel()),
            ).to(dtype=torch.float32)
        weights_t = torch.tensor(
            self._initial_weights,
            device=get_torch_device(X),
            dtype=torch.float32,
        )
        scores = _combine_scores_torch(
            predictions_idx,
            weights_t,
            n_classes=int(self._classes_t.numel()),
            min_confidence=float(self.spec.min_confidence),
        )
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self._clfs:
            raise RuntimeError("DemocraticCoLearningMethod is not fitted yet. Call fit() first.")
        if self.spec.training_mode == "legacy":
            return self._predict_proba_standardized(X)
        if self.spec.control_mode != "dcl":
            return self.predict_proba_control(
                X,
                control_mode=self.spec.control_mode,
            )
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")
        if self._weights is None:
            raise RuntimeError("DemocraticCoLearningMethod missing weights; fit() was not called.")

        if backend == "numpy":
            weights = np.asarray(self._weights, dtype=np.float64)
            if self._classes is None:
                raise RuntimeError(
                    "DemocraticCoLearningMethod missing classes; fit() was not called."
                )
            preds = [predict_in_batches(clf, X, backend="numpy") for clf in self._clfs]
            preds_idx = _encode_predictions_numpy(preds, self._classes)
            scores = _combine_scores_numpy(
                preds_idx,
                weights,
                n_classes=int(self._classes.size),
                min_confidence=float(self.spec.min_confidence),
            )
            row_sum = scores.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)

        torch = optional_import("torch", extra="inductive-torch")
        weights_t = torch.tensor(self._weights, device=get_torch_device(X), dtype=torch.float32)
        if self._classes_t is None:
            raise RuntimeError("DemocraticCoLearningMethod missing classes; fit() was not called.")
        preds = [predict_in_batches(clf, X, backend="torch") for clf in self._clfs]
        preds_idx = _encode_predictions_torch(preds, self._classes_t)
        scores = _combine_scores_torch(
            preds_idx,
            weights_t,
            n_classes=int(self._classes_t.numel()),
            min_confidence=float(self.spec.min_confidence),
        )
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def _predict_proba_standardized(self, X: Any):
        """Predict with the historical ModSSC final ensemble."""
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")
        if self._weights is None:
            raise RuntimeError("DemocraticCoLearningMethod missing weights; fit() was not called.")

        if backend == "numpy":
            weights = np.asarray(self._weights, dtype=np.float64)
            if self._classes is None:
                raise RuntimeError(
                    "DemocraticCoLearningMethod missing classes; fit() was not called."
                )
            preds = [clf.predict(X) for clf in self._clfs]
            preds_idx = _encode_predictions_numpy(preds, self._classes)
            scores = _standardized_combine_scores_numpy(
                preds_idx,
                weights,
                n_classes=int(self._classes.size),
                min_confidence=float(self.spec.min_confidence),
            )
            row_sum = scores.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)

        torch = optional_import("torch", extra="inductive-torch")
        weights_t = torch.tensor(self._weights, device=get_torch_device(X), dtype=torch.float32)
        if self._classes_t is None:
            raise RuntimeError("DemocraticCoLearningMethod missing classes; fit() was not called.")
        preds = [clf.predict(X) for clf in self._clfs]
        preds_idx = _encode_predictions_torch(preds, self._classes_t)
        scores = _standardized_combine_scores_torch(
            preds_idx,
            weights_t,
            n_classes=int(self._classes_t.numel()),
            min_confidence=float(self.spec.min_confidence),
        )
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        backend = self._backend or detect_backend(X)
        if backend == "numpy":
            idx = proba.argmax(axis=1)
            if self._classes is None:
                return idx
            return np.asarray(self._classes)[idx]
        idx = proba.argmax(dim=1)
        if self._classes_t is None:
            return idx
        return self._classes_t[idx]
