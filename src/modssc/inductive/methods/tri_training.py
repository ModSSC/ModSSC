from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import ceil, floor
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    concat_data,
    get_torch_device,
    get_torch_len,
    slice_data,
)
from modssc.inductive.methods.helpers.tri_training_standardized import (
    fit_standardized_tri_training,
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
    flatten_if_numpy,
)
from modssc.inductive.methods.utils import (
    predict_scores_in_batches as predict_scores,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec
from modssc.supervised.base import classifier_capabilities

logger = logging.getLogger(__name__)

_TRAINING_MODES = frozenset({"legacy", "error_bound_subsample"})


@dataclass(frozen=True)
class _ErrorEstimate:
    rate: float
    agreements: int
    wrong_agreements: int


@dataclass(frozen=True)
class _UpdateDecision:
    previous_size: int
    selected_size: int
    accepted: bool
    subsample: bool
    reason: str


def _as_numpy_labels(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1)


def _global_class_order(y_l: Any, *ensembles: list[Any]) -> np.ndarray:
    """Resolve one sorted class order from labels and fitted ensemble metadata."""

    class_arrays = [_as_numpy_labels(y_l)]
    for classifiers in ensembles:
        for classifier in classifiers:
            classes = getattr(classifier, "classes_", None)
            if classes is None:
                classes = getattr(classifier, "classes_t_", None)
            if classes is not None:
                class_arrays.append(_as_numpy_labels(classes))
    return np.unique(np.concatenate(class_arrays))


def _measure_error(pred_j: Any, pred_k: Any, y_true: Any) -> _ErrorEstimate:
    """Estimate the error of ``h_j & h_k`` on the original labeled set.

    Zhou and Li define the estimate as the fraction of agreeing predictions for
    which both classifiers are wrong.  With no agreement, the error is left at
    the initial conservative value 0.5 so the pair cannot trigger an update.
    """

    pred_j_arr = _as_numpy_labels(pred_j)
    pred_k_arr = _as_numpy_labels(pred_k)
    y_arr = _as_numpy_labels(y_true)
    if pred_j_arr.shape != y_arr.shape or pred_k_arr.shape != y_arr.shape:
        raise InductiveValidationError(
            "TriTraining MeasureError inputs must have the same one-dimensional shape."
        )

    agree = pred_j_arr == pred_k_arr
    agreements = int(agree.sum())
    if agreements == 0:
        return _ErrorEstimate(rate=0.5, agreements=0, wrong_agreements=0)

    both_wrong = agree & (pred_j_arr != y_arr) & (pred_k_arr != y_arr)
    wrong_agreements = int(both_wrong.sum())
    return _ErrorEstimate(
        rate=float(wrong_agreements / agreements),
        agreements=agreements,
        wrong_agreements=wrong_agreements,
    )


def _paper_update_decision(
    *,
    error: float,
    previous_error: float,
    previous_size: int,
    candidate_size: int,
) -> _UpdateDecision:
    """Apply Table I and Eqs. 9--11 of Zhou and Li (2005)."""

    error = float(error)
    previous_error = float(previous_error)
    previous_size = int(previous_size)
    candidate_size = int(candidate_size)
    if not (0.0 <= error <= 1.0 and 0.0 <= previous_error <= 1.0):
        raise ValueError("error and previous_error must be in [0, 1].")
    if previous_size < 0 or candidate_size < 0:
        raise ValueError("previous_size and candidate_size must be non-negative.")
    if error >= previous_error:
        return _UpdateDecision(previous_size, 0, False, False, "error_not_improved")

    delta = previous_error - error
    effective_previous_size = previous_size
    if effective_previous_size == 0:
        # Table I: l'_i <- floor(e_i / (e'_i - e_i) + 1).
        effective_previous_size = max(1, int(floor(error / delta + 1.0)))

    if effective_previous_size >= candidate_size:
        return _UpdateDecision(
            effective_previous_size,
            0,
            False,
            False,
            "insufficient_candidates",
        )

    previous_bound = previous_error * effective_previous_size
    if error * candidate_size < previous_bound:
        return _UpdateDecision(
            effective_previous_size,
            candidate_size,
            True,
            False,
            "accepted_full",
        )

    # Eq. 11 permits Eq. 10 subsampling while retaining |L_i| > l'_i.
    if error > 0.0 and effective_previous_size > error / delta:
        selected_size = int(ceil(previous_bound / error - 1.0))
        selected_size = min(selected_size, candidate_size)
        if selected_size > effective_previous_size and error * selected_size < previous_bound:
            return _UpdateDecision(
                effective_previous_size,
                selected_size,
                True,
                selected_size < candidate_size,
                "accepted_subsample",
            )

    return _UpdateDecision(
        effective_previous_size,
        0,
        False,
        False,
        "noise_bound_not_improved",
    )


def _subsample_positions(
    n_candidates: int,
    target_size: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a seed-controlled, order-stable random subset of candidate positions."""

    n_candidates = int(n_candidates)
    target_size = int(target_size)
    if not 0 <= target_size <= n_candidates:
        raise ValueError("target_size must be between zero and n_candidates.")
    if target_size == n_candidates:
        return np.arange(n_candidates, dtype=np.int64)
    selected = rng.choice(n_candidates, size=target_size, replace=False)
    return np.sort(np.asarray(selected, dtype=np.int64))


def _numpy_labels_from_scores(classifier: Any, scores: np.ndarray) -> np.ndarray:
    indices = np.asarray(scores).argmax(axis=1)
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        classes = getattr(classifier, "classes_t_", None)
    if classes is None:
        return indices
    classes_arr = _as_numpy_labels(classes)
    if int(classes_arr.shape[0]) != int(scores.shape[1]):
        raise InductiveValidationError(
            "TriTraining classifier classes must align with score columns."
        )
    return classes_arr[indices]


def _torch_labels_from_scores(classifier: Any, scores: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    indices = scores.argmax(dim=1)
    classes = getattr(classifier, "classes_t_", None)
    if not isinstance(classes, torch.Tensor):
        classes = getattr(classifier, "classes_", None)
    if classes is None:
        return indices
    if isinstance(classes, torch.Tensor):
        classes_t = classes.to(device=scores.device)
    else:
        classes_t = torch.as_tensor(classes, device=scores.device)
    classes_t = classes_t.reshape(-1)
    if int(classes_t.shape[0]) != int(scores.shape[1]):
        raise InductiveValidationError(
            "TriTraining classifier classes must align with score columns."
        )
    return classes_t[indices]


def _cap_numpy_candidates(
    indices: np.ndarray,
    scores_j: np.ndarray,
    scores_k: np.ndarray,
    *,
    max_new_labels: int | None,
) -> np.ndarray:
    if max_new_labels is None or int(indices.size) <= int(max_new_labels):
        return indices
    limit = max(0, int(max_new_labels))
    if limit == 0:
        return np.empty((0,), dtype=np.int64)
    confidence = (scores_j[indices].max(axis=1) + scores_k[indices].max(axis=1)) / 2.0
    order = np.lexsort((indices, -confidence))
    return indices[order[:limit]]


def _cap_torch_candidates(
    indices: Any,
    scores_j: Any,
    scores_k: Any,
    *,
    max_new_labels: int | None,
) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if max_new_labels is None or int(indices.numel()) <= int(max_new_labels):
        return indices
    limit = max(0, int(max_new_labels))
    if limit == 0:
        return torch.empty((0,), dtype=torch.long, device=indices.device)
    confidence = (scores_j[indices].max(dim=1).values + scores_k[indices].max(dim=1).values) / 2.0
    indices_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    confidence_np = confidence.detach().cpu().numpy()
    order = np.lexsort((indices_np, -confidence_np))[:limit]
    order_t = torch.as_tensor(order, dtype=torch.long, device=indices.device)
    return indices[order_t]


@dataclass(frozen=True)
class TriTrainingSpec(BaseClassifierSpec):
    max_iter: int = 20
    confidence_threshold: float | None = None
    max_new_labels: int | None = None
    bootstrap_ratio: float = 1.0
    retain_initial_ensemble: bool = field(default=False, kw_only=True)
    prediction_rule: Literal["score_average", "soft_average", "majority_vote"] = field(
        default="score_average",
        kw_only=True,
    )
    training_mode: str = field(default="legacy", kw_only=True)


class TriTrainingMethod(InductiveMethod):
    """Tri-training with three classifiers (CPU/GPU)."""

    info = MethodInfo(
        method_id="tri_training",
        name="Tri-Training",
        year=2005,
        family="classic",
        supports_gpu=True,
        paper_title="Tri-Training: Exploiting Unlabeled Data Using Three Classifiers",
        paper_pdf="https://www.lamda.nju.edu.cn/publication/tkde05.pdf",
        official_code="https://www.lamda.nju.edu.cn/files/TriTrain.rar",
    )

    def __init__(self, spec: TriTrainingSpec | None = None) -> None:
        self.spec = spec or TriTrainingSpec()
        self._clfs: list[Any] = []
        self._initial_clfs: list[Any] = []
        self._backend: str | None = None
        self.n_iter_: int = 0
        self.changed_rounds_: int = 0
        self.converged_: bool = False
        self.updates_per_learner_: tuple[int, ...] = (0, 0, 0)
        self.pseudo_labels_selected_per_learner_: tuple[int, ...] = (0, 0, 0)
        self.subsample_events_per_learner_: tuple[int, ...] = (0, 0, 0)
        self.last_errors_: tuple[float, ...] = (0.5, 0.5, 0.5)
        self.last_pseudo_label_sizes_: tuple[int, ...] = (0, 0, 0)
        self.diagnostics_: dict[str, Any] = {}
        self._round_history: list[dict[str, Any]] = []

    def _reset_diagnostics(self) -> None:
        self.n_iter_ = 0
        self.changed_rounds_ = 0
        self.converged_ = False
        self.updates_per_learner_ = (0, 0, 0)
        self.pseudo_labels_selected_per_learner_ = (0, 0, 0)
        self.subsample_events_per_learner_ = (0, 0, 0)
        self.last_errors_ = (0.5, 0.5, 0.5)
        self.last_pseudo_label_sizes_ = (0, 0, 0)
        self.diagnostics_ = {}
        self._round_history = []
        self._initial_clfs = []
        # Fitted class-order attributes must not leak across repeated fits.  They
        # are deliberately created by ``fit`` (rather than ``__init__``), like
        # scikit-learn's ``classes_`` attributes.
        self.__dict__.pop("classes_", None)
        self.__dict__.pop("initial_classes_", None)

    def _require_probability_ensemble(self, classifiers: list[Any]) -> None:
        if self.spec.prediction_rule != "soft_average":
            return
        unsupported = [
            index
            for index, classifier in enumerate(classifiers)
            if not classifier_capabilities(classifier).probabilities
        ]
        if unsupported:
            raise InductiveValidationError(
                "TriTraining prediction_rule='soft_average' requires native class "
                f"probabilities from every classifier; unsupported learners: {unsupported}. "
                "Use prediction_rule='majority_vote' for predict-only backends."
            )

    def _finalize_diagnostics(
        self,
        *,
        previous_errors: Any,
        previous_sizes: Any,
        updates: list[int],
        selected_counts: list[int],
        subsample_counts: list[int],
    ) -> None:
        self.updates_per_learner_ = tuple(int(value) for value in updates)
        self.pseudo_labels_selected_per_learner_ = tuple(int(value) for value in selected_counts)
        self.subsample_events_per_learner_ = tuple(int(value) for value in subsample_counts)
        self.last_errors_ = tuple(float(value) for value in previous_errors)
        self.last_pseudo_label_sizes_ = tuple(int(value) for value in previous_sizes)
        self.diagnostics_ = {
            "n_iter": int(self.n_iter_),
            "changed_rounds": int(self.changed_rounds_),
            "converged": bool(self.converged_),
            "initial_ensemble_retained": bool(self._initial_clfs),
            "prediction_rule": str(self.spec.prediction_rule),
            "updates_per_learner": list(self.updates_per_learner_),
            "pseudo_labels_selected_per_learner": list(self.pseudo_labels_selected_per_learner_),
            "pseudo_labels_selected_total": int(sum(self.pseudo_labels_selected_per_learner_)),
            "subsample_events_per_learner": list(self.subsample_events_per_learner_),
            "last_errors": list(self.last_errors_),
            "last_pseudo_label_sizes": list(self.last_pseudo_label_sizes_),
            "rounds": self._round_history,
        }
        logger.info(
            "Tri-training diagnostics rounds=%s changed_rounds=%s updates=%s "
            "pseudo_labels_selected=%s subsamples=%s converged=%s",
            self.n_iter_,
            self.changed_rounds_,
            self.updates_per_learner_,
            self.pseudo_labels_selected_per_learner_,
            self.subsample_events_per_learner_,
            self.converged_,
        )

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> TriTrainingMethod:
        if self.spec.training_mode not in _TRAINING_MODES:
            raise InductiveValidationError(
                f"training_mode must be one of {sorted(_TRAINING_MODES)!r}."
            )
        if self.spec.training_mode == "legacy":
            if self.spec.retain_initial_ensemble or self.spec.prediction_rule != "score_average":
                raise InductiveValidationError(
                    "retain_initial_ensemble and non-default prediction rules are available "
                    "only with training_mode='error_bound_subsample'."
                )
            self._reset_diagnostics()
            return fit_standardized_tri_training(
                self,
                data,
                device=device,
                seed=int(seed),
            )
        return self._fit_error_bound_subsample(data, device=device, seed=seed)

    def _fit_error_bound_subsample(
        self,
        data: Any,
        *,
        device: DeviceSpec,
        seed: int,
    ) -> TriTrainingMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        self._reset_diagnostics()
        if self.spec.prediction_rule not in {"score_average", "soft_average", "majority_vote"}:
            raise InductiveValidationError(
                "prediction_rule must be 'score_average', 'soft_average', or 'majority_vote'."
            )
        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")

            if ds.X_u is None:
                raise InductiveValidationError("TriTraining requires X_u (unlabeled data).")

            X_l = np.asarray(ds.X_l)
            X_u = np.asarray(ds.X_u)
            y_l = np.asarray(y_l)
            logger.info(
                "Tri-training sizes: n_labeled=%s n_unlabeled=%s",
                int(X_l.shape[0]),
                int(X_u.shape[0]),
            )

            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            # Flatten features if >2D for standard classifiers
            X_l = flatten_if_numpy(X_l)
            X_u = flatten_if_numpy(X_u)

            rng = np.random.default_rng(int(seed))
            n_l = int(X_l.shape[0])
            n_boot = max(1, int(round(float(self.spec.bootstrap_ratio) * n_l)))

            clfs = [build_classifier(self.spec, seed=seed + i) for i in range(3)]
            boot_idx = [rng.choice(n_l, size=n_boot, replace=True) for _ in range(3)]
            for i in range(3):
                clfs[i].fit(X_l[boot_idx[i]], y_l[boot_idx[i]])
            if bool(self.spec.retain_initial_ensemble):
                self._initial_clfs = [build_classifier(self.spec, seed=seed + i) for i in range(3)]
                for i in range(3):
                    self._initial_clfs[i].fit(X_l[boot_idx[i]], y_l[boot_idx[i]])

            previous_errors = np.full((3,), 0.5, dtype=np.float64)
            previous_sizes = np.zeros((3,), dtype=np.int64)
            updates = [0, 0, 0]
            selected_counts = [0, 0, 0]
            subsample_counts = [0, 0, 0]

            for round_idx in range(max(0, int(self.spec.max_iter))):
                pending: list[tuple[np.ndarray, np.ndarray, float] | None] = [None, None, None]
                learner_records: list[dict[str, Any]] = []

                for i in range(3):
                    j, k = [learner for learner in range(3) if learner != i]
                    scores_j_l = predict_scores(clfs[j], X_l, backend=backend)
                    scores_k_l = predict_scores(clfs[k], X_l, backend=backend)
                    pred_j_l = _numpy_labels_from_scores(clfs[j], scores_j_l)
                    pred_k_l = _numpy_labels_from_scores(clfs[k], scores_k_l)
                    estimate = _measure_error(pred_j_l, pred_k_l, y_l)

                    previous_error = float(previous_errors[i])
                    raw_candidate_size = 0
                    candidate_size = 0
                    decision = _paper_update_decision(
                        error=estimate.rate,
                        previous_error=previous_error,
                        previous_size=int(previous_sizes[i]),
                        candidate_size=0,
                    )

                    if estimate.rate < previous_error:
                        scores_j_u = predict_scores(clfs[j], X_u, backend=backend)
                        scores_k_u = predict_scores(clfs[k], X_u, backend=backend)
                        pred_j_u = _numpy_labels_from_scores(clfs[j], scores_j_u)
                        pred_k_u = _numpy_labels_from_scores(clfs[k], scores_k_u)
                        agree = pred_j_u == pred_k_u
                        if self.spec.confidence_threshold is not None:
                            threshold = float(self.spec.confidence_threshold)
                            agree &= scores_j_u.max(axis=1) >= threshold
                            agree &= scores_k_u.max(axis=1) >= threshold

                        candidate_idx = np.flatnonzero(agree).astype(np.int64, copy=False)
                        raw_candidate_size = int(candidate_idx.size)
                        candidate_idx = _cap_numpy_candidates(
                            candidate_idx,
                            scores_j_u,
                            scores_k_u,
                            max_new_labels=self.spec.max_new_labels,
                        )
                        candidate_size = int(candidate_idx.size)
                        decision = _paper_update_decision(
                            error=estimate.rate,
                            previous_error=previous_error,
                            previous_size=int(previous_sizes[i]),
                            candidate_size=candidate_size,
                        )
                        previous_sizes[i] = decision.previous_size

                        if decision.accepted:
                            selected_idx = candidate_idx
                            if decision.subsample:
                                positions = _subsample_positions(
                                    candidate_size,
                                    decision.selected_size,
                                    rng=rng,
                                )
                                selected_idx = candidate_idx[positions]
                                subsample_counts[i] += 1
                            selected_labels = pred_j_u[selected_idx]
                            pending[i] = (selected_idx, selected_labels, estimate.rate)

                    learner_records.append(
                        {
                            "learner": int(i),
                            "error": float(estimate.rate),
                            "previous_error": previous_error,
                            "labeled_agreements": int(estimate.agreements),
                            "wrong_labeled_agreements": int(estimate.wrong_agreements),
                            "agreement_candidates": raw_candidate_size,
                            "candidates_after_cap": candidate_size,
                            "previous_size": int(decision.previous_size),
                            "selected_size": int(decision.selected_size),
                            "accepted": bool(decision.accepted),
                            "subsampled": bool(decision.subsample),
                            "reason": decision.reason,
                        }
                    )

                changed = False
                for i, update in enumerate(pending):
                    if update is None:
                        continue
                    selected_idx, selected_labels, error = update
                    X_train = np.concatenate([X_l, X_u[selected_idx]], axis=0)
                    y_train = np.concatenate([y_l, selected_labels.astype(y_l.dtype)], axis=0)
                    clfs[i].fit(X_train, y_train)
                    previous_errors[i] = error
                    previous_sizes[i] = int(selected_idx.size)
                    updates[i] += 1
                    selected_counts[i] += int(selected_idx.size)
                    changed = True

                self.n_iter_ += 1
                self._round_history.append(
                    {
                        "round": int(round_idx),
                        "changed": bool(changed),
                        "learners": learner_records,
                    }
                )
                if not changed:
                    self.converged_ = True
                    break
                self.changed_rounds_ += 1

            self._finalize_diagnostics(
                previous_errors=previous_errors,
                previous_sizes=previous_sizes,
                updates=updates,
                selected_counts=selected_counts,
                subsample_counts=subsample_counts,
            )

            self._clfs = clfs
            self._backend = backend
            self.classes_ = _global_class_order(y_l, clfs, self._initial_clfs)
            if self._initial_clfs:
                self.initial_classes_ = self.classes_.copy()
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u is None:
            raise InductiveValidationError("TriTraining requires X_u (unlabeled data).")

        X_l = ds.X_l
        X_u = ds.X_u
        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        logger.info(
            "Tri-training sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u)),
        )

        n_l = int(get_torch_len(X_l))
        n_boot = max(1, int(round(float(self.spec.bootstrap_ratio) * n_l)))
        data_device = get_torch_device(X_l)
        rng = np.random.default_rng(int(seed))

        clfs = [build_classifier(self.spec, seed=seed + i) for i in range(3)]
        boot_idx = [
            torch.as_tensor(
                rng.integers(0, n_l, size=n_boot),
                dtype=torch.long,
                device=data_device,
            )
            for _ in range(3)
        ]
        for i in range(3):
            clfs[i].fit(slice_data(X_l, boot_idx[i]), y_l[boot_idx[i]])
        if bool(self.spec.retain_initial_ensemble):
            self._initial_clfs = [build_classifier(self.spec, seed=seed + i) for i in range(3)]
            for i in range(3):
                self._initial_clfs[i].fit(slice_data(X_l, boot_idx[i]), y_l[boot_idx[i]])

        previous_errors = np.full((3,), 0.5, dtype=np.float64)
        previous_sizes = np.zeros((3,), dtype=np.int64)
        updates = [0, 0, 0]
        selected_counts = [0, 0, 0]
        subsample_counts = [0, 0, 0]

        for round_idx in range(max(0, int(self.spec.max_iter))):
            pending: list[tuple[Any, Any, float] | None] = [None, None, None]
            learner_records = []

            for i in range(3):
                j, k = [learner for learner in range(3) if learner != i]
                scores_j_l = predict_scores(clfs[j], X_l, backend=backend)
                scores_k_l = predict_scores(clfs[k], X_l, backend=backend)
                pred_j_l = _torch_labels_from_scores(clfs[j], scores_j_l)
                pred_k_l = _torch_labels_from_scores(clfs[k], scores_k_l)
                estimate = _measure_error(pred_j_l, pred_k_l, y_l)

                previous_error = float(previous_errors[i])
                raw_candidate_size = 0
                candidate_size = 0
                decision = _paper_update_decision(
                    error=estimate.rate,
                    previous_error=previous_error,
                    previous_size=int(previous_sizes[i]),
                    candidate_size=0,
                )

                if estimate.rate < previous_error:
                    scores_j_u = predict_scores(clfs[j], X_u, backend=backend)
                    scores_k_u = predict_scores(clfs[k], X_u, backend=backend)
                    pred_j_u = _torch_labels_from_scores(clfs[j], scores_j_u)
                    pred_k_u = _torch_labels_from_scores(clfs[k], scores_k_u)
                    agree = pred_j_u == pred_k_u
                    if self.spec.confidence_threshold is not None:
                        threshold = float(self.spec.confidence_threshold)
                        agree &= scores_j_u.max(dim=1).values >= threshold
                        agree &= scores_k_u.max(dim=1).values >= threshold

                    candidate_idx = agree.nonzero(as_tuple=False).reshape(-1)
                    raw_candidate_size = int(candidate_idx.numel())
                    candidate_idx = _cap_torch_candidates(
                        candidate_idx,
                        scores_j_u,
                        scores_k_u,
                        max_new_labels=self.spec.max_new_labels,
                    )
                    candidate_size = int(candidate_idx.numel())
                    decision = _paper_update_decision(
                        error=estimate.rate,
                        previous_error=previous_error,
                        previous_size=int(previous_sizes[i]),
                        candidate_size=candidate_size,
                    )
                    previous_sizes[i] = decision.previous_size

                    if decision.accepted:
                        selected_idx = candidate_idx
                        if decision.subsample:
                            positions = _subsample_positions(
                                candidate_size,
                                decision.selected_size,
                                rng=rng,
                            )
                            positions_t = torch.as_tensor(
                                positions,
                                dtype=torch.long,
                                device=data_device,
                            )
                            selected_idx = candidate_idx[positions_t]
                            subsample_counts[i] += 1
                        selected_labels = pred_j_u[selected_idx]
                        pending[i] = (selected_idx, selected_labels, estimate.rate)

                learner_records.append(
                    {
                        "learner": int(i),
                        "error": float(estimate.rate),
                        "previous_error": previous_error,
                        "labeled_agreements": int(estimate.agreements),
                        "wrong_labeled_agreements": int(estimate.wrong_agreements),
                        "agreement_candidates": raw_candidate_size,
                        "candidates_after_cap": candidate_size,
                        "previous_size": int(decision.previous_size),
                        "selected_size": int(decision.selected_size),
                        "accepted": bool(decision.accepted),
                        "subsampled": bool(decision.subsample),
                        "reason": decision.reason,
                    }
                )

            changed = False
            for i, update in enumerate(pending):
                if update is None:
                    continue
                selected_idx, selected_labels, error = update
                X_train = concat_data([X_l, slice_data(X_u, selected_idx)])
                y_train = torch.cat(
                    [y_l, selected_labels.to(dtype=y_l.dtype, device=y_l.device)],
                    dim=0,
                )
                clfs[i].fit(X_train, y_train)
                previous_errors[i] = error
                previous_sizes[i] = int(selected_idx.numel())
                updates[i] += 1
                selected_counts[i] += int(selected_idx.numel())
                changed = True

            self.n_iter_ += 1
            self._round_history.append(
                {
                    "round": int(round_idx),
                    "changed": bool(changed),
                    "learners": learner_records,
                }
            )
            if not changed:
                self.converged_ = True
                break
            self.changed_rounds_ += 1

        self._finalize_diagnostics(
            previous_errors=previous_errors,
            previous_sizes=previous_sizes,
            updates=updates,
            selected_counts=selected_counts,
            subsample_counts=subsample_counts,
        )

        self._clfs = clfs
        self._backend = backend
        self.classes_ = _global_class_order(y_l, clfs, self._initial_clfs)
        if self._initial_clfs:
            self.initial_classes_ = self.classes_.copy()
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def _predict_proba_with_classes(
        self,
        X: Any,
        classifiers: list[Any],
        *,
        preferred_classes: Any | None,
        require_probabilities: bool = False,
    ) -> tuple[Any, np.ndarray | None]:
        """Average an ensemble and return the exact order of its score columns."""

        if not classifiers:
            raise RuntimeError("TriTrainingMethod is not fitted yet. Call fit() first.")
        backend = detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")

        if backend == "numpy":
            X = flatten_if_numpy(X)

        if require_probabilities:
            self._require_probability_ensemble(classifiers)
            scores_list = []
            for classifier in classifiers:
                scores = classifier.predict_proba(X)
                if backend == "numpy":
                    scores = np.asarray(scores)
                else:
                    torch = optional_import("torch", extra="inductive-torch")
                    if not isinstance(scores, torch.Tensor):
                        raise InductiveValidationError(
                            "Torch classifier predict_proba() must return a torch.Tensor."
                        )
                    x_device = X["x"].device if isinstance(X, dict) else X.device
                    if scores.device != x_device:
                        raise InductiveValidationError(
                            "Torch classifier returned probabilities on a different device."
                        )
                if scores.ndim != 2:
                    raise InductiveValidationError(
                        "predict_proba must return shape (n_samples, n_classes)."
                    )
                scores_list.append(scores)
        else:
            scores_list = [predict_scores(clf, X, backend=backend) for clf in classifiers]

        # Robustly align scores if shapes differ
        shapes = [s.shape[1] for s in scores_list]
        distinct_shapes = set(shapes)

        def _resolve_score_classes(clf: Any, scores: Any) -> np.ndarray | None:
            classes = getattr(clf, "classes_", None)
            if classes is None:
                classes = getattr(clf, "classes_t_", None)
            if classes is None:
                return None
            if hasattr(classes, "detach"):
                classes_arr = classes.detach().cpu().numpy()
            else:
                classes_arr = np.asarray(classes)
            if classes_arr.ndim != 1 or int(classes_arr.shape[0]) != int(scores.shape[1]):
                raise InductiveValidationError(
                    "TriTraining classifiers disagree on class counts "
                    f"{shapes}, and one classifier exposes incompatible class labels "
                    f"with shape {tuple(classes_arr.shape)} for {scores.shape[1]} score columns."
                )
            return classes_arr

        class_labels = [
            _resolve_score_classes(clf, scores)
            for clf, scores in zip(classifiers, scores_list, strict=True)
        ]
        labels_available = [classes is not None for classes in class_labels]

        class_order: np.ndarray | None = None
        if preferred_classes is not None:
            class_order = _as_numpy_labels(preferred_classes)
        elif all(labels_available):
            class_order = np.unique(
                np.concatenate([classes for classes in class_labels if classes is not None])
            )

        if class_order is None:
            if len(distinct_shapes) > 1:
                raise InductiveValidationError(
                    f"TriTraining classifiers disagree on class counts {shapes}, "
                    "and not all classifiers expose non-null, score-aligned class labels "
                    "via 'classes_' or 'classes_t_' to allow alignment. "
                    "Cannot safely merge predictions."
                )
        else:
            if class_order.size == 0 or np.unique(class_order).size != class_order.size:
                raise InductiveValidationError(
                    "TriTraining global class order must contain distinct class labels."
                )
            global_map = {label: idx for idx, label in enumerate(class_order.tolist())}
            final_n_classes = int(class_order.size)

            # Common case: every classifier already uses the fitted global order.
            already_aligned = all(int(score.shape[1]) == final_n_classes for score in scores_list)
            already_aligned = already_aligned and all(
                classes is None or np.array_equal(classes, class_order) for classes in class_labels
            )
            if already_aligned:
                if backend == "numpy":
                    avg = np.mean(np.stack(scores_list, axis=0), axis=0)
                    row_sum = avg.sum(axis=1, keepdims=True)
                    row_sum[row_sum == 0.0] = 1.0
                    return (avg / row_sum).astype(np.float32, copy=False), class_order
                torch = optional_import("torch", extra="inductive-torch")
                avg = torch.mean(torch.stack(scores_list, dim=0), dim=0)
                row_sum = avg.sum(dim=1, keepdim=True)
                row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
                return avg / row_sum, class_order

            aligned_scores = []
            if backend == "numpy":
                for classes, s in zip(class_labels, scores_list, strict=False):
                    if classes is None:
                        if int(s.shape[1]) != final_n_classes:
                            raise InductiveValidationError(
                                f"TriTraining classifiers disagree on class counts {shapes}, "
                                "and a classifier without class labels cannot be aligned to "
                                f"the global {final_n_classes}-class order."
                            )
                        aligned_scores.append(s)
                        continue
                    target = np.zeros((s.shape[0], final_n_classes), dtype=s.dtype)
                    # Map local columns to global columns
                    for local_idx, cls_label in enumerate(classes):
                        try:
                            global_idx = global_map[cls_label]
                        except KeyError as exc:
                            raise InductiveValidationError(
                                f"TriTraining classifier exposes class {cls_label!r} outside "
                                "the fitted global class order."
                            ) from exc
                        target[:, global_idx] = s[:, local_idx]
                    aligned_scores.append(target)

                avg = np.mean(np.stack(aligned_scores, axis=0), axis=0)
            else:
                torch = optional_import("torch", extra="inductive-torch")
                for classes, s in zip(class_labels, scores_list, strict=False):
                    if classes is None:
                        if int(s.shape[1]) != final_n_classes:
                            raise InductiveValidationError(
                                f"TriTraining classifiers disagree on class counts {shapes}, "
                                "and a classifier without class labels cannot be aligned to "
                                f"the global {final_n_classes}-class order."
                            )
                        aligned_scores.append(s)
                        continue
                    target = torch.zeros(
                        (s.shape[0], final_n_classes), dtype=s.dtype, device=s.device
                    )
                    for local_idx, cls_label in enumerate(classes):
                        val = cls_label.item() if hasattr(cls_label, "item") else cls_label
                        try:
                            global_idx = global_map[val]
                        except KeyError as exc:
                            raise InductiveValidationError(
                                f"TriTraining classifier exposes class {val!r} outside "
                                "the fitted global class order."
                            ) from exc
                        target[:, global_idx] = s[:, local_idx]
                    aligned_scores.append(target)

                avg = torch.mean(torch.stack(aligned_scores, dim=0), dim=0)

            # Normalize row sums
            if backend == "numpy":
                row_sum = avg.sum(axis=1, keepdims=True)
                row_sum[row_sum == 0.0] = 1.0
                return (avg / row_sum).astype(np.float32, copy=False), class_order
            else:
                row_sum = avg.sum(dim=1, keepdim=True)
                row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
                return avg / row_sum, class_order

        # Fast path if shapes match
        if backend == "numpy":
            avg = np.mean(np.stack(scores_list, axis=0), axis=0)
            row_sum = avg.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (avg / row_sum).astype(np.float32, copy=False), None
        else:
            torch = optional_import("torch", extra="inductive-torch")
            avg = torch.mean(torch.stack(scores_list, dim=0), dim=0)
            row_sum = avg.sum(dim=1, keepdim=True)
            row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
            return avg / row_sum, None

    def _predict_vote_proba_with_classes(
        self,
        X: Any,
        classifiers: list[Any],
        *,
        preferred_classes: Any | None,
    ) -> tuple[Any, np.ndarray]:
        """Return hard-vote proportions for Table I's final majority vote."""

        if not classifiers:
            raise RuntimeError("TriTrainingMethod is not fitted yet. Call fit() first.")
        backend = detect_backend(X)
        aligned = [
            self._predict_proba_with_classes(
                X,
                [classifier],
                preferred_classes=preferred_classes,
            )
            for classifier in classifiers
        ]
        class_order = aligned[0][1]
        if class_order is None:
            class_order = np.arange(int(aligned[0][0].shape[1]), dtype=np.int64)
        if any(
            classes is not None and not np.array_equal(classes, class_order)
            for _scores, classes in aligned
        ):
            raise InductiveValidationError(
                "TriTraining classifiers disagree on the global class order."
            )

        if backend == "numpy":
            hard_votes = np.stack(
                [scores.argmax(axis=1) for scores, _classes in aligned],
                axis=0,
            )
            one_hot = np.eye(int(class_order.size), dtype=np.float32)[hard_votes]
            return one_hot.mean(axis=0), class_order

        torch = optional_import("torch", extra="inductive-torch")
        hard_votes = torch.stack(
            [scores.argmax(dim=1) for scores, _classes in aligned],
            dim=0,
        )
        one_hot = torch.nn.functional.one_hot(
            hard_votes,
            num_classes=int(class_order.size),
        ).to(dtype=aligned[0][0].dtype)
        return one_hot.mean(dim=0), class_order

    def _prediction_with_classes(
        self,
        X: Any,
        classifiers: list[Any],
        *,
        preferred_classes: Any | None,
    ) -> tuple[Any, np.ndarray | None]:
        if self.spec.prediction_rule == "majority_vote":
            return self._predict_vote_proba_with_classes(
                X,
                classifiers,
                preferred_classes=preferred_classes,
            )
        if self.spec.prediction_rule == "score_average":
            return self._predict_proba_with_classes(
                X,
                classifiers,
                preferred_classes=preferred_classes,
            )
        if self.spec.prediction_rule != "soft_average":
            raise InductiveValidationError(
                "prediction_rule must be 'score_average', 'soft_average', or 'majority_vote'."
            )
        return self._predict_proba_with_classes(
            X,
            classifiers,
            preferred_classes=preferred_classes,
            require_probabilities=True,
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        proba, _ = self._prediction_with_classes(
            X,
            self._clfs,
            preferred_classes=getattr(self, "classes_", None),
        )
        return proba

    def predict_proba_initial(self, X: Any) -> np.ndarray:
        """Predict with the retained round-zero ensemble."""

        if not self._initial_clfs:
            raise RuntimeError(
                "TriTrainingMethod did not retain its initial ensemble; set "
                "retain_initial_ensemble=true before fit()."
            )
        proba, _ = self._prediction_with_classes(
            X,
            self._initial_clfs,
            preferred_classes=getattr(
                self,
                "initial_classes_",
                getattr(self, "classes_", None),
            ),
        )
        return proba

    @staticmethod
    def _labels_from_proba(proba: Any, classes: np.ndarray | None, *, backend: str) -> Any:
        if backend == "numpy":
            idx = proba.argmax(axis=1)
            return idx if classes is None else classes[idx]
        idx = proba.argmax(dim=1)
        if classes is None:
            return idx
        torch = optional_import("torch", extra="inductive-torch")
        classes_t = torch.as_tensor(classes, device=proba.device)
        return classes_t[idx]

    def predict_initial(self, X: Any) -> np.ndarray:
        """Predict labels with the retained round-0 ensemble."""

        if not self._initial_clfs:
            raise RuntimeError(
                "TriTrainingMethod did not retain its initial ensemble; set "
                "retain_initial_ensemble=true before fit()."
            )
        proba, classes = self._prediction_with_classes(
            X,
            self._initial_clfs,
            preferred_classes=getattr(
                self,
                "initial_classes_",
                getattr(self, "classes_", None),
            ),
        )
        return self._labels_from_proba(proba, classes, backend=detect_backend(X))

    def predict(self, X: Any) -> np.ndarray:
        proba, classes = self._prediction_with_classes(
            X,
            self._clfs,
            preferred_classes=getattr(self, "classes_", None),
        )
        backend = detect_backend(X)
        return self._labels_from_proba(proba, classes, backend=backend)
