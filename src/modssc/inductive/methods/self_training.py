from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
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
    predict_scores,
    unwrap_torch_x,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)

_GROUP_KEYS_U = (
    "group_u",
    "groups_u",
    "group_ids_u",
    "u_group_ids",
    "u_groups",
    "discourse_u",
    "discourse_ids_u",
    "u_discourse_ids",
    "group_ids",
    "groups",
    "discourse_ids",
    "discourse",
)
_GROUP_KEYS_L = (
    "group_l",
    "groups_l",
    "group_ids_l",
    "l_group_ids",
    "l_groups",
    "discourse_l",
    "discourse_ids_l",
    "l_discourse_ids",
)


@dataclass(frozen=True)
class SelfTrainingSpec(BaseClassifierSpec):
    """Configuration for generic and historical self-training.

    ``selection_strategy="li_zhou_2005_1nn_distance"`` implements the
    Self-training baseline described in the SETRED experiments.  The paper
    specifies 1-NN, one most-confident example per class, and at most 40
    iterations.  It does *not* publish the numerical distance-confidence
    formula, feature scaling, or the unlabeled-pool size.  Those reproduction
    assumptions are therefore explicit in fields whose names end in
    ``_unspecified``.

    The historical profile must set ``classifier_params={"k": 1}``,
    ``max_iter=40``, and ``confidence_threshold=None``.  Defaults intentionally
    retain ModSSC's pre-existing generic behavior.
    """

    max_iter: int = 10
    confidence_threshold: float | None = 0.95
    max_new_labels: int | None = None
    min_new_labels: int = 1
    use_group_propagation: bool | None = None
    group_key: str | None = None
    group_min_count: int = 2
    group_min_fraction: float = 1.0
    group_confidence_threshold: float | None = None
    selection_strategy: Literal["classifier_confidence", "li_zhou_2005_1nn_distance"] = field(
        default="classifier_confidence", kw_only=True
    )
    paper_pool_size_unspecified: int | None = field(default=None, kw_only=True)
    paper_candidates_per_class_unspecified: int | Mapping[int, int] = field(default=1, kw_only=True)
    paper_distance_confidence_unspecified: Literal[
        "margin", "ratio", "nearest_neighbor_distance"
    ] = field(default="margin", kw_only=True)
    paper_feature_scaling_unspecified: Literal["external", "dynamic_labeled_minmax"] = field(
        default="external", kw_only=True
    )


@dataclass(frozen=True)
class SelfTrainingRoundTrace:
    """Immutable audit record for one attempted self-training round.

    Pool and candidate indices refer to the original ordering of ``X_u`` even
    after accepted examples have been removed from the working pool.
    """

    iteration: int
    labeled_before: int
    unlabeled_before: int
    pool_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    candidate_labels: tuple[int, ...]
    accepted_indices: tuple[int, ...]
    accepted_labels: tuple[int, ...]
    labeled_after: int
    remaining_unlabeled: int


def _validate_paper_selection_spec(spec: SelfTrainingSpec) -> None:
    if spec.classifier_id != "knn":
        raise InductiveValidationError("li_zhou_2005_1nn_distance requires classifier_id='knn'.")
    classifier_k = spec.classifier_params.get("k", 5)
    if (
        isinstance(classifier_k, bool)
        or not isinstance(classifier_k, (int, np.integer))
        or int(classifier_k) != 1
    ):
        raise InductiveValidationError("li_zhou_2005_1nn_distance requires classifier_params.k=1.")
    if str(spec.classifier_params.get("metric", "euclidean")) != "euclidean":
        raise InductiveValidationError("li_zhou_2005_1nn_distance requires the Euclidean metric.")
    if spec.confidence_threshold is not None:
        raise InductiveValidationError(
            "li_zhou_2005_1nn_distance requires confidence_threshold=None."
        )
    if spec.max_new_labels is not None:
        raise InductiveValidationError(
            "li_zhou_2005_1nn_distance uses per-class quotas; max_new_labels must be None."
        )
    if spec.use_group_propagation is True:
        raise InductiveValidationError(
            "li_zhou_2005_1nn_distance is incompatible with group propagation."
        )
    if spec.paper_pool_size_unspecified is not None:
        pool_size = spec.paper_pool_size_unspecified
        if (
            isinstance(pool_size, bool)
            or not isinstance(pool_size, (int, np.integer))
            or int(pool_size) <= 0
        ):
            raise InductiveValidationError("paper_pool_size_unspecified must be >= 1 or None.")
    quotas = spec.paper_candidates_per_class_unspecified
    if isinstance(quotas, Mapping):
        quota_values = list(quotas.values())
        if not quota_values:
            raise InductiveValidationError(
                "paper_candidates_per_class_unspecified must not be empty."
            )
    else:
        quota_values = [quotas]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0
        for value in quota_values
    ):
        raise InductiveValidationError(
            "paper_candidates_per_class_unspecified values must be non-negative integers."
        )
    if not any(int(value) > 0 for value in quota_values):
        raise InductiveValidationError(
            "paper_candidates_per_class_unspecified must contain a positive quota."
        )
    if spec.paper_distance_confidence_unspecified not in {
        "margin",
        "ratio",
        "nearest_neighbor_distance",
    }:
        raise InductiveValidationError(
            "paper_distance_confidence_unspecified must be 'margin', 'ratio', or "
            "'nearest_neighbor_distance'."
        )
    if spec.paper_feature_scaling_unspecified not in {
        "external",
        "dynamic_labeled_minmax",
    }:
        raise InductiveValidationError(
            "paper_feature_scaling_unspecified must be 'external' or 'dynamic_labeled_minmax'."
        )


def _paper_quota_for_label(value: int | Mapping[int, int], label: int) -> int:
    if isinstance(value, Mapping):
        return int(value.get(int(label), 0))
    return int(value)


def _paper_pool_ids(
    remaining_ids: np.ndarray,
    *,
    previous_pool_ids: np.ndarray | None,
    pool_size: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep a seeded fixed-size pool and replenish only vacated positions."""

    remaining = np.asarray(remaining_ids, dtype=np.int64)
    if pool_size is None or int(pool_size) >= int(remaining.size):
        return remaining.copy()
    size = int(pool_size)
    if previous_pool_ids is None:
        return np.asarray(rng.choice(remaining, size=size, replace=False), dtype=np.int64)
    retained = previous_pool_ids[np.isin(previous_pool_ids, remaining)]
    missing = size - int(retained.size)
    if missing <= 0:
        return retained[:size].copy()
    available = remaining[~np.isin(remaining, retained)]
    replenished = np.asarray(
        rng.choice(available, size=min(missing, int(available.size)), replace=False),
        dtype=np.int64,
    )
    return np.concatenate([retained, replenished])


def _dynamic_labeled_minmax_parameters_numpy(
    X_l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit per-feature min-max parameters on the current labeled set only.

    A feature that is constant in the current labeled set contributes zero to
    every distance.  This matches the historical normalizable-distance
    convention and avoids allowing an unlabeled query to define the scale.
    """

    values = np.asarray(X_l, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise InductiveValidationError(
            "dynamic_labeled_minmax requires a non-empty 2D labeled matrix."
        )
    minimum = values.min(axis=0)
    width = values.max(axis=0) - minimum
    varying = width > 0.0
    denominator = np.where(varying, width, 1.0)
    return minimum, denominator, varying


def _apply_dynamic_labeled_minmax_numpy(
    X: np.ndarray,
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    minimum, denominator, varying = parameters
    if values.ndim != 2 or values.shape[1] != minimum.shape[0]:
        raise InductiveValidationError(
            "dynamic_labeled_minmax input must be 2D with the fitted feature width."
        )
    transformed = (values - minimum) / denominator
    transformed[:, ~varying] = 0.0
    return transformed


def _dynamic_labeled_minmax_parameters_torch(X_l: Any) -> tuple[Any, Any, Any]:
    torch = optional_import("torch", extra="inductive-torch")
    if X_l.ndim != 2 or int(X_l.shape[0]) == 0:
        raise InductiveValidationError(
            "dynamic_labeled_minmax requires a non-empty 2D labeled matrix."
        )
    minimum = X_l.amin(dim=0)
    width = X_l.amax(dim=0) - minimum
    varying = width > 0
    denominator = torch.where(varying, width, torch.ones_like(width))
    return minimum, denominator, varying


def _apply_dynamic_labeled_minmax_torch(X: Any, parameters: tuple[Any, Any, Any]) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    minimum, denominator, varying = parameters
    if X.ndim != 2 or int(X.shape[1]) != int(minimum.shape[0]):
        raise InductiveValidationError(
            "dynamic_labeled_minmax input must be 2D with the fitted feature width."
        )
    transformed = (X - minimum) / denominator
    return torch.where(varying.unsqueeze(0), transformed, torch.zeros_like(transformed))


def _select_li_zhou_2005_1nn_candidates_numpy(
    X_l: np.ndarray,
    y_l: np.ndarray,
    X_pool: np.ndarray,
    pool_ids: np.ndarray,
    *,
    per_class_unspecified: int | Mapping[int, int],
    distance_confidence_unspecified: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Select paper-style 1-NN candidates using an explicit confidence assumption.

    Li and Zhou only describe confidence as being near labeled examples of the
    predicted class and far from other classes. ``margin`` instantiates that as
    ``d_other - d_same``; ``ratio`` uses ``d_other / d_same``.  The confirmation
    reconstruction ``nearest_neighbor_distance`` ranks the closest predicted
    neighbor first.  None of the numerical formulas is claimed to have been
    published by the authors.
    """

    X_train = np.asarray(X_l, dtype=np.float64)
    X_query = np.asarray(X_pool, dtype=np.float64)
    labels = np.asarray(y_l)
    original_ids = np.asarray(pool_ids, dtype=np.int64)
    if X_train.ndim != 2 or X_query.ndim != 2:
        raise InductiveValidationError("li_zhou_2005_1nn_distance requires 2D feature matrices.")
    if X_train.shape[1] != X_query.shape[1]:
        raise InductiveValidationError("Labeled and unlabeled features must have the same width.")
    if X_query.shape[0] != original_ids.shape[0]:
        raise InductiveValidationError("pool_ids must match the unlabeled pool size.")
    classes = np.unique(labels)
    if classes.size < 2:
        raise InductiveValidationError(
            "li_zhou_2005_1nn_distance requires at least two labeled classes."
        )
    if X_query.shape[0] == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=labels.dtype)

    differences = X_query[:, None, :] - X_train[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    class_distances = np.column_stack(
        [distances[:, labels == class_label].min(axis=1) for class_label in classes]
    )
    predicted_positions = class_distances.argmin(axis=1)
    predicted = classes[predicted_positions]
    same_distance = class_distances[np.arange(X_query.shape[0]), predicted_positions]
    other_distances = class_distances.copy()
    other_distances[np.arange(X_query.shape[0]), predicted_positions] = np.inf
    other_distance = other_distances.min(axis=1)
    if distance_confidence_unspecified == "margin":
        confidence = other_distance - same_distance
    elif distance_confidence_unspecified == "ratio":
        confidence = other_distance / np.maximum(same_distance, 1e-12)
    elif distance_confidence_unspecified == "nearest_neighbor_distance":
        confidence = -same_distance
    else:
        raise InductiveValidationError(
            "paper_distance_confidence_unspecified must be 'margin', 'ratio', or "
            "'nearest_neighbor_distance'."
        )

    selected: list[int] = []
    selected_labels: list[int] = []
    for class_label in classes.tolist():
        quota = _paper_quota_for_label(per_class_unspecified, int(class_label))
        if quota <= 0:
            continue
        eligible = np.where(predicted == class_label)[0]
        if eligible.size == 0:
            continue
        if distance_confidence_unspecified == "nearest_neighbor_distance":
            order = np.lexsort((original_ids[eligible], same_distance[eligible]))
        else:
            order = np.lexsort(
                (
                    original_ids[eligible],
                    same_distance[eligible],
                    -other_distance[eligible],
                    -confidence[eligible],
                )
            )
        take = eligible[order[:quota]]
        selected.extend(int(index) for index in take.tolist())
        selected_labels.extend([int(class_label)] * int(take.size))
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(selected_labels, dtype=labels.dtype),
    )


def _normalize_group_ids_numpy(value: Any, *, n_expected: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise InductiveValidationError(f"{name} must be 1D group ids.")
    if arr.shape[0] != n_expected:
        raise InductiveValidationError(
            f"{name} must have {n_expected} entries, got {arr.shape[0]}."
        )
    return arr


def _normalize_group_ids_torch(value: Any, *, n_expected: int, name: str):
    torch = optional_import("torch", extra="inductive-torch")
    if not isinstance(value, torch.Tensor):
        raise InductiveValidationError(f"{name} must be a torch.Tensor.")
    if value.ndim != 1:
        raise InductiveValidationError(f"{name} must be 1D group ids.")
    if int(value.shape[0]) != n_expected:
        raise InductiveValidationError(
            f"{name} must have {n_expected} entries, got {int(value.shape[0])}."
        )
    if value.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise InductiveValidationError(f"{name} must have an integer dtype.")
    return value


def _resolve_group_ids(
    meta: Mapping[str, Any] | None,
    *,
    group_key: str | None,
    n_expected: int,
    backend: str,
    name: str,
    key_candidates: tuple[str, ...],
) -> Any | None:
    if meta is None:
        return None
    if not isinstance(meta, Mapping):
        raise InductiveValidationError("meta must be a mapping when provided.")
    if group_key is not None:
        if group_key not in meta:
            raise InductiveValidationError(f"meta is missing key {group_key!r}.")
        value = meta[group_key]
        if backend == "numpy":
            return _normalize_group_ids_numpy(value, n_expected=n_expected, name=name)
        return _normalize_group_ids_torch(value, n_expected=n_expected, name=name)
    for key in key_candidates:
        if key not in meta:
            continue
        value = meta[key]
        try:
            if backend == "numpy":
                return _normalize_group_ids_numpy(value, n_expected=n_expected, name=name)
            return _normalize_group_ids_torch(value, n_expected=n_expected, name=name)
        except InductiveValidationError:
            continue
    return None


def _select_candidates_numpy(
    scores: np.ndarray,
    pred: np.ndarray,
    *,
    threshold: float | None,
    max_new: int | None,
    use_group: bool,
    group_u: np.ndarray | None,
    group_l: np.ndarray | None,
    y_l: np.ndarray | None,
    group_min_count: int,
    group_min_fraction: float,
    group_conf_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    conf = scores.max(axis=1)
    if threshold is None:
        direct_mask = np.ones(conf.shape[0], dtype=bool)
    else:
        direct_mask = conf >= float(threshold)
    direct_idx = np.where(direct_mask)[0]

    candidates: dict[int, tuple[Any, float]] = {
        int(i): (pred[int(i)], float(conf[int(i)])) for i in direct_idx
    }
    group_added = 0

    if use_group and group_u is not None:
        if group_conf_threshold is None:
            group_conf_threshold = threshold
        if group_conf_threshold is None:
            group_conf_threshold = -np.inf

        for gid in np.unique(group_u):
            group_idx = np.where(group_u == gid)[0]
            if group_idx.size == 0:
                continue

            votes: list[Any] = []
            vote_conf: list[float] = []

            if group_l is not None and y_l is not None:
                group_l_idx = np.where(group_l == gid)[0]
                if group_l_idx.size:
                    votes.extend(y_l[group_l_idx].tolist())
                    vote_conf.extend([1.0] * int(group_l_idx.size))

            group_direct_idx = group_idx[direct_mask[group_idx]]
            if group_direct_idx.size:
                votes.extend(pred[group_direct_idx].tolist())
                vote_conf.extend(conf[group_direct_idx].tolist())

            if len(votes) < int(group_min_count):
                continue

            labels, counts = np.unique(np.asarray(votes, dtype=object), return_counts=True)
            major_pos = int(counts.argmax())
            major_label = labels[major_pos]
            fraction = float(counts[major_pos]) / float(counts.sum())
            if fraction < float(group_min_fraction):
                continue

            major_conf = float(
                np.mean([c for v, c in zip(votes, vote_conf, strict=False) if v == major_label])
            )
            if major_conf < float(group_conf_threshold):
                continue

            for idx in group_idx.tolist():
                if int(idx) in candidates:
                    continue
                candidates[int(idx)] = (major_label, major_conf)
                group_added += 1

    items = list(candidates.items())
    if not items:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=object), int(direct_idx.size), 0

    items.sort(key=lambda item: item[1][1], reverse=True)
    if max_new is not None and len(items) > int(max_new):
        items = items[: int(max_new)]

    idx = np.asarray([i for i, _ in items], dtype=np.int64)
    labels = np.asarray([label for _, (label, _) in items], dtype=object)
    return idx, labels, int(direct_idx.size), group_added


class SelfTrainingMethod(InductiveMethod):
    """Self-training with group propagation or SETRED-paper 1-NN selection."""

    info = MethodInfo(
        method_id="self_training",
        name="Self-Training",
        year=1995,
        family="classic",
        supports_gpu=True,
        paper_title="Unsupervised Word Sense Disambiguation Rivaling Supervised Methods",
        paper_pdf="docs/article_code/inductive/1995-Self Training/4-1995-Unsupervised word sense disambiguation rivaling supervised methods.pdf",
        official_code=None,
    )

    def __init__(self, spec: SelfTrainingSpec | None = None) -> None:
        self.spec = spec or SelfTrainingSpec()
        self._clf: Any | None = None
        self._backend: str | None = None
        self._paper_minmax_parameters: tuple[Any, Any, Any] | None = None
        self._round_trace: tuple[SelfTrainingRoundTrace, ...] = ()
        self.diagnostics_: dict[str, Any] = {}

    @property
    def round_trace_(self) -> tuple[SelfTrainingRoundTrace, ...]:
        """Per-round audit trace, exposed as an immutable tuple."""

        return self._round_trace

    def _fit_classifier(
        self,
        classifier: Any,
        X_l: Any,
        y_l: Any,
        *,
        backend: str,
    ) -> Any:
        """Fit one round, optionally refreshing the labeled-only min-max state."""

        if self.spec.paper_feature_scaling_unspecified != "dynamic_labeled_minmax":
            self._paper_minmax_parameters = None
            classifier.fit(X_l, y_l)
            return X_l
        if backend == "numpy":
            parameters = _dynamic_labeled_minmax_parameters_numpy(X_l)
            transformed = _apply_dynamic_labeled_minmax_numpy(X_l, parameters)
        else:
            features = unwrap_torch_x(X_l)
            parameters = _dynamic_labeled_minmax_parameters_torch(features)
            transformed = _apply_dynamic_labeled_minmax_torch(features, parameters)
        self._paper_minmax_parameters = parameters
        classifier.fit(transformed, y_l)
        return transformed

    def _transform_with_current_paper_scaling(self, X: Any, *, backend: str) -> Any:
        parameters = self._paper_minmax_parameters
        if parameters is None:
            return X
        if backend == "numpy":
            return _apply_dynamic_labeled_minmax_numpy(X, parameters)
        return _apply_dynamic_labeled_minmax_torch(unwrap_torch_x(X), parameters)

    def _finalize_paper_diagnostics(
        self,
        *,
        seed: int,
        initial_labeled_size: int,
        initial_unlabeled_count: int,
        final_labeled_size: int,
        remaining_unlabeled_count: int,
    ) -> None:
        if self.spec.selection_strategy != "li_zhou_2005_1nn_distance":
            return

        quotas = self.spec.paper_candidates_per_class_unspecified
        if isinstance(quotas, Mapping):
            serialized_quotas: int | dict[str, int] = {
                str(label): int(quota)
                for label, quota in sorted(quotas.items(), key=lambda item: str(item[0]))
            }
        else:
            serialized_quotas = int(quotas)

        round_trace = []
        for trace in self._round_trace:
            trace_payload = asdict(trace)
            round_trace.append(
                {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in trace_payload.items()
                }
            )

        selection_parameters = {
            "selection_strategy": self.spec.selection_strategy,
            "classifier_id": self.spec.classifier_id,
            "classifier_k": int(self.spec.classifier_params.get("k", 1)),
            "classifier_metric": str(self.spec.classifier_params.get("metric", "euclidean")),
            "max_iter": int(self.spec.max_iter),
            "confidence_threshold": self.spec.confidence_threshold,
            "max_new_labels": self.spec.max_new_labels,
            "min_new_labels": int(self.spec.min_new_labels),
            "use_group_propagation": bool(self.spec.use_group_propagation),
            "paper_pool_size_unspecified": (
                None
                if self.spec.paper_pool_size_unspecified is None
                else int(self.spec.paper_pool_size_unspecified)
            ),
            "paper_candidates_per_class_unspecified": serialized_quotas,
            "paper_distance_confidence_unspecified": (
                self.spec.paper_distance_confidence_unspecified
            ),
        }
        if self.spec.paper_feature_scaling_unspecified != "external":
            selection_parameters["paper_feature_scaling_unspecified"] = (
                self.spec.paper_feature_scaling_unspecified
            )

        self.diagnostics_ = {
            "protocol": "li_zhou_2005_1nn_distance",
            "seed": int(seed),
            "n_iter": len(round_trace),
            "initial_labeled_size": int(initial_labeled_size),
            "initial_unlabeled_count": int(initial_unlabeled_count),
            "final_labeled_size": int(final_labeled_size),
            "remaining_unlabeled_count": int(remaining_unlabeled_count),
            "pseudo_labels_added": sum(len(trace.accepted_indices) for trace in self._round_trace),
            "selection_parameters": selection_parameters,
            "round_trace": round_trace,
        }

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> SelfTrainingMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        self._clf = None
        self._backend = None
        self._round_trace = ()
        self.diagnostics_ = {}
        self._paper_minmax_parameters = None

        if self.spec.group_min_count < 1:
            raise InductiveValidationError("group_min_count must be >= 1.")
        if not (0.0 <= float(self.spec.group_min_fraction) <= 1.0):
            raise InductiveValidationError("group_min_fraction must be in [0, 1].")
        if self.spec.selection_strategy not in {
            "classifier_confidence",
            "li_zhou_2005_1nn_distance",
        }:
            raise InductiveValidationError(
                f"Unknown selection_strategy: {self.spec.selection_strategy!r}."
            )
        paper_selection = self.spec.selection_strategy == "li_zhou_2005_1nn_distance"
        if paper_selection:
            _validate_paper_selection_spec(self.spec)
        elif self.spec.paper_feature_scaling_unspecified != "external":
            raise InductiveValidationError(
                "paper_feature_scaling_unspecified is only valid with "
                "selection_strategy='li_zhou_2005_1nn_distance'."
            )
        paper_rng = np.random.default_rng(int(seed))

        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")
            initial_labeled_size = int(y_l.shape[0])

            if ds.X_u is None or np.asarray(ds.X_u).size == 0:
                clf = build_classifier(self.spec, seed=seed)
                self._fit_classifier(clf, ds.X_l, y_l, backend=backend)
                self._clf = clf
                self._backend = backend
                self._finalize_paper_diagnostics(
                    seed=seed,
                    initial_labeled_size=initial_labeled_size,
                    initial_unlabeled_count=0,
                    final_labeled_size=initial_labeled_size,
                    remaining_unlabeled_count=0,
                )
                logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
                return self

            X_l = np.asarray(ds.X_l)
            X_u = np.asarray(ds.X_u)
            y_l = np.asarray(y_l)
            initial_unlabeled_count = int(X_u.shape[0])
            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            group_u = _resolve_group_ids(
                ds.meta,
                group_key=self.spec.group_key,
                n_expected=int(X_u.shape[0]),
                backend=backend,
                name="meta[group_u]",
                key_candidates=_GROUP_KEYS_U,
            )
            group_l = _resolve_group_ids(
                ds.meta,
                group_key=None,
                n_expected=int(X_l.shape[0]),
                backend=backend,
                name="meta[group_l]",
                key_candidates=_GROUP_KEYS_L,
            )

            use_group = self.spec.use_group_propagation
            if use_group is None:
                use_group = False if paper_selection else group_u is not None
            if use_group and group_u is None:
                raise InductiveValidationError(
                    "SelfTraining requires meta group ids for group propagation."
                )

            clf = build_classifier(self.spec, seed=seed)
            X_u_curr = X_u
            group_u_curr = group_u
            u_ids_curr = np.arange(int(X_u.shape[0]), dtype=np.int64)
            paper_pool_ids: np.ndarray | None = None
            iter_count = 0

            while iter_count < int(self.spec.max_iter):
                X_l_classifier = self._fit_classifier(clf, X_l, y_l, backend=backend)
                if X_u_curr.shape[0] == 0:
                    break

                if paper_selection:
                    paper_pool_ids = _paper_pool_ids(
                        u_ids_curr,
                        previous_pool_ids=paper_pool_ids,
                        pool_size=self.spec.paper_pool_size_unspecified,
                        rng=paper_rng,
                    )
                    pool_positions = np.searchsorted(u_ids_curr, paper_pool_ids)
                    X_pool = X_u_curr[pool_positions]
                    X_pool_classifier = self._transform_with_current_paper_scaling(
                        X_pool,
                        backend=backend,
                    )
                    pool_idx, labels = _select_li_zhou_2005_1nn_candidates_numpy(
                        X_l_classifier,
                        y_l,
                        X_pool_classifier,
                        paper_pool_ids,
                        per_class_unspecified=(self.spec.paper_candidates_per_class_unspecified),
                        distance_confidence_unspecified=(
                            self.spec.paper_distance_confidence_unspecified
                        ),
                    )
                    idx = pool_positions[pool_idx]
                    direct_count = int(idx.size)
                    group_added = 0
                    trace_pool_ids = paper_pool_ids
                else:
                    scores = predict_scores(clf, X_u_curr, backend=backend)
                    pred = clf.predict(X_u_curr)
                    idx, labels, direct_count, group_added = _select_candidates_numpy(
                        scores,
                        pred,
                        threshold=self.spec.confidence_threshold,
                        max_new=self.spec.max_new_labels,
                        use_group=bool(use_group),
                        group_u=group_u_curr,
                        group_l=group_l,
                        y_l=y_l,
                        group_min_count=self.spec.group_min_count,
                        group_min_fraction=self.spec.group_min_fraction,
                        group_conf_threshold=self.spec.group_confidence_threshold,
                    )
                    trace_pool_ids = u_ids_curr
                labels = labels.astype(y_l.dtype, copy=False)

                logger.debug(
                    "Self-training iter=%s direct=%s group_added=%s total_new=%s remaining=%s",
                    iter_count,
                    int(direct_count),
                    int(group_added),
                    int(idx.size),
                    int(X_u_curr.shape[0]),
                )
                accepted = int(idx.size) >= int(self.spec.min_new_labels)
                candidate_ids = u_ids_curr[idx]
                accepted_ids = candidate_ids if accepted else np.empty((0,), dtype=np.int64)
                accepted_labels = labels if accepted else np.empty((0,), dtype=y_l.dtype)
                self._round_trace = (
                    *self._round_trace,
                    SelfTrainingRoundTrace(
                        iteration=int(iter_count),
                        labeled_before=int(X_l.shape[0]),
                        unlabeled_before=int(X_u_curr.shape[0]),
                        pool_indices=tuple(int(value) for value in trace_pool_ids.tolist()),
                        candidate_indices=tuple(int(value) for value in candidate_ids.tolist()),
                        candidate_labels=tuple(int(value) for value in labels.tolist()),
                        accepted_indices=tuple(int(value) for value in accepted_ids.tolist()),
                        accepted_labels=tuple(int(value) for value in accepted_labels.tolist()),
                        labeled_after=int(X_l.shape[0]) + int(accepted_ids.size),
                        remaining_unlabeled=(int(X_u_curr.shape[0]) - int(accepted_ids.size)),
                    ),
                )
                if not accepted:
                    break

                X_l = np.concatenate([X_l, X_u_curr[idx]], axis=0)
                y_l = np.concatenate([y_l, labels], axis=0)

                keep = np.ones((X_u_curr.shape[0],), dtype=bool)
                keep[idx] = False
                X_u_curr = X_u_curr[keep]
                u_ids_curr = u_ids_curr[keep]
                if group_u_curr is not None:
                    group_u_curr = group_u_curr[keep]

                iter_count += 1

            self._fit_classifier(clf, X_l, y_l, backend=backend)
            self._clf = clf
            self._backend = backend
            self._finalize_paper_diagnostics(
                seed=seed,
                initial_labeled_size=initial_labeled_size,
                initial_unlabeled_count=initial_unlabeled_count,
                final_labeled_size=int(X_l.shape[0]),
                remaining_unlabeled_count=int(X_u_curr.shape[0]),
            )
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")
        initial_labeled_size = int(get_torch_len(y_l))

        if ds.X_u is None or int(get_torch_len(ds.X_u)) == 0:
            clf = build_classifier(self.spec, seed=seed)
            self._fit_classifier(clf, ds.X_l, y_l, backend=backend)
            self._clf = clf
            self._backend = backend
            self._finalize_paper_diagnostics(
                seed=seed,
                initial_labeled_size=initial_labeled_size,
                initial_unlabeled_count=0,
                final_labeled_size=initial_labeled_size,
                remaining_unlabeled_count=0,
            )
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        X_l = ds.X_l
        X_u = ds.X_u
        initial_unlabeled_count = int(get_torch_len(X_u))
        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")

        group_u_t = _resolve_group_ids(
            ds.meta,
            group_key=self.spec.group_key,
            n_expected=int(get_torch_len(X_u)),
            backend=backend,
            name="meta[group_u]",
            key_candidates=_GROUP_KEYS_U,
        )
        group_l_t = _resolve_group_ids(
            ds.meta,
            group_key=None,
            n_expected=int(get_torch_len(X_l)),
            backend=backend,
            name="meta[group_l]",
            key_candidates=_GROUP_KEYS_L,
        )

        use_group = self.spec.use_group_propagation
        if use_group is None:
            use_group = False if paper_selection else group_u_t is not None
        if use_group and group_u_t is None:
            raise InductiveValidationError(
                "SelfTraining requires meta group ids for group propagation."
            )

        group_u_curr = group_u_t.detach().cpu().numpy() if group_u_t is not None else None
        group_l = group_l_t.detach().cpu().numpy() if group_l_t is not None else None

        clf = build_classifier(self.spec, seed=seed)
        X_u_curr = X_u
        u_ids_curr = np.arange(int(get_torch_len(X_u)), dtype=np.int64)
        paper_pool_ids = None
        iter_count = 0

        while iter_count < int(self.spec.max_iter):
            X_l_classifier = self._fit_classifier(clf, X_l, y_l, backend=backend)
            if int(get_torch_len(X_u_curr)) == 0:
                break

            y_l_np = y_l.detach().cpu().numpy()
            if paper_selection:
                paper_pool_ids = _paper_pool_ids(
                    u_ids_curr,
                    previous_pool_ids=paper_pool_ids,
                    pool_size=self.spec.paper_pool_size_unspecified,
                    rng=paper_rng,
                )
                pool_positions = np.searchsorted(u_ids_curr, paper_pool_ids)
                pool_positions_t = torch.tensor(
                    pool_positions,
                    dtype=torch.long,
                    device=get_torch_device(X_u_curr),
                )
                X_pool = slice_data(X_u_curr, pool_positions_t)
                X_l_features = unwrap_torch_x(X_l_classifier)
                X_pool_features = self._transform_with_current_paper_scaling(
                    X_pool,
                    backend=backend,
                )
                pool_idx, labels_np = _select_li_zhou_2005_1nn_candidates_numpy(
                    X_l_features.detach().cpu().numpy(),
                    y_l_np,
                    X_pool_features.detach().cpu().numpy(),
                    paper_pool_ids,
                    per_class_unspecified=(self.spec.paper_candidates_per_class_unspecified),
                    distance_confidence_unspecified=(
                        self.spec.paper_distance_confidence_unspecified
                    ),
                )
                idx_np = pool_positions[pool_idx]
                direct_count = int(idx_np.size)
                group_added = 0
                trace_pool_ids = paper_pool_ids
            else:
                scores = predict_scores(clf, X_u_curr, backend=backend)
                pred = clf.predict(X_u_curr)
                scores_np = scores.detach().cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                idx_np, labels_np, direct_count, group_added = _select_candidates_numpy(
                    scores_np,
                    pred_np,
                    threshold=self.spec.confidence_threshold,
                    max_new=self.spec.max_new_labels,
                    use_group=bool(use_group),
                    group_u=group_u_curr,
                    group_l=group_l,
                    y_l=y_l_np,
                    group_min_count=self.spec.group_min_count,
                    group_min_fraction=self.spec.group_min_fraction,
                    group_conf_threshold=self.spec.group_confidence_threshold,
                )
                trace_pool_ids = u_ids_curr

            logger.debug(
                "Self-training iter=%s direct=%s group_added=%s total_new=%s remaining=%s",
                iter_count,
                int(direct_count),
                int(group_added),
                int(idx_np.size),
                int(get_torch_len(X_u_curr)),
            )
            labels_np = labels_np.astype(y_l_np.dtype, copy=False)
            accepted = int(idx_np.size) >= int(self.spec.min_new_labels)
            candidate_ids = u_ids_curr[idx_np]
            accepted_ids = candidate_ids if accepted else np.empty((0,), dtype=np.int64)
            accepted_labels_np = labels_np if accepted else np.empty((0,), dtype=y_l_np.dtype)
            self._round_trace = (
                *self._round_trace,
                SelfTrainingRoundTrace(
                    iteration=int(iter_count),
                    labeled_before=int(get_torch_len(X_l)),
                    unlabeled_before=int(get_torch_len(X_u_curr)),
                    pool_indices=tuple(int(value) for value in trace_pool_ids.tolist()),
                    candidate_indices=tuple(int(value) for value in candidate_ids.tolist()),
                    candidate_labels=tuple(int(value) for value in labels_np.tolist()),
                    accepted_indices=tuple(int(value) for value in accepted_ids.tolist()),
                    accepted_labels=tuple(int(value) for value in accepted_labels_np.tolist()),
                    labeled_after=int(get_torch_len(X_l)) + int(accepted_ids.size),
                    remaining_unlabeled=(int(get_torch_len(X_u_curr)) - int(accepted_ids.size)),
                ),
            )
            if not accepted:
                break

            idx = torch.tensor(idx_np, dtype=torch.long, device=get_torch_device(X_u_curr))
            labels = torch.tensor(labels_np, dtype=y_l.dtype, device=get_torch_device(X_u_curr))

            X_l = concat_data([X_l, slice_data(X_u_curr, idx)])
            y_l = torch.cat([y_l, labels], dim=0)

            mask = torch.ones(
                (int(get_torch_len(X_u_curr)),),
                dtype=torch.bool,
                device=get_torch_device(X_u_curr),
            )
            mask[idx] = False
            X_u_curr = slice_data(X_u_curr, mask)
            u_ids_curr = u_ids_curr[mask.detach().cpu().numpy()]
            if group_u_curr is not None:
                group_u_curr = group_u_curr[mask.detach().cpu().numpy()]

            iter_count += 1

        self._fit_classifier(clf, X_l, y_l, backend=backend)
        self._clf = clf
        self._backend = backend
        self._finalize_paper_diagnostics(
            seed=seed,
            initial_labeled_size=initial_labeled_size,
            initial_unlabeled_count=initial_unlabeled_count,
            final_labeled_size=int(get_torch_len(X_l)),
            remaining_unlabeled_count=int(get_torch_len(X_u_curr)),
        )
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("SelfTrainingMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")
        X_classifier = self._transform_with_current_paper_scaling(X, backend=backend)
        scores = predict_scores(self._clf, X_classifier, backend=backend)
        if backend == "numpy":
            row_sum = scores.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)
        torch = optional_import("torch", extra="inductive-torch")
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def predict(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("SelfTrainingMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict input backend mismatch.")
        X_classifier = self._transform_with_current_paper_scaling(X, backend=backend)
        return self._clf.predict(X_classifier)
