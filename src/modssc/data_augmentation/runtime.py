from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.data_loader.selection import select_rows

from .api import AugmentationStrategy, build_strategy
from .errors import DataAugmentationValidationError
from .plan import AugmentationPlan, parse_augmentation_plan
from .registry import get_online_augmenter
from .types import AugmentationContext
from .utils import is_torch_tensor

__all__ = [
    "OnlineAugmentation",
    "UnlabeledAugmentationResult",
    "build_online_augmentation",
    "materialize_views",
    "prepare_unlabeled_augmentation",
    "validate_augmentation_regime",
]

PlanLike = AugmentationPlan | Mapping[str, Any]


def validate_augmentation_regime(*, regime: str, configured: bool) -> None:
    """Reject configured augmentation for regimes that cannot consume it.

    The distinction between ``configured=False`` and a missing materialized
    result is intentional: a configuration adapter must report that the user
    requested augmentation even when it has not attempted materialization yet.
    This prevents transductive augmentation blocks from becoming silent no-ops.
    """

    if regime not in {"inductive", "transductive"}:
        raise DataAugmentationValidationError(
            "augmentation regime must be 'inductive' or 'transductive'"
        )
    if bool(configured) and regime != "inductive":
        raise DataAugmentationValidationError(
            "configured augmentation cannot be delivered to a transductive method"
        )


@dataclass(frozen=True)
class UnlabeledAugmentationResult:
    """Materialized method inputs for one unlabeled population."""

    weak: Any | None
    strong: Any | None
    second_strong: Any | None
    online: Any | None
    sample_ids: np.ndarray


def _index_list(indices: Any) -> list[int]:
    """Return device-independent integer indices without changing their order."""

    if is_torch_tensor(indices):
        return [int(value) for value in indices.detach().cpu().reshape(-1).tolist()]
    return [int(value) for value in np.asarray(indices, dtype=np.int64).reshape(-1).tolist()]


def _stack_like(reference: Any, values: list[Any]) -> Any:
    if not values:
        return reference[:0]
    if is_torch_tensor(reference):
        import importlib

        torch = importlib.import_module("torch")
        return torch.stack(values, dim=0)
    return np.stack(values, axis=0)


@dataclass(frozen=True)
class OnlineAugmentation:
    """Recompute deterministic weak/strong views for every optimization step.

    Randomness is keyed by ``(seed, optimization_step, absolute_sample_id)``.
    Consequently a resumed task obtains the same view for the same logical step,
    independently of the order in which samples are materialized.
    """

    strategy: AugmentationStrategy
    seed: int
    modality: str | None = None

    def _context(self, *, sample_id: int, step: int, view: int = 0) -> AugmentationContext:
        if int(step) < 0:
            raise ValueError("step must be >= 0")
        return AugmentationContext(
            seed=int(self.seed) + int(view),
            sample_id=int(sample_id),
            epoch=int(step),
            modality=self.modality,
        )

    def weak_batch(self, X: Any, *, indices: Any, sample_ids: Any, step: int) -> Any:
        local = _index_list(indices)
        absolute = _index_list(sample_ids)
        if len(local) != len(absolute):
            raise ValueError("indices and sample_ids must have the same length")
        values = [
            self.strategy.weak.apply(
                X[index],
                ctx=self._context(sample_id=sample_id, step=int(step)),
            )
            for index, sample_id in zip(local, absolute, strict=True)
        ]
        return _stack_like(X, values)

    def pair_batch(
        self,
        X: Any,
        *,
        indices: Any,
        sample_ids: Any,
        step: int,
    ) -> tuple[Any, Any]:
        local = _index_list(indices)
        absolute = _index_list(sample_ids)
        if len(local) != len(absolute):
            raise ValueError("indices and sample_ids must have the same length")
        weak: list[Any] = []
        strong: list[Any] = []
        for index, sample_id in zip(local, absolute, strict=True):
            xw, xs = self.strategy.apply(
                X[index],
                ctx=self._context(sample_id=sample_id, step=int(step)),
            )
            weak.append(xw)
            strong.append(xs)
        return _stack_like(X, weak), _stack_like(X, strong)


def build_online_augmentation(
    *,
    weak_plan: PlanLike,
    strong_plan: PlanLike,
    seed: int,
    modality: str | None,
    online_augmenter_id: str | None = None,
    online_augmenter_params: Mapping[str, Any] | None = None,
) -> Any:
    """Build a native online weak/strong augmenter from a declarative plan.

    A registered runtime can be selected by ``online_augmenter_id``.  Otherwise
    the generic deterministic runtime is compiled from the two plans.
    """

    if modality == "graph":
        raise DataAugmentationValidationError(
            "Online per-sample augmentation is not supported for graph inputs"
        )
    if online_augmenter_params is not None and not isinstance(online_augmenter_params, Mapping):
        raise DataAugmentationValidationError("online augmenter params must be a mapping")
    params = dict(online_augmenter_params or {})
    if "seed" in params:
        raise DataAugmentationValidationError("online augmenter params must not redefine seed")
    if online_augmenter_id is not None:
        return get_online_augmenter(
            online_augmenter_id,
            modality=modality,  # type: ignore[arg-type]
            seed=int(seed),
            **params,
        )
    if params:
        raise DataAugmentationValidationError("online augmenter params require online_augmenter_id")
    weak = parse_augmentation_plan(weak_plan, modality=modality)  # type: ignore[arg-type]
    strong = parse_augmentation_plan(strong_plan, modality=modality)  # type: ignore[arg-type]
    return OnlineAugmentation(
        strategy=build_strategy(weak=weak, strong=strong),
        seed=int(seed),
        modality=modality,
    )


def _is_graph_like_sample(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "x" in value and "edge_index" in value
    return hasattr(value, "x") and hasattr(value, "edge_index")


def _sample_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[0])
    try:
        return len(value)
    except TypeError as exc:
        raise DataAugmentationValidationError(
            "unlabeled augmentation input must be an indexable batch"
        ) from exc


def _sample_id_values(sample_ids: Any, *, n_samples: int) -> list[int]:
    if sample_ids is None:
        return list(range(n_samples))
    if is_torch_tensor(sample_ids):
        values = sample_ids.detach().cpu().reshape(-1).tolist()
    else:
        values = np.asarray(sample_ids, dtype=np.int64).reshape(-1).tolist()
    if len(values) != n_samples:
        raise DataAugmentationValidationError(
            "sample_ids must contain one stable id per unlabeled sample"
        )
    return [int(value) for value in values]


def _copy_batch_item(value: Any) -> Any:
    if is_torch_tensor(value):
        return value.clone()
    return value.copy()


class _ViewCollector:
    """Preallocate fixed-shape tensor/array views and fall back to stacking."""

    def __init__(self, n_samples: int) -> None:
        self._n_samples = int(n_samples)
        self._buffer: Any | None = None
        self._values: list[Any] = []
        self._size = 0

    @staticmethod
    def _allocate(first: Any, n_samples: int) -> Any | None:
        if is_torch_tensor(first):
            return first.new_empty((n_samples,) + tuple(first.shape))
        if isinstance(first, np.ndarray):
            return np.empty((n_samples,) + first.shape, dtype=first.dtype)
        return None

    @staticmethod
    def _compatible(buffer: Any, value: Any) -> bool:
        if is_torch_tensor(buffer):
            return is_torch_tensor(value) and tuple(value.shape) == tuple(buffer.shape[1:])
        return isinstance(value, np.ndarray) and tuple(value.shape) == tuple(buffer.shape[1:])

    def _fall_back_to_values(self) -> None:
        assert self._buffer is not None
        self._values = [_copy_batch_item(self._buffer[index]) for index in range(self._size)]
        self._buffer = None

    def append(self, value: Any) -> None:
        if self._size == 0:
            self._buffer = self._allocate(value, self._n_samples)
            if self._buffer is not None:
                self._buffer[0] = value
            else:
                self._values.append(value)
            self._size = 1
            return

        if self._buffer is not None:
            if self._compatible(self._buffer, value):
                self._buffer[self._size] = value
                self._size += 1
                return
            self._fall_back_to_values()
        self._values.append(value)
        self._size += 1

    def finish(self) -> Any:
        if self._buffer is not None:
            return self._buffer
        if self._values and all(is_torch_tensor(value) for value in self._values):
            import importlib

            torch = importlib.import_module("torch")
            try:
                return torch.stack(self._values, dim=0)
            except RuntimeError:
                return self._values
        try:
            return np.stack(self._values, axis=0)
        except (TypeError, ValueError):
            return self._values


def materialize_views(
    X_u: Any,
    *,
    weak_plan: PlanLike,
    strong_plan: PlanLike,
    seed: int,
    mode: str = "fixed",
    modality: str | None = None,
    sample_ids: Any | None = None,
    strong_views: int = 1,
) -> tuple[Any, Any, Any | None]:
    """Materialize deterministic weak and one or two strong unlabeled views.

    Graph-like objects are transformed as a single logical sample.  Ordinary
    batches use stable absolute ``sample_ids`` and preallocate fixed-shape NumPy
    or torch outputs before falling back to stacking for dynamic shapes.
    """

    if mode != "fixed":
        raise DataAugmentationValidationError("Only augmentation.mode='fixed' is supported")
    if int(strong_views) not in {1, 2}:
        raise DataAugmentationValidationError("strong_views must be 1 or 2")
    if X_u is None:
        return None, None, None

    weak = parse_augmentation_plan(weak_plan, modality=modality)  # type: ignore[arg-type]
    strong = parse_augmentation_plan(strong_plan, modality=modality)  # type: ignore[arg-type]
    strategy = build_strategy(weak=weak, strong=strong)

    if modality == "graph" and _is_graph_like_sample(X_u):
        if sample_ids is None:
            sample_id = 0
        elif is_torch_tensor(sample_ids):
            ids = sample_ids.detach().cpu().reshape(-1).tolist()
            sample_id = 0 if not ids else int(ids[0])
        else:
            ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1).tolist()
            sample_id = 0 if not ids else int(ids[0])
        context = AugmentationContext(
            seed=int(seed),
            sample_id=sample_id,
            epoch=0,
            modality=modality,  # type: ignore[arg-type]
        )
        weak_view, strong_view = strategy.apply(X_u, ctx=context)
        second_strong = None
        if int(strong_views) == 2:
            second_strong = strategy.strong.apply(
                X_u,
                ctx=AugmentationContext(
                    seed=int(seed) + 1,
                    sample_id=sample_id,
                    epoch=0,
                    modality=modality,  # type: ignore[arg-type]
                ),
            )
        return weak_view, strong_view, second_strong

    n_samples = _sample_count(X_u)
    if n_samples == 0:
        return X_u, X_u, X_u if int(strong_views) == 2 else None
    absolute_ids = _sample_id_values(sample_ids, n_samples=n_samples)

    weak_output = _ViewCollector(n_samples)
    strong_output = _ViewCollector(n_samples)
    second_strong_output = _ViewCollector(n_samples) if int(strong_views) == 2 else None

    for index, sample_id in enumerate(absolute_ids):
        sample = X_u[index]
        context = AugmentationContext(
            seed=int(seed),
            sample_id=sample_id,
            epoch=0,
            modality=modality,  # type: ignore[arg-type]
        )
        weak_view, strong_view = strategy.apply(sample, ctx=context)
        weak_output.append(weak_view)
        strong_output.append(strong_view)
        if second_strong_output is not None:
            second_strong_output.append(
                strategy.strong.apply(
                    sample,
                    ctx=AugmentationContext(
                        seed=int(seed) + 1,
                        sample_id=sample_id,
                        epoch=0,
                        modality=modality,  # type: ignore[arg-type]
                    ),
                )
            )

    return (
        weak_output.finish(),
        strong_output.finish(),
        None if second_strong_output is None else second_strong_output.finish(),
    )


def prepare_unlabeled_augmentation(
    primary_input: Any,
    *,
    unlabeled_indices: Any,
    weak_plan: PlanLike,
    strong_plan: PlanLike,
    seed: int,
    mode: str,
    modality: str | None,
    strong_views: int = 1,
    online_augmenter_id: str | None = None,
    online_augmenter_params: Mapping[str, Any] | None = None,
) -> UnlabeledAugmentationResult:
    """Select the unlabeled pool and build all augmentation method inputs.

    Structured graph-like feature mappings retain their topology and metadata;
    only their ``x`` field is passed through non-graph augmentations.
    """

    sample_ids = np.asarray(unlabeled_indices, dtype=np.int64).reshape(-1)
    selected = select_rows(
        primary_input,
        sample_ids,
        context="data_augmentation.unlabeled",
    )
    wrapped = modality != "graph" and isinstance(selected, Mapping) and "x" in selected
    augmentation_input = selected["x"] if wrapped else selected

    online = None
    if mode == "online":
        if int(strong_views) != 1:
            raise DataAugmentationValidationError(
                "augmentation mode 'online' supports exactly one strong view"
            )
        online = build_online_augmentation(
            weak_plan=weak_plan,
            strong_plan=strong_plan,
            seed=seed,
            modality=modality,
            online_augmenter_id=online_augmenter_id,
            online_augmenter_params=online_augmenter_params,
        )
        weak = augmentation_input
        strong = augmentation_input
        second_strong = None
    else:
        weak, strong, second_strong = materialize_views(
            augmentation_input,
            weak_plan=weak_plan,
            strong_plan=strong_plan,
            seed=seed,
            mode=mode,
            modality=modality,
            sample_ids=sample_ids,
            strong_views=strong_views,
        )

    if wrapped:

        def restore(value: Any | None) -> Any | None:
            if value is None:
                return None
            restored = dict(selected)
            restored["x"] = value
            return restored

        weak = restore(weak)
        strong = restore(strong)
        second_strong = restore(second_strong)

    return UnlabeledAugmentationResult(
        weak=weak,
        strong=strong,
        second_strong=second_strong,
        online=online,
        sample_ids=sample_ids,
    )
