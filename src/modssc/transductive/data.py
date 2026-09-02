"""Native preparation of graph artifacts, masks, and node-classification data."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from modssc.graph.artifacts import GraphArtifact, NodeDataset
from modssc.graph.errors import GraphValidationError

from .errors import TransductiveDataError

_INDEX_KEYS = frozenset({"train", "val", "test", "train_labeled", "train_unlabeled"})
_INDEX_REFS = frozenset({"train", "test"})
_MASK_KEYS = frozenset({"train_mask", "val_mask", "test_mask", "unlabeled_mask", "labeled_mask"})


@dataclass(frozen=True)
class NodeEvaluationData:
    """Ground truth and split masks reserved exclusively for evaluation."""

    y_true: np.ndarray
    masks: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        raw_truth = np.asarray(self.y_true)
        if not np.issubdtype(raw_truth.dtype, np.integer):
            raise TransductiveDataError(
                "evaluation truth must contain integer class ids",
                code="E_TRANSDUCTIVE_LABELS",
            )
        truth = raw_truth.astype(np.int64, copy=True)
        if truth.ndim != 1:
            raise TransductiveDataError(
                f"evaluation truth must have shape (n_nodes,), got {truth.shape}",
                code="E_TRANSDUCTIVE_SHAPE",
            )
        normalized_masks = _canonical_masks(self.masks, n_nodes=int(truth.shape[0]))
        for mask in normalized_masks.values():
            mask.setflags(write=False)
        truth.setflags(write=False)
        object.__setattr__(self, "y_true", truth)
        object.__setattr__(self, "masks", MappingProxyType(normalized_masks))


@dataclass(frozen=True)
class PreparedNodeData:
    """Physically separate method-visible data from evaluation-only truth."""

    fit: NodeDataset
    evaluation: NodeEvaluationData


def to_numpy(value: Any) -> np.ndarray:
    """Convert an array-like value to NumPy without assuming a Torch backend."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _scipy_sparse() -> Any | None:
    try:
        return importlib.import_module("scipy.sparse")
    except ModuleNotFoundError:
        return None


def _combine_splits(train: Any, test: Any | None) -> Any:
    sparse = _scipy_sparse()
    train_is_sparse = sparse is not None and sparse.issparse(train)
    test_is_sparse = sparse is not None and test is not None and sparse.issparse(test)
    if test is None:
        return train.copy() if train_is_sparse else to_numpy(train)
    if train_is_sparse or test_is_sparse:
        assert sparse is not None
        train_part = train if train_is_sparse else sparse.csr_matrix(to_numpy(train))
        test_part = test if test_is_sparse else sparse.csr_matrix(to_numpy(test))
        return sparse.vstack((train_part, test_part), format="csr")
    return np.concatenate([to_numpy(train), to_numpy(test)], axis=0)


def _normalize_size(name: str, value: int | None, *, optional: bool = False) -> int | None:
    if value is None:
        if optional:
            return None
        raise TransductiveDataError(
            f"{name} is required",
            code="E_TRANSDUCTIVE_SHAPE",
        )
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TransductiveDataError(
            f"{name} must be a non-negative integer",
            code="E_TRANSDUCTIVE_SHAPE",
        )
    normalized = int(value)
    if normalized < 0:
        raise TransductiveDataError(
            f"{name} must be a non-negative integer",
            code="E_TRANSDUCTIVE_SHAPE",
        )
    return normalized


def _normalize_indices(
    name: str,
    value: Any,
    *,
    reference: str,
    n_train: int,
    n_test: int | None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise TransductiveDataError(
            f"indices[{name!r}] must be one-dimensional, got shape {array.shape}",
            code="E_TRANSDUCTIVE_INDICES",
        )
    if array.dtype.kind not in {"i", "u"}:
        raise TransductiveDataError(
            f"indices[{name!r}] must contain integers, got dtype {array.dtype}",
            code="E_TRANSDUCTIVE_INDICES",
        )
    limit = n_train if reference == "train" else n_test
    if limit is None:
        raise TransductiveDataError(
            f"indices[{name!r}] references test but n_test is unavailable",
            code="E_TRANSDUCTIVE_REFS",
        )
    if array.size:
        if int(array.min()) < 0 or int(array.max()) >= int(limit):
            raise TransductiveDataError(
                f"indices[{name!r}] contains values outside [0, {int(limit)})",
                code="E_TRANSDUCTIVE_INDICES",
            )
        if np.unique(array).size != array.size:
            raise TransductiveDataError(
                f"indices[{name!r}] contains duplicate values",
                code="E_TRANSDUCTIVE_INDICES",
            )
    return array.astype(np.int64, copy=False)


def _require_exact_keys(
    values: Mapping[str, Any],
    *,
    expected: frozenset[str],
    label: str,
    code: str,
) -> None:
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise TransductiveDataError(
            f"{label} keys mismatch: missing={missing} unknown={unknown}",
            code=code,
        )


def _validate_partition_masks(masks: Mapping[str, np.ndarray]) -> None:
    train = masks["train_mask"]
    val = masks["val_mask"]
    test = masks["test_mask"]
    labeled = masks["labeled_mask"]
    unlabeled = masks["unlabeled_mask"]

    if np.any(train & val) or np.any(train & test) or np.any(val & test):
        raise TransductiveDataError(
            "train, val, and test masks must be pairwise disjoint",
            code="E_TRANSDUCTIVE_MASK_OVERLAP",
        )
    if np.any(labeled & ~train):
        raise TransductiveDataError(
            "labeled_mask must be a subset of train_mask",
            code="E_TRANSDUCTIVE_MASK_OVERLAP",
        )
    complement_unlabeled = train & ~labeled
    inclusive_unlabeled = train
    if not (
        np.array_equal(unlabeled, complement_unlabeled)
        or np.array_equal(unlabeled, inclusive_unlabeled)
    ):
        raise TransductiveDataError(
            "unlabeled_mask must equal train_mask minus labeled_mask, or the inclusive train pool",
            code="E_TRANSDUCTIVE_MASK_OVERLAP",
        )


def _canonical_masks(
    masks: Mapping[str, Any],
    *,
    n_nodes: int,
) -> dict[str, np.ndarray]:
    _require_exact_keys(
        masks,
        expected=_MASK_KEYS,
        label="mask",
        code="E_TRANSDUCTIVE_MASKS",
    )
    canonical: dict[str, np.ndarray] = {}
    for name in sorted(_MASK_KEYS):
        raw = np.asarray(masks[name])
        if raw.dtype != np.bool_:
            raise TransductiveDataError(
                f"{name} must have bool dtype, got {raw.dtype}",
                code="E_TRANSDUCTIVE_MASKS",
            )
        if raw.shape != (int(n_nodes),):
            raise TransductiveDataError(
                f"{name} must have shape ({int(n_nodes)},), got {raw.shape}",
                code="E_TRANSDUCTIVE_SHAPE",
            )
        canonical[name] = raw.astype(bool, copy=True)
    _validate_partition_masks(canonical)
    return canonical


def _mask_from_indices(size: int, indices: np.ndarray, *, offset: int = 0) -> np.ndarray:
    mask = np.zeros((int(size),), dtype=bool)
    if indices.size:
        mask[indices + int(offset)] = True
    return mask


def masks_from_indices(
    *,
    n_train: int,
    n_test: int | None,
    indices: Mapping[str, np.ndarray],
    refs: Mapping[str, str],
) -> dict[str, np.ndarray]:
    """Materialize canonical masks from a strict sampled-index contract."""

    train_size = _normalize_size("n_train", n_train)
    test_size = _normalize_size("n_test", n_test, optional=True)
    assert train_size is not None
    _require_exact_keys(
        indices,
        expected=_INDEX_KEYS,
        label="indices",
        code="E_TRANSDUCTIVE_INDICES",
    )
    _require_exact_keys(
        refs,
        expected=_INDEX_KEYS,
        label="refs",
        code="E_TRANSDUCTIVE_REFS",
    )

    invalid_refs = {name: ref for name, ref in refs.items() if ref not in _INDEX_REFS}
    if invalid_refs:
        raise TransductiveDataError(
            f"unknown split references: {invalid_refs}",
            code="E_TRANSDUCTIVE_REFS",
        )
    invalid_train_refs = {
        name: refs[name]
        for name in ("train", "val", "train_labeled", "train_unlabeled")
        if refs[name] != "train"
    }
    if invalid_train_refs:
        raise TransductiveDataError(
            f"training index sets must reference train: {invalid_train_refs}",
            code="E_TRANSDUCTIVE_REFS",
        )

    normalized = {
        name: _normalize_indices(
            name,
            indices[name],
            reference=refs[name],
            n_train=train_size,
            n_test=test_size,
        )
        for name in sorted(_INDEX_KEYS)
    }
    n_total = train_size + int(test_size or 0)

    def offset(name: str) -> int:
        return 0 if refs[name] == "train" else train_size

    canonical = {
        "train_mask": _mask_from_indices(
            n_total,
            normalized["train"],
            offset=offset("train"),
        ),
        "val_mask": _mask_from_indices(
            n_total,
            normalized["val"],
            offset=offset("val"),
        ),
        "test_mask": _mask_from_indices(
            n_total,
            normalized["test"],
            offset=offset("test"),
        ),
        "unlabeled_mask": _mask_from_indices(
            n_total,
            normalized["train_unlabeled"],
            offset=offset("train_unlabeled"),
        ),
        "labeled_mask": _mask_from_indices(
            n_total,
            normalized["train_labeled"],
            offset=offset("train_labeled"),
        ),
    }
    _validate_partition_masks(canonical)
    return canonical


def masks_from_sampling(
    sampling: Any,
    *,
    n_train: int,
    n_test: int | None,
) -> dict[str, np.ndarray]:
    """Convert either graph masks or ordinary split indices to one mask contract."""

    if sampling.is_graph():
        train_size = _normalize_size("n_train", n_train)
        test_size = _normalize_size("n_test", n_test, optional=True)
        assert train_size is not None
        raw_masks = sampling.masks
        _require_exact_keys(
            raw_masks,
            expected=frozenset({"train", "val", "test", "unlabeled", "labeled"}),
            label="graph mask",
            code="E_TRANSDUCTIVE_MASKS",
        )
        return _canonical_masks(
            {
                "train_mask": raw_masks["train"],
                "val_mask": raw_masks["val"],
                "test_mask": raw_masks["test"],
                "unlabeled_mask": raw_masks["unlabeled"],
                "labeled_mask": raw_masks["labeled"],
            },
            n_nodes=train_size + int(test_size or 0),
        )
    return masks_from_indices(
        n_train=n_train,
        n_test=n_test,
        indices=sampling.indices,
        refs=sampling.refs,
    )


def graph_from_dataset(dataset: Any, *, n_nodes: int) -> GraphArtifact:
    """Normalize a provider-owned edge container to :class:`GraphArtifact`."""

    try:
        node_count = _normalize_size("n_nodes", n_nodes)
        assert node_count is not None
        edges = dataset.train.edges
        if isinstance(edges, GraphArtifact):
            if int(edges.n_nodes) != node_count:
                raise TransductiveDataError(
                    f"graph n_nodes mismatch: expected {node_count}, got {int(edges.n_nodes)}",
                    code="E_TRANSDUCTIVE_SHAPE",
                )
            return edges

        edge_weight = None
        if isinstance(edges, Mapping):
            if "edge_index" not in edges:
                raise TransductiveDataError(
                    "graph edge mapping is missing edge_index",
                    code="E_TRANSDUCTIVE_GRAPH",
                )
            edge_index = edges["edge_index"]
            edge_weight = edges.get("edge_weight")
        else:
            edge_index = edges
        normalized_index = np.asarray(edge_index)
        if (
            normalized_index.ndim == 2
            and normalized_index.shape[0] != 2
            and normalized_index.shape[1] == 2
        ):
            normalized_index = normalized_index.T
        return GraphArtifact(
            n_nodes=node_count,
            edge_index=normalized_index,
            edge_weight=None if edge_weight is None else np.asarray(edge_weight),
            directed=True,
            meta={},
        )
    except TransductiveDataError:
        raise
    except (
        AttributeError,
        GraphValidationError,
        ValueError,
        TypeError,
        IndexError,
        OverflowError,
    ) as exc:
        raise TransductiveDataError(
            f"invalid graph data: {exc}",
            code="E_TRANSDUCTIVE_GRAPH",
        ) from exc


def _labels_array(train: Any, test: Any | None) -> np.ndarray:
    combined = _combine_splits(train, test)
    try:
        labels = to_numpy(combined)
    except (ValueError, TypeError) as exc:
        raise TransductiveDataError(
            f"failed to materialize labels: {exc}",
            code="E_TRANSDUCTIVE_LABELS",
        ) from exc
    if labels.ndim != 1:
        raise TransductiveDataError(
            f"transductive labels must have shape (n_nodes,), got {labels.shape}",
            code="E_TRANSDUCTIVE_SHAPE",
        )
    if np.issubdtype(labels.dtype, np.integer):
        return labels.astype(np.int64, copy=True)
    if np.issubdtype(labels.dtype, np.floating):
        if not np.isfinite(labels).all():
            raise TransductiveDataError(
                "transductive labels must be finite integer class ids",
                code="E_TRANSDUCTIVE_LABELS",
            )
        converted = labels.astype(np.int64)
        if np.all(labels == converted):
            return converted
    raise TransductiveDataError(
        "transductive labels must contain integer class ids",
        code="E_TRANSDUCTIVE_LABELS",
    )


def prepare_node_data(
    *,
    dataset: Any,
    graph: GraphArtifact | None,
    masks: Mapping[str, np.ndarray],
    use_test_split: bool,
    expected_labeled_count: int | None = None,
) -> PreparedNodeData:
    """Prepare isolated method-visible data and evaluation-only ground truth."""

    try:
        X_test = dataset.test.X if use_test_split and dataset.test is not None else None
        y_test = dataset.test.y if use_test_split and dataset.test is not None else None
        X_all = _combine_splits(dataset.train.X, X_test)
        y_true = _labels_array(dataset.train.y, y_test)
        n_nodes = int(X_all.shape[0])
        if graph is not None and int(graph.n_nodes) != n_nodes:
            raise TransductiveDataError(
                f"graph n_nodes mismatch: expected {n_nodes}, got {int(graph.n_nodes)}",
                code="E_TRANSDUCTIVE_SHAPE",
            )
        if y_true.shape != (n_nodes,):
            raise TransductiveDataError(
                f"feature/label row mismatch: X has {n_nodes}, y has {y_true.shape[0]}",
                code="E_TRANSDUCTIVE_SHAPE",
            )
        canonical_masks = _canonical_masks(masks, n_nodes=n_nodes)
        labeled_mask = canonical_masks["labeled_mask"]
        labeled_count = int(labeled_mask.sum())
        if expected_labeled_count is not None:
            expected = _normalize_size("expected_labeled_count", expected_labeled_count)
            if labeled_count != expected:
                raise TransductiveDataError(
                    "transductive labeled mask mismatch: expected "
                    f"{expected} from sampling stats, got {labeled_count}",
                    code="E_TRANSDUCTIVE_LABELED_MASK",
                )

        y_observed = y_true.copy()
        y_observed[~labeled_mask] = -1
        fit_masks = {
            "train_mask": labeled_mask.copy(),
            "unlabeled_mask": canonical_masks["unlabeled_mask"].copy(),
            "labeled_mask": labeled_mask.copy(),
            "train_all_mask": canonical_masks["train_mask"].copy(),
        }
        fit_data = NodeDataset(
            X=X_all,
            y=y_observed,
            graph=graph,
            masks=fit_masks,
            meta={"label_visibility": "labeled_only"},
        )
        evaluation = NodeEvaluationData(y_true=y_true, masks=canonical_masks)
        return PreparedNodeData(fit=fit_data, evaluation=evaluation)
    except TransductiveDataError:
        raise
    except (
        AttributeError,
        GraphValidationError,
        ValueError,
        TypeError,
        IndexError,
        OverflowError,
    ) as exc:
        raise TransductiveDataError(
            f"invalid transductive node data: {exc}",
            code="E_TRANSDUCTIVE_DATA",
        ) from exc


def build_node_dataset(
    *,
    dataset: Any,
    graph: GraphArtifact | None,
    masks: Mapping[str, np.ndarray],
    use_test_split: bool,
    expected_labeled_count: int | None = None,
) -> NodeDataset:
    """Build only the fit-visible dataset; evaluation truth is never attached."""

    return prepare_node_data(
        dataset=dataset,
        graph=graph,
        masks=masks,
        use_test_split=use_test_split,
        expected_labeled_count=expected_labeled_count,
    ).fit


__all__ = [
    "NodeEvaluationData",
    "PreparedNodeData",
    "build_node_dataset",
    "graph_from_dataset",
    "masks_from_indices",
    "masks_from_sampling",
    "prepare_node_data",
    "to_numpy",
]
