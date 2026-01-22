from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset, Split
from modssc.preprocess import preprocess as run_preprocess

from .errors import ViewsValidationError
from .plan import ColumnSelectSpec, ViewsPlan
from .types import ViewsResult

logger = logging.getLogger(__name__)


def _as_numpy(x: Any) -> np.ndarray:
    """Convert common tensor/array containers to a NumPy array without copying when possible."""

    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _stable_u32(text: str) -> int:
    """Stable 32-bit hash (independent of PYTHONHASHSEED)."""

    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _resolve_columns(
    *,
    spec: ColumnSelectSpec | None,
    n_features: int,
    seed: int,
    view_name: str,
    resolved: dict[str, np.ndarray],
    n_features_map: dict[str, int],
) -> np.ndarray:
    if n_features <= 0:
        raise ViewsValidationError("Cannot select columns when n_features <= 0")

    if spec is None or spec.mode == "all":
        return np.arange(n_features, dtype=np.int64)

    spec.validate()

    if spec.mode == "indices":
        cols = np.asarray([int(i) for i in spec.indices], dtype=np.int64)
        if cols.size == 0:
            raise ViewsValidationError("ColumnSelectSpec(indices) resolved to an empty list")
        if np.unique(cols).size != cols.size:
            raise ViewsValidationError("ColumnSelectSpec.indices contains duplicates")
        if cols.min() < 0 or cols.max() >= n_features:
            raise ViewsValidationError(
                f"ColumnSelectSpec.indices must be within [0, {n_features}), got min={cols.min()}, max={cols.max()}"
            )
        return np.sort(cols)

    if spec.mode == "random":
        frac = float(spec.fraction)
        k = int(round(frac * float(n_features)))
        k = max(1, min(int(n_features), k))
        local_seed = int(seed) ^ _stable_u32(view_name) ^ int(spec.seed_offset)
        rng = np.random.default_rng(local_seed)
        cols = rng.choice(np.arange(n_features, dtype=np.int64), size=k, replace=False)
        return np.sort(cols.astype(np.int64, copy=False))

    if spec.mode == "complement":
        other = str(spec.complement_of)
        if other not in resolved:
            raise ViewsValidationError(
                f"ColumnSelectSpec(mode='complement') refers to view {other!r} which is not resolved yet"
            )
        if n_features_map.get(other) != int(n_features):
            raise ViewsValidationError(
                f"complement_of={other!r} has n_features={n_features_map.get(other)}, "
                f"but current view has n_features={n_features}"
            )
        base = resolved[other]
        cols = np.setdiff1d(np.arange(n_features, dtype=np.int64), base, assume_unique=False)
        if cols.size == 0:
            raise ViewsValidationError(
                f"Complement of view {other!r} is empty (n_features={n_features}). "
                "Use a smaller fraction, or specify explicit indices."
            )
        return cols.astype(np.int64, copy=False)

    raise ViewsValidationError(f"Unhandled ColumnSelectSpec.mode={spec.mode!r}")


def generate_views(
    dataset: LoadedDataset,
    *,
    plan: ViewsPlan,
    seed: int = 0,
    cache: bool = True,
    fit_indices: np.ndarray | None = None,
) -> ViewsResult:
    """Generate multiple feature views from a dataset.

    Parameters
    ----------
    dataset:
        Input dataset from :mod:`modssc.data_loader` (train/test splits).
    plan:
        ViewsPlan describing how to create each view.
    seed:
        Global seed controlling stochastic view operations (e.g. random feature split).
    cache:
        Passed through to :func:`modssc.preprocess.preprocess` when preprocessing is used.
    fit_indices:
        Indices (relative to the *train* split) to use when fitting preprocessing steps
        (e.g. PCA). Defaults to ``np.arange(len(train))``.

    Returns
    -------
    ViewsResult
        Each view is returned as a `LoadedDataset` where `.train.X` and `.test.X` are view-specific
        feature matrices, while labels / edges / masks are preserved.
    """

    start = perf_counter()
    plan.validate()

    dataset_fp = None
    if hasattr(dataset, "meta") and isinstance(dataset.meta, dict):
        dataset_fp = dataset.meta.get("dataset_fingerprint")
    logger.info(
        "Views start: views=%s seed=%s cache=%s dataset_fp=%s",
        [v.name for v in plan.views],
        seed,
        bool(cache),
        dataset_fp,
    )

    train_y = _as_numpy(dataset.train.y)
    n_train = int(train_y.shape[0])
    if fit_indices is None:
        fit_indices = np.arange(n_train, dtype=np.int64)

    views: dict[str, LoadedDataset] = {}
    columns: dict[str, np.ndarray] = {}
    n_features_map: dict[str, int] = {}

    for view in plan.views:
        view_start = perf_counter()
        # 1) Optional preprocessing (cached, deterministic)
        ds = dataset
        if view.preprocess is not None:
            res = run_preprocess(
                ds, plan=view.preprocess, seed=int(seed), fit_indices=fit_indices, cache=bool(cache)
            )
            ds = res.dataset

        def _get_feats(x):
            if isinstance(x, dict) and "x" in x:
                return _as_numpy(x["x"])
            return _as_numpy(x)

        X_train = _get_feats(ds.train.X)
        X_test = _get_feats(ds.test.X) if ds.test is not None else None

        if X_train.ndim < 2:
            raise ViewsValidationError(
                f"View {view.name!r}: expected train.X to be at least 2D, got shape={X_train.shape}"
            )
        if X_test is not None and X_test.ndim < 2:
            raise ViewsValidationError(
                f"View {view.name!r}: expected test.X to be at least 2D, got shape={X_test.shape}"
            )

        n_features = int(X_train.shape[1])
        cols = _resolve_columns(
            spec=view.columns,
            n_features=n_features,
            seed=int(seed),
            view_name=str(view.name),
            resolved=columns,
            n_features_map=n_features_map,
        )
        n_features_map[str(view.name)] = n_features
        columns[str(view.name)] = cols

        X_train_v_sub = X_train[:, cols]
        X_test_v_sub = X_test[:, cols] if X_test is not None else None

        def _reconstruct(orig, feats):
            if isinstance(orig, dict) and "x" in orig:
                new_d = dict(orig)
                new_d["x"] = feats
                return new_d
            return feats

        X_train_v = _reconstruct(ds.train.X, X_train_v_sub)
        X_test_v = _reconstruct(ds.test.X, X_test_v_sub) if ds.test is not None else None

        # 2) Preserve y/edges/masks (do NOT copy large arrays)
        train_split = Split(
            X=X_train_v,
            y=ds.train.y,
            edges=ds.train.edges,
            masks=ds.train.masks,
        )
        test_split = (
            Split(X=X_test_v, y=ds.test.y, edges=ds.test.edges, masks=ds.test.masks)
            if ds.test is not None
            else None
        )

        # 3) Meta
        meta: dict[str, Any] = dict(ds.meta) if isinstance(ds.meta, dict) else {}
        meta.setdefault("views", {})
        meta["views"][str(view.name)] = {
            "columns": cols.tolist(),
            "columns_mode": (view.columns.mode if view.columns is not None else "all"),
            "preprocess": (asdict(view.preprocess) if view.preprocess is not None else None),
        }
        if view.meta:
            # view-level metadata override/additions
            meta.setdefault("view_meta", {})
            meta["view_meta"][str(view.name)] = dict(view.meta)

        views[str(view.name)] = LoadedDataset(train=train_split, test=test_split, meta=meta)
        logger.debug(
            "View built: name=%s train_shape=%s test_shape=%s duration_s=%.3f",
            view.name,
            _shape_of(X_train_v),
            _shape_of(X_test_v),
            perf_counter() - view_start,
        )

    result = ViewsResult(
        views=views,
        columns=columns,
        seed=int(seed),
        plan=plan,
        meta={"n_views": len(views)},
    )
    logger.info(
        "Views done: count=%s duration_s=%.3f",
        len(views),
        perf_counter() - start,
    )
    return result


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(s) for s in shape)
    except Exception:
        return None
