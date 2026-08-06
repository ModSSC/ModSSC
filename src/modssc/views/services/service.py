from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset, Split
from modssc.preprocess import preprocess as run_preprocess
from modssc.preprocess.fingerprint import fingerprint
from modssc.utils.numpy import to_numpy as _as_numpy
from modssc.utils.shape import shape_of as _shape_of

from ..errors import ViewsValidationError
from ..plan import ColumnSelectSpec, ViewsPlan
from ..types import ViewsResult

logger = logging.getLogger(__name__)


def _stable_u32(text: str) -> int:
    """Stable 32-bit hash (independent of PYTHONHASHSEED)."""

    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _update_content_digest(digest: Any, value: Any) -> None:
    """Add a deterministic, full-content representation to ``digest``."""

    def _framed(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    if value is None:
        _framed(b"none")
        return
    if isinstance(value, Mapping):
        _framed(b"mapping")
        for key in sorted(value, key=lambda item: str(item)):
            _framed(str(key).encode("utf-8"))
            _update_content_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        _framed(b"sequence")
        for item in value:
            _update_content_digest(digest, item)
        return
    if hasattr(value, "tocoo"):
        coo = value.tocoo()
        _framed(b"sparse-coo")
        _framed(str(tuple(int(dim) for dim in coo.shape)).encode("ascii"))
        _update_content_digest(digest, np.asarray(coo.row, dtype=np.int64))
        _update_content_digest(digest, np.asarray(coo.col, dtype=np.int64))
        _update_content_digest(digest, np.asarray(coo.data))
        return

    array = _as_numpy(value)
    if array.ndim == 0 and array.dtype.hasobject:
        item = array.item()
        if item is not value:
            _update_content_digest(digest, item)
            return
        _framed(type(item).__qualname__.encode("utf-8"))
        _framed(repr(item).encode("utf-8"))
        return
    _framed(b"array")
    _framed(str(array.dtype).encode("ascii"))
    _framed(str(tuple(int(dim) for dim in array.shape)).encode("ascii"))
    if array.dtype.hasobject:
        for item in array.flat:
            if isinstance(item, np.generic):
                item = item.item()
            if isinstance(item, str):
                _framed(b"str")
                _framed(item.encode("utf-8"))
            elif isinstance(item, bytes):
                _framed(b"bytes")
                _framed(item)
            elif item is None:
                _framed(b"none")
            elif isinstance(item, (bool, int, float, complex)):
                _framed(type(item).__name__.encode("ascii"))
                _framed(repr(item).encode("ascii"))
            else:
                _update_content_digest(digest, item)
        return
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))


def _dataset_content_sha256(dataset: LoadedDataset) -> str:
    digest = hashlib.sha256()
    for split_name, split in (("train", dataset.train), ("test", dataset.test)):
        digest.update(split_name.encode("ascii"))
        if split is None:
            digest.update(b"absent")
            continue
        _update_content_digest(digest, split.X)
        _update_content_digest(digest, split.y)
        _update_content_digest(digest, split.edges)
        _update_content_digest(digest, split.masks)
    return digest.hexdigest()


def _source_dataset_fingerprint(dataset: LoadedDataset) -> str:
    """Return the source identity used when deriving per-input-view cache keys."""

    meta = dataset.meta if hasattr(dataset, "meta") and isinstance(dataset.meta, Mapping) else {}
    dataset_fp = meta.get("dataset_fingerprint")
    content_sha256 = meta.get("dataset_content_sha256")
    if not (
        isinstance(content_sha256, str)
        and len(content_sha256) == 64
        and all(character in "0123456789abcdef" for character in content_sha256)
    ):
        content_sha256 = _dataset_content_sha256(dataset)
    return fingerprint(
        {
            "dataset_fingerprint": dataset_fp if isinstance(dataset_fp, str) else None,
            "dataset_content_sha256": content_sha256,
            "modality": meta.get("modality"),
        },
        prefix="dataset:",
    )


def _input_view_fingerprint(
    *, source_dataset_fingerprint: str, source_width: int, source_columns: np.ndarray
) -> str:
    """Derive a deterministic preprocessing identity for a source-column selection."""

    return fingerprint(
        {
            "source_dataset_fingerprint": source_dataset_fingerprint,
            "source_width": int(source_width),
            "input_columns": [int(column) for column in source_columns],
        },
        prefix="view-input:",
    )


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
    if hasattr(dataset, "meta") and isinstance(dataset.meta, Mapping):
        dataset_fp = dataset.meta.get("dataset_fingerprint")
    source_dataset_fp: str | None = None
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
    input_columns: dict[str, np.ndarray] = {}
    input_n_features_map: dict[str, int] = {}

    for view in plan.views:
        view_start = perf_counter()
        restore_dataset_fingerprint = False
        # 1) Optional source-column selection.  This happens before per-view
        # preprocessing so a canonical dataset may carry native raw views in
        # separate columns (for example WebKB full text and inbound anchors).
        ds = dataset
        if view.input_columns is not None:
            source_train = _as_numpy(ds.train.X)
            source_test = _as_numpy(ds.test.X) if ds.test is not None else None
            if source_train.ndim < 2:
                raise ViewsValidationError(
                    f"View {view.name!r}: input_columns requires train.X to be at least 2D, "
                    f"got shape={source_train.shape}"
                )
            if source_test is not None and source_test.ndim < 2:
                raise ViewsValidationError(
                    f"View {view.name!r}: input_columns requires test.X to be at least 2D, "
                    f"got shape={source_test.shape}"
                )
            source_width = int(source_train.shape[1])
            source_cols = _resolve_columns(
                spec=view.input_columns,
                n_features=source_width,
                seed=int(seed),
                view_name=str(view.name),
                resolved=input_columns,
                n_features_map=input_n_features_map,
            )
            input_n_features_map[str(view.name)] = source_width
            input_columns[str(view.name)] = source_cols
            selected_train = source_train[:, source_cols]
            selected_test = source_test[:, source_cols] if source_test is not None else None
            selected_meta = dict(ds.meta) if isinstance(ds.meta, Mapping) else {}
            selected_meta["views"] = dict(selected_meta.get("views", {}))
            selected_meta["views"][str(view.name)] = {
                "input_columns": source_cols.tolist(),
                "input_columns_mode": view.input_columns.mode,
            }
            if view.preprocess is not None:
                if source_dataset_fp is None:
                    source_dataset_fp = _source_dataset_fingerprint(dataset)
                selected_meta["dataset_fingerprint"] = _input_view_fingerprint(
                    source_dataset_fingerprint=source_dataset_fp,
                    source_width=source_width,
                    source_columns=source_cols,
                )
                restore_dataset_fingerprint = True
            ds = LoadedDataset(
                train=Split(
                    X=selected_train,
                    y=ds.train.y,
                    edges=ds.train.edges,
                    masks=ds.train.masks,
                ),
                test=(
                    Split(
                        X=selected_test,
                        y=ds.test.y,
                        edges=ds.test.edges,
                        masks=ds.test.masks,
                    )
                    if ds.test is not None
                    else None
                ),
                meta=selected_meta,
            )

        # 2) Optional preprocessing (cached, deterministic)
        if view.preprocess is not None:
            res = run_preprocess(
                ds, plan=view.preprocess, seed=int(seed), fit_indices=fit_indices, cache=bool(cache)
            )
            ds = res.dataset
            if restore_dataset_fingerprint:
                restored_meta = dict(ds.meta)
                source_meta = dataset.meta if isinstance(dataset.meta, Mapping) else {}
                if "dataset_fingerprint" in source_meta:
                    restored_meta["dataset_fingerprint"] = source_meta["dataset_fingerprint"]
                else:
                    restored_meta.pop("dataset_fingerprint", None)
                ds = LoadedDataset(train=ds.train, test=ds.test, meta=restored_meta)

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

        # 3) Preserve y/edges/masks (do NOT copy large arrays)
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

        # 4) Meta
        meta: dict[str, Any] = dict(ds.meta) if isinstance(ds.meta, Mapping) else {}
        existing_views = meta.get("views")
        views_meta = dict(existing_views) if isinstance(existing_views, Mapping) else {}
        existing_view = views_meta.get(str(view.name))
        current_view_meta = dict(existing_view) if isinstance(existing_view, Mapping) else {}
        current_view_meta.update(
            {
                "columns": cols.tolist(),
                "columns_mode": (view.columns.mode if view.columns is not None else "all"),
                "preprocess": (asdict(view.preprocess) if view.preprocess is not None else None),
            }
        )
        views_meta[str(view.name)] = current_view_meta
        meta["views"] = views_meta
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

    result_meta: dict[str, Any] = {"n_views": len(views)}
    if input_columns:
        result_meta["input_columns"] = {
            name: values.tolist() for name, values in sorted(input_columns.items())
        }

    result = ViewsResult(
        views=views,
        columns=columns,
        seed=int(seed),
        plan=plan,
        meta=result_meta,
    )
    logger.info(
        "Views done: count=%s duration_s=%.3f",
        len(views),
        perf_counter() - start,
    )
    return result
