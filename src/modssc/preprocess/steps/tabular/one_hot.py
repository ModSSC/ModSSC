from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.optional import require
from modssc.preprocess.store import ArtifactStore

_MISSING_TOKENS = {"", "?", "nan", "NaN", "NA", "N/A", "null", "None"}


def _as_object_array(raw: Any) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        arr = raw
    elif isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=object)
    else:
        arr = np.asarray(list(raw), dtype=object)

    if arr.ndim == 0:
        raise PreprocessValidationError("raw.X must not be a scalar")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, (bytes, bytearray)):
        value = value.decode(errors="ignore")
    if isinstance(value, str):
        return value.strip() in _MISSING_TOKENS
    return False


def _is_numeric_column(values: np.ndarray) -> bool:
    for val in values:
        if _is_missing(val):
            continue
        try:
            float(val)
        except Exception:
            return False
    return True


def _to_numeric_column(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.float32)
    for idx, val in enumerate(values):
        if _is_missing(val):
            out[idx] = np.nan
        else:
            out[idx] = float(val)
    return out


def _to_categorical_column(values: np.ndarray) -> list[str]:
    out = []
    for val in values:
        if _is_missing(val):
            out.append("__missing__")
        else:
            out.append(str(val))
    return out


@dataclass
class TabularOneHotStep:
    """One-hot encode categorical columns and pass through numeric columns."""

    handle_unknown: str = "ignore"
    _encoder: Any = field(default=None, init=False, repr=False)
    _numeric_cols: list[int] = field(default_factory=list, init=False, repr=False)
    _categorical_cols: list[int] = field(default_factory=list, init=False, repr=False)

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        try:
            pre = require(
                module="sklearn.preprocessing",
                extra="preprocess-sklearn",
                purpose="One-hot encoding for tabular data",
            )
        except OptionalDependencyError:
            raise

        raw = store.require("raw.X")
        arr = _as_object_array(raw)
        idx = np.asarray(fit_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise PreprocessValidationError("fit_indices must be 1D")
        arr_fit = np.take(arr, idx, axis=0)

        self._numeric_cols = []
        self._categorical_cols = []
        for col in range(arr_fit.shape[1]):
            if _is_numeric_column(arr_fit[:, col]):
                self._numeric_cols.append(col)
            else:
                self._categorical_cols.append(col)

        if self._categorical_cols:
            cat = np.column_stack(
                [_to_categorical_column(arr_fit[:, col]) for col in self._categorical_cols]
            )
            try:
                self._encoder = pre.OneHotEncoder(
                    handle_unknown=self.handle_unknown,
                    sparse_output=False,
                    dtype=np.float32,
                )
            except TypeError:
                self._encoder = pre.OneHotEncoder(
                    handle_unknown=self.handle_unknown,
                    sparse=False,
                    dtype=np.float32,
                )
            self._encoder.fit(cat)
        else:
            self._encoder = None

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if not self._numeric_cols and not self._categorical_cols:
            raise PreprocessValidationError("TabularOneHotStep.transform called before fit()")

        raw = store.require("raw.X")
        arr = _as_object_array(raw)
        parts: list[np.ndarray] = []

        if self._numeric_cols:
            numeric = np.column_stack(
                [_to_numeric_column(arr[:, col]) for col in self._numeric_cols]
            )
            parts.append(numeric)

        if self._categorical_cols:
            if self._encoder is None:
                raise PreprocessValidationError("TabularOneHotStep missing fitted encoder")
            cat = np.column_stack(
                [_to_categorical_column(arr[:, col]) for col in self._categorical_cols]
            )
            cat_feat = self._encoder.transform(cat)
            parts.append(np.asarray(cat_feat, dtype=np.float32))

        if not parts:
            raise PreprocessValidationError("TabularOneHotStep produced no features")

        features = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        return {"features.X": np.asarray(features, dtype=np.float32)}
