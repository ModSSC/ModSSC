from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.store import ArtifactStore


def _is_scipy_sparse(x: Any) -> bool:
    return hasattr(x, "tocsr") and hasattr(x, "multiply")


@dataclass
class RowNormalizeStep:
    """Normalize each feature row independently."""

    norm: Literal["l1"] = "l1"
    eps: float = 1e-12

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if self.norm != "l1":
            raise PreprocessValidationError("core.row_normalize only supports norm='l1'")
        if float(self.eps) <= 0.0:
            raise PreprocessValidationError("eps must be > 0")

        x = store.require("features.X")
        if _is_scipy_sparse(x):
            return {"features.X": self._transform_sparse(x)}
        return {"features.X": self._transform_dense(x)}

    def _transform_dense(self, x: Any) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim != 2:
            raise PreprocessValidationError("core.row_normalize expects 2D features.X")
        if arr.dtype == object:
            raise PreprocessValidationError("core.row_normalize expects numeric features.X")

        out = arr.astype(np.float32, copy=True)
        row_sum = np.abs(out).sum(axis=1, keepdims=True)
        mask = row_sum[:, 0] > float(self.eps)
        out[mask] /= row_sum[mask]
        out[~mask] = 0.0
        return out

    def _transform_sparse(self, x: Any) -> Any:
        mat = x.tocsr(copy=True).astype(np.float32)
        row_sum = np.asarray(np.abs(mat).sum(axis=1)).reshape(-1)
        inv = np.zeros_like(row_sum, dtype=np.float32)
        mask = row_sum > float(self.eps)
        inv[mask] = 1.0 / row_sum[mask]
        return mat.multiply(inv[:, None]).tocsr()
