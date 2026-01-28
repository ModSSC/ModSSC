from __future__ import annotations

import random
from contextlib import suppress
from typing import Any

import numpy as np

from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.optional import optional_import


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Best-effort deterministic seeding across random, numpy, and torch (if available)."""
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)

    try:
        torch = optional_import("torch", extra="supervised-torch", feature="supervised:seed")
    except Exception:
        return

    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)

    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            with suppress(Exception):
                torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def as_numpy(x: Any) -> np.ndarray:
    """Convert common array-likes (numpy, torch, lists) to numpy without copying when possible."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def ensure_2d(X: Any) -> np.ndarray:
    arr = as_numpy(X)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3:
        # flatten everything but the first axis
        n = int(arr.shape[0])
        return arr.reshape(n, -1)
    raise SupervisedValidationError(f"X must be array-like, got ndim={arr.ndim}")


def encode_labels(y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Encode arbitrary labels to contiguous int64 indices.

    Returns
    -------
    y_enc:
        int64 labels in [0, n_classes)
    classes:
        array of original class labels (sorted unique)
    """
    y_arr = as_numpy(y).reshape(-1)
    classes = np.unique(y_arr)
    # searchsorted expects sorted unique, which np.unique provides
    y_enc = np.searchsorted(classes, y_arr).astype(np.int64)
    return y_enc, classes


def onehot(y_enc: np.ndarray, n_classes: int) -> np.ndarray:
    y_enc = np.asarray(y_enc, dtype=np.int64).reshape(-1)
    out = np.zeros((int(y_enc.size), int(n_classes)), dtype=np.float32)
    if y_enc.size:
        out[np.arange(int(y_enc.size)), y_enc] = 1.0
    return out
