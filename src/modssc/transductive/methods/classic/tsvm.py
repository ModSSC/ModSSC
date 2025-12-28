from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)


def _encode_binary(
    y: np.ndarray,
    *,
    labeled_mask: np.ndarray,
    full_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    labeled_y = y[np.asarray(labeled_mask, dtype=bool)]
    labeled_y = labeled_y[labeled_y >= 0]
    classes = np.unique(labeled_y)
    if classes.size != 2:
        pool = y if full_y is None else np.asarray(full_y).reshape(-1)
        pool = pool[pool >= 0]
        classes = np.unique(pool)
    if classes.size != 2:
        raise ValueError(f"TSVM supports binary classification only (got {classes.size} classes).")
    y_enc = np.zeros_like(y, dtype=np.float32)
    y_enc[y == classes[0]] = -1.0
    y_enc[y == classes[1]] = 1.0
    return y_enc, classes


def _batch_indices(rng: np.random.Generator, idx: np.ndarray, batch_size: int):
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return
    perm = rng.permutation(idx)
    bs = int(batch_size)
    for i in range(0, perm.size, bs):
        yield perm[i : i + bs]


class _LinearSVM:
    """Linear SVM trained with SGD on hinge loss."""

    def __init__(self, n_features: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(int(seed))
        self.w = (0.01 * rng.normal(size=(int(n_features),))).astype(np.float32)
        self.b = np.float32(0.0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X @ self.w + self.b).astype(np.float32, copy=False)

    def fit_sgd(
        self,
        X: np.ndarray,
        y_pm1: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        C: float,
        l2: float,
        rng: np.random.Generator,
    ) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y_pm1, dtype=np.float32).reshape(-1)
        n = int(X.shape[0])
        idx = np.arange(n, dtype=np.int64)

        for _ in range(int(epochs)):
            for bidx in _batch_indices(rng, idx, int(batch_size)):
                Xb = X[bidx]
                yb = y[bidx]
                margin = yb * (Xb @ self.w + self.b)
                active = margin < 1.0
                if np.any(active):
                    Xa = Xb[active]
                    ya = yb[active]
                    gw = self.w * float(l2) - float(C) * (Xa.T @ ya) / max(1.0, float(Xa.shape[0]))
                    gb = -float(C) * ya.mean()
                else:
                    gw = self.w * float(l2)
                    gb = 0.0
                self.w = (self.w - float(lr) * gw).astype(np.float32, copy=False)
                self.b = np.float32(self.b - float(lr) * gb)


@dataclass(frozen=True)
class TSVMTransductiveSpec:
    """Hyperparameters for transductive SVM (binary)."""

    max_iter: int = 10
    epochs_per_iter: int = 5
    batch_size: int = 256
    lr: float = 0.1
    C_l: float = 1.0
    C_u_max: float = 1.0
    l2: float = 1.0
    balance: bool = True


class TSVMMethod(TransductiveMethod):
    """Transductive SVM (TSVM) baseline (binary)."""

    info = MethodInfo(
        method_id="tsvm",
        name="TSVM",
        year=1999,
        family="classic",
        supports_gpu=False,
        paper_title="Transductive inference for text classification using support vector machines",
        paper_pdf="https://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf",
    )

    def __init__(self, spec: TSVMTransductiveSpec | None = None) -> None:
        self.spec = spec or TSVMTransductiveSpec()
        self._svm: _LinearSVM | None = None
        self._classes: np.ndarray | None = None
        self._fitted = False

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> TSVMMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        validate_node_dataset(data)

        X = np.asarray(data.X, dtype=np.float32)
        y = np.asarray(data.y, dtype=np.int64).reshape(-1)
        masks = data.masks or {}
        if "train_mask" not in masks:
            raise ValueError("TSVM requires data.masks['train_mask'] for labeled nodes.")

        train_mask = np.asarray(masks["train_mask"], dtype=bool).reshape(-1)
        unlabeled_mask = np.asarray(masks.get("unlabeled_mask", ~train_mask), dtype=bool).reshape(
            -1
        )

        if train_mask.shape[0] != X.shape[0] or unlabeled_mask.shape[0] != X.shape[0]:
            raise ValueError("train_mask/unlabeled_mask must match number of nodes")

        if not train_mask.any():
            raise ValueError("TSVM requires at least 1 labeled sample.")
        logger.info(
            "TSVM sizes: n_nodes=%s labeled=%s unlabeled=%s",
            int(X.shape[0]),
            int(train_mask.sum()),
            int(unlabeled_mask.sum()),
        )

        full_y = y
        meta = getattr(data, "meta", None)
        if isinstance(meta, Mapping) and "y_true" in meta:
            full_y = np.asarray(meta["y_true"])
        y_pm1, classes = _encode_binary(y, labeled_mask=train_mask, full_y=full_y)
        self._classes = classes

        labeled_idx = np.flatnonzero(train_mask).astype(np.int64)
        unlabeled_idx = np.flatnonzero(unlabeled_mask).astype(np.int64)

        rng = np.random.default_rng(int(seed))
        svm = _LinearSVM(n_features=int(X.shape[1]), seed=int(seed))

        svm.fit_sgd(
            X[labeled_idx],
            y_pm1[labeled_idx],
            epochs=int(self.spec.epochs_per_iter),
            batch_size=int(self.spec.batch_size),
            lr=float(self.spec.lr),
            C=float(self.spec.C_l),
            l2=float(self.spec.l2),
            rng=rng,
        )

        C_u = 1e-3
        for _ in range(int(self.spec.max_iter)):
            if unlabeled_idx.size == 0:
                break

            scores_u = svm.decision_function(X[unlabeled_idx])
            y_u = np.where(scores_u >= 0, 1.0, -1.0).astype(np.float32)

            if self.spec.balance:
                n_pos = int((y_u > 0).sum())
                n_neg = int((y_u < 0).sum())
                if n_pos == 0 or n_neg == 0:
                    order = np.argsort(np.abs(scores_u))
                    half = int(order.size // 2)
                    y_u[order[:half]] = -1.0
                    y_u[order[half:]] = 1.0

            X_all = np.vstack([X[labeled_idx], X[unlabeled_idx]])
            y_all = np.concatenate([y_pm1[labeled_idx], y_u])

            rep = int(max(1, round(float(self.spec.C_l) / max(C_u, 1e-6))))
            if rep > 1:
                X_rep = np.repeat(X[labeled_idx], rep, axis=0)
                y_rep = np.repeat(y_pm1[labeled_idx], rep, axis=0)
                X_all = np.vstack([X_rep, X[unlabeled_idx]])
                y_all = np.concatenate([y_rep, y_u])

            svm.fit_sgd(
                X_all,
                y_all,
                epochs=int(self.spec.epochs_per_iter),
                batch_size=int(self.spec.batch_size),
                lr=float(self.spec.lr),
                C=float(max(C_u, 1e-6)),
                l2=float(self.spec.l2),
                rng=rng,
            )

            C_u = min(float(self.spec.C_u_max), float(C_u) * 10.0)

        self._svm = svm
        self._fitted = True
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if not self._fitted or self._svm is None or self._classes is None:
            raise RuntimeError("TSVMMethod is not fitted yet. Call fit() first.")

        X = np.asarray(data.X, dtype=np.float32)
        scores = self._svm.decision_function(X).astype(np.float64)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        proba = np.stack([1.0 - p1, p1], axis=1)
        return proba.astype(np.float32, copy=False)
