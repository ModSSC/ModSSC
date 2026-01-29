from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.utils import (
    detect_backend,
    ensure_1d_labels,
    ensure_1d_labels_torch,
    ensure_cpu_device,
    ensure_numpy_data,
    ensure_torch_data,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


def _get_torch_x(obj: Any) -> Any:
    if isinstance(obj, dict) and "x" in obj:
        return obj["x"]
    return obj


def _encode_binary(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    if classes.size != 2:
        raise InductiveValidationError(
            f"TSVM supports binary classification only (got {classes.size} classes)."
        )
    y_enc = np.where(y == classes[0], -1.0, 1.0).astype(np.float32)
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
class TSVMSpec:
    """Hyperparameters for transductive SVM (binary, CPU/GPU)."""

    max_iter: int = 10
    epochs_per_iter: int = 5
    batch_size: int = 256
    lr: float = 0.1
    C_l: float = 1.0
    C_u_max: float = 1.0
    l2: float = 1.0
    balance: bool = True


class TSVMMethod(InductiveMethod):
    """Transductive SVM (TSVM) baseline (binary, CPU/GPU)."""

    info = MethodInfo(
        method_id="tsvm",
        name="TSVM",
        year=1999,
        family="classic",
        supports_gpu=True,
        paper_title="Transductive inference for text classification using support vector machines",
        paper_pdf="https://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf",
        official_code="",
    )

    def __init__(self, spec: TSVMSpec | None = None) -> None:
        self.spec = spec or TSVMSpec()
        self._svm: _LinearSVM | None = None
        self._classes: np.ndarray | None = None
        self._fitted = False
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> TSVMMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        backend = detect_backend(data.X_l)
        logger.debug("backend=%s", backend)
        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")

            if ds.X_u is None:
                raise InductiveValidationError("TSVM requires X_u (unlabeled data).")

            X_l = np.asarray(ds.X_l, dtype=np.float32)
            y_l = np.asarray(y_l)
            X_u = np.asarray(ds.X_u, dtype=np.float32)
            logger.info(
                "TSVM sizes: n_labeled=%s n_unlabeled=%s",
                int(X_l.shape[0]),
                int(X_u.shape[0]),
            )

            if X_l.shape[0] == 0:
                raise InductiveValidationError("X_l must be non-empty.")

            y_pm1, classes = _encode_binary(y_l)
            self._classes = classes

            rng = np.random.default_rng(int(seed))
            svm = _LinearSVM(n_features=int(X_l.shape[1]), seed=int(seed))

            svm.fit_sgd(
                X_l,
                y_pm1,
                epochs=int(self.spec.epochs_per_iter),
                batch_size=int(self.spec.batch_size),
                lr=float(self.spec.lr),
                C=float(self.spec.C_l),
                l2=float(self.spec.l2),
                rng=rng,
            )

            C_u = 1e-3
            for _ in range(int(self.spec.max_iter)):
                if X_u.shape[0] == 0:
                    break

                scores_u = svm.decision_function(X_u)
                y_u = np.where(scores_u >= 0, 1.0, -1.0).astype(np.float32)

                if self.spec.balance:
                    n_pos = int((y_u > 0).sum())
                    n_neg = int((y_u < 0).sum())
                    if n_pos == 0 or n_neg == 0:
                        order = np.argsort(np.abs(scores_u))
                        half = int(order.size // 2)
                        y_u[order[:half]] = -1.0
                        y_u[order[half:]] = 1.0

                X_all = np.vstack([X_l, X_u])
                y_all = np.concatenate([y_pm1, y_u])

                rep = int(max(1, round(float(self.spec.C_l) / max(C_u, 1e-6))))
                if rep > 1:
                    X_rep = np.repeat(X_l, rep, axis=0)
                    y_rep = np.repeat(y_pm1, rep, axis=0)
                    X_all = np.vstack([X_rep, X_u])
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
            self._backend = backend
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u is None:
            raise InductiveValidationError("TSVM requires X_u (unlabeled data).")

        X_l = _get_torch_x(ds.X_l)
        X_u = _get_torch_x(ds.X_u)
        if int(X_l.shape[0]) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        logger.info(
            "TSVM sizes: n_labeled=%s n_unlabeled=%s",
            int(X_l.shape[0]),
            int(X_u.shape[0]),
        )

        if X_l.dtype not in (torch.float32, torch.float64):
            raise InductiveValidationError("X_l must be float32 or float64 for TSVM.")

        classes = torch.unique(y_l)
        if int(classes.numel()) != 2:
            raise InductiveValidationError("TSVM supports binary classification only.")
        classes_sorted = torch.sort(classes).values
        y_pm1 = torch.where(y_l == classes_sorted[0], -1.0, 1.0).to(X_l.dtype)

        gen = torch.Generator(device=X_l.device).manual_seed(int(seed))
        w = (0.01 * torch.randn((int(X_l.shape[1]),), generator=gen, device=X_l.device)).to(
            X_l.dtype
        )
        b = torch.tensor(0.0, device=X_l.device, dtype=X_l.dtype)

        def fit_sgd(X, y_pm1_t, *, epochs, batch_size, lr, C, l2):
            nonlocal w, b
            n = int(X.shape[0])
            for _ in range(int(epochs)):
                perm = torch.randperm(n, generator=gen, device=X.device)
                for start in range(0, n, int(batch_size)):
                    idx = perm[start : start + int(batch_size)]
                    Xb = X[idx]
                    yb = y_pm1_t[idx]
                    margin = yb * (Xb @ w + b)
                    active = margin < 1.0
                    if bool(active.any()):
                        Xa = Xb[active]
                        ya = yb[active]
                        gw = w * float(l2) - float(C) * (Xa.T @ ya) / max(1.0, float(Xa.shape[0]))
                        gb = -float(C) * ya.mean()
                    else:
                        gw = w * float(l2)
                        gb = 0.0
                    w -= float(lr) * gw
                    b -= float(lr) * gb

        fit_sgd(
            X_l,
            y_pm1,
            epochs=int(self.spec.epochs_per_iter),
            batch_size=int(self.spec.batch_size),
            lr=float(self.spec.lr),
            C=float(self.spec.C_l),
            l2=float(self.spec.l2),
        )

        C_u = 1e-3
        for _ in range(int(self.spec.max_iter)):
            if int(X_u.shape[0]) == 0:
                break
            scores_u = X_u @ w + b
            y_u = torch.where(scores_u >= 0, 1.0, -1.0).to(X_l.dtype)

            if self.spec.balance:
                n_pos = int((y_u > 0).sum().item())
                n_neg = int((y_u < 0).sum().item())
                if n_pos == 0 or n_neg == 0:
                    order = torch.argsort(torch.abs(scores_u))
                    half = int(order.numel() // 2)
                    y_u[order[:half]] = -1.0
                    y_u[order[half:]] = 1.0

            X_all = torch.cat([X_l, X_u], dim=0)
            y_all = torch.cat([y_pm1, y_u], dim=0)

            rep = int(max(1, round(float(self.spec.C_l) / max(C_u, 1e-6))))
            if rep > 1:
                X_rep = X_l.repeat((rep, 1))
                y_rep = y_pm1.repeat(rep)
                X_all = torch.cat([X_rep, X_u], dim=0)
                y_all = torch.cat([y_rep, y_u], dim=0)

            fit_sgd(
                X_all,
                y_all,
                epochs=int(self.spec.epochs_per_iter),
                batch_size=int(self.spec.batch_size),
                lr=float(self.spec.lr),
                C=float(max(C_u, 1e-6)),
                l2=float(self.spec.l2),
            )

            C_u = min(float(self.spec.C_u_max), float(C_u) * 10.0)

        self._svm = (w, b, classes_sorted)
        self._fitted = True
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self._fitted or self._svm is None:
            raise RuntimeError("TSVMMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")
        if backend == "numpy":
            if not isinstance(X, np.ndarray):
                raise InductiveValidationError(
                    "Numpy backend requires numpy.ndarray inputs. Use preprocess core.to_numpy."
                )
            if self._classes is None:
                raise RuntimeError("TSVMMethod missing classes.")
            Xn = np.asarray(X, dtype=np.float32)
            scores = self._svm.decision_function(Xn).astype(np.float64)
            p1 = 1.0 / (1.0 + np.exp(-scores))
            proba = np.stack([1.0 - p1, p1], axis=1)
            return proba.astype(np.float32, copy=False)
        torch = optional_import("torch", extra="inductive-torch")
        if not isinstance(X, torch.Tensor) and not isinstance(X, dict):
            raise InductiveValidationError(
                "Torch backend requires torch.Tensor or dict inputs. Use preprocess core.to_torch."
            )
        X_t = _get_torch_x(X)
        w, b, classes = self._svm  # type: ignore[misc]
        if X_t.dtype not in (torch.float32, torch.float64):
            raise InductiveValidationError("X must be float32 or float64 for TSVM.")
        scores = X_t @ w + b
        p1 = torch.sigmoid(scores)
        proba = torch.stack([1.0 - p1, p1], dim=1)
        return proba

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        backend = self._backend or detect_backend(X)
        if backend == "numpy":
            idx = proba.argmax(axis=1)
            return np.asarray(self._classes)[idx]
        idx = proba.argmax(dim=1)
        w, b, classes = self._svm  # type: ignore[misc]
        return classes[idx]
