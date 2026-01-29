from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.evaluation import accuracy as accuracy_score
from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import concat_data, get_torch_len
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
    flatten_if_numpy,
    predict_scores,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class S4VMSpec(BaseClassifierSpec):
    k_candidates: int = 6
    flip_rate: float = 0.05


class S4VMMethod(InductiveMethod):
    """S4VM-style candidate selection (simplified, CPU/GPU, binary)."""

    info = MethodInfo(
        method_id="s4vm",
        name="S4VM",
        year=2010,
        family="classic",
        supports_gpu=True,
        paper_title="Large Scale Transductive SVMs",
        paper_pdf="https://icml.cc/Conferences/2010/papers/472.pdf",
        official_code="",
    )

    def __init__(self, spec: S4VMSpec | None = None) -> None:
        self.spec = spec or S4VMSpec()
        self._clf: Any | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> S4VMMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        backend = detect_backend(data.X_l)
        ensure_classifier_backend(self.spec, backend=backend)
        logger.debug("backend=%s", backend)

        if backend == "numpy":
            ensure_cpu_device(device)
            ds = ensure_numpy_data(data)
            y_l = ensure_1d_labels(ds.y_l, name="y_l")

            if ds.X_u is None:
                raise InductiveValidationError("S4VM requires X_u (unlabeled data).")

            X_l = np.asarray(ds.X_l)
            X_u = np.asarray(ds.X_u)
            X_l = flatten_if_numpy(X_l)
            X_u = flatten_if_numpy(X_u)
            y_l = np.asarray(y_l)
            logger.info(
                "S4VM sizes: n_labeled=%s n_unlabeled=%s",
                int(X_l.shape[0]),
                int(X_u.shape[0]),
            )

            classes = np.unique(y_l)
            if classes.size != 2:
                raise InductiveValidationError("S4VM is defined for binary classification only.")

            baseline = build_classifier(self.spec, seed=seed)
            baseline.fit(X_l, y_l)
            base_pred = np.asarray(baseline.predict(X_l))
            base_score = accuracy_score(y_l, base_pred)

            scores = predict_scores(baseline, X_u, backend=backend)
            pred = np.asarray(baseline.predict(X_u))
            conf = scores.max(axis=1)

            best_model = baseline
            best_improve = 0.0

            for i in range(int(self.spec.k_candidates)):
                flip_rate = min(float(self.spec.flip_rate) * float(i + 1), 0.5)
                n_flip = int(round(flip_rate * pred.shape[0]))
                if n_flip <= 0:
                    y_u = pred
                else:
                    order = np.argsort(conf)[:n_flip]
                    y_u = pred.copy()
                    a, b = classes[0], classes[1]
                    y_u[order] = np.where(y_u[order] == a, b, a)

                model = build_classifier(self.spec, seed=seed + i + 1)
                X_train = np.concatenate([X_l, X_u], axis=0)
                y_train = np.concatenate([y_l, y_u], axis=0)
                model.fit(X_train, y_train)
                pred_l = np.asarray(model.predict(X_l))
                score = accuracy_score(y_l, pred_l) - base_score
                if score > best_improve:
                    best_improve = float(score)
                    best_model = model

            self._clf = best_model
            self._backend = backend
            logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
            return self

        ds = ensure_torch_data(data, device=device)
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u is None:
            raise InductiveValidationError("S4VM requires X_u (unlabeled data).")

        X_l = ds.X_l
        X_u = ds.X_u
        classes = torch.unique(y_l, sorted=True)
        if int(classes.numel()) != 2:
            raise InductiveValidationError("S4VM is defined for binary classification only.")
        logger.info(
            "S4VM sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u)),
        )

        baseline = build_classifier(self.spec, seed=seed)
        baseline.fit(X_l, y_l)
        base_pred = baseline.predict(X_l)
        base_score = float((base_pred == y_l).float().mean().item())

        scores = predict_scores(baseline, X_u, backend=backend)
        pred = baseline.predict(X_u)
        conf = scores.max(dim=1).values

        best_model = baseline
        best_improve = 0.0

        for i in range(int(self.spec.k_candidates)):
            flip_rate = min(float(self.spec.flip_rate) * float(i + 1), 0.5)
            n_flip = int(round(flip_rate * int(pred.shape[0])))
            if n_flip <= 0:
                y_u = pred
            else:
                order = torch.argsort(conf)[:n_flip]
                y_u = pred.clone()
                a, b = classes[0], classes[1]
                y_u[order] = torch.where(y_u[order] == a, b, a)

            model = build_classifier(self.spec, seed=seed + i + 1)
            X_train = concat_data([X_l, X_u])
            # Ensure y_u is on the same device as y_l before cat
            y_u = y_u.to(y_l.device)
            y_train = torch.cat([y_l, y_u], dim=0)
            model.fit(X_train, y_train)
            pred_l = model.predict(X_l)
            score = float((pred_l == y_l).float().mean().item()) - base_score
            if score > best_improve:
                best_improve = float(score)
                best_model = model

        self._clf = best_model
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("S4VMMethod is not fitted yet. Call fit() first.")
        backend = detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict_proba input backend mismatch.")

        if backend == "numpy":
            X = flatten_if_numpy(X)

        scores = predict_scores(self._clf, X, backend=backend)
        if backend == "numpy":
            row_sum = scores.sum(axis=1, keepdims=True)
            if np.any(row_sum == 0.0):
                row_sum[row_sum == 0.0] = 1.0
            return (scores / row_sum).astype(np.float32, copy=False)
        torch = optional_import("torch", extra="inductive-torch")
        row_sum = scores.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return scores / row_sum

    def predict(self, X: Any) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("S4VMMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if self._backend is not None and backend != self._backend:
            raise InductiveValidationError("predict input backend mismatch.")
        if backend == "numpy":
            X = flatten_if_numpy(X)
        return self._clf.predict(X)
