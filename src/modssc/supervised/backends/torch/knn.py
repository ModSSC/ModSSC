from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Literal

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.optional import optional_import

logger = logging.getLogger(__name__)


def _torch():
    return optional_import("torch", extra="supervised-torch", feature="supervised:knn")


class TorchKNNClassifier(BaseSupervisedClassifier):
    """Torch kNN classifier (supports CPU/GPU depending on tensors device)."""

    classifier_id = "knn"
    backend = "torch"

    def __init__(
        self,
        *,
        k: int = 5,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
        batch_size: int = 1024,
        eps: float = 1e-12,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.k = int(k)
        self.metric = str(metric)
        self.weights = str(weights)
        self.batch_size = int(batch_size)
        self.eps = float(eps)

        self._X_train: Any | None = None
        self._y_train_enc: Any | None = None
        self._classes_t: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    @property
    def classes_t_(self):
        return self._classes_t

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params k=%s metric=%s weights=%s batch_size=%s eps=%s seed=%s n_jobs=%s",
            self.k,
            self.metric,
            self.weights,
            self.batch_size,
            self.eps,
            self.seed,
            self.n_jobs,
        )
        torch = _torch()
        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError("TorchKNNClassifier requires torch.Tensor X.")
        if not isinstance(y, torch.Tensor):
            raise SupervisedValidationError("TorchKNNClassifier requires torch.Tensor y.")
        if X.ndim == 1:
            X = X.view(-1, 1)
        if X.ndim != 2:
            raise SupervisedValidationError("X must be 2D for TorchKNNClassifier.")
        if y.ndim != 1:
            y = y.view(-1)
        if X.shape[0] != y.shape[0]:
            raise SupervisedValidationError("X and y must have matching first dimension.")
        if X.numel() == 0:
            raise SupervisedValidationError("X must be non-empty.")
        if y.device != X.device:
            raise SupervisedValidationError("X and y must be on the same device.")
        if y.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise SupervisedValidationError("y must be an integer tensor.")
        if self.k <= 0:
            raise SupervisedValidationError("k must be >= 1")
        if self.metric not in {"euclidean", "cosine"}:
            raise SupervisedValidationError(f"Unknown metric: {self.metric!r}")
        if self.weights not in {"uniform", "distance"}:
            raise SupervisedValidationError(f"Unknown weights: {self.weights!r}")

        classes, y_enc = torch.unique(y, sorted=True, return_inverse=True)
        self._classes_t = classes
        self.classes_ = classes.detach().cpu().numpy()

        self._X_train = X
        self._y_train_enc = y_enc.to(torch.long)

        self._fit_result = FitResult(
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            n_classes=int(classes.numel()),
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def _pairwise_scores(self, Q):
        if self._X_train is None:
            raise RuntimeError("Model is not fitted (X_train is None)")
        X = self._X_train
        if self.metric == "euclidean":
            q2 = (Q * Q).sum(dim=1, keepdim=True)
            x2 = (X * X).sum(dim=1, keepdim=True).T
            dist2 = q2 + x2 - 2.0 * (Q @ X.T)
            return -dist2
        Qn = Q / (Q.norm(dim=1, keepdim=True) + float(self.eps))
        Xn = X / (X.norm(dim=1, keepdim=True) + float(self.eps))
        return Qn @ Xn.T

    def predict_scores(self, X: Any):
        torch = _torch()
        if self._X_train is None or self._y_train_enc is None or self._classes_t is None:
            raise RuntimeError("Model is not fitted")
        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError("TorchKNNClassifier requires torch.Tensor input.")
        if X.ndim == 1:
            X = X.view(-1, 1)
        if X.ndim != 2:
            raise SupervisedValidationError("X must be 2D for TorchKNNClassifier.")
        if X.device != self._X_train.device:
            raise SupervisedValidationError("X must be on the same device as training data.")

        n_query = int(X.shape[0])
        n_train = int(self._X_train.shape[0])
        n_classes = int(self._classes_t.numel())

        k = min(int(self.k), n_train)
        out = torch.zeros((n_query, n_classes), device=X.device, dtype=torch.float32)

        for start in range(0, n_query, int(self.batch_size)):
            stop = min(n_query, start + int(self.batch_size))
            Qb = X[start:stop]

            scores = self._pairwise_scores(Qb)
            top_scores, idx = torch.topk(scores, k=k, dim=1)
            neigh = self._y_train_enc[idx]

            if self.weights == "uniform":
                w = torch.ones_like(top_scores, dtype=torch.float32)
            else:
                if self.metric == "euclidean":
                    dist = torch.clamp(-top_scores, min=0.0)
                else:
                    dist = torch.clamp(1.0 - top_scores, min=0.0)
                w = 1.0 / (dist.to(torch.float32) + float(self.eps))

            out_batch = torch.zeros(
                (int(stop - start), n_classes), device=X.device, dtype=torch.float32
            )
            out_batch.scatter_add_(1, neigh, w)
            out[start:stop] = out_batch

        row_sum = out.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return out / row_sum

    def predict_proba(self, X: Any):
        return self.predict_scores(X)

    def predict(self, X: Any):
        proba = self.predict_scores(X)
        idx = proba.argmax(dim=1)
        return self._classes_t[idx]
