from __future__ import annotations

import logging
from collections.abc import Iterable
from time import perf_counter
from typing import Any

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.optional import optional_import

logger = logging.getLogger(__name__)


def _torch():
    return optional_import("torch", extra="supervised-torch", feature="supervised:mlp")


def _normalize_hidden_sizes(hidden_sizes: Any) -> tuple[int, ...]:
    if hidden_sizes is None:
        return ()
    if isinstance(hidden_sizes, int):
        return (int(hidden_sizes),)
    if isinstance(hidden_sizes, (list, tuple)):
        return tuple(int(h) for h in hidden_sizes)
    raise SupervisedValidationError("hidden_sizes must be an int or a sequence of ints.")


def _make_activation(name: str, torch):
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise SupervisedValidationError(f"Unknown activation: {name!r}")


class TorchMLPClassifier(BaseSupervisedClassifier):
    """Torch MLP classifier for vector features."""

    classifier_id = "mlp"
    backend = "torch"

    def __init__(
        self,
        *,
        hidden_sizes: Iterable[int] | int | None = (256, 128),
        activation: str = "relu",
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        max_epochs: int = 50,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.hidden_sizes = _normalize_hidden_sizes(hidden_sizes)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self._model: Any | None = None
        self._classes_t: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params hidden_sizes=%s activation=%s dropout=%s lr=%s weight_decay=%s batch_size=%s "
            "max_epochs=%s seed=%s n_jobs=%s",
            self.hidden_sizes,
            self.activation,
            self.dropout,
            self.lr,
            self.weight_decay,
            self.batch_size,
            self.max_epochs,
            self.seed,
            self.n_jobs,
        )
        torch = _torch()

        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError("TorchMLPClassifier requires torch.Tensor X.")
        if not isinstance(y, torch.Tensor):
            raise SupervisedValidationError("TorchMLPClassifier requires torch.Tensor y.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            X = X.reshape(int(X.shape[0]), -1)
        if X.ndim != 2:
            raise SupervisedValidationError("X must be 2D for TorchMLPClassifier.")

        if y.ndim != 1:
            y = y.view(-1)
        if X.shape[0] != y.shape[0]:
            raise SupervisedValidationError("X and y must have matching first dimension.")
        if X.numel() == 0:
            raise SupervisedValidationError("X must be non-empty.")
        if X.device != y.device:
            raise SupervisedValidationError("X and y must be on the same device.")

        if int(self.batch_size) <= 0:
            raise SupervisedValidationError("batch_size must be >= 1.")
        if int(self.max_epochs) <= 0:
            raise SupervisedValidationError("max_epochs must be >= 1.")
        if float(self.lr) <= 0:
            raise SupervisedValidationError("lr must be > 0.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise SupervisedValidationError("dropout must be in [0, 1).")

        for h in self.hidden_sizes:
            if int(h) <= 0:
                raise SupervisedValidationError("hidden_sizes must be positive.")

        classes, y_enc = torch.unique(y, sorted=True, return_inverse=True)
        self._classes_t = classes
        self.classes_ = classes.detach().cpu().numpy()

        torch.manual_seed(int(self.seed or 0))

        n_features = int(X.shape[1])
        n_classes = int(classes.numel())

        layers: list[Any] = []
        in_features = n_features
        for h in self.hidden_sizes:
            layers.append(torch.nn.Linear(in_features, int(h)))
            layers.append(_make_activation(self.activation, torch))
            if float(self.dropout) > 0.0:
                layers.append(torch.nn.Dropout(p=float(self.dropout)))
            in_features = int(h)
        layers.append(torch.nn.Linear(in_features, n_classes))

        model = torch.nn.Sequential(*layers).to(X.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay)
        )

        model.train()
        n = int(X.shape[0])
        for _epoch in range(int(self.max_epochs)):
            order = torch.randperm(n, device=X.device)
            for i in range(0, n, int(self.batch_size)):
                idx = order[i : i + int(self.batch_size)]
                logits = model(X[idx].to(dtype=torch.float32))
                loss = torch.nn.functional.cross_entropy(logits, y_enc[idx].to(torch.long))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model = model
        self._fit_result = FitResult(
            n_samples=int(X.shape[0]),
            n_features=n_features,
            n_classes=n_classes,
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def _scores(self, X: Any):
        torch = _torch()
        if self._model is None or self._classes_t is None:
            raise RuntimeError("Model is not fitted")
        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError("TorchMLPClassifier requires torch.Tensor input.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            X = X.reshape(int(X.shape[0]), -1)
        if X.ndim != 2:
            raise SupervisedValidationError("X must be 2D for TorchMLPClassifier.")
        if X.device != self._classes_t.device:
            raise SupervisedValidationError("X must be on the same device as the model.")
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X.to(dtype=torch.float32))
            return torch.softmax(logits, dim=1)

    def predict_scores(self, X: Any):
        return self._scores(X)

    def predict_proba(self, X: Any):
        return self._scores(X)

    def predict(self, X: Any):
        if self._classes_t is None:
            raise RuntimeError("Model is not fitted")
        scores = self._scores(X)
        idx = scores.argmax(dim=1)
        return self._classes_t[idx]
