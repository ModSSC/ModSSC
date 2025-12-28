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
    return optional_import("torch", extra="supervised-torch", feature="supervised:text_cnn")


def _make_activation(name: str, torch):
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise SupervisedValidationError(f"Unknown activation: {name!r}")


torch = _torch()


class _TextCNN(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        kernel_sizes: Iterable[int],
        num_filters: int,
        activation: str,
        dropout: float,
        n_classes: int,
    ) -> None:
        super().__init__()
        torch = _torch()
        self._torch = torch
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    in_channels,
                    int(num_filters),
                    kernel_size=int(k),
                )
                for k in kernel_sizes
            ]
        )
        self.act = _make_activation(activation, torch)
        self.dropout = torch.nn.Dropout(p=float(dropout))
        self.fc = torch.nn.Linear(int(num_filters) * int(len(self.convs)), int(n_classes))

    def forward(self, x: Any):
        torch = self._torch
        feats = []
        for conv in self.convs:
            out = conv(x)
            out = self.act(out)
            out = torch.max(out, dim=2).values
            feats.append(out)
        merged = torch.cat(feats, dim=1)
        merged = self.dropout(merged)
        return self.fc(merged)


class TorchTextCNNClassifier(BaseSupervisedClassifier):
    """Text CNN for sequence embeddings (N, L, D) or (N, D, L)."""

    classifier_id = "text_cnn"
    backend = "torch"

    def __init__(
        self,
        *,
        kernel_sizes: Iterable[int] = (3, 4, 5),
        num_filters: int = 100,
        activation: str = "relu",
        dropout: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 20,
        input_layout: str = "channels_last",
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.num_filters = int(num_filters)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.input_layout = str(input_layout)
        self._model: Any | None = None
        self._classes_t: Any | None = None
        self._kernel_sizes: tuple[int, ...] | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def _prepare_X(self, X: Any, torch) -> Any:
        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError("TorchTextCNNClassifier requires torch.Tensor X.")
        if X.ndim == 3:
            if self.input_layout == "channels_last":
                X3 = X.transpose(1, 2)
            elif self.input_layout == "channels_first":
                X3 = X
            else:
                raise SupervisedValidationError(
                    "input_layout must be 'channels_last' or 'channels_first'."
                )
        elif X.ndim == 2:
            X3 = X.unsqueeze(1)
        else:
            raise SupervisedValidationError("X must be 2D or 3D for text_cnn.")
        return X3

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params kernel_sizes=%s num_filters=%s activation=%s dropout=%s lr=%s weight_decay=%s "
            "batch_size=%s max_epochs=%s input_layout=%s seed=%s n_jobs=%s",
            self.kernel_sizes,
            self.num_filters,
            self.activation,
            self.dropout,
            self.lr,
            self.weight_decay,
            self.batch_size,
            self.max_epochs,
            self.input_layout,
            self.seed,
            self.n_jobs,
        )
        torch = _torch()

        if not isinstance(y, torch.Tensor):
            raise SupervisedValidationError("TorchTextCNNClassifier requires torch.Tensor y.")
        if self.input_layout not in {"channels_last", "channels_first"}:
            raise SupervisedValidationError(
                "input_layout must be 'channels_last' or 'channels_first'."
            )
        X3 = self._prepare_X(X, torch)

        if y.ndim != 1:
            y = y.view(-1)
        if X3.shape[0] != y.shape[0]:
            raise SupervisedValidationError("X and y must have matching first dimension.")
        if X3.numel() == 0:
            raise SupervisedValidationError("X must be non-empty.")
        if X3.device != y.device:
            raise SupervisedValidationError("X and y must be on the same device.")

        if not self.kernel_sizes:
            raise SupervisedValidationError("kernel_sizes must be non-empty.")
        if int(self.num_filters) <= 0:
            raise SupervisedValidationError("num_filters must be >= 1.")
        if int(self.batch_size) <= 0:
            raise SupervisedValidationError("batch_size must be >= 1.")
        if int(self.max_epochs) <= 0:
            raise SupervisedValidationError("max_epochs must be >= 1.")
        if float(self.lr) <= 0:
            raise SupervisedValidationError("lr must be > 0.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise SupervisedValidationError("dropout must be in [0, 1).")
        for k in self.kernel_sizes:
            if int(k) <= 0:
                raise SupervisedValidationError("kernel_sizes must be positive.")

        seq_len = int(X3.shape[2])
        usable = tuple(k for k in self.kernel_sizes if int(k) <= seq_len)
        if not usable:
            raise SupervisedValidationError("All kernel_sizes are larger than sequence length.")

        classes, y_enc = torch.unique(y, sorted=True, return_inverse=True)
        self._classes_t = classes
        self.classes_ = classes.detach().cpu().numpy()

        torch.manual_seed(int(self.seed or 0))

        n_classes = int(classes.numel())
        in_channels = int(X3.shape[1])

        model = _TextCNN(
            in_channels=in_channels,
            kernel_sizes=usable,
            num_filters=self.num_filters,
            activation=self.activation,
            dropout=self.dropout,
            n_classes=n_classes,
        ).to(X3.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay)
        )

        model.train()
        n = int(X3.shape[0])
        for _epoch in range(int(self.max_epochs)):
            order = torch.randperm(n, device=X3.device)
            for i in range(0, n, int(self.batch_size)):
                idx = order[i : i + int(self.batch_size)]
                logits = model(X3[idx].to(dtype=torch.float32))
                loss = torch.nn.functional.cross_entropy(logits, y_enc[idx].to(torch.long))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model = model
        self._kernel_sizes = usable
        n_features = int(X3.shape[1] * X3.shape[2])
        self._fit_result = FitResult(
            n_samples=int(X3.shape[0]),
            n_features=n_features,
            n_classes=n_classes,
        )
        logger.info("Finished %s.fit in %.3fs", self.classifier_id, perf_counter() - start)
        return self._fit_result

    def _scores(self, X: Any):
        torch = _torch()
        if self._model is None or self._classes_t is None or self._kernel_sizes is None:
            raise RuntimeError("Model is not fitted")
        X3 = self._prepare_X(X, torch)
        if X3.device != self._classes_t.device:
            raise SupervisedValidationError("X must be on the same device as the model.")
        if int(X3.shape[2]) < max(self._kernel_sizes):
            raise SupervisedValidationError("Sequence length too short for fitted kernel_sizes.")
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X3.to(dtype=torch.float32))
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
