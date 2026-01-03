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
    return optional_import("torch", extra="supervised-torch", feature="supervised:audio_cnn")


def _make_activation(name: str, torch):
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise SupervisedValidationError(f"Unknown activation: {name!r}")


def _parse_input_shape(input_shape: Any) -> tuple[int, int] | None:
    if input_shape is None:
        return None
    if not isinstance(input_shape, (list, tuple)):
        raise SupervisedValidationError("input_shape must be a list or tuple.")
    shape = tuple(int(s) for s in input_shape)
    if len(shape) == 1:
        c, length = 1, shape[0]
    elif len(shape) == 2:
        c, length = shape
    else:
        raise SupervisedValidationError("input_shape must be (L,) or (C, L).")
    if c <= 0 or length <= 0:
        raise SupervisedValidationError("input_shape entries must be positive.")
    return int(c), int(length)


torch = _torch()


class _AudioCNN(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        conv_channels: Iterable[int],
        kernel_size: int,
        activation: str,
        dropout: float,
        fc_dim: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        torch = _torch()
        self._torch = torch
        layers: list[Any] = []
        current = int(in_channels)
        for out_ch in conv_channels:
            layers.append(
                torch.nn.Conv1d(
                    current,
                    int(out_ch),
                    kernel_size=int(kernel_size),
                    padding=int(kernel_size) // 2,
                )
            )
            layers.append(_make_activation(activation, torch))
            current = int(out_ch)
        self.conv = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)

        head: list[Any] = []
        if int(fc_dim) > 0:
            head.append(torch.nn.Linear(current, int(fc_dim)))
            head.append(_make_activation(activation, torch))
            if float(dropout) > 0.0:
                head.append(torch.nn.Dropout(p=float(dropout)))
            head.append(torch.nn.Linear(int(fc_dim), int(n_classes)))
        else:
            head.append(torch.nn.Linear(current, int(n_classes)))
        self.head = torch.nn.Sequential(*head)

    def forward(self, x: Any):
        torch = self._torch
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class TorchAudioCNNClassifier(BaseSupervisedClassifier):
    """Small 1D CNN for audio tensors (N, C, L)."""

    classifier_id = "audio_cnn"
    backend = "torch"

    def __init__(
        self,
        *,
        conv_channels: Iterable[int] = (16, 32),
        kernel_size: int = 5,
        fc_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 20,
        input_shape: Iterable[int] | None = None,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.conv_channels = tuple(int(c) for c in conv_channels)
        self.kernel_size = int(kernel_size)
        self.fc_dim = int(fc_dim)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.input_shape = input_shape
        self._model: Any | None = None
        self._classes_t: Any | None = None
        self._input_shape: tuple[int, int] | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def _prepare_X(self, X: Any, torch, *, allow_infer: bool) -> Any:
        if not isinstance(X, torch.Tensor):
            raise SupervisedValidationError("TorchAudioCNNClassifier requires torch.Tensor X.")

        if X.ndim == 3:
            shape = (int(X.shape[1]), int(X.shape[2]))
            X3 = X
        elif X.ndim == 2:
            parsed = self._input_shape or _parse_input_shape(self.input_shape)
            if parsed is not None:
                c, length = parsed
                if int(c) == 1 and int(X.shape[1]) == int(length):
                    shape = (1, int(length))
                    X3 = X.unsqueeze(1)
                elif int(X.shape[1]) == int(c * length):
                    shape = (int(c), int(length))
                    X3 = X.reshape(int(X.shape[0]), int(c), int(length))
                else:
                    raise SupervisedValidationError(
                        "input_shape does not match X feature dimension."
                    )
            else:
                shape = (1, int(X.shape[1]))
                X3 = X.unsqueeze(1)
        elif X.ndim == 1:
            X2 = X.view(1, -1)
            shape = (1, int(X2.shape[1]))
            X3 = X2.unsqueeze(1)
        else:
            raise SupervisedValidationError("X must be 1D, 2D, or 3D for audio_cnn.")

        if allow_infer:
            self._input_shape = shape
        elif self._input_shape is not None and shape != self._input_shape:
            raise SupervisedValidationError("X shape does not match fitted input_shape.")
        return X3

    def fit(self, X: Any, y: Any) -> FitResult:
        start = perf_counter()
        logger.info("Starting %s.fit", self.classifier_id)
        logger.debug(
            "params conv_channels=%s kernel_size=%s fc_dim=%s activation=%s dropout=%s lr=%s "
            "weight_decay=%s batch_size=%s max_epochs=%s input_shape=%s seed=%s n_jobs=%s",
            self.conv_channels,
            self.kernel_size,
            self.fc_dim,
            self.activation,
            self.dropout,
            self.lr,
            self.weight_decay,
            self.batch_size,
            self.max_epochs,
            self.input_shape,
            self.seed,
            self.n_jobs,
        )
        torch = _torch()

        if not isinstance(y, torch.Tensor):
            raise SupervisedValidationError("TorchAudioCNNClassifier requires torch.Tensor y.")
        X3 = self._prepare_X(X, torch, allow_infer=True)

        if y.ndim != 1:
            y = y.view(-1)
        if X3.shape[0] != y.shape[0]:
            raise SupervisedValidationError("X and y must have matching first dimension.")
        if X3.numel() == 0:
            raise SupervisedValidationError("X must be non-empty.")
        if X3.device != y.device:
            raise SupervisedValidationError("X and y must be on the same device.")

        if not self.conv_channels:
            raise SupervisedValidationError("conv_channels must be non-empty.")
        if int(self.kernel_size) <= 0:
            raise SupervisedValidationError("kernel_size must be >= 1.")
        if int(self.fc_dim) < 0:
            raise SupervisedValidationError("fc_dim must be >= 0.")
        if int(self.batch_size) <= 0:
            raise SupervisedValidationError("batch_size must be >= 1.")
        if int(self.max_epochs) <= 0:
            raise SupervisedValidationError("max_epochs must be >= 1.")
        if float(self.lr) <= 0:
            raise SupervisedValidationError("lr must be > 0.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise SupervisedValidationError("dropout must be in [0, 1).")
        for c in self.conv_channels:
            if int(c) <= 0:
                raise SupervisedValidationError("conv_channels must be positive.")

        classes, y_enc = torch.unique(y, sorted=True, return_inverse=True)
        self._classes_t = classes
        self.classes_ = classes.detach().cpu().numpy()

        torch.manual_seed(int(self.seed or 0))

        n_classes = int(classes.numel())
        in_channels = int(self._input_shape[0]) if self._input_shape else int(X3.shape[1])

        model = _AudioCNN(
            in_channels=in_channels,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            activation=self.activation,
            dropout=self.dropout,
            fc_dim=self.fc_dim,
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
        if self._model is None or self._classes_t is None:
            raise RuntimeError("Model is not fitted")
        X3 = self._prepare_X(X, torch, allow_infer=False)
        if X3.device != self._classes_t.device:
            raise SupervisedValidationError("X must be on the same device as the model.")
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
