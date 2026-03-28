from __future__ import annotations

import logging
from typing import Any

import numpy as np

from modssc.supervised.backends.torch.common import (
    TorchScoresClassifierBase,
)
from modssc.supervised.base import FitResult
from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import seed_everything

logger = logging.getLogger(__name__)


def _torch():
    return optional_import(
        "torch", extra="supervised-torch", feature="supervised:audio_cnn_scratch"
    )


class TorchAudioCNNClassifier(TorchScoresClassifierBase):
    """Simple 2D CNN for Audio Spectrograms (Tabula Rasa context)."""

    classifier_id = "audio_cnn_scratch"
    backend = "torch"

    def __init__(
        self,
        *,
        n_mels: int = 128,
        # Default architecture params similar to M5 or simple ResNet
        base_channels: int = 32,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 50,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.n_mels = int(n_mels)
        self.base_channels = int(base_channels)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)

        self._model: Any | None = None
        self._classes_t: Any | None = None

    def _prepare_X(self, X: Any, torch) -> Any:
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        else:
            X = X.to(dtype=torch.float32)
        if X.ndim == 3:
            X = X.unsqueeze(1)
        if X.ndim != 4:
            raise SupervisedValidationError("X must be 3D or 4D for TorchAudioCNNClassifier.")
        if int(X.shape[0]) == 0:
            raise SupervisedValidationError("X must be non-empty.")
        return X

    def _prepare_y(self, y: Any, torch) -> Any:
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        else:
            y = y.to(dtype=torch.long)
        if y.ndim != 1:
            y = y.view(-1)
        if int(y.shape[0]) == 0:
            raise SupervisedValidationError("y must be non-empty.")
        return y

    def _fit_device(self, X_raw: Any, y_raw: Any, torch) -> Any:
        if (
            isinstance(X_raw, torch.Tensor)
            and isinstance(y_raw, torch.Tensor)
            and X_raw.device != y_raw.device
        ):
            raise SupervisedValidationError("X and y must be on the same device.")
        for tensor in (X_raw, y_raw):
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        if torch.cuda.is_available() and self.n_jobs != 0:
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(self, X: Any, y: Any) -> FitResult:
        torch = _torch()
        seed_value = None if self.seed is None else int(self.seed)
        if seed_value is not None:
            seed_everything(seed_value, deterministic=True)

        X_raw = X
        y_raw = y
        X = self._prepare_X(X, torch)
        y = self._prepare_y(y, torch)
        if int(X.shape[0]) != int(y.shape[0]):
            raise SupervisedValidationError("X and y must have matching first dimension.")

        device = self._fit_device(X_raw, y_raw, torch)

        classes, y_enc = torch.unique(y, sorted=True, return_inverse=True)
        num_classes = int(classes.numel())
        self._classes_t = classes.to(device)
        self.classes_ = classes.detach().cpu().numpy()

        class ConvBlock(torch.nn.Module):
            def __init__(self, in_c, out_c, kernel_size, stride=1, padding=1, pool=True):
                super().__init__()
                self.conv = torch.nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
                self.bn = torch.nn.BatchNorm2d(out_c)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.MaxPool2d(2) if pool else torch.nn.Identity()

            def forward(self, x):
                return self.pool(self.relu(self.bn(self.conv(x))))

        class AudioCNN(torch.nn.Module):
            def __init__(self, in_channels, base_channels, num_blocks, num_classes, dropout):
                super().__init__()
                layers = []
                c = in_channels
                features = base_channels

                for _i in range(num_blocks):
                    layers.append(ConvBlock(c, features, kernel_size=3, pool=True))
                    c = features
                    features *= 2

                self.convs = torch.nn.Sequential(*layers)
                self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.dropout = torch.nn.Dropout(dropout)
                self.fc = torch.nn.Linear(c, num_classes)

            def forward(self, x):
                # x: (N, 1, F, T)
                x = self.convs(x)
                x = self.global_pool(x)
                x = x.flatten(1)
                x = self.dropout(x)
                return self.fc(x)

        self._model = AudioCNN(
            in_channels=1,
            base_channels=self.base_channels,
            num_blocks=self.num_blocks,
            num_classes=num_classes,
            dropout=self.dropout,
        )
        self._model.to(device)
        X = X.to(device)
        y_enc = y_enc.to(device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X, y_enc)
        generator = None
        if seed_value is not None:
            generator = torch.Generator().manual_seed(seed_value)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self._model.train()
        for _epoch in range(self.max_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        n_features = int(X.shape[-2] * X.shape[-1]) if X.ndim >= 3 else int(X.shape[1])
        self._fit_result = FitResult(
            n_samples=int(X.shape[0]),
            n_features=n_features,
            n_classes=num_classes,
        )
        return self._fit_result

    def _scores(self, X: Any):
        torch = _torch()
        if self._model is None or self._classes_t is None:
            raise RuntimeError("Model is not fitted")
        input_is_tensor = isinstance(X, torch.Tensor)
        X = self._prepare_X(X, torch)

        device = next(self._model.parameters()).device
        if input_is_tensor and X.device != device:
            raise SupervisedValidationError("X must be on the same device as the model.")
        X = X.to(device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X)
            return torch.softmax(logits, dim=1)
