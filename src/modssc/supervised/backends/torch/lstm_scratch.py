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
    return optional_import("torch", extra="supervised-torch", feature="supervised:lstm_scratch")


class TorchLSTMClassifier(TorchScoresClassifierBase):
    """LSTM classifier for token sequence features (Tabula Rasa context)."""

    classifier_id = "lstm_scratch"
    backend = "torch"

    def __init__(
        self,
        *,
        vocab_size: int = 20001,  # 20000 + 1 for safety or alignment with tokenizer default
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str | None = None,
        dropout: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 20,
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = activation
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)

        self._model: Any | None = None
        self._classes_t: Any | None = None

    def _prepare_X(self, X: Any, torch) -> Any:
        if hasattr(X, "to_dense"):
            X = X.to_dense()
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(np.asarray(X), dtype=torch.long)
        else:
            X = X.to(dtype=torch.long)
        if X.ndim != 2:
            raise SupervisedValidationError("X must be 2D token ids for TorchLSTMClassifier.")
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

        # Model definition
        classes, y_enc = torch.unique(y, sorted=True, return_inverse=True)
        num_classes = int(classes.numel())
        self._classes_t = classes.to(device)
        self.classes_ = classes.detach().cpu().numpy()

        class LSTMModel(torch.nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.lstm = torch.nn.LSTM(
                    embed_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.fc = torch.nn.Linear(hidden_dim, num_classes)
                self.dropout = torch.nn.Dropout(dropout)

            def forward(self, x):
                # x: (B, L)
                emb = self.embedding(x)  # (B, L, D)
                _, (hn, _) = self.lstm(emb)
                # hn: (layers, B, H) -> take last layer
                out = hn[-1]
                out = self.dropout(out)
                return self.fc(out)

        self._model = LSTMModel(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.num_layers,
            num_classes,
            self.dropout,
        )
        self._model.to(device)
        X = X.to(device)
        y_enc = y_enc.to(device)

        optimizer = torch.optim.Adam(
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

        n_features = int(X.shape[1]) if X.ndim >= 2 else 1
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
