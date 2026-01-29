from __future__ import annotations

import logging
from typing import Any

import numpy as np

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import seed_everything

logger = logging.getLogger(__name__)


def _torch():
    return optional_import("torch", extra="supervised-torch", feature="supervised:lstm_scratch")


class TorchLSTMClassifier(BaseSupervisedClassifier):
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
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)

        self._model: Any | None = None

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any) -> FitResult:
        torch = _torch()
        seed_value = None if self.seed is None else int(self.seed)
        if seed_value is not None:
            seed_everything(seed_value, deterministic=True)

        # Ensure input is LongTensor (batch, seq_len)
        if hasattr(X, "to_dense"):  # sparse handling if needed (unlikely for input_ids)
            X = X.to_dense()

        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(np.asarray(X), dtype=torch.long)
        else:
            X = X.long()

        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(np.asarray(y), dtype=torch.long)

        # Detect device
        device = "cuda" if torch.cuda.is_available() and self.n_jobs != 0 else "cpu"

        # Model definition
        num_classes = int(y.max().item()) + 1

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
        y = y.to(device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X, y)
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

    def predict(self, X: Any) -> Any:
        torch = _torch()
        probs = self.predict_proba(X)
        if isinstance(probs, np.ndarray):
            return np.argmax(probs, axis=1)
        return torch.argmax(probs, dim=1).cpu().numpy()

    def predict_proba(self, X: Any) -> Any:
        torch = _torch()
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(np.asarray(X), dtype=torch.long)
        else:
            X = X.long()

        device = next(self._model.parameters()).device
        X = X.to(device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()
