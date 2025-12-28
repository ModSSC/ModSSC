from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.device import resolve_device_name
from modssc.preprocess.errors import OptionalDependencyError
from modssc.preprocess.optional import require


@dataclass
class SentenceTransformerEncoder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            st = require(
                module="sentence_transformers",
                extra="preprocess-text",
                purpose="SentenceTransformer",
            )
        except OptionalDependencyError:
            raise
        # Store module to avoid re-importing
        self._SentenceTransformer = st.SentenceTransformer  # type: ignore[attr-defined]
        self.device = resolve_device_name(self.device)
        self._model = self._SentenceTransformer(self.model_name, device=self.device)

    def encode(
        self, X: Any, *, batch_size: int = 32, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        # sentence-transformers expects list[str]
        texts = X if isinstance(X, list) else list(X)
        emb = self._model.encode(
            texts,
            batch_size=int(batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(emb, dtype=np.float32)
