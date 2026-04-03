from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.cache.model import resolve_hf_local_files_only, resolve_sentence_transformers_cache
from modssc.preprocess.errors import OptionalDependencyError
from modssc.preprocess.optional import require
from modssc.runtime.device import resolve_device_name


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
        self.cache_folder = resolve_sentence_transformers_cache()
        self.local_files_only = resolve_hf_local_files_only()
        try:
            self._model = self._SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            mode = "local cache only" if self.local_files_only else "cache/network resolution"
            raise RuntimeError(
                f"Failed to load SentenceTransformer model {self.model_name!r} using {mode} "
                f"(cache_folder={self.cache_folder!r}). "
                "On offline compute nodes, pre-populate the cache and export "
                "HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1."
            ) from e

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
