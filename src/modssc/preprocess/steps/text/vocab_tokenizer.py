from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.store import ArtifactStore


def _tokenize(text: str) -> list[str]:
    # Simple whitespace tokenizer.
    # In a real scenario, use spaCy or NLTK, but this is zero-dependency "from scratch" logic.
    return str(text).lower().split()


def _as_text_array(raw: Any) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        return raw.astype(str)
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=object)
    return np.asarray(list(map(str, raw)), dtype=object)


@dataclass
class VocabTokenizerStep:
    """Tokenize text into integer sequences using a learned vocabulary (Tabula Rasa)."""

    vocab_size: int = 20000
    max_length: int = 256
    min_freq: int = 2
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    _vocab: dict[str, int] = field(default=None, init=False, repr=False)

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        raw = store.require("raw.X")
        texts = _as_text_array(raw)

        # Select fitting subset
        idx = np.asarray(fit_indices, dtype=np.int64)
        texts_fit = np.take(texts, idx, axis=0)

        counts = Counter()
        for t in texts_fit:
            counts.update(_tokenize(t))

        # Build vocab. 0 is reserved for padding, 1 for unknown.
        self._vocab = {self.pad_token: 0, self.unk_token: 1}

        # We need (vocab_size - 2) slots
        candidates = counts.most_common(self.vocab_size - 2)
        for token, count in candidates:
            if count >= self.min_freq:
                self._vocab[token] = len(self._vocab)

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if self._vocab is None:
            raise PreprocessValidationError("VocabTokenizerStep.transform called before fit()")

        raw = store.require("raw.X")
        texts = _as_text_array(raw)

        n_samples = len(texts)
        L = self.max_length

        # Prepare output arrays
        input_ids = np.zeros((n_samples, L), dtype=np.int64)  # 0-init is effectively PAD
        attention_mask = np.zeros((n_samples, L), dtype=np.int64)

        unk_idx = self._vocab[self.unk_token]

        for i, t in enumerate(texts):
            tokens = _tokenize(t)[:L]
            for j, token in enumerate(tokens):
                idx = self._vocab.get(token, unk_idx)
                input_ids[i, j] = idx
                attention_mask[i, j] = 1

        # Expose as features.X for compatibility with existing methods that expect a primary input
        # Also expose specific tensor keys if needed by specialized models
        return {
            "features.X": input_ids,
            "tokens.input_ids": input_ids,
            "tokens.attention_mask": attention_mask,
            "metadata.vocab_size": len(self._vocab),
        }
