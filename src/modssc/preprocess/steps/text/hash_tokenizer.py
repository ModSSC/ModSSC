from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.store import ArtifactStore


def _hash_token(token: str, vocab_size: int) -> int:
    h = hashlib.md5(token.encode("utf-8", errors="ignore")).digest()
    v = int.from_bytes(h[:4], "big", signed=False)
    # Reserve 0 for PAD, 1 for UNK.
    return 2 + (v % max(1, vocab_size - 2))


@dataclass
class HashTokenizerStep:
    vocab_size: int = 20000
    max_length: int = 64
    lowercase: bool = True

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        texts = store.require("raw.X")
        if isinstance(texts, np.ndarray):
            if texts.ndim == 0:
                seq_list = [texts.item()]
                n = len(seq_list)
                iter_seq = (str(t) for t in seq_list)
            else:
                n = int(texts.shape[0])
                iter_seq = (str(t) for t in texts)
        elif isinstance(texts, (list, tuple)):
            n = len(texts)
            iter_seq = (str(t) for t in texts)
        else:
            seq_list = list(texts)
            n = len(seq_list)
            iter_seq = (str(t) for t in seq_list)

        if self.lowercase:
            iter_seq = (t.lower() for t in iter_seq)
        L = int(self.max_length)
        ids = np.zeros((n, L), dtype=np.int64)
        mask = np.zeros((n, L), dtype=np.int8)

        for i, t in enumerate(iter_seq):
            toks = [tok for tok in t.split() if tok]
            toks = toks[:L]
            if not toks:
                continue
            for j, tok in enumerate(toks):
                ids[i, j] = _hash_token(tok, int(self.vocab_size))
                mask[i, j] = 1

        return {"tokens.input_ids": ids, "tokens.attention_mask": mask}
