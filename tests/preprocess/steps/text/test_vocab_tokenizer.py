from __future__ import annotations

import numpy as np
import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.text import vocab_tokenizer
from modssc.preprocess.steps.text.vocab_tokenizer import VocabTokenizerStep
from modssc.preprocess.store import ArtifactStore


def test_as_text_array_variants():
    arr = np.array(["a", "b"], dtype=object)
    out = vocab_tokenizer._as_text_array(arr)
    assert out.dtype.kind in {"U", "O"}

    out2 = vocab_tokenizer._as_text_array(["x", "y"])
    assert out2.shape == (2,)

    out3 = vocab_tokenizer._as_text_array(iter(["m", "n"]))
    assert out3.shape == (2,)


def test_vocab_tokenizer_fit_transform_and_unknowns():
    store = ArtifactStore({"raw.X": ["hello world", "hello there"]})
    step = VocabTokenizerStep(vocab_size=10, max_length=3, min_freq=2)
    step.fit(store, fit_indices=np.array([0, 1]), rng=np.random.default_rng(0))
    out = step.transform(store, rng=np.random.default_rng(1))

    assert out["features.X"].shape == (2, 3)
    assert out["tokens.input_ids"].shape == (2, 3)
    assert out["tokens.attention_mask"].shape == (2, 3)
    assert out["metadata.vocab_size"] >= 2

    unk_idx = step._vocab[step.unk_token]
    # "world" appears once and should be unknown with min_freq=2
    assert out["tokens.input_ids"][0, 1] == unk_idx


def test_vocab_tokenizer_transform_requires_fit():
    store = ArtifactStore({"raw.X": ["a", "b"]})
    step = VocabTokenizerStep()
    with pytest.raises(PreprocessValidationError, match="before fit"):
        step.transform(store, rng=np.random.default_rng(0))
