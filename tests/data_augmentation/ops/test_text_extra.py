from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation.ops.text import (
    Lowercase,
    RandomSwap,
    WordDropout,
    _as_list,
)
from modssc.data_augmentation.types import AugmentationContext


@pytest.fixture
def ctx():
    return AugmentationContext(seed=0, epoch=0, sample_id=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_text_as_list():
    assert _as_list("abc") == (["abc"], False)
    assert _as_list(["a", "b"]) == (["a", "b"], True)
    with pytest.raises(TypeError):
        _as_list(123)


def test_text_lowercase(ctx, rng):
    op = Lowercase()
    assert op.apply("ABC", rng=rng, ctx=ctx) == "abc"
    assert op.apply(["A", "B"], rng=rng, ctx=ctx) == ["a", "b"]


def test_text_word_dropout(ctx, rng):
    with pytest.raises(ValueError):
        WordDropout(p=-0.1).apply("a b", rng=rng, ctx=ctx)

    op = WordDropout(p=0.5)
    assert op.apply("", rng=rng, ctx=ctx) == ""
    assert op.apply("   ", rng=rng, ctx=ctx) == "   "

    op_high = WordDropout(p=1.0)
    assert op_high.apply("word", rng=rng, ctx=ctx) == "word"

    out = op.apply(["a b", "c d"], rng=rng, ctx=ctx)
    assert isinstance(out, list)
    assert len(out) == 2


def test_text_random_swap(ctx, rng):
    op = RandomSwap(n_swaps=1)
    assert op.apply("word", rng=rng, ctx=ctx) == "word"

    op_neg = RandomSwap(n_swaps=-1)
    assert op_neg.apply("a b c", rng=rng, ctx=ctx) == "a b c"

    out = op.apply(["a b c", "d e f"], rng=rng, ctx=ctx)
    assert isinstance(out, list)
    assert len(out) == 2
