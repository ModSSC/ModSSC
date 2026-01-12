from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation import AugmentationContext
from modssc.data_augmentation.ops.text import (
    Lowercase,
    RandomSwap,
    WordDropout,
    _as_list,
)
from modssc.data_augmentation.registry import get_op
from modssc.data_augmentation.utils import make_numpy_rng


def _make_ctx_rng(seed: int = 0) -> tuple[AugmentationContext, np.random.Generator]:
    ctx = AugmentationContext(seed=seed, epoch=0, sample_id=0, modality="text")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    return ctx, rng


def test_word_dropout_p1_keeps_one_token() -> None:
    op = get_op("text.word_dropout", p=1.0)
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=0, modality="text")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    out = op.apply("a b c", rng=rng, ctx=ctx)
    assert out in {"a", "b", "c"}


def test_random_swap_deterministic() -> None:
    op = get_op("text.random_swap", n_swaps=2)
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=42, modality="text")

    rng1 = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    rng2 = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)

    s1 = op.apply("one two three four", rng=rng1, ctx=ctx)
    s2 = op.apply("one two three four", rng=rng2, ctx=ctx)

    assert s1 == s2


def test_text_as_list():
    assert _as_list("abc") == (["abc"], False)
    assert _as_list(["a", "b"]) == (["a", "b"], True)
    with pytest.raises(TypeError):
        _as_list(123)


def test_text_lowercase():
    ctx, rng = _make_ctx_rng()
    op = Lowercase()
    assert op.apply("ABC", rng=rng, ctx=ctx) == "abc"
    assert op.apply(["A", "B"], rng=rng, ctx=ctx) == ["a", "b"]


def test_text_word_dropout():
    ctx, rng = _make_ctx_rng()
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


def test_text_random_swap():
    ctx, rng = _make_ctx_rng()
    op = RandomSwap(n_swaps=1)
    assert op.apply("word", rng=rng, ctx=ctx) == "word"

    op_neg = RandomSwap(n_swaps=-1)
    assert op_neg.apply("a b c", rng=rng, ctx=ctx) == "a b c"

    out = op.apply(["a b c", "d e f"], rng=rng, ctx=ctx)
    assert isinstance(out, list)
    assert len(out) == 2
