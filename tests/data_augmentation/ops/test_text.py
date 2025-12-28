from __future__ import annotations

from modssc.data_augmentation import AugmentationContext
from modssc.data_augmentation.registry import get_op
from modssc.data_augmentation.utils import make_numpy_rng


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
