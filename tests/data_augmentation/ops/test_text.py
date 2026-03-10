from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation import AugmentationContext
from modssc.data_augmentation.ops.text import (
    Lowercase,
    RandomSwap,
    TokenMask,
    TokenSwap,
    WordDropout,
    _as_list,
    _PerItemTextOp,
    _swap_token_positions_numpy,
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


def test_text_token_mask_masks_only_non_pad_tokens_numpy() -> None:
    ctx, rng = _make_ctx_rng()
    op = TokenMask(p=1.0, mask_token_id=1, pad_token_id=0)
    x = np.array([[5, 6, 0], [7, 0, 0]], dtype=np.int64)
    out = op.apply(x, rng=rng, ctx=ctx)
    expected = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_text_token_mask_masks_only_non_pad_tokens_torch() -> None:
    torch = pytest.importorskip("torch")
    ctx, rng = _make_ctx_rng()
    op = TokenMask(p=1.0, mask_token_id=1, pad_token_id=0)
    x = torch.tensor([[9, 3, 0], [4, 0, 2]], dtype=torch.long)
    out = op.apply(x, rng=rng, ctx=ctx)
    expected = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.long)
    assert torch.equal(out, expected)


def test_text_token_mask_validation_and_passthrough() -> None:
    ctx, rng = _make_ctx_rng()
    x = np.array([5, 6, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="p must be in \\[0, 1\\]"):
        TokenMask(p=1.5).apply(x, rng=rng, ctx=ctx)

    out = TokenMask(p=0.0, mask_token_id=1, pad_token_id=0).apply(x, rng=rng, ctx=ctx)
    assert out is x


def test_text_token_swap_preserves_pad_suffix_numpy() -> None:
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=42, modality="text")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    op = TokenSwap(n_swaps=2, pad_token_id=0)
    x = np.array([10, 20, 30, 0, 0], dtype=np.int64)
    out = op.apply(x, rng=rng, ctx=ctx)
    assert out.shape == x.shape
    np.testing.assert_array_equal(np.sort(out[:3]), np.sort(x[:3]))
    np.testing.assert_array_equal(out[3:], x[3:])


def test_text_token_swap_batch_torch() -> None:
    torch = pytest.importorskip("torch")
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=7, modality="text")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    op = TokenSwap(n_swaps=1, pad_token_id=0)
    x = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    out = op.apply(x, rng=rng, ctx=ctx)
    assert tuple(out.shape) == (2, 4)
    assert torch.equal(out[:, 2:] == 0, x[:, 2:] == 0)
    assert torch.equal(torch.sort(out[0, :3]).values, torch.sort(x[0, :3]).values)
    assert torch.equal(torch.sort(out[1, :2]).values, torch.sort(x[1, :2]).values)


def test_text_token_swap_numpy_edge_cases() -> None:
    ctx, rng = _make_ctx_rng()

    x = np.array([9, 0, 0], dtype=np.int64)
    out = TokenSwap(n_swaps=1, pad_token_id=0).apply(x, rng=rng, ctx=ctx)
    np.testing.assert_array_equal(out, x)

    batch = np.array([[1, 2, 3, 0], [4, 0, 0, 0]], dtype=np.int64)
    out_batch = TokenSwap(n_swaps=1, pad_token_id=0).apply(batch, rng=rng, ctx=ctx)
    assert out_batch.shape == batch.shape
    np.testing.assert_array_equal(np.sort(out_batch[0, :3]), np.sort(batch[0, :3]))
    np.testing.assert_array_equal(out_batch[1], batch[1])

    passthrough = TokenSwap(n_swaps=0, pad_token_id=0).apply(batch, rng=rng, ctx=ctx)
    assert passthrough is batch

    with pytest.raises(TypeError, match="1D or 2D token array"):
        TokenSwap(n_swaps=1, pad_token_id=0).apply(
            np.zeros((1, 2, 3), dtype=np.int64), rng=rng, ctx=ctx
        )


def test_text_token_swap_torch_edge_cases() -> None:
    torch = pytest.importorskip("torch")
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=11, modality="text")

    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    x = torch.tensor([1, 2, 3, 0], dtype=torch.long)
    out = TokenSwap(n_swaps=1, pad_token_id=0).apply(x, rng=rng, ctx=ctx)
    assert torch.equal(torch.sort(out[:3]).values, torch.sort(x[:3]).values)
    assert torch.equal(out[3:], x[3:])

    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    one_valid = torch.tensor([5, 0, 0], dtype=torch.long)
    out_one_valid = TokenSwap(n_swaps=1, pad_token_id=0).apply(one_valid, rng=rng, ctx=ctx)
    assert torch.equal(out_one_valid, one_valid)

    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    batch = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]], dtype=torch.long)
    out_batch = TokenSwap(n_swaps=1, pad_token_id=0).apply(batch, rng=rng, ctx=ctx)
    assert torch.equal(torch.sort(out_batch[0, :3]).values, torch.sort(batch[0, :3]).values)
    assert torch.equal(out_batch[1], batch[1])

    with pytest.raises(TypeError, match="1D or 2D token tensor"):
        TokenSwap(n_swaps=1, pad_token_id=0).apply(
            torch.zeros((1, 2, 3), dtype=torch.long), rng=rng, ctx=ctx
        )


def test_swap_token_positions_numpy_passthrough() -> None:
    rng = np.random.default_rng(0)
    x = np.array([1, 2, 0], dtype=np.int64)
    out = _swap_token_positions_numpy(x, rng=rng, n_swaps=0, pad_token_id=0)
    np.testing.assert_array_equal(out, x)
    assert out is not x


def test_per_item_text_op_default_apply_one_raises():
    op = _PerItemTextOp(op_id="text._per_item")
    with pytest.raises(NotImplementedError):
        op._apply_one("abc", rng=np.random.default_rng(0))
