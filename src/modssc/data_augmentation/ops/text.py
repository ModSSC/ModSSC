from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry import register_op
from ..types import AugmentationContext, Modality
from ..utils import is_torch_tensor
from .base import AugmentationOp


def _as_list(x: Any) -> tuple[list[str], bool]:
    if isinstance(x, str):
        return [x], False
    if isinstance(x, (list, tuple)):
        return [str(s) for s in x], True
    raise TypeError(f"Expected str or list[str], got {type(x).__name__}")


@register_op("text.lowercase")
@dataclass
class Lowercase(AugmentationOp):
    """Lowercase text."""

    op_id: str = "text.lowercase"
    modality: Modality = "text"

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        items, many = _as_list(x)
        out = [s.lower() for s in items]
        return out if many else out[0]


class _PerItemTextOp(AugmentationOp):
    def _apply_one(self, s: str, *, rng: np.random.Generator) -> str:
        raise NotImplementedError

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        items, many = _as_list(x)
        out = [self._apply_one(s, rng=rng) for s in items]
        return out if many else out[0]


@register_op("text.word_dropout")
@dataclass
class WordDropout(_PerItemTextOp):
    """Randomly drop words from whitespace-tokenized text."""

    op_id: str = "text.word_dropout"
    modality: Modality = "text"
    p: float = 0.1

    def _apply_one(self, s: str, *, rng: np.random.Generator) -> str:
        tokens = s.split()
        if not tokens:
            return s
        p = float(self.p)
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        keep = [t for t in tokens if rng.random() >= p]
        if not keep:
            keep = [tokens[int(rng.integers(0, len(tokens)))]]
        return " ".join(keep)


@register_op("text.random_swap")
@dataclass
class RandomSwap(_PerItemTextOp):
    """Randomly swap two words N times."""

    op_id: str = "text.random_swap"
    modality: Modality = "text"
    n_swaps: int = 1

    def _apply_one(self, s: str, *, rng: np.random.Generator) -> str:
        tokens = s.split()
        if len(tokens) < 2:
            return s
        n_swaps = max(0, int(self.n_swaps))
        for _ in range(n_swaps):
            i, j = rng.integers(0, len(tokens), size=(2,))
            i = int(i)
            j = int(j)
            tokens[i], tokens[j] = tokens[j], tokens[i]
        return " ".join(tokens)


def _swap_token_positions_numpy(
    seq: np.ndarray,
    *,
    rng: np.random.Generator,
    n_swaps: int,
    pad_token_id: int,
) -> np.ndarray:
    out = np.array(seq, copy=True)
    valid = np.flatnonzero(out != pad_token_id)
    if valid.size < 2 or n_swaps <= 0:
        return out
    for _ in range(int(n_swaps)):
        i, j = rng.choice(valid, size=2, replace=False)
        out[int(i)], out[int(j)] = out[int(j)], out[int(i)]
    return out


@register_op("text.token_mask")
@dataclass
class TokenMask(AugmentationOp):
    """Mask token ids after tokenization by replacing them with a sentinel id."""

    op_id: str = "text.token_mask"
    modality: Modality = "text"
    p: float = 0.1
    mask_token_id: int = 1
    pad_token_id: int = 0

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        p = float(self.p)
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if p == 0.0:
            return x

        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")
            out = x.clone()
            mask_np = rng.random(size=tuple(out.shape)) < p
            mask = torch.from_numpy(mask_np).to(device=out.device, dtype=torch.bool)
            eligible = out.ne(int(self.pad_token_id))
            out[mask & eligible] = int(self.mask_token_id)
            return out

        arr = np.asarray(x)
        out = np.array(arr, copy=True)
        mask = rng.random(size=out.shape) < p
        eligible = out != int(self.pad_token_id)
        out[mask & eligible] = np.asarray(self.mask_token_id, dtype=out.dtype)
        return out


@register_op("text.token_swap")
@dataclass
class TokenSwap(AugmentationOp):
    """Swap token-id positions within each tokenized sequence."""

    op_id: str = "text.token_swap"
    modality: Modality = "text"
    n_swaps: int = 1
    pad_token_id: int = 0

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        n_swaps = int(self.n_swaps)
        if n_swaps <= 0:
            return x

        if is_torch_tensor(x):
            out = x.clone()
            if int(out.ndim) == 1:
                valid = (
                    out.ne(int(self.pad_token_id))
                    .nonzero(as_tuple=False)
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                if valid.size < 2:
                    return out
                for _ in range(n_swaps):
                    i, j = rng.choice(valid, size=2, replace=False)
                    tmp = out[int(i)].clone()
                    out[int(i)] = out[int(j)]
                    out[int(j)] = tmp
                return out
            if int(out.ndim) == 2:
                for row in range(int(out.shape[0])):
                    valid = (
                        out[row]
                        .ne(int(self.pad_token_id))
                        .nonzero(as_tuple=False)
                        .reshape(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    if valid.size < 2:
                        continue
                    for _ in range(n_swaps):
                        i, j = rng.choice(valid, size=2, replace=False)
                        tmp = out[row, int(i)].clone()
                        out[row, int(i)] = out[row, int(j)]
                        out[row, int(j)] = tmp
                return out
            raise TypeError("text.token_swap expects a 1D or 2D token tensor")

        arr = np.asarray(x)
        if arr.ndim == 1:
            return _swap_token_positions_numpy(
                arr,
                rng=rng,
                n_swaps=n_swaps,
                pad_token_id=int(self.pad_token_id),
            )
        if arr.ndim == 2:
            out = np.array(arr, copy=True)
            for row in range(out.shape[0]):
                out[row] = _swap_token_positions_numpy(
                    out[row],
                    rng=rng,
                    n_swaps=n_swaps,
                    pad_token_id=int(self.pad_token_id),
                )
            return out
        raise TypeError("text.token_swap expects a 1D or 2D token array")
