from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry import register_op
from ..types import AugmentationContext, Modality
from ..utils import is_torch_tensor, split_image_channels_last
from .base import AugmentationOp


def _torch_hw_layout(x: Any) -> tuple[int, int, str]:
    """Return (H, W, layout) for a torch tensor image."""
    shape = tuple(int(s) for s in x.shape)
    if len(shape) == 2:
        return shape[0], shape[1], "hw"
    if len(shape) == 3:
        # Heuristic: PyTorch is typically CHW. Assume HWC only if channels (last dim)
        # are small (<=4) and spatial dims are larger.
        if shape[2] <= 4 and shape[0] > shape[2]:
            return shape[0], shape[1], "hwc"
        return shape[1], shape[2], "chw"
    raise ValueError(f"Expected image ndim 2 or 3, got shape={shape}")


def _numpy_hw_layout(arr: np.ndarray) -> tuple[int, int, str]:
    arr, layout = split_image_channels_last(arr)
    if layout == "hw" or layout == "hwc":
        H, W = int(arr.shape[0]), int(arr.shape[1])
    else:  # chw
        H, W = int(arr.shape[1]), int(arr.shape[2])
    return H, W, layout


@register_op("vision.random_horizontal_flip")
@dataclass
class RandomHorizontalFlip(AugmentationOp):
    """Random horizontal flip."""

    op_id: str = "vision.random_horizontal_flip"
    modality: Modality = "vision"
    p: float = 0.5

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        if not (0.0 <= float(self.p) <= 1.0):
            raise ValueError("p must be in [0, 1]")
        do = bool(rng.random() < float(self.p))
        if not do:
            return x
        if is_torch_tensor(x):
            _, _, layout = _torch_hw_layout(x)
            axis = -1 if layout in ("hw", "chw") else -2
            return x.flip((axis,))
        arr = np.asarray(x)
        _, _, layout = _numpy_hw_layout(arr)
        axis = 1 if layout in ("hw", "hwc") else 2
        return np.flip(arr, axis=axis)


@register_op("vision.gaussian_noise")
@dataclass
class GaussianNoise(AugmentationOp):
    """Add zero-mean gaussian noise."""

    op_id: str = "vision.gaussian_noise"
    modality: Modality = "vision"
    std: float = 0.05

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        std = float(self.std)
        if std < 0:
            raise ValueError("std must be >= 0")
        if std == 0:
            return x
        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")
            # Use torch generator for performance on GPU while keeping seed deterministic from rng
            seed = int(rng.integers(0, 1 << 31))
            gen = torch.Generator(device=x.device).manual_seed(seed)
            noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype).mul_(std)
            return x.add(noise)
        arr = np.asarray(x)
        noise = rng.normal(0.0, std, size=arr.shape).astype(arr.dtype, copy=False)
        return arr + noise


@register_op("vision.cutout")
@dataclass
class Cutout(AugmentationOp):
    """Randomly zero out a square region of the image."""

    op_id: str = "vision.cutout"
    modality: Modality = "vision"
    frac: float = 0.25
    fill: float = 0.0

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        frac = float(self.frac)
        if not (0.0 <= frac <= 1.0):
            raise ValueError("frac must be in [0, 1]")
        if frac == 0.0:
            return x

        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")

            H, W, layout = _torch_hw_layout(x)
            size = max(1, int(round(frac * min(H, W))))
            top = int(rng.integers(0, max(1, H - size + 1)))
            left = int(rng.integers(0, max(1, W - size + 1)))

            out = x.clone()
            fill = torch.as_tensor(self.fill, device=x.device, dtype=x.dtype)
            if layout == "hw":
                out[top : top + size, left : left + size] = fill
            elif layout == "hwc":
                out[top : top + size, left : left + size, :] = fill
            else:  # chw
                out[:, top : top + size, left : left + size] = fill
            return out

        arr = np.asarray(x).copy()
        H, W, layout = _numpy_hw_layout(arr)
        size = max(1, int(round(frac * min(H, W))))
        top = int(rng.integers(0, max(1, H - size + 1)))
        left = int(rng.integers(0, max(1, W - size + 1)))

        if layout == "hw":
            arr[top : top + size, left : left + size] = self.fill
        elif layout == "hwc":
            arr[top : top + size, left : left + size, :] = self.fill
        else:  # chw
            arr[:, top : top + size, left : left + size] = self.fill
        return arr


@register_op("vision.random_crop_pad")
@dataclass
class RandomCropPad(AugmentationOp):
    """Pad then crop back to original size (common in CIFAR-style pipelines)."""

    op_id: str = "vision.random_crop_pad"
    modality: Modality = "vision"
    pad: int = 4

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        pad = int(self.pad)
        if pad < 0:
            raise ValueError("pad must be >= 0")
        if pad == 0:
            return x

        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")
            H, W, layout = _torch_hw_layout(x)
            if layout == "hw":
                # F.pad with mode='reflect' expects >=3D tensors for 2D padding.
                chw = x.unsqueeze(0)  # (1, H, W)
                padded = torch.nn.functional.pad(chw, (pad, pad, pad, pad), mode="reflect")
                H2, W2 = int(padded.shape[1]), int(padded.shape[2])
                top = int(rng.integers(0, H2 - H + 1))
                left = int(rng.integers(0, W2 - W + 1))
                return padded[:, top : top + H, left : left + W].squeeze(0)
            if layout == "hwc":
                # pad H/W dims by permuting to CHW
                chw = x.permute(2, 0, 1)
                padded = torch.nn.functional.pad(chw, (pad, pad, pad, pad), mode="reflect").permute(
                    1, 2, 0
                )
                H2, W2 = int(padded.shape[0]), int(padded.shape[1])
                top = int(rng.integers(0, H2 - H + 1))
                left = int(rng.integers(0, W2 - W + 1))
                return padded[top : top + H, left : left + W, :]
            # chw
            padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
            H2, W2 = int(padded.shape[1]), int(padded.shape[2])
            top = int(rng.integers(0, H2 - H + 1))
            left = int(rng.integers(0, W2 - W + 1))
            return padded[:, top : top + H, left : left + W]

        arr = np.asarray(x)
        H, W, layout = _numpy_hw_layout(arr)

        if layout == "hw":
            padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
            top = int(rng.integers(0, 2 * pad + 1))
            left = int(rng.integers(0, 2 * pad + 1))
            return padded[top : top + H, left : left + W]
        if layout == "hwc":
            padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            top = int(rng.integers(0, 2 * pad + 1))
            left = int(rng.integers(0, 2 * pad + 1))
            return padded[top : top + H, left : left + W, :]
        # chw
        padded = np.pad(arr, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        top = int(rng.integers(0, 2 * pad + 1))
        left = int(rng.integers(0, 2 * pad + 1))
        return padded[:, top : top + H, left : left + W]
