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
    # Backward-compatible parameter. If provided, overrides length.
    frac: float | None = None
    n_holes: int = 1
    length: int = 16
    fill: float = 0.0

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        frac = self.frac
        n_holes = int(self.n_holes)
        length = int(self.length)
        if frac is not None:
            if self.length != 16 or self.n_holes != 1:
                raise ValueError("Use either frac or length/n_holes, not both.")
            frac = float(frac)
            if not (0.0 <= frac <= 1.0):
                raise ValueError("frac must be in [0, 1]")
            if frac == 0.0:
                return x
        else:
            if length <= 0 or n_holes <= 0:
                return x

        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")

            H, W, layout = _torch_hw_layout(x)
            if frac is not None:
                length = max(1, int(round(float(frac) * min(H, W))))
            out = x.clone()
            fill = torch.as_tensor(self.fill, device=x.device, dtype=x.dtype)

            for _ in range(n_holes):
                top = int(rng.integers(0, max(1, H - length + 1)))
                left = int(rng.integers(0, max(1, W - length + 1)))

                if layout == "hw":
                    out[top : top + length, left : left + length] = fill
                elif layout == "hwc":
                    out[top : top + length, left : left + length, :] = fill
                else:  # chw
                    out[:, top : top + length, left : left + length] = fill
            return out

        arr = np.asarray(x).copy()
        H, W, layout = _numpy_hw_layout(arr)
        if frac is not None:
            length = max(1, int(round(float(frac) * min(H, W))))

        for _ in range(n_holes):
            top = int(rng.integers(0, max(1, H - length + 1)))
            left = int(rng.integers(0, max(1, W - length + 1)))

            if layout == "hw":
                arr[top : top + length, left : left + length] = self.fill
            elif layout == "hwc":
                arr[top : top + length, left : left + length, :] = self.fill
            else:  # chw
                arr[:, top : top + length, left : left + length] = self.fill
        return arr


@register_op("vision.random_crop_pad")
@dataclass
class RandomCropPad(AugmentationOp):
    """Pad then crop back to original size (common in CIFAR-style pipelines)."""

    op_id: str = "vision.random_crop_pad"
    modality: Modality = "vision"
    # Backward-compatible parameter. If provided, overrides padding.
    pad: int | None = None
    padding: int = 4

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        pad_val = self.pad
        if pad_val is not None:
            if self.padding != 4:
                raise ValueError("Use either pad or padding, not both.")
            pad = int(pad_val)
        else:
            pad = int(self.padding)
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


@register_op("vision.randaugment")
@dataclass
class RandAugment(AugmentationOp):
    """RandAugment strong policy used by the FixMatch paper ablation.

    The paper setting is ``N=2, M=10``.  Torchvision supplies the canonical
    geometric/photometric operation space; its RNG is seeded from ModSSC's
    replayable augmentation context for deterministic task resumption.
    """

    op_id: str = "vision.randaugment"
    modality: Modality = "vision"
    num_ops: int = 2
    magnitude: int = 10
    num_magnitude_bins: int = 31
    fill: float | tuple[float, ...] | None = None

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        if int(self.num_ops) <= 0:
            raise ValueError("num_ops must be >= 1")
        if int(self.num_magnitude_bins) <= 1:
            raise ValueError("num_magnitude_bins must be >= 2")
        if not (0 <= int(self.magnitude) < int(self.num_magnitude_bins)):
            raise ValueError("magnitude must be in [0, num_magnitude_bins)")

        import importlib

        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")

        is_numpy = not is_torch_tensor(x)
        tensor = torch.from_numpy(np.asarray(x)) if is_numpy else x
        height, width, layout = _torch_hw_layout(tensor)
        if height <= 0 or width <= 0:
            raise ValueError("RandAugment requires non-empty spatial dimensions")
        if layout == "hw":
            policy_input = tensor.unsqueeze(0)
        elif layout == "hwc":
            policy_input = tensor.permute(2, 0, 1)
        else:
            policy_input = tensor
        if int(policy_input.shape[0]) not in (1, 3):
            raise ValueError("RandAugment requires one or three image channels")

        floating_input = bool(policy_input.is_floating_point())
        input_dtype = policy_input.dtype
        if floating_input:
            if not bool(torch.isfinite(policy_input).all().item()):
                raise ValueError("RandAugment floating-point input must contain finite values")
            min_value = float(policy_input.min().item())
            max_value = float(policy_input.max().item())
            if min_value < 0.0 or max_value > 1.0:
                raise ValueError("RandAugment floating-point input must be in [0, 1]")
            policy_input = policy_input.mul(255.0).round().clamp_(0.0, 255.0).to(torch.uint8)

        fill: float | list[float] | None
        if isinstance(self.fill, tuple):
            fill = [float(value) for value in self.fill]
        else:
            fill = None if self.fill is None else float(self.fill)
        if floating_input and fill is not None:
            fill_values = fill if isinstance(fill, list) else [fill]
            if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in fill_values):
                raise ValueError("RandAugment fill must be finite and in [0, 1] for float input")
            scaled_fill = [int(round(value * 255.0)) for value in fill_values]
            fill = scaled_fill if isinstance(fill, list) else scaled_fill[0]
        transform = torchvision.transforms.RandAugment(
            num_ops=int(self.num_ops),
            magnitude=int(self.magnitude),
            num_magnitude_bins=int(self.num_magnitude_bins),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            fill=fill,
        )
        policy_seed = int(rng.integers(0, (1 << 31) - 1))
        devices = [policy_input.device] if policy_input.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(policy_seed)
            if policy_input.device.type == "cuda":
                torch.cuda.manual_seed_all(policy_seed)
            augmented = transform(policy_input)
        if tuple(augmented.shape) != tuple(policy_input.shape):
            raise ValueError("RandAugment changed the image shape")
        if floating_input:
            augmented = augmented.to(dtype=input_dtype).div_(255.0)

        if layout == "hw":
            augmented = augmented.squeeze(0)
        elif layout == "hwc":
            augmented = augmented.permute(1, 2, 0)
        if is_numpy:
            return augmented.cpu().numpy()
        return augmented
