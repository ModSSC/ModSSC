from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .errors import DataAugmentationValidationError

CifarAugmentationProfile = Literal["google_fixmatch_ra", "torchssl_ra"]
CifarAugmentationView = Literal[
    "labeled_weak",
    "unlabeled_weak",
    "unlabeled_strong",
]

_VIEWS: dict[CifarAugmentationView, int] = {
    "labeled_weak": 0xA24BAED4,
    "unlabeled_weak": 0x9FB21C65,
    "unlabeled_strong": 0xC13FA9A9,
}
_GOOGLE_OPERATIONS = (
    "Identity",
    "AutoContrast",
    "Equalize",
    "Rotate",
    "Solarize",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "TranslateX",
    "TranslateY",
    "Posterize",
    "ShearY",
)
_TORCHSSL_OPERATIONS = (
    "AutoContrast",
    "Brightness",
    "Color",
    "Contrast",
    "Equalize",
    "Identity",
    "Posterize",
    "Rotate",
    "Sharpness",
    "ShearX",
    "ShearY",
    "Solarize",
    "TranslateX",
    "TranslateY",
)


def resolve_cifar_augmentation_profile(profile: str) -> CifarAugmentationProfile:
    """Validate and return a selectable CIFAR augmentation implementation."""

    if profile == "google_fixmatch_ra":
        return "google_fixmatch_ra"
    if profile == "torchssl_ra":
        return "torchssl_ra"
    raise DataAugmentationValidationError(
        f"Unknown CIFAR reference augmentation profile: {profile!r}."
    )


@dataclass(frozen=True)
class CifarAugmentationDraws:
    """Vectorized, per-example random choices for one batch and one view."""

    profile: CifarAugmentationProfile
    view: CifarAugmentationView
    sample_ids: np.ndarray
    occurrence_ids: np.ndarray
    crop_top: np.ndarray
    crop_left: np.ndarray
    flip: np.ndarray
    operation_indices: np.ndarray
    magnitudes: np.ndarray
    apply_operation: np.ndarray
    signs: np.ndarray
    cutout_size_fraction: np.ndarray
    cutout_center_y: np.ndarray
    cutout_center_x: np.ndarray


def _stateless_uniform(
    sample_ids: np.ndarray,
    *,
    occurrence_ids: np.ndarray,
    seed: int,
    step: int,
    view_code: int,
    columns: int,
) -> np.ndarray:
    """Generate order-independent random values with a vectorized SplitMix64 PRF."""

    ids = np.asarray(sample_ids, dtype=np.int64)
    unsigned_ids = ids.astype(np.uint64, copy=False).reshape(-1, 1)
    unsigned_occurrences = (
        np.asarray(occurrence_ids, dtype=np.int64).astype(np.uint64, copy=False).reshape(-1, 1)
    )
    counters = np.arange(columns, dtype=np.uint64).reshape(1, -1)
    mask = (1 << 64) - 1
    seed_word = np.uint64(int(seed) & mask)
    step_word = np.uint64(int(step) & mask)
    view_word = np.uint64(int(view_code) & mask)
    with np.errstate(over="ignore"):
        value = (
            unsigned_ids * np.uint64(0xD6E8FEB86659FD93)
            + unsigned_occurrences * np.uint64(0x8EBC6AF09C88C6E3)
            + counters * np.uint64(0x9E3779B97F4A7C15)
            + seed_word * np.uint64(0xA0761D6478BD642F)
            + step_word * np.uint64(0xE7037ED1A0B428DB)
            + view_word
        )
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        value = value ^ (value >> np.uint64(31))
    return (value >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))


class CifarReferenceAugmentation:
    """Compiled, deterministic CIFAR augmentation for fixed-step Match runs.

    Policy draws are generated for the complete batch in one vectorized call
    and are keyed by ``(seed, step, sample_id, occurrence_id, view)``.
    ``occurrence_id`` is essential for replacement sampling: two occurrences
    of the same CIFAR example in one batch must receive independent transforms,
    as they do in the pinned Dataset implementations.  Strong pixel operations
    use the exact Pillow primitives and ordering of the pinned sources.
    """

    padding = 4

    def __init__(self, profile: str, *, seed: int = 0) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise DataAugmentationValidationError("seed must be an integer.")
        self.profile = resolve_cifar_augmentation_profile(profile)
        self.seed = seed
        self.operation_names = (
            _GOOGLE_OPERATIONS if self.profile == "google_fixmatch_ra" else _TORCHSSL_OPERATIONS
        )
        self._backend: tuple[Any, Any, Any] | None = None
        configured_threads = os.environ.get("MODSSC_AUGMENT_THREADS")
        if configured_threads is None:
            self._worker_count = max(1, min(8, int(os.cpu_count() or 1)))
        else:
            try:
                self._worker_count = int(configured_threads)
            except ValueError as exc:
                raise DataAugmentationValidationError(
                    "MODSSC_AUGMENT_THREADS must be an integer."
                ) from exc
            if self._worker_count <= 0:
                raise DataAugmentationValidationError("MODSSC_AUGMENT_THREADS must be positive.")
        self._executor: ThreadPoolExecutor | None = None

    def _load_backend(self) -> tuple[Any, Any, Any]:
        if self._backend is None:
            import importlib

            torch = importlib.import_module("torch")
            functional = importlib.import_module("torchvision.transforms.functional")
            transforms = importlib.import_module("torchvision.transforms")
            self._backend = (torch, functional, transforms.InterpolationMode)
        return self._backend

    def sample_batch(
        self,
        sample_ids: Any,
        *,
        occurrence_ids: Any | None = None,
        step: int,
        view: CifarAugmentationView,
    ) -> CifarAugmentationDraws:
        ids = _sample_ids(sample_ids)
        occurrences = (
            np.arange(ids.size, dtype=np.int64)
            if occurrence_ids is None
            else _sample_ids(occurrence_ids)
        )
        if occurrences.shape != ids.shape:
            raise DataAugmentationValidationError(
                "sample_ids and occurrence_ids must have equal size."
            )
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise DataAugmentationValidationError("step must be a non-negative integer.")
        if view not in _VIEWS:
            raise DataAugmentationValidationError(f"Unknown CIFAR augmentation view: {view!r}.")

        strong = view == "unlabeled_strong"
        num_ops = 2 if self.profile == "google_fixmatch_ra" and strong else 0
        if self.profile == "torchssl_ra" and strong:
            num_ops = 3
        # crop(2), flip(1), then four values per policy operation and three
        # cutout values. Keeping a fixed layout makes the draw oracle explicit.
        random_values = _stateless_uniform(
            ids,
            occurrence_ids=occurrences,
            seed=self.seed,
            step=step,
            view_code=_VIEWS[view],
            columns=3 + 4 * num_ops + 3,
        )
        crop_top = np.floor(random_values[:, 0] * (2 * self.padding + 1)).astype(np.int64)
        crop_left = np.floor(random_values[:, 1] * (2 * self.padding + 1)).astype(np.int64)
        flip = random_values[:, 2] < 0.5
        if num_ops:
            policy = random_values[:, 3 : 3 + 4 * num_ops].reshape(-1, num_ops, 4)
            operation_indices = np.floor(policy[:, :, 0] * len(self.operation_names)).astype(
                np.int64
            )
            if self.profile == "google_fixmatch_ra":
                magnitudes = 1.0 + np.floor(policy[:, :, 1] * 9.0)
                apply_operation = policy[:, :, 2] < 0.5
            else:
                magnitudes = policy[:, :, 1]
                apply_operation = np.ones((ids.size, num_ops), dtype=bool)
            signs = np.where(policy[:, :, 3] < 0.5, -1.0, 1.0)
        else:
            operation_indices = np.empty((ids.size, 0), dtype=np.int64)
            magnitudes = np.empty((ids.size, 0), dtype=np.float64)
            apply_operation = np.empty((ids.size, 0), dtype=bool)
            signs = np.empty((ids.size, 0), dtype=np.float64)

        cutout_values = random_values[:, -3:]
        if not strong:
            cutout_size = np.zeros(ids.size, dtype=np.float64)
        elif self.profile == "google_fixmatch_ra":
            # Google AugmentPoolRAMC calls cutout_numpy with its default
            # 16-pixel square on CIFAR.
            cutout_size = np.full(ids.size, 16.0 / 32.0, dtype=np.float64)
        else:
            # TorchSSL's custom RandAugment ignores m and samples Cutout in
            # [0, 0.5) after three operations.
            cutout_size = cutout_values[:, 0] * 0.5
        return CifarAugmentationDraws(
            profile=self.profile,
            view=view,
            sample_ids=ids.copy(),
            occurrence_ids=occurrences.copy(),
            crop_top=crop_top,
            crop_left=crop_left,
            flip=flip,
            operation_indices=operation_indices,
            magnitudes=magnitudes,
            apply_operation=apply_operation,
            signs=signs,
            cutout_size_fraction=cutout_size,
            cutout_center_y=cutout_values[:, 1],
            cutout_center_x=cutout_values[:, 2],
        )

    def apply_batch(
        self,
        batch: Any,
        *,
        sample_ids: Any,
        occurrence_ids: Any | None = None,
        step: int,
        view: CifarAugmentationView,
    ) -> Any:
        """Apply a compiled reference policy to an already selected batch."""

        draws = self.sample_batch(
            sample_ids,
            occurrence_ids=occurrence_ids,
            step=step,
            view=view,
        )
        return self.apply_draws(batch, draws)

    def apply_draws(self, batch: Any, draws: CifarAugmentationDraws) -> Any:
        """Apply precomputed draws, useful for trajectory oracles and replay."""

        torch, _, _ = self._load_backend()
        is_numpy = isinstance(batch, np.ndarray)
        tensor = torch.from_numpy(batch) if is_numpy else batch
        if not isinstance(tensor, torch.Tensor) or int(tensor.ndim) != 4:
            raise DataAugmentationValidationError(
                "CIFAR reference augmentation expects a 4D NumPy array or torch tensor."
            )
        channels_last = int(tensor.shape[1]) != 3 and int(tensor.shape[-1]) == 3
        if int(tensor.shape[1]) != 3 and not channels_last:
            raise DataAugmentationValidationError(
                "CIFAR reference augmentation requires 3 channels."
            )
        nchw = tensor.permute(0, 3, 1, 2) if channels_last else tensor
        if int(nchw.shape[0]) != int(draws.sample_ids.size):
            raise DataAugmentationValidationError(
                "batch and augmentation draws must have equal size."
            )
        if draws.profile != self.profile:
            raise DataAugmentationValidationError("augmentation draws use a different profile.")
        if int(nchw.shape[0]) == 0:
            return batch.copy() if is_numpy else batch.clone()

        floating = bool(nchw.is_floating_point())
        original_dtype = nchw.dtype
        if floating:
            if not bool(torch.isfinite(nchw).all().item()):
                raise DataAugmentationValidationError("floating CIFAR input must be finite.")
            lower, upper = (-1.0, 1.0) if self.profile == "google_fixmatch_ra" else (0.0, 1.0)
            if float(nchw.min().item()) < lower or float(nchw.max().item()) > upper:
                raise DataAugmentationValidationError(
                    f"floating CIFAR input must be in [{lower:g}, {upper:g}]."
                )
            if self.profile == "google_fixmatch_ra":
                # Exact inverse used by the pinned pil_wrap helper: NumPy's
                # uint8 cast truncates instead of rounding.
                working = nchw.add(1.0).mul(127.5).to(torch.uint8)
            else:
                working = nchw.mul(255.0).round().to(torch.uint8)
        elif nchw.dtype == torch.uint8:
            working = nchw.clone()
        else:
            raise DataAugmentationValidationError("CIFAR input dtype must be uint8 or floating.")

        height, width = int(working.shape[-2]), int(working.shape[-1])
        if height <= self.padding or width <= self.padding:
            raise DataAugmentationValidationError("CIFAR images are too small for reflect padding.")
        strong = draws.view == "unlabeled_strong"
        if self.profile == "torchssl_ra" and strong:
            working = self._apply_torchssl_policy(working, draws)

        # Both sources flip before cropping. TorchSSL applies its strong
        # RandAugment/Cutout to the PIL image before this weak transform,
        # whereas Google applies the shared weak path first.
        flip_mask = torch.as_tensor(draws.flip, device=working.device).reshape(-1, 1, 1, 1)
        working = torch.where(flip_mask, working.flip(-1), working)
        padded = torch.nn.functional.pad(
            working,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )
        working = torch.stack(
            [
                padded[
                    index,
                    :,
                    int(draws.crop_top[index]) : int(draws.crop_top[index]) + height,
                    int(draws.crop_left[index]) : int(draws.crop_left[index]) + width,
                ]
                for index in range(int(working.shape[0]))
            ],
            dim=0,
        )

        if self.profile == "google_fixmatch_ra" and strong:
            normalized = self._apply_google_policy(working, draws, dtype=original_dtype)
            normalized = self._apply_cutout(normalized, draws)
            if floating:
                working = normalized
            else:
                working = normalized.add(1.0).mul(127.5).round().clamp_(0, 255).to(torch.uint8)
        elif floating:
            if self.profile == "google_fixmatch_ra":
                working = working.to(dtype=original_dtype).div_(127.5).sub_(1.0)
            else:
                working = working.to(dtype=original_dtype).div_(255.0)
        output = working.permute(0, 2, 3, 1) if channels_last else working
        return output.cpu().numpy() if is_numpy else output

    def _thread_map(self, tasks: list[tuple[Any, ...]]) -> list[np.ndarray]:
        if self._worker_count == 1 or len(tasks) <= 1:
            return [self._pillow_task(task) for task in tasks]
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._worker_count,
                thread_name_prefix="modssc-cifar-reference",
            )
        return list(self._executor.map(self._pillow_task, tasks))

    def _apply_google_policy(self, batch: Any, draws: CifarAugmentationDraws, *, dtype: Any) -> Any:
        torch, _, _ = self._load_backend()
        device = batch.device
        cpu_images = batch.detach().cpu().permute(0, 2, 3, 1).numpy()
        tasks = [
            ("google", cpu_images[index], draws, index) for index in range(int(batch.shape[0]))
        ]
        output = np.stack(self._thread_map(tasks), axis=0)
        return torch.from_numpy(output).to(device=device, dtype=dtype)

    def _apply_torchssl_policy(self, batch: Any, draws: CifarAugmentationDraws) -> Any:
        torch, _, _ = self._load_backend()
        device = batch.device
        cpu_images = batch.detach().cpu().permute(0, 2, 3, 1).numpy()
        tasks = [
            ("torchssl", cpu_images[index], draws, index) for index in range(int(batch.shape[0]))
        ]
        output = np.stack(self._thread_map(tasks), axis=0)
        return torch.from_numpy(output).permute(0, 3, 1, 2).to(device=device)

    def _pillow_task(self, task: tuple[Any, ...]) -> np.ndarray:
        import PIL.Image
        import PIL.ImageDraw

        stack, raw_image, draws, sample_index = task
        if stack == "google":
            image = PIL.Image.fromarray(raw_image).convert("RGBA")
        else:
            image = PIL.Image.fromarray(raw_image).convert("RGB")
        for policy_index in range(int(draws.operation_indices.shape[1])):
            if bool(draws.apply_operation[sample_index, policy_index]):
                image = self._apply_pillow_operation(
                    image,
                    operation_index=int(draws.operation_indices[sample_index, policy_index]),
                    magnitude=float(draws.magnitudes[sample_index, policy_index]),
                    sign=float(draws.signs[sample_index, policy_index]),
                )
        if stack == "torchssl":
            fraction = float(draws.cutout_size_fraction[sample_index])
            if fraction > 0.0:
                width, height = image.size
                size = fraction * float(width)
                x0 = int(
                    max(
                        0.0,
                        float(draws.cutout_center_x[sample_index]) * width - size / 2.0,
                    )
                )
                y0 = int(
                    max(
                        0.0,
                        float(draws.cutout_center_y[sample_index]) * height - size / 2.0,
                    )
                )
                x1 = min(float(width), float(x0) + size)
                y1 = min(float(height), float(y0) + size)
                image = image.copy()
                PIL.ImageDraw.Draw(image).rectangle(
                    (x0, y0, x1, y1),
                    (125, 123, 114),
                )
            return np.asarray(image, dtype=np.uint8).copy()

        rgba = np.asarray(image, dtype=np.float64) / 255.0
        normalized = (rgba[:, :, :3] - 0.5) / 0.5
        normalized[rgba[:, :, 3] == 0] = 0.0
        return np.transpose(normalized.astype(np.float32), (2, 0, 1))

    def _apply_pillow_operation(
        self,
        image: Any,
        *,
        operation_index: int,
        magnitude: float,
        sign: float,
    ) -> Any:
        import PIL.Image
        import PIL.ImageEnhance
        import PIL.ImageOps

        name = self.operation_names[operation_index]
        google = self.profile == "google_fixmatch_ra"
        if google:
            level = magnitude
            rotate = sign * int(level * 30.0 / 10.0)
            shear = sign * (level * 0.3 / 10.0)
            translate = sign * int(level * 10.0 / 10.0)
            enhancement = level * 1.8 / 10.0 + 0.1
            solarize = 256 - int(level * 256.0 / 10.0)
            posterize = 4 - int(level * 4.0 / 10.0)
        else:
            rotate = -30.0 + 60.0 * magnitude
            shear = -0.3 + 0.6 * magnitude
            translate = (-0.3 + 0.6 * magnitude) * float(image.size[0])
            enhancement = 0.05 + 0.9 * magnitude
            solarize = 256.0 * magnitude
            posterize = max(1, int(4.0 + 4.0 * magnitude))

        def rgb_result(value: Any) -> Any:
            return value.convert("RGBA") if google else value

        if name == "Identity":
            return image
        if name == "AutoContrast":
            return rgb_result(PIL.ImageOps.autocontrast(image.convert("RGB")))
        if name == "Equalize":
            return rgb_result(PIL.ImageOps.equalize(image.convert("RGB")))
        if name == "Rotate":
            return image.rotate(rotate)
        if name == "Solarize":
            return rgb_result(
                PIL.ImageOps.solarize(
                    image.convert("RGB"),
                    int(solarize) if google else solarize,
                )
            )
        if name == "Color":
            return PIL.ImageEnhance.Color(image).enhance(enhancement)
        if name == "Contrast":
            return PIL.ImageEnhance.Contrast(image).enhance(enhancement)
        if name == "Brightness":
            return PIL.ImageEnhance.Brightness(image).enhance(enhancement)
        if name == "Sharpness":
            return PIL.ImageEnhance.Sharpness(image).enhance(enhancement)
        if name == "Posterize":
            return rgb_result(PIL.ImageOps.posterize(image.convert("RGB"), int(posterize)))
        affine = PIL.Image.Transform.AFFINE
        if name == "ShearX":
            matrix = (1, shear, 0, 0, 1, 0)
        elif name == "ShearY":
            matrix = (1, 0, 0, shear, 1, 0)
        elif name == "TranslateX":
            matrix = (1, 0, translate, 0, 1, 0)
        elif name == "TranslateY":
            matrix = (1, 0, 0, 0, 1, translate)
        else:  # pragma: no cover - operation_index is constrained by sampled table
            raise AssertionError(name)
        return image.transform(image.size, affine, matrix)

    def _apply_cutout(self, batch: Any, draws: CifarAugmentationDraws) -> Any:
        torch, _, _ = self._load_backend()
        height, width = int(batch.shape[-2]), int(batch.shape[-1])
        fractions = torch.as_tensor(
            draws.cutout_size_fraction,
            device=batch.device,
            dtype=torch.float32,
        )
        if not bool((fractions > 0).any().item()):
            return batch
        size = fractions * float(min(height, width))
        center_y_values = torch.as_tensor(
            draws.cutout_center_y,
            device=batch.device,
            dtype=torch.float32,
        )
        center_x_values = torch.as_tensor(
            draws.cutout_center_x,
            device=batch.device,
            dtype=torch.float32,
        )
        if self.profile == "google_fixmatch_ra":
            center_y = torch.floor(center_y_values * float(height))
            center_x = torch.floor(center_x_values * float(width))
            half_size = torch.floor(size / 2.0)
            top = (center_y - half_size).clamp(min=0.0)
            left = (center_x - half_size).clamp(min=0.0)
            bottom = (center_y + half_size).clamp(max=float(height))
            right = (center_x + half_size).clamp(max=float(width))
            inclusive_end = False
        else:
            center_y = center_y_values * float(height)
            center_x = center_x_values * float(width)
            top = torch.trunc((center_y - size / 2.0).clamp(min=0.0))
            left = torch.trunc((center_x - size / 2.0).clamp(min=0.0))
            bottom = (top + size).clamp(max=float(height))
            right = (left + size).clamp(max=float(width))
            inclusive_end = True
        rows = torch.arange(height, device=batch.device, dtype=torch.float32).reshape(1, height, 1)
        cols = torch.arange(width, device=batch.device, dtype=torch.float32).reshape(1, 1, width)
        if inclusive_end:
            mask = (
                (rows >= top[:, None, None])
                & (rows <= bottom[:, None, None])
                & (cols >= left[:, None, None])
                & (cols <= right[:, None, None])
            )
        else:
            mask = (
                (rows >= top[:, None, None])
                & (rows < bottom[:, None, None])
                & (cols >= left[:, None, None])
                & (cols < right[:, None, None])
            )
        if self.profile == "torchssl_ra":
            fill = torch.tensor([125, 123, 114], device=batch.device, dtype=batch.dtype).reshape(
                1, 3, 1, 1
            )
        else:
            fill = torch.zeros((1, 3, 1, 1), device=batch.device, dtype=batch.dtype)
        return torch.where(mask[:, None, :, :], fill, batch)


def _sample_ids(sample_ids: Any) -> np.ndarray:
    if hasattr(sample_ids, "detach"):
        sample_ids = sample_ids.detach().cpu().numpy()
    raw = np.asarray(sample_ids)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise DataAugmentationValidationError("sample_ids must be a 1D integer array.")
    ids = raw.astype(np.int64, copy=False)
    if bool((ids < 0).any()):
        raise DataAugmentationValidationError("sample_ids must be non-negative.")
    return ids


__all__ = [
    "CifarAugmentationDraws",
    "CifarAugmentationProfile",
    "CifarAugmentationView",
    "CifarReferenceAugmentation",
    "resolve_cifar_augmentation_profile",
]
