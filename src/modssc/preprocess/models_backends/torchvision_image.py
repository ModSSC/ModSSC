from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.device import resolve_device_name
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.optional import require
from modssc.supervised.backends.torch import image_pretrained as image_pretrained_backend


def _resolve_feature_module(model: Any, torch: Any) -> Any:
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        return model.fc
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, torch.nn.Linear):
            return clf
        if isinstance(clf, torch.nn.Sequential):
            for idx in range(len(clf) - 1, -1, -1):
                if isinstance(clf[idx], torch.nn.Linear):
                    return clf[idx]
    if hasattr(model, "heads"):
        heads = model.heads
        if hasattr(heads, "head") and isinstance(heads.head, torch.nn.Linear):
            return heads.head
        if isinstance(heads, torch.nn.Linear):
            return heads
    if hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
        return model.head
    raise PreprocessValidationError("Unable to resolve classifier head for torchvision encoder.")


def _to_nchw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise PreprocessValidationError(
            f"TorchvisionImageEncoder expects 2D/3D images, got shape={arr.shape}."
        )
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        return arr
    if arr.shape[-1] in (1, 3, 4):
        return np.transpose(arr, (2, 0, 1))
    raise PreprocessValidationError(
        f"Could not infer image channel layout for shape={arr.shape}. "
        "Run vision.channels_order explicitly before embedding."
    )


@dataclass
class TorchvisionImageEncoder:
    model_name: str = "resnet18"
    weights: str | None = "DEFAULT"
    device: str | None = None
    auto_channel_repeat: bool = True

    def __post_init__(self) -> None:
        torch = require(module="torch", extra="preprocess-vision", purpose="torchvision encoder")
        require(module="torchvision", extra="preprocess-vision", purpose="torchvision encoder")

        self._torch = torch
        self.device = resolve_device_name(self.device, torch=torch)
        self._model = image_pretrained_backend._load_model(self.model_name, self.weights)
        self._model.eval()
        self._model = self._model.to(self.device or "cpu")
        self._expected_in_channels = image_pretrained_backend._infer_in_channels(self._model, torch)
        self._feature_module = _resolve_feature_module(self._model, torch)

    def _prepare_batch(self, batch: list[Any]) -> Any:
        torch = self._torch
        tensors: list[Any] = []
        expected = self._expected_in_channels
        for sample in batch:
            arr = to_numpy(sample)
            arr = _to_nchw(arr).astype(np.float32, copy=False)
            tensor = torch.from_numpy(np.ascontiguousarray(arr))
            if expected is not None and int(tensor.shape[0]) != int(expected):
                if self.auto_channel_repeat and int(tensor.shape[0]) == 1 and int(expected) == 3:
                    tensor = tensor.repeat(3, 1, 1)
                else:
                    raise PreprocessValidationError(
                        f"Model expects {expected} channels, got {int(tensor.shape[0])}."
                    )
            tensors.append(tensor)
        return torch.stack(tensors, dim=0).to(self.device or "cpu", dtype=torch.float32)

    def _encode_batch(self, batch: Any) -> np.ndarray:
        captured: list[Any] = []

        def _capture(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs:
                raise PreprocessValidationError(
                    "Torchvision encoder failed to capture penultimate features."
                )
            feat = inputs[0]
            if feat.ndim > 2:
                feat = feat.reshape(int(feat.shape[0]), -1)
            captured.append(feat.detach())

        hook = self._feature_module.register_forward_hook(_capture)
        try:
            with self._torch.no_grad():
                _ = self._model(batch)
        finally:
            hook.remove()

        if not captured:
            raise PreprocessValidationError(
                "Torchvision encoder failed to capture penultimate features."
            )
        return captured[-1].cpu().numpy().astype(np.float32, copy=False)

    def _split_samples(self, X: np.ndarray) -> list[np.ndarray]:
        if X.ndim <= 2:
            return [X]
        if X.ndim == 4:
            return [X[i] for i in range(X.shape[0])]
        if X.ndim != 3:
            return [X]

        # Treat 3D arrays as a single image only when their layout is unambiguous.
        if X.shape[-1] in (1, 3, 4):
            return [X]
        expected = self._expected_in_channels
        if expected is not None and int(X.shape[0]) == int(expected):
            return [X]
        return [X[i] for i in range(X.shape[0])]

    def encode(
        self, X: Any, *, batch_size: int = 32, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        del rng
        if isinstance(X, np.ndarray):
            samples = self._split_samples(X)
        elif isinstance(X, list):
            samples = X
        else:
            samples = list(X)

        if not samples:
            return np.empty((0, 0), dtype=np.float32)

        feats: list[np.ndarray] = []
        bs = int(batch_size)
        for start in range(0, len(samples), bs):
            batch = self._prepare_batch(samples[start : start + bs])
            feats.append(self._encode_batch(batch))

        return np.concatenate(feats, axis=0)
