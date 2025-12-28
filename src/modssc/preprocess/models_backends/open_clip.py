from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.device import resolve_device_name
from modssc.preprocess.errors import OptionalDependencyError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.optional import require


@dataclass
class OpenClipEncoder:
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            open_clip = require(module="open_clip", extra="preprocess-vision", purpose="OpenCLIP")
            torch = require(module="torch", extra="preprocess-vision", purpose="OpenCLIP")
        except OptionalDependencyError:
            raise

        self._torch = torch
        self._open_clip = open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model.eval()
        self.device = resolve_device_name(self.device, torch=torch)
        self._model = model.to(self.device or "cpu")
        self._preprocess = preprocess

    def encode(
        self, X: Any, *, batch_size: int = 32, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        torch = self._torch
        # X: array-like of images
        if isinstance(X, np.ndarray) and X.ndim >= 3:
            samples = [X[i] for i in range(X.shape[0])] if X.ndim == 4 else [X]
        elif isinstance(X, list):
            samples = X
        else:
            samples = list(X)

        feats: list[np.ndarray] = []
        bs = int(batch_size)
        for start in range(0, len(samples), bs):
            batch = samples[start : start + bs]
            imgs = []
            for img in batch:
                arr = to_numpy(img)
                if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    # likely CHW -> HWC
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                from PIL import Image  # pillow is part of preprocess-vision

                pil = Image.fromarray(arr)
                imgs.append(self._preprocess(pil))

            device = next(self._model.parameters()).device
            t = torch.stack(imgs, dim=0).to(device)
            with torch.no_grad():
                emb = self._model.encode_image(t).cpu().numpy()
            feats.append(np.asarray(emb, dtype=np.float32))
        return np.concatenate(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)
