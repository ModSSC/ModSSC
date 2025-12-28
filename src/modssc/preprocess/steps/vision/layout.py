from __future__ import annotations

from typing import Literal

import numpy as np

Layout4D = Literal["NCHW", "NHWC"]


def _infer_layout_3d(arr: np.ndarray) -> Literal["CHW", "HWC"] | None:
    if arr.ndim != 3:
        return None
    first_is_ch = arr.shape[0] in (1, 3, 4)
    last_is_ch = arr.shape[2] in (1, 2, 3, 4)
    if first_is_ch and not last_is_ch:
        return "CHW"
    if last_is_ch and not first_is_ch:
        return "HWC"
    if first_is_ch and last_is_ch:
        # Ambiguous: prefer channels-last for numpy data.
        return "HWC"
    return None


def _infer_layout_4d(arr: np.ndarray) -> Layout4D | None:
    if arr.ndim != 4:
        return None
    first_is_ch = arr.shape[1] in (1, 2, 3, 4)
    last_is_ch = arr.shape[3] in (1, 2, 3, 4)
    if first_is_ch and not last_is_ch:
        return "NCHW"
    if last_is_ch and not first_is_ch:
        return "NHWC"
    if first_is_ch and last_is_ch:
        # Ambiguous: prefer channels-last for numpy data.
        return "NHWC"
    return None


def prepare_image_array(arr: np.ndarray) -> tuple[np.ndarray, bool, Layout4D] | None:
    """Return (arr4, single, layout) or None if input is not image-like.

    - single=True indicates the input was a single image (no batch dimension).
    - layout is the detected 4D layout for arr4.
    """
    if arr.ndim == 4:
        layout = _infer_layout_4d(arr)
        if layout is None:
            return None
        return arr, False, layout

    if arr.ndim == 3:
        layout3 = _infer_layout_3d(arr)
        if layout3 is not None:
            arr4 = arr[None, ...]
            layout4: Layout4D = "NCHW" if layout3 == "CHW" else "NHWC"
            return arr4, True, layout4
        # Grayscale batch (N, H, W) -> NCHW with explicit channel.
        return arr[:, None, :, :], False, "NCHW"

    return None
