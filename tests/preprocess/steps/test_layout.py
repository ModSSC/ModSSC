from __future__ import annotations

import numpy as np

from modssc.preprocess.steps.vision.layout import (
    _infer_layout_3d,
    _infer_layout_4d,
    prepare_image_array,
)


def test_infer_layout_3d_non_3d_returns_none() -> None:
    assert _infer_layout_3d(np.zeros((2, 2, 2, 2))) is None


def test_infer_layout_3d_chw_and_ambiguous() -> None:
    arr_chw = np.zeros((3, 10, 12))
    assert _infer_layout_3d(arr_chw) == "CHW"

    arr_amb = np.zeros((3, 5, 3))
    assert _infer_layout_3d(arr_amb) == "HWC"


def test_infer_layout_4d_non_4d_returns_none() -> None:
    assert _infer_layout_4d(np.zeros((2, 2, 2))) is None


def test_prepare_image_array_handles_single_and_batch() -> None:
    single = np.zeros((3, 8, 8))
    arr4, is_single, layout = prepare_image_array(single)
    assert is_single is True
    assert layout in {"NCHW", "NHWC"}
    assert arr4.shape[0] == 1

    batch = np.zeros((2, 8, 8))
    arr4b, is_single_b, layout_b = prepare_image_array(batch)
    assert is_single_b is False
    assert layout_b == "NCHW"
    assert arr4b.shape[0] == 2
