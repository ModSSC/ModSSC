from __future__ import annotations

import pytest

from modssc.data_augmentation.errors import DataAugmentationValidationError
from modssc.data_augmentation.registry import available_ops, get_op, op_info


def test_available_ops_contains_some_builtins() -> None:
    ops = available_ops()
    assert "tabular.gaussian_noise" in ops
    assert "vision.random_horizontal_flip" in ops
    assert "text.word_dropout" in ops
    assert "graph.edge_dropout" in ops


def test_get_op_unknown_raises() -> None:
    with pytest.raises(DataAugmentationValidationError):
        get_op("does.not.exist")


def test_op_info_has_defaults_and_doc() -> None:
    info = op_info("tabular.gaussian_noise")
    assert info["op_id"] == "tabular.gaussian_noise"
    assert info["modality"] == "tabular"
    assert "std" in info["defaults"]
