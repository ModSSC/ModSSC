from __future__ import annotations

import pytest

import modssc.data_augmentation.registry as registry
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


def test_available_ops_filters_builtins_by_modality() -> None:
    vision_ops = available_ops(modality="vision")
    assert "vision.random_horizontal_flip" in vision_ops
    assert "tabular.gaussian_noise" not in vision_ops


def test_load_builtin_cls_returns_builtin_class() -> None:
    cls = registry._load_builtin_cls("tabular.gaussian_noise")
    assert cls.__name__ == "GaussianNoise"


def test_get_op_rejects_builtin_entries_that_do_not_resolve_to_classes(monkeypatch) -> None:
    monkeypatch.setitem(registry._BUILTIN_OPS, "bad.fake", ("bad.module:obj", "any"))
    monkeypatch.setattr(registry, "load_object", lambda _path: 123)

    with pytest.raises(DataAugmentationValidationError, match="did not resolve to a class"):
        get_op("bad.fake")


def test_available_ops_uses_builtin_modality_when_runtime_registry_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_RUNTIME_OPS", {})
    assert "vision.random_crop_pad" in registry.available_ops(modality="vision")
