from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from modssc.data_augmentation import (
    AugmentationContext,
    AugmentationPlan,
    OnlineAugmentation,
    StepConfig,
    build_online_augmentation,
    materialize_views,
    register_op,
)
from modssc.data_augmentation.api import build_strategy
from modssc.data_augmentation.errors import DataAugmentationValidationError
from modssc.data_augmentation.registry import _OPS
from modssc.data_augmentation.types import AugmentationOp


def _runtime() -> OnlineAugmentation:
    weak = AugmentationPlan(
        steps=(StepConfig(op_id="core.identity", params={}),),
        modality="tabular",
    )
    strong = AugmentationPlan(
        steps=(StepConfig(op_id="tabular.gaussian_noise", params={"std": 0.5}),),
        modality="tabular",
    )
    return OnlineAugmentation(
        strategy=build_strategy(weak=weak, strong=strong),
        seed=17,
        modality="tabular",
    )


def test_online_augmentation_is_replayable_per_step_and_sample() -> None:
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    runtime = _runtime()
    indices = np.array([4, 1, 5], dtype=np.int64)
    sample_ids = np.array([104, 101, 105], dtype=np.int64)

    weak_a, strong_a = runtime.pair_batch(X, indices=indices, sample_ids=sample_ids, step=8)
    weak_b, strong_b = runtime.pair_batch(X, indices=indices, sample_ids=sample_ids, step=8)
    _, strong_next = runtime.pair_batch(X, indices=indices, sample_ids=sample_ids, step=9)

    np.testing.assert_array_equal(weak_a, X[indices])
    np.testing.assert_array_equal(weak_a, weak_b)
    np.testing.assert_array_equal(strong_a, strong_b)
    assert not np.array_equal(strong_a, strong_next)


def test_online_augmentation_weak_batch_uses_absolute_sample_ids() -> None:
    X = np.zeros((3, 5), dtype=np.float32)
    runtime = _runtime()
    first = runtime.weak_batch(
        X,
        indices=np.array([0, 1]),
        sample_ids=np.array([20, 21]),
        step=0,
    )
    np.testing.assert_array_equal(first, np.zeros((2, 5), dtype=np.float32))


def test_online_augmentation_rejects_misaligned_ids_and_negative_step() -> None:
    runtime = _runtime()
    X = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="same length"):
        runtime.pair_batch(
            X,
            indices=np.array([0, 1]),
            sample_ids=np.array([10]),
            step=0,
        )
    with pytest.raises(ValueError, match="step must be"):
        runtime.weak_batch(
            X,
            indices=np.array([0]),
            sample_ids=np.array([10]),
            step=-1,
        )
    with pytest.raises(ValueError, match="same length"):
        runtime.weak_batch(
            X,
            indices=np.array([0, 1]),
            sample_ids=np.array([10]),
            step=0,
        )


def test_online_augmentation_supports_torch_indices_and_empty_batches() -> None:
    torch = pytest.importorskip("torch")
    runtime = _runtime()
    X = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    weak = runtime.weak_batch(
        X,
        indices=torch.tensor([2, 0]),
        sample_ids=torch.tensor([12, 10]),
        step=1,
    )
    empty_weak, empty_strong = runtime.pair_batch(
        X,
        indices=torch.empty(0, dtype=torch.int64),
        sample_ids=torch.empty(0, dtype=torch.int64),
        step=1,
    )

    torch.testing.assert_close(weak, X[torch.tensor([2, 0])])
    assert empty_weak.shape == empty_strong.shape == (0, 4)


def _plan(op_id: str, **params) -> dict:
    return {"steps": [{"id": op_id, "params": params}]}


def test_build_online_augmentation_compiles_generic_native_runtime() -> None:
    runtime = build_online_augmentation(
        weak_plan=_plan("core.identity"),
        strong_plan=_plan("tabular.gaussian_noise", std=0.5),
        seed=31,
        modality="tabular",
    )

    assert isinstance(runtime, OnlineAugmentation)
    assert runtime.seed == 31
    assert runtime.modality == "tabular"


def test_build_online_augmentation_delegates_registered_runtime() -> None:
    sentinel = object()
    with patch(
        "modssc.data_augmentation.runtime.get_online_augmenter",
        return_value=sentinel,
    ) as factory:
        result = build_online_augmentation(
            weak_plan=_plan("core.identity"),
            strong_plan=_plan("core.identity"),
            seed=9,
            modality="vision",
            online_augmenter_id="vision.reference",
            online_augmenter_params={"profile": "paper"},
        )

    assert result is sentinel
    factory.assert_called_once_with(
        "vision.reference",
        modality="vision",
        seed=9,
        profile="paper",
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"modality": "graph"}, "not supported for graph"),
        ({"online_augmenter_params": []}, "must be a mapping"),
        ({"online_augmenter_params": {"seed": 1}}, "must not redefine seed"),
        ({"online_augmenter_params": {"profile": "paper"}}, "require online_augmenter_id"),
    ],
)
def test_build_online_augmentation_rejects_invalid_native_contract(overrides, match: str) -> None:
    kwargs = {
        "weak_plan": _plan("core.identity"),
        "strong_plan": _plan("core.identity"),
        "seed": 0,
        "modality": "tabular",
    }
    kwargs.update(overrides)
    with pytest.raises(DataAugmentationValidationError, match=match):
        build_online_augmentation(**kwargs)


def test_materialize_views_is_replayable_and_uses_absolute_sample_ids() -> None:
    X = np.zeros((3, 5), dtype=np.float32)
    kwargs = {
        "weak_plan": _plan("core.identity"),
        "strong_plan": _plan("tabular.gaussian_noise", std=0.5),
        "seed": 4,
        "modality": "tabular",
        "sample_ids": np.array([91, 15, 72]),
        "strong_views": 2,
    }

    weak_a, strong_a, strong_a_1 = materialize_views(X, **kwargs)
    weak_b, strong_b, strong_b_1 = materialize_views(X, **kwargs)

    np.testing.assert_array_equal(weak_a, X)
    np.testing.assert_array_equal(strong_a, strong_b)
    np.testing.assert_array_equal(strong_a_1, strong_b_1)
    assert not np.array_equal(strong_a, strong_a_1)


def test_materialize_views_handles_none_and_empty_batches() -> None:
    plan = _plan("core.identity")
    assert materialize_views(
        None,
        weak_plan=plan,
        strong_plan=plan,
        seed=0,
    ) == (None, None, None)

    empty = np.empty((0, 3), dtype=np.float32)
    weak, strong, second = materialize_views(
        empty,
        weak_plan=plan,
        strong_plan=plan,
        seed=0,
        strong_views=2,
    )
    assert weak is strong is second is empty


def test_materialize_views_supports_indexable_python_batches() -> None:
    X = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    plan = _plan("core.identity")

    weak, strong, second = materialize_views(
        X,
        weak_plan=plan,
        strong_plan=plan,
        seed=0,
        modality="tabular",
    )

    np.testing.assert_array_equal(weak, np.asarray(X))
    np.testing.assert_array_equal(strong, np.asarray(X))
    assert second is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"mode": "online"}, "mode='fixed'"),
        ({"strong_views": 3}, "must be 1 or 2"),
        ({"sample_ids": [1]}, "one stable id"),
    ],
)
def test_materialize_views_rejects_invalid_runtime_contract(overrides, match: str) -> None:
    kwargs = {
        "weak_plan": _plan("core.identity"),
        "strong_plan": _plan("core.identity"),
        "seed": 0,
    }
    kwargs.update(overrides)
    with pytest.raises(DataAugmentationValidationError, match=match):
        materialize_views(np.zeros((2, 3)), **kwargs)


def test_materialize_views_rejects_non_batch_input() -> None:
    plan = _plan("core.identity")
    with pytest.raises(DataAugmentationValidationError, match="indexable batch"):
        materialize_views(
            3.0,
            weak_plan=plan,
            strong_plan=plan,
            seed=0,
        )


def test_materialize_views_falls_back_for_dynamic_shapes() -> None:
    @register_op("test.dynamic_shape")
    @dataclass
    class DynamicShape(AugmentationOp):
        op_id: str = "test.dynamic_shape"
        modality: str = "tabular"

        def apply(self, x, *, rng, ctx: AugmentationContext):
            del rng
            return x[: 1 + int(ctx.sample_id)]

    try:
        X = np.arange(12, dtype=np.float32).reshape(3, 4)
        plan = _plan("test.dynamic_shape")
        weak, strong, _ = materialize_views(
            X,
            weak_plan=plan,
            strong_plan=plan,
            seed=0,
            modality="tabular",
            sample_ids=[0, 1, 2],
        )

        assert isinstance(weak, list)
        assert isinstance(strong, list)
        assert [value.shape for value in weak] == [(1,), (2,), (3,)]
    finally:
        _OPS.pop("test.dynamic_shape", None)


def test_materialize_views_stacks_non_array_outputs() -> None:
    @register_op("test.scalar_output")
    @dataclass
    class ScalarOutput(AugmentationOp):
        op_id: str = "test.scalar_output"
        modality: str = "tabular"

        def apply(self, x, *, rng, ctx):
            del rng, ctx
            return float(np.asarray(x).sum())

    try:
        X = np.arange(6, dtype=np.float32).reshape(2, 3)
        plan = _plan("test.scalar_output")
        weak, strong, _ = materialize_views(
            X,
            weak_plan=plan,
            strong_plan=plan,
            seed=0,
            modality="tabular",
        )

        np.testing.assert_array_equal(weak, np.array([3.0, 12.0]))
        np.testing.assert_array_equal(strong, weak)
    finally:
        _OPS.pop("test.scalar_output", None)


def test_materialize_views_falls_back_for_dynamic_torch_shapes() -> None:
    torch = pytest.importorskip("torch")

    @register_op("test.dynamic_torch_shape")
    @dataclass
    class DynamicTorchShape(AugmentationOp):
        op_id: str = "test.dynamic_torch_shape"
        modality: str = "tabular"

        def apply(self, x, *, rng, ctx: AugmentationContext):
            del rng
            return x[: 1 + int(ctx.sample_id)]

    try:
        X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        plan = _plan("test.dynamic_torch_shape")
        weak, strong, _ = materialize_views(
            X,
            weak_plan=plan,
            strong_plan=plan,
            seed=0,
            modality="tabular",
            sample_ids=[0, 1, 2],
        )

        assert isinstance(weak, list)
        assert isinstance(strong, list)
        assert [tuple(value.shape) for value in weak] == [(1,), (2,), (3,)]
    finally:
        _OPS.pop("test.dynamic_torch_shape", None)


def test_materialize_views_preallocates_torch_outputs() -> None:
    torch = pytest.importorskip("torch")
    X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    plan = _plan("core.identity")

    weak, strong, second = materialize_views(
        X,
        weak_plan=plan,
        strong_plan=plan,
        seed=0,
        sample_ids=torch.tensor([8, 9, 10]),
        strong_views=2,
    )

    torch.testing.assert_close(weak, X)
    torch.testing.assert_close(strong, X)
    torch.testing.assert_close(second, X)

    graph = {
        "x": torch.ones((2, 2)),
        "edge_index": torch.tensor([[0], [1]]),
    }
    graph_weak, graph_strong, graph_second = materialize_views(
        graph,
        weak_plan=plan,
        strong_plan=plan,
        seed=0,
        modality="graph",
        sample_ids=torch.tensor([42]),
    )
    assert graph_weak is graph_strong is graph
    assert graph_second is None
