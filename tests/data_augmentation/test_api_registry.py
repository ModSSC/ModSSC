from __future__ import annotations

from dataclasses import dataclass

import pytest

from modssc.data_augmentation.api import (
    AugmentationPipeline,
    AugmentationStrategy,
    build_pipeline,
    build_strategy,
    get_op,
)
from modssc.data_augmentation.errors import (
    DataAugmentationValidationError,
    OptionalDependencyError,
)
from modssc.data_augmentation.plan import AugmentationPlan, StepConfig
from modssc.data_augmentation.registry import (
    _OPS,
    available_ops,
    op_info,
    register_op,
)
from modssc.data_augmentation.types import AugmentationContext, AugmentationOp


def test_build_pipeline_validation() -> None:
    plan = AugmentationPlan(steps=[])  # type: ignore
    with pytest.raises(DataAugmentationValidationError, match="must be a tuple"):
        build_pipeline(plan)

    plan = AugmentationPlan(steps=("not_a_step",))  # type: ignore
    with pytest.raises(DataAugmentationValidationError, match="must be a StepConfig"):
        build_pipeline(plan)

    @register_op("test.vision_op")
    @dataclass
    class VisionOp(AugmentationOp):
        op_id: str = "test.vision_op"
        modality: str = "vision"

        def apply(self, x, **k):
            return x

    plan = AugmentationPlan(
        steps=(StepConfig(op_id="test.vision_op"),),
        modality="text",
    )
    with pytest.raises(
        DataAugmentationValidationError, match="has modality 'vision' but plan expects 'text'"
    ):
        build_pipeline(plan)

    plan_match = AugmentationPlan(steps=(StepConfig(op_id="test.vision_op"),), modality="vision")
    pipeline = build_pipeline(plan_match)
    assert len(pipeline.ops) == 1

    @register_op("test.any_op")
    @dataclass
    class AnyOp(AugmentationOp):
        op_id: str = "test.any_op"
        modality: str = "any"

        def apply(self, x, **k):
            return x

    plan_any = AugmentationPlan(steps=(StepConfig(op_id="test.any_op"),), modality="vision")
    pipeline_any = build_pipeline(plan_any)
    assert len(pipeline_any.ops) == 1

    plan_none = AugmentationPlan(steps=(StepConfig(op_id="test.any_op"),), modality=None)
    pipeline_none = build_pipeline(plan_none)
    assert len(pipeline_none.ops) == 1

    if "test.vision_op" in _OPS:
        del _OPS["test.vision_op"]

    if "test.any_op" in _OPS:
        del _OPS["test.any_op"]


def test_augmentation_strategy_apply() -> None:
    @dataclass
    class MockPipeline:
        def apply(self, x, ctx):
            return x + "_processed"

    weak = MockPipeline()
    strong = MockPipeline()
    strategy = AugmentationStrategy(weak=weak, strong=strong)  # type: ignore

    ctx = AugmentationContext(seed=0, epoch=0, sample_id=0)
    xw, xs = strategy.apply("input", ctx=ctx)
    assert xw == "input_processed"
    assert xs == "input_processed"


def test_build_strategy() -> None:
    plan = AugmentationPlan(steps=())
    strategy = build_strategy(weak=plan, strong=plan)
    assert isinstance(strategy, AugmentationStrategy)
    assert isinstance(strategy.weak, AugmentationPipeline)
    assert isinstance(strategy.strong, AugmentationPipeline)


def test_registry_validation() -> None:
    @register_op("test.dup_op")
    class Op1:
        op_id = "test.dup_op"
        modality = "any"

    with pytest.raises(DataAugmentationValidationError, match="Duplicate op_id"):

        @register_op("test.dup_op")
        class Op2:
            pass

    if "test.dup_op" in _OPS:
        del _OPS["test.dup_op"]

    with pytest.raises(DataAugmentationValidationError, match="must define 'op_id' and 'modality'"):

        @register_op("test.bad_op")
        class BadOp:
            pass


def test_available_ops_filtering() -> None:
    @register_op("test.text_op")
    class TextOp:
        op_id = "test.text_op"
        modality = "text"

    @register_op("test.any_op")
    class AnyOp:
        op_id = "test.any_op"
        modality = "any"

    ops_text = available_ops(modality="text")
    assert "test.text_op" in ops_text
    assert "test.any_op" in ops_text

    ops_vision = available_ops(modality="vision")
    assert "test.text_op" not in ops_vision
    assert "test.any_op" in ops_vision

    for k in ["test.text_op", "test.any_op"]:
        if k in _OPS:
            del _OPS[k]


def test_get_op_invalid_params() -> None:
    @register_op("test.param_op")
    @dataclass
    class ParamOp:
        op_id = "test.param_op"
        modality = "any"
        val: int = 0

    with pytest.raises(DataAugmentationValidationError, match="Invalid parameters"):
        get_op("test.param_op", invalid_param=1)

    if "test.param_op" in _OPS:
        del _OPS["test.param_op"]


def test_op_info_unknown() -> None:
    with pytest.raises(DataAugmentationValidationError, match="Unknown op_id"):
        op_info("unknown.op")


def test_errors_coverage() -> None:
    err = OptionalDependencyError("some_extra")
    assert "some_extra" in str(err)
    assert isinstance(err, ImportError)
