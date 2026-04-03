from __future__ import annotations

import numpy as np

from bench.orchestrators import augmentation as bench_augmentation
from modssc.data_augmentation import (
    AugmentationContext,
    AugmentationPlan,
    StepConfig,
    build_pipeline,
)
from modssc.data_augmentation.utils import seed_to_rng_seed


def test_seed_to_rng_seed_changes_with_sample_id() -> None:
    s1 = seed_to_rng_seed(seed=0, epoch=0, sample_id=1)
    s2 = seed_to_rng_seed(seed=0, epoch=0, sample_id=2)
    assert s1 != s2


def test_pipeline_is_deterministic_for_same_context() -> None:
    x = np.zeros((8,), dtype=np.float32)
    plan = AugmentationPlan(
        modality="tabular",
        steps=(StepConfig("tabular.gaussian_noise", {"std": 1.0}),),
    )
    pipe = build_pipeline(plan)

    ctx = AugmentationContext(seed=0, epoch=0, sample_id=123, modality="tabular")
    y1 = pipe(x, ctx=ctx)
    y2 = pipe(x, ctx=ctx)

    assert np.allclose(y1, y2)


def test_pipeline_changes_with_different_sample_id() -> None:
    x = np.zeros((8,), dtype=np.float32)
    plan = AugmentationPlan(
        modality="tabular",
        steps=(StepConfig("tabular.gaussian_noise", {"std": 1.0}),),
    )
    pipe = build_pipeline(plan)

    y1 = pipe(x, ctx=AugmentationContext(seed=0, epoch=0, sample_id=1, modality="tabular"))
    y2 = pipe(x, ctx=AugmentationContext(seed=0, epoch=0, sample_id=2, modality="tabular"))

    assert not np.allclose(y1, y2)


def test_bench_augmentation_runs_graph_as_whole_sample() -> None:
    graph = {
        "x": np.ones((4, 2), dtype=np.float32),
        "edge_index": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
    }

    weak, strong, strong_1 = bench_augmentation.run(
        graph,
        weak_plan={"steps": [{"id": "graph.feature_mask", "params": {"p": 1.0}}]},
        strong_plan={"steps": [{"id": "graph.edge_dropout", "params": {"p": 1.0}}]},
        seed=0,
        mode="fixed",
        modality="graph",
        sample_ids=np.array([7], dtype=np.int64),
        strong_views=2,
    )

    assert isinstance(weak, dict)
    assert isinstance(strong, dict)
    assert isinstance(strong_1, dict)
    assert np.allclose(np.asarray(weak["x"]), 0.0)
    assert tuple(np.asarray(strong["edge_index"]).shape) == (2, 0)
    assert tuple(np.asarray(strong_1["edge_index"]).shape) == (2, 0)
