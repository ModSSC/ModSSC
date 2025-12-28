from __future__ import annotations

import numpy as np

from modssc.data_augmentation import AugmentationContext, GraphSample
from modssc.data_augmentation.registry import get_op
from modssc.data_augmentation.utils import make_numpy_rng


def test_edge_dropout_p1_drops_all_edges() -> None:
    x = np.ones((4, 2), dtype=np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    g = GraphSample(x=x, edge_index=edge_index)

    op = get_op("graph.edge_dropout", p=1.0)
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=0, modality="graph")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    out = op.apply(g, rng=rng, ctx=ctx)

    assert isinstance(out, GraphSample)
    assert out.edge_index.shape == (2, 0)


def test_feature_mask_p1_masks_all_features() -> None:
    x = np.ones((4, 2), dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    g = GraphSample(x=x, edge_index=edge_index)

    op = get_op("graph.feature_mask", p=1.0)
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=0, modality="graph")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    out = op.apply(g, rng=rng, ctx=ctx)

    assert np.allclose(np.asarray(out.x), 0.0)
