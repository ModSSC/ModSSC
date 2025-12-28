from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation.types import AugmentationContext, AugmentationOp, GraphSample


def test_graph_sample_num_nodes():
    x = np.zeros((10, 2))
    ei = np.zeros((2, 0))
    gs = GraphSample(x=x, edge_index=ei)
    assert gs.num_nodes() == 10


def test_augmentation_op_base():
    op = AugmentationOp(op_id="test")
    ctx = AugmentationContext(seed=0)
    rng = np.random.default_rng(0)

    with pytest.raises(NotImplementedError):
        op(None, rng=rng, ctx=ctx)
