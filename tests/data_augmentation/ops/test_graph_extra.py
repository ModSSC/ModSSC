from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation.ops.graph import (
    EdgeDropout,
    FeatureMask,
    _extract_graph,
    _rebuild_graph,
)
from modssc.data_augmentation.types import AugmentationContext, GraphSample


@pytest.fixture
def ctx():
    return AugmentationContext(seed=0, epoch=0, sample_id=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_graph_extract_rebuild():
    x = np.zeros((2, 2))
    ei = np.zeros((2, 1), dtype=np.int64)
    ew = np.ones((1,))

    gs = GraphSample(x=x, edge_index=ei, edge_weight=ew)
    x2, ei2, ew2, kind = _extract_graph(gs)
    assert kind == "graphsample"
    assert x2 is x
    assert ei2 is ei
    assert ew2 is ew

    gs2 = _rebuild_graph(gs, kind, x=x, edge_index=ei, edge_weight=ew)
    assert isinstance(gs2, GraphSample)

    d = {"x": x, "edge_index": ei, "edge_weight": ew}
    x2, ei2, ew2, kind = _extract_graph(d)
    assert kind == "dict"

    d2 = _rebuild_graph(d, kind, x=x, edge_index=ei, edge_weight=ew)
    assert isinstance(d2, dict)
    assert "edge_weight" in d2

    d3 = _rebuild_graph(d, kind, x=x, edge_index=ei, edge_weight=None)
    assert "edge_weight" not in d3

    class ObjWithEW:
        def __init__(self):
            self.x = x
            self.edge_index = ei
            self.edge_weight = ew

    obj = ObjWithEW()

    obj2 = _rebuild_graph(obj, "attr", x=x, edge_index=ei, edge_weight=None)
    assert not hasattr(obj2, "edge_weight")

    d_noweight = {"x": x, "edge_index": ei}
    x2, ei2, ew2, kind = _extract_graph(d_noweight)
    assert ew2 is None
    d2 = _rebuild_graph(d_noweight, kind, x=x, edge_index=ei, edge_weight=None)
    assert "edge_weight" not in d2

    class GObj:
        def __init__(self):
            self.x = x
            self.edge_index = ei
            self.edge_weight = ew

    obj = GObj()
    x2, ei2, ew2, kind = _extract_graph(obj)
    assert kind == "attr"

    obj2 = _rebuild_graph(obj, kind, x=x, edge_index=ei, edge_weight=ew)
    assert obj2.edge_weight is ew

    class GObjNoW:
        def __init__(self):
            self.x = x
            self.edge_index = ei

    obj = GObjNoW()
    x2, ei2, ew2, kind = _extract_graph(obj)
    assert ew2 is None

    obj2 = _rebuild_graph(obj, kind, x=x, edge_index=ei, edge_weight=None)
    assert not hasattr(obj2, "edge_weight")

    with pytest.raises(TypeError):
        _extract_graph("invalid")


def test_graph_edge_dropout(ctx, rng):
    with pytest.raises(ValueError):
        EdgeDropout(p=1.1).apply(None, rng=rng, ctx=ctx)

    op = EdgeDropout(p=0)
    assert op.apply("g", rng=rng, ctx=ctx) == "g"

    x = np.zeros((2, 2))
    ei = np.array([[0, 1], [1, 0]], dtype=np.int64)
    ew = np.array([1.0, 2.0], dtype=np.float32)
    gs = GraphSample(x=x, edge_index=ei, edge_weight=ew)

    op = EdgeDropout(p=0.5)
    out = op.apply(gs, rng=rng, ctx=ctx)
    assert out.edge_index.shape[1] <= 2
    if out.edge_weight is not None:
        assert out.edge_weight.shape[0] == out.edge_index.shape[1]


def test_graph_feature_mask(ctx, rng):
    with pytest.raises(ValueError):
        FeatureMask(p=1.1).apply(None, rng=rng, ctx=ctx)

    op = FeatureMask(p=0)
    assert op.apply("g", rng=rng, ctx=ctx) == "g"

    x = np.ones((10, 2))
    ei = np.zeros((2, 0), dtype=np.int64)
    gs = GraphSample(x=x, edge_index=ei)

    op = FeatureMask(p=0.5)
    out = op.apply(gs, rng=rng, ctx=ctx)

    assert (out.x == 0).any()
