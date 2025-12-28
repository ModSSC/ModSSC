from __future__ import annotations

import pytest

from modssc.transductive.methods.advanced import graphmae2, llm_gnn, nodeformer, oft, sgformer


@pytest.mark.parametrize(
    ("method_cls", "spec_cls", "method_id"),
    [
        (nodeformer.NodeFormerMethod, nodeformer.NodeFormerSpec, "nodeformer"),
        (graphmae2.GraphMAE2Method, graphmae2.GraphMAE2Spec, "graphmae2"),
        (sgformer.SGFormerMethod, sgformer.SGFormerSpec, "sgformer"),
        (oft.OFTMethod, oft.OFTSpec, "oft"),
        (llm_gnn.LLMGNNMethod, llm_gnn.LLMGNNSpec, "llm_gnn"),
    ],
)
def test_placeholder_methods_raise_and_expose_info(method_cls, spec_cls, method_id) -> None:
    method = method_cls()
    assert isinstance(method.spec, spec_cls)
    assert method.info.method_id == method_id

    with pytest.raises(NotImplementedError):
        method.fit(None)

    with pytest.raises(RuntimeError):
        method.predict_proba(None)
