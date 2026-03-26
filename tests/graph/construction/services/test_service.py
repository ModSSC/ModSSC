from __future__ import annotations

import modssc.graph.construction.api as api
import modssc.graph.construction.services.service as service


def test_api_module_aliases_internal_service() -> None:
    assert api is service
    assert api.build_graph is service.build_graph
    assert api.build_raw_edges is service.build_raw_edges
