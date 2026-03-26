from __future__ import annotations

import modssc.graph.construction.services as services
import modssc.graph.construction.services.service as service


def test_services_exports_build_graph() -> None:
    assert services.build_graph is service.build_graph
