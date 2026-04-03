from __future__ import annotations

import modssc.views.services as services
import modssc.views.services.service as service


def test_services_exports_generate_views() -> None:
    assert services.generate_views is service.generate_views
