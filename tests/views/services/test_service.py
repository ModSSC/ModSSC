from __future__ import annotations

import modssc.views.api as api
import modssc.views.services.service as service


def test_api_module_aliases_internal_service() -> None:
    assert api is service
    assert api.generate_views is service.generate_views
    assert api._resolve_columns is service._resolve_columns
