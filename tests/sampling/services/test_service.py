from __future__ import annotations

import modssc.sampling.api as api
import modssc.sampling.services.service as service


def test_api_module_aliases_internal_service() -> None:
    assert api is service
    assert api.sample is service.sample
    assert api.default_split_cache_dir is service.default_split_cache_dir
    assert api._idx_to_mask is service._idx_to_mask
