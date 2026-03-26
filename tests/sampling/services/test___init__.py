from __future__ import annotations

import modssc.sampling.services as services
import modssc.sampling.services.service as service


def test_services_exports_sampling_api() -> None:
    assert services.sample is service.sample
    assert services.load_split is service.load_split
    assert services.save_split is service.save_split
    assert services.default_split_cache_dir is service.default_split_cache_dir
