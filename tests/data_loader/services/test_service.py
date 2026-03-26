from __future__ import annotations

import modssc.data_loader.api as api
import modssc.data_loader.services.service as service


def test_api_module_aliases_internal_service() -> None:
    assert api is service
    assert api.load_dataset is service.load_dataset
    assert api.download_dataset is service.download_dataset
    assert api.dataset_info is service.dataset_info
