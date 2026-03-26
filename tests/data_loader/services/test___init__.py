from __future__ import annotations

import modssc.data_loader.services as services
import modssc.data_loader.services.service as service


def test_services_exports_public_api() -> None:
    assert services.load_dataset is service.load_dataset
    assert services.download_dataset is service.download_dataset
    assert services.download_all_datasets is service.download_all_datasets
    assert services.available_datasets is service.available_datasets
    assert services.provider_names is service.provider_names
