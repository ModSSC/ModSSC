from __future__ import annotations

import modssc.preprocess.services as services
import modssc.preprocess.services.pipeline as pipeline


def test_services_exports_pipeline_api() -> None:
    assert services.preprocess is pipeline.preprocess
    assert services.fit_transform is pipeline.fit_transform
    assert services.resolve_plan is pipeline.resolve_plan
