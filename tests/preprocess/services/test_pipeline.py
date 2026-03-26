from __future__ import annotations

import modssc.preprocess.api as api
import modssc.preprocess.services.pipeline as pipeline


def test_api_module_aliases_internal_pipeline() -> None:
    assert api is pipeline
    assert api.preprocess is pipeline.preprocess
    assert api.resolve_plan is pipeline.resolve_plan
    assert api._build_purge_keep_sets is pipeline._build_purge_keep_sets
