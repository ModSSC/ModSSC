from __future__ import annotations

import modssc.inductive.methods.helpers.classifier_bridge as classifier_bridge
import modssc.inductive.methods.utils as legacy_utils


def test_legacy_utils_module_aliases_classifier_bridge() -> None:
    assert legacy_utils is classifier_bridge
    assert legacy_utils.detect_backend is classifier_bridge.detect_backend
    assert legacy_utils.predict_scores is classifier_bridge.predict_scores
