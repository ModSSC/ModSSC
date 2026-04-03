from __future__ import annotations

import modssc.inductive.methods.helpers as helpers
import modssc.inductive.methods.helpers.classifier_bridge as classifier_bridge
import modssc.inductive.methods.helpers.torch_support as torch_support


def test_helpers_package_reexports_bridge_and_torch_support() -> None:
    assert helpers.BaseClassifierSpec is classifier_bridge.BaseClassifierSpec
    assert helpers.detect_backend is classifier_bridge.detect_backend
    assert helpers.ensure_model_bundle is torch_support.ensure_model_bundle
    assert helpers.predict_proba_from_bundle is torch_support.predict_proba_from_bundle
