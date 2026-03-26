from __future__ import annotations

import modssc.inductive.methods.deep_utils as legacy_deep_utils
import modssc.inductive.methods.helpers.torch_support as torch_support


def test_legacy_deep_utils_module_aliases_torch_support() -> None:
    assert legacy_deep_utils is torch_support
    assert legacy_deep_utils.ensure_model_bundle is torch_support.ensure_model_bundle
    assert legacy_deep_utils.predict_proba_from_bundle is torch_support.predict_proba_from_bundle
