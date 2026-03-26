from __future__ import annotations

import modssc.inductive.deep.bundle_factories.factory as factory
import modssc.inductive.deep.bundles as bundles


def test_legacy_bundles_module_aliases_new_factory_module() -> None:
    assert bundles is factory
    assert bundles.build_torch_bundle_from_classifier is factory.build_torch_bundle_from_classifier
