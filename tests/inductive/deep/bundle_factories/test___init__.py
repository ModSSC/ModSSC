from __future__ import annotations

import modssc.inductive.deep.bundle_factories as bundle_factories
import modssc.inductive.deep.bundle_factories.factory as factory


def test_package_exports_bundle_factory() -> None:
    assert bundle_factories.build_torch_bundle_from_classifier is (
        factory.build_torch_bundle_from_classifier
    )
