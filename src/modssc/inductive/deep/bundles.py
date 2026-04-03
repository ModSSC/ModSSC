from __future__ import annotations

import sys

from modssc.inductive.deep.bundle_factories import factory as _factory

# Keep the historical module path as a true module alias so monkeypatching
# `modssc.inductive.deep.bundles.*` still affects the implementation used at runtime.
sys.modules[__name__] = _factory
