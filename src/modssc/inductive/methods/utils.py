from __future__ import annotations

import sys

from modssc.inductive.methods.helpers import classifier_bridge as _classifier_bridge

# Keep the historical module path as a true module alias so monkeypatching
# `modssc.inductive.methods.utils.*` still affects the implementation used at runtime.
sys.modules[__name__] = _classifier_bridge
