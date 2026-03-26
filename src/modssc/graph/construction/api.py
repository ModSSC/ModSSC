from __future__ import annotations

import sys

from modssc.graph.construction.services import service as _service

# Keep the historical module path as a true module alias so monkeypatching
# `modssc.graph.construction.api.*` still affects the implementation used at runtime.
sys.modules[__name__] = _service
