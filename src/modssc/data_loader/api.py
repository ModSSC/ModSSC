from __future__ import annotations

import sys

from modssc.data_loader.services import service as _service

# Keep the historical module path as a true module alias so monkeypatching
# `modssc.data_loader.api.*` still affects the implementation used at runtime.
sys.modules[__name__] = _service
