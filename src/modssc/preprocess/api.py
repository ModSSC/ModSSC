from __future__ import annotations

import sys

from modssc.preprocess.services import pipeline as _pipeline

# Keep the historical module path as a true module alias so monkeypatching
# `modssc.preprocess.api.*` still affects the implementation used at runtime.
sys.modules[__name__] = _pipeline
