from __future__ import annotations

import sys

from modssc.inductive.methods.helpers import torch_support as _torch_support

# Keep the historical module path as a true module alias so monkeypatching
# `modssc.inductive.methods.deep_utils.*` still affects the implementation used at runtime.
sys.modules[__name__] = _torch_support
