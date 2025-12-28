from __future__ import annotations

from .patching import deep_merge, flatten_patch
from .space import Space
from .types import HpoError, Trial

__all__ = [
    "Space",
    "Trial",
    "HpoError",
    "deep_merge",
    "flatten_patch",
]
