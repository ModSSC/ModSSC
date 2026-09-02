from __future__ import annotations

from .patching import deep_merge, flatten_patch
from .runtime import (
    RUNTIME_CONTRACT_FIELDS,
    PreparedTrial,
    SearchResult,
    SearchStatus,
    TrialResult,
    TrialStatus,
    run_search,
    validate_space_targets,
)
from .space import Space
from .types import HpoError, Trial

__all__ = [
    "HpoError",
    "PreparedTrial",
    "RUNTIME_CONTRACT_FIELDS",
    "SearchResult",
    "SearchStatus",
    "Space",
    "Trial",
    "TrialResult",
    "TrialStatus",
    "deep_merge",
    "flatten_patch",
    "run_search",
    "validate_space_targets",
]
