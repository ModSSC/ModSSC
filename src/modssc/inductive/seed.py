from __future__ import annotations

import random
from contextlib import suppress

import numpy as np

from .errors import OptionalDependencyError
from .optional import optional_import


def make_numpy_rng(seed: int) -> np.random.Generator:
    """Create a numpy Generator seeded with the given seed."""
    return np.random.default_rng(int(seed))


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Best-effort deterministic seeding across random, numpy, and torch (if available)."""
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)

    try:
        torch = optional_import("torch", extra="inductive-torch")
    except OptionalDependencyError:
        return

    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)

    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            with suppress(Exception):
                torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
