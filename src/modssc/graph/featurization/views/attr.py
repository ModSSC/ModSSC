from __future__ import annotations

from typing import Any


def attr_view(X: Any) -> Any:
    """Attribute view: returns X as-is."""
    # Keep sparse/dense type to let downstream choose how to handle it.
    return X
