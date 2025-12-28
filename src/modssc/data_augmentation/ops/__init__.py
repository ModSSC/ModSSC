"""Built-in augmentation operations.

Operations are registered on import.
"""

from . import (
    audio,  # noqa: F401
    core,  # noqa: F401
    graph,  # noqa: F401
    tabular,  # noqa: F401
    text,  # noqa: F401
    vision,  # noqa: F401
)

__all__ = ["audio", "core", "graph", "tabular", "text", "vision"]
