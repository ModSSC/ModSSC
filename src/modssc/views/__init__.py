"""Multi-view feature generation.

This brick focuses on *feature views* (classic multi-view SSL methods such as Co-Training),
not on augmentation-based multi-view training (handled by :mod:`modssc.data_augmentation`).

The core entry-point is :func:`modssc.views.generate_views`.
"""

from .api import generate_views
from .errors import ViewsError, ViewsValidationError
from .plan import ColumnSelectSpec, ViewSpec, ViewsPlan, two_view_random_feature_split
from .types import ViewsResult

__all__ = [
    "generate_views",
    "ViewsError",
    "ViewsValidationError",
    "ColumnSelectSpec",
    "ViewSpec",
    "ViewsPlan",
    "two_view_random_feature_split",
    "ViewsResult",
]
