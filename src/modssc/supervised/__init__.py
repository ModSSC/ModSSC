"""Supervised baselines for ModSSC.

This brick provides classic supervised classifiers used as baselines in SSL papers.
It is designed to be backend-agnostic (numpy, scikit-learn, torch, etc.).
"""

from __future__ import annotations

from modssc.supervised.api import (
    available_classifiers,
    classifier_info,
    create_classifier,
)

__all__ = [
    "available_classifiers",
    "classifier_info",
    "create_classifier",
]
