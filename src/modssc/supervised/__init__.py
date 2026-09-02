"""Supervised baselines for ModSSC.

This brick provides classic supervised classifiers used as baselines in SSL papers.
It is designed to be backend-agnostic (numpy, scikit-learn, torch, etc.).
"""

from __future__ import annotations

from modssc.supervised.api import (
    available_classifiers,
    classifier_info,
    create_classifier,
    resolve_classifier_backend_spec,
)
from modssc.supervised.base import ClassifierCapabilities, classifier_capabilities

__all__ = [
    "available_classifiers",
    "classifier_info",
    "ClassifierCapabilities",
    "classifier_capabilities",
    "create_classifier",
    "resolve_classifier_backend_spec",
]
