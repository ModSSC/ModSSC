"""Preprocessing and representation utilities for ModSSC.

This module provides a deterministic, cacheable, and extensible preprocessing pipeline that
consumes datasets from :mod:`modssc.data_loader` and (optionally) split information from
:mod:`modssc.sampling`.

Design goals:
- minimal imports (optional dependencies are loaded lazily)
- reproducibility via plan + dataset fingerprint + fit scope fingerprint
- step registry for easy contribution
"""

from modssc.preprocess.api import fit_transform, preprocess, resolve_plan
from modssc.preprocess.models import available_models, model_info
from modssc.preprocess.plan import PreprocessPlan, StepConfig
from modssc.preprocess.registry import available_steps, step_info

__all__ = [
    "PreprocessPlan",
    "StepConfig",
    "available_models",
    "available_steps",
    "fit_transform",
    "model_info",
    "preprocess",
    "resolve_plan",
    "step_info",
]
