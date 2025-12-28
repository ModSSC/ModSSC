"""Inductive methods (classic and deep baselines)."""

from .adamatch import AdaMatchMethod
from .co_training import CoTrainingMethod
from .fixmatch import FixMatchMethod
from .flexmatch import FlexMatchMethod
from .free_match import FreeMatchMethod
from .mean_teacher import MeanTeacherMethod
from .mixmatch import MixMatchMethod
from .pi_model import PiModelMethod
from .pseudo_label import PseudoLabelMethod
from .s4vm import S4VMMethod
from .self_training import SelfTrainingMethod
from .softmatch import SoftMatchMethod
from .tri_training import TriTrainingMethod
from .tsvm import TSVMMethod
from .uda import UDAMethod

__all__ = [
    "CoTrainingMethod",
    "AdaMatchMethod",
    "FixMatchMethod",
    "FlexMatchMethod",
    "FreeMatchMethod",
    "MeanTeacherMethod",
    "MixMatchMethod",
    "PiModelMethod",
    "PseudoLabelMethod",
    "S4VMMethod",
    "SelfTrainingMethod",
    "SoftMatchMethod",
    "TriTrainingMethod",
    "TSVMMethod",
    "UDAMethod",
]
