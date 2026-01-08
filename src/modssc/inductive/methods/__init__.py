"""Inductive methods (classic and deep baselines)."""

from .adamatch import AdaMatchMethod
from .co_training import CoTrainingMethod
from .fixmatch import FixMatchMethod
from .flexmatch import FlexMatchMethod
from .free_match import FreeMatchMethod
from .mean_teacher import MeanTeacherMethod
from .mixmatch import MixMatchMethod
from .noisy_student import NoisyStudentMethod
from .pi_model import PiModelMethod
from .pseudo_label import PseudoLabelMethod
from .s4vm import S4VMMethod
from .softmatch import SoftMatchMethod
from .tri_training import TriTrainingMethod
from .tsvm import TSVMMethod
from .uda import UDAMethod
from .vat import VATMethod

__all__ = [
    "CoTrainingMethod",
    "AdaMatchMethod",
    "FixMatchMethod",
    "FlexMatchMethod",
    "FreeMatchMethod",
    "MeanTeacherMethod",
    "MixMatchMethod",
    "NoisyStudentMethod",
    "PiModelMethod",
    "PseudoLabelMethod",
    "S4VMMethod",
    "SoftMatchMethod",
    "TriTrainingMethod",
    "TSVMMethod",
    "UDAMethod",
    "VATMethod",
]
