"""Inductive methods (classic and deep baselines)."""

from .adamatch import AdaMatchMethod
from .adsh import ADSHMethod
from .co_training import CoTrainingMethod
from .comatch import CoMatchMethod
from .daso import DASOMethod
from .deep_co_training import DeepCoTrainingMethod
from .defixmatch import DeFixMatchMethod
from .democratic_co_learning import DemocraticCoLearningMethod
from .fixmatch import FixMatchMethod
from .flexmatch import FlexMatchMethod
from .free_match import FreeMatchMethod
from .mean_teacher import MeanTeacherMethod
from .meta_pseudo_labels import MetaPseudoLabelsMethod
from .mixmatch import MixMatchMethod
from .noisy_student import NoisyStudentMethod
from .pi_model import PiModelMethod
from .pseudo_label import PseudoLabelMethod
from .s4vm import S4VMMethod
from .self_training import SelfTrainingMethod
from .setred import SetredMethod
from .simclr_v2 import SimCLRv2Method
from .softmatch import SoftMatchMethod
from .temporal_ensembling import TemporalEnsemblingMethod
from .tri_training import TriTrainingMethod
from .trinet import TriNetMethod
from .tsvm import TSVMMethod
from .uda import UDAMethod
from .vat import VATMethod

__all__ = [
    "CoTrainingMethod",
    "CoMatchMethod",
    "DeepCoTrainingMethod",
    "AdaMatchMethod",
    "ADSHMethod",
    "DemocraticCoLearningMethod",
    "DeFixMatchMethod",
    "DASOMethod",
    "FixMatchMethod",
    "FlexMatchMethod",
    "FreeMatchMethod",
    "MeanTeacherMethod",
    "MetaPseudoLabelsMethod",
    "MixMatchMethod",
    "NoisyStudentMethod",
    "PiModelMethod",
    "PseudoLabelMethod",
    "SetredMethod",
    "SelfTrainingMethod",
    "S4VMMethod",
    "SimCLRv2Method",
    "SoftMatchMethod",
    "TemporalEnsemblingMethod",
    "TriNetMethod",
    "TriTrainingMethod",
    "TSVMMethod",
    "UDAMethod",
    "VATMethod",
]
