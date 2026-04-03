"""Inductive methods (classic and deep baselines)."""

from __future__ import annotations

from modssc.utils.imports import load_object

_EXPORTS = {
    "AdaMatchMethod": "modssc.inductive.methods.adamatch:AdaMatchMethod",
    "ADSHMethod": "modssc.inductive.methods.adsh:ADSHMethod",
    "CoTrainingMethod": "modssc.inductive.methods.co_training:CoTrainingMethod",
    "CoMatchMethod": "modssc.inductive.methods.comatch:CoMatchMethod",
    "DASOMethod": "modssc.inductive.methods.daso:DASOMethod",
    "DeepCoTrainingMethod": "modssc.inductive.methods.deep_co_training:DeepCoTrainingMethod",
    "DeFixMatchMethod": "modssc.inductive.methods.defixmatch:DeFixMatchMethod",
    "DemocraticCoLearningMethod": (
        "modssc.inductive.methods.democratic_co_learning:DemocraticCoLearningMethod"
    ),
    "FixMatchMethod": "modssc.inductive.methods.fixmatch:FixMatchMethod",
    "FlexMatchMethod": "modssc.inductive.methods.flexmatch:FlexMatchMethod",
    "FreeMatchMethod": "modssc.inductive.methods.free_match:FreeMatchMethod",
    "MeanTeacherMethod": "modssc.inductive.methods.mean_teacher:MeanTeacherMethod",
    "MetaPseudoLabelsMethod": (
        "modssc.inductive.methods.meta_pseudo_labels:MetaPseudoLabelsMethod"
    ),
    "MixMatchMethod": "modssc.inductive.methods.mixmatch:MixMatchMethod",
    "NoisyStudentMethod": "modssc.inductive.methods.noisy_student:NoisyStudentMethod",
    "PiModelMethod": "modssc.inductive.methods.pi_model:PiModelMethod",
    "PseudoLabelMethod": "modssc.inductive.methods.pseudo_label:PseudoLabelMethod",
    "S4VMMethod": "modssc.inductive.methods.s4vm:S4VMMethod",
    "SelfTrainingMethod": "modssc.inductive.methods.self_training:SelfTrainingMethod",
    "SetredMethod": "modssc.inductive.methods.setred:SetredMethod",
    "SimCLRv2Method": "modssc.inductive.methods.simclr_v2:SimCLRv2Method",
    "SoftMatchMethod": "modssc.inductive.methods.softmatch:SoftMatchMethod",
    "TemporalEnsemblingMethod": (
        "modssc.inductive.methods.temporal_ensembling:TemporalEnsemblingMethod"
    ),
    "TriNetMethod": "modssc.inductive.methods.trinet:TriNetMethod",
    "TriTrainingMethod": "modssc.inductive.methods.tri_training:TriTrainingMethod",
    "UDAMethod": "modssc.inductive.methods.uda:UDAMethod",
    "VATMethod": "modssc.inductive.methods.vat:VATMethod",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        import_path = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return load_object(import_path)
