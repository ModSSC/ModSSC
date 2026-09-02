"""Method registry for inductive SSL.

This registry stores import strings and avoids importing optional heavyweight
dependencies until a specific method is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from modssc.inductive.base import MethodInfo
from modssc.utils.registry import (
    make_available_methods,
    make_debug_registry,
    make_get_method_class,
    make_get_method_info,
    make_register_method,
)


@dataclass(frozen=True)
class MethodRef:
    method_id: str
    import_path: str  # "pkg.module:ClassName"
    status: Literal["implemented", "planned"] = "implemented"


_REGISTRY: dict[str, MethodRef] = {}
register_method = make_register_method(registry=_REGISTRY, ref_factory=MethodRef)
_BUILTIN_METHOD_IDS: set[str] = set()


def _register_builtin_method(
    method_id: str,
    import_path: str,
    *,
    status: Literal["implemented", "planned"] = "implemented",
) -> None:
    register_method(method_id, import_path, status=status)
    _BUILTIN_METHOD_IDS.add(method_id)


def register_builtin_methods() -> None:
    """Register built-in inductive methods (planned/implemented).

    This function is idempotent and safe to call multiple times.
    """
    # Keep the declarations below as the single source of truth: runtime
    # extensions still use the public ``register_method`` but are not added to
    # the built-in inventory used by release audits.
    register_method = _register_builtin_method
    register_method(
        "supervised",
        "modssc.inductive.methods.supervised:SupervisedMethod",
        status="implemented",
    )
    register_method(
        "pseudo_label",
        "modssc.inductive.methods.pseudo_label:PseudoLabelMethod",
        status="implemented",
    )
    register_method(
        "self_training",
        "modssc.inductive.methods.self_training:SelfTrainingMethod",
        status="implemented",
    )
    register_method(
        "setred",
        "modssc.inductive.methods.setred:SetredMethod",
        status="implemented",
    )
    register_method(
        "pi_model", "modssc.inductive.methods.pi_model:PiModelMethod", status="implemented"
    )
    register_method(
        "fixmatch", "modssc.inductive.methods.fixmatch:FixMatchMethod", status="implemented"
    )
    register_method(
        "comatch", "modssc.inductive.methods.comatch:CoMatchMethod", status="implemented"
    )
    register_method(
        "defixmatch",
        "modssc.inductive.methods.defixmatch:DeFixMatchMethod",
        status="implemented",
    )
    register_method(
        "daso",
        "modssc.inductive.methods.daso:DASOMethod",
        status="implemented",
    )
    register_method("adsh", "modssc.inductive.methods.adsh:ADSHMethod", status="implemented")
    register_method(
        "flexmatch", "modssc.inductive.methods.flexmatch:FlexMatchMethod", status="implemented"
    )
    register_method(
        "adamatch", "modssc.inductive.methods.adamatch:AdaMatchMethod", status="implemented"
    )
    register_method(
        "free_match", "modssc.inductive.methods.free_match:FreeMatchMethod", status="implemented"
    )
    register_method(
        "softmatch", "modssc.inductive.methods.softmatch:SoftMatchMethod", status="implemented"
    )
    register_method(
        "mixmatch", "modssc.inductive.methods.mixmatch:MixMatchMethod", status="implemented"
    )
    register_method(
        "simclr_v2",
        "modssc.inductive.methods.simclr_v2:SimCLRv2Method",
        status="implemented",
    )
    register_method(
        "mean_teacher",
        "modssc.inductive.methods.mean_teacher:MeanTeacherMethod",
        status="implemented",
    )
    register_method(
        "meta_pseudo_labels",
        "modssc.inductive.methods.meta_pseudo_labels:MetaPseudoLabelsMethod",
        status="implemented",
    )
    register_method(
        "temporal_ensembling",
        "modssc.inductive.methods.temporal_ensembling:TemporalEnsemblingMethod",
        status="implemented",
    )
    register_method("uda", "modssc.inductive.methods.uda:UDAMethod", status="implemented")
    register_method("vat", "modssc.inductive.methods.vat:VATMethod", status="implemented")
    register_method(
        "noisy_student",
        "modssc.inductive.methods.noisy_student:NoisyStudentMethod",
        status="implemented",
    )
    register_method(
        "co_training", "modssc.inductive.methods.co_training:CoTrainingMethod", status="implemented"
    )
    register_method(
        "democratic_co_learning",
        "modssc.inductive.methods.democratic_co_learning:DemocraticCoLearningMethod",
        status="implemented",
    )
    register_method(
        "deep_co_training",
        "modssc.inductive.methods.deep_co_training:DeepCoTrainingMethod",
        status="implemented",
    )
    register_method(
        "tri_training",
        "modssc.inductive.methods.tri_training:TriTrainingMethod",
        status="implemented",
    )
    register_method(
        "trinet",
        "modssc.inductive.methods.trinet:TriNetMethod",
        status="implemented",
    )
    register_method("s4vm", "modssc.inductive.methods.s4vm:S4VMMethod", status="implemented")


def builtin_methods(*, available_only: bool = True) -> list[str]:
    """Return method IDs shipped by ModSSC, excluding runtime extensions."""

    register_builtin_methods()
    method_ids = sorted(_BUILTIN_METHOD_IDS)
    if not available_only:
        return method_ids
    return [method_id for method_id in method_ids if _REGISTRY[method_id].status != "planned"]


available_methods = make_available_methods(
    registry=_REGISTRY, ensure_builtins=register_builtin_methods
)
get_method_class = make_get_method_class(
    registry=_REGISTRY,
    ensure_builtins=register_builtin_methods,
    available_methods=available_methods,
)
get_method_info = make_get_method_info(
    get_method_class=get_method_class,
    method_info_cls=MethodInfo,
)


def _debug_registry():
    return make_debug_registry(
        registry=_REGISTRY,
        ensure_builtins=register_builtin_methods,
    )()
