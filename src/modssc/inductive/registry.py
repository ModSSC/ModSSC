from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

from modssc.inductive.base import InductiveMethod, MethodInfo

"""Method registry for inductive SSL.

This registry stores import strings and avoids importing heavy dependencies
until a specific method is requested.
"""


@dataclass(frozen=True)
class MethodRef:
    method_id: str
    import_path: str  # "pkg.module:ClassName"
    status: Literal["implemented", "planned"] = "implemented"


_REGISTRY: dict[str, MethodRef] = {}


def register_method(
    method_id: str,
    import_path: str,
    *,
    status: Literal["implemented", "planned"] = "implemented",
) -> None:
    """Register a method by id and a lazy import string."""
    if not method_id or not isinstance(method_id, str):
        raise ValueError("method_id must be a non-empty string")
    if ":" not in import_path:
        raise ValueError("import_path must be of the form 'pkg.module:ClassName'")
    existing = _REGISTRY.get(method_id)
    if existing is not None and existing.import_path != import_path:
        raise ValueError(
            f"method_id {method_id!r} already registered with import_path={existing.import_path!r}"
        )
    if status not in {"implemented", "planned"}:
        raise ValueError("status must be 'implemented' or 'planned'")
    _REGISTRY[method_id] = MethodRef(method_id=method_id, import_path=import_path, status=status)


def register_builtin_methods() -> None:
    """Register built-in inductive methods (planned/implemented).

    This function is idempotent and safe to call multiple times.
    """
    register_method(
        "pseudo_label",
        "modssc.inductive.methods.pseudo_label:PseudoLabelMethod",
        status="implemented",
    )
    register_method(
        "pi_model", "modssc.inductive.methods.pi_model:PiModelMethod", status="implemented"
    )
    register_method(
        "fixmatch", "modssc.inductive.methods.fixmatch:FixMatchMethod", status="implemented"
    )
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
        "mean_teacher",
        "modssc.inductive.methods.mean_teacher:MeanTeacherMethod",
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
        "tri_training",
        "modssc.inductive.methods.tri_training:TriTrainingMethod",
        status="implemented",
    )
    register_method("tsvm", "modssc.inductive.methods.tsvm:TSVMMethod", status="implemented")
    register_method("s4vm", "modssc.inductive.methods.s4vm:S4VMMethod", status="implemented")


def available_methods(*, available_only: bool = True) -> list[str]:
    register_builtin_methods()
    methods = sorted(_REGISTRY.keys())
    if not available_only:
        return methods
    return [m for m in methods if _REGISTRY[m].status != "planned"]


def get_method_class(method_id: str) -> type[InductiveMethod]:
    register_builtin_methods()
    if method_id not in _REGISTRY:
        raise KeyError(f"Unknown method_id: {method_id!r}. Available: {available_methods()}")
    ref = _REGISTRY[method_id]
    mod_name, cls_name = ref.import_path.split(":")
    module = import_module(mod_name)
    return getattr(module, cls_name)


def get_method_info(method_id: str) -> MethodInfo:
    cls = get_method_class(method_id)
    info = getattr(cls, "info", None)
    if not isinstance(info, MethodInfo):
        raise TypeError(f"Method class {cls} must expose a class attribute `info: MethodInfo`")
    return info


def _debug_registry() -> dict[str, Any]:
    register_builtin_methods()
    return {k: v.import_path for k, v in _REGISTRY.items()}
