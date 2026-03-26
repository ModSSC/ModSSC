from __future__ import annotations

from types import ModuleType
from typing import Any

from modssc.dependencies.optional import has_module as _has_module
from modssc.dependencies.optional import make_optional_import
from modssc.supervised.errors import OptionalDependencyError

_EXTRA_TO_MODULE = {
    "audio": "torchaudio",
    "hf": "datasets",
    "openml": "sklearn",
    "preprocess-text": "transformers",
    "sklearn": "sklearn",
    "supervised-torch": "torch",
    "supervised-torch-geometric": "torch_geometric",
    "tfds": "tensorflow_datasets",
    "vision": "torchvision",
}


def _error_factory(**kwargs) -> Exception:
    return OptionalDependencyError(extra=kwargs["extra"], feature=str(kwargs.get("feature")))


_optional_import_impl = make_optional_import(error_factory=_error_factory)


def optional_import(module: str, *, extra: str, feature: str) -> ModuleType:
    return _optional_import_impl(module, extra=extra, feature=feature)


def has_module(module: str) -> bool:
    return _has_module(module)


def module_for_extra(extra: str) -> str:
    if extra in _EXTRA_TO_MODULE:
        return _EXTRA_TO_MODULE[extra]
    return extra


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)
