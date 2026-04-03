from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from modssc.supervised.errors import UnknownBackendError, UnknownClassifierError
from modssc.supervised.registry_data import BUILTIN_CLASSIFIERS
from modssc.supervised.types import BackendSpec, ClassifierSpec

_REGISTRY: dict[str, ClassifierSpec] = {}
_BOOTSTRAPPED = False


def register_classifier(
    *,
    key: str,
    description: str,
    preferred_backends: tuple[str, ...] = ("sklearn", "numpy"),
) -> None:
    if key in _REGISTRY:
        existing = _REGISTRY[key]
        if existing.description != description or existing.preferred_backends != preferred_backends:
            raise ValueError(f"Classifier already registered with different metadata: {key!r}")
        return
    _REGISTRY[key] = ClassifierSpec(
        key=key,
        description=description,
        backends={},
        preferred_backends=preferred_backends,
    )


def register_backend(
    *,
    classifier_id: str,
    backend: str,
    factory: str,
    required_extra: str | None = None,
    supports_gpu: bool = False,
    notes: str = "",
) -> None:
    if classifier_id not in _REGISTRY:
        raise UnknownClassifierError(classifier_id)
    spec = _REGISTRY[classifier_id]
    if backend in spec.backends:
        raise ValueError(f"Backend already registered for {classifier_id!r}: {backend!r}")
    new_backends = dict(spec.backends)
    new_backends[backend] = BackendSpec(
        backend=backend,
        factory=factory,
        required_extra=required_extra,
        supports_gpu=bool(supports_gpu),
        notes=str(notes),
    )
    _REGISTRY[classifier_id] = replace(spec, backends=new_backends)


def _register_builtin_classifier(entry: dict[str, object]) -> None:
    register_classifier(
        key=str(entry["key"]),
        description=str(entry["description"]),
        preferred_backends=tuple(entry["preferred_backends"]),
    )
    for backend in entry["backends"]:
        register_backend(
            classifier_id=str(entry["key"]),
            backend=str(backend["backend"]),
            factory=str(backend["factory"]),
            required_extra=backend["required_extra"],
            supports_gpu=bool(backend["supports_gpu"]),
            notes=str(backend["notes"]),
        )


def ensure_bootstrap() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    for entry in BUILTIN_CLASSIFIERS:
        _register_builtin_classifier(entry)
    _BOOTSTRAPPED = True


def list_classifiers() -> list[str]:
    ensure_bootstrap()
    return sorted(_REGISTRY.keys())


def iter_specs() -> Iterable[ClassifierSpec]:
    ensure_bootstrap()
    for key in sorted(_REGISTRY.keys()):
        yield _REGISTRY[key]


def get_spec(classifier_id: str) -> ClassifierSpec:
    ensure_bootstrap()
    if classifier_id not in _REGISTRY:
        raise UnknownClassifierError(classifier_id)
    return _REGISTRY[classifier_id]


def get_backend_spec(classifier_id: str, backend: str) -> BackendSpec:
    spec = get_spec(classifier_id)
    if backend not in spec.backends:
        raise UnknownBackendError(classifier_id, backend)
    return spec.backends[backend]
