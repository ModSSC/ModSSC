from __future__ import annotations

from typing import Any

from modssc.supervised.errors import OptionalDependencyError, UnknownBackendError
from modssc.supervised.optional import has_module, module_for_extra
from modssc.supervised.registry import get_backend_spec, get_spec, iter_specs
from modssc.supervised.types import BackendSpec, ClassifierRuntime
from modssc.utils.imports import load_object as _load_object


def _normalize_classifier_params(classifier_id: str, params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)

    # Keep bench/deep-config aliases compatible with direct supervised constructors.
    if classifier_id == "lstm_scratch":
        alias = out.get("hidden_size")
        if alias is None and "hidden_sizes" in out:
            alias = out.get("hidden_sizes")
            if isinstance(alias, (list, tuple)):
                alias = alias[0] if alias else None
        if alias is not None and "hidden_dim" not in out:
            out["hidden_dim"] = int(alias)
        out.pop("hidden_sizes", None)
        out.pop("hidden_size", None)

    return out


def available_classifiers(*, available_only: bool = False) -> list[dict[str, Any]]:
    """List classifiers and their backends.

    Parameters
    ----------
    available_only:
        If True, filter out backends whose required module is not importable.
    """
    out: list[dict[str, Any]] = []
    for spec in iter_specs():
        d = spec.to_dict()
        if available_only:
            backends = {}
            for b, bs in d["backends"].items():
                extra = bs.get("required_extra")
                if extra is None:
                    backends[b] = bs
                    continue
                module = module_for_extra(str(extra))
                if has_module(module):
                    backends[b] = bs
            d["backends"] = backends
        out.append(d)
    return out


def classifier_info(classifier_id: str) -> dict[str, Any]:
    spec = get_spec(classifier_id)
    return spec.to_dict()


def resolve_classifier_backend_spec(
    classifier_id: str,
    *,
    backend: str = "auto",
) -> BackendSpec:
    """Resolve a classifier backend with the policy used by construction.

    Keeping availability-aware ``auto`` selection in one public native helper
    ensures dependency discovery and classifier construction select the same
    backend on a given host.
    """

    spec = get_spec(classifier_id)
    if backend != "auto":
        return get_backend_spec(classifier_id, backend)

    for preferred_backend in spec.preferred_backends:
        if preferred_backend not in spec.backends:
            continue
        backend_spec = spec.backends[preferred_backend]
        if backend_spec.required_extra is None:
            return get_backend_spec(classifier_id, preferred_backend)
        module = module_for_extra(str(backend_spec.required_extra))
        if has_module(module):
            return get_backend_spec(classifier_id, preferred_backend)

    first = spec.preferred_backends[0] if spec.preferred_backends else "unknown"
    if first in spec.backends and spec.backends[first].required_extra:
        raise OptionalDependencyError(
            extra=str(spec.backends[first].required_extra),
            feature=f"supervised:{classifier_id}",
        )
    raise UnknownBackendError(classifier_id, "auto")


def create_classifier(
    classifier_id: str,
    *,
    backend: str = "auto",
    params: dict[str, Any] | None = None,
    runtime: ClassifierRuntime | None = None,
) -> Any:
    """Instantiate a classifier.

    Notes
    -----
    - backend="auto" selects the first available backend from preferred_backends.
    - params are passed to the backend constructor (after runtime injection).
    """
    params = _normalize_classifier_params(classifier_id, dict(params or {}))
    runtime = runtime or ClassifierRuntime()
    bs = resolve_classifier_backend_spec(classifier_id, backend=backend)

    # runtime injection (do not override explicit params)
    if "seed" not in params and runtime.seed is not None:
        params["seed"] = int(runtime.seed)
    if "n_jobs" not in params and runtime.n_jobs is not None:
        params["n_jobs"] = int(runtime.n_jobs)

    cls = _load_object(bs.factory)
    return cls(**params)
