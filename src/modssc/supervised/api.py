from __future__ import annotations

from typing import Any

from modssc.import_utils import load_object as _load_object
from modssc.supervised.errors import OptionalDependencyError, UnknownBackendError
from modssc.supervised.optional import has_module
from modssc.supervised.registry import get_backend_spec, get_spec, iter_specs
from modssc.supervised.types import ClassifierRuntime


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
                extra_to_module = {
                    "sklearn": "sklearn",
                    "vision": "torchvision",
                    "audio": "torchaudio",
                    "preprocess-text": "transformers",
                }
                module = extra_to_module.get(extra, extra)
                if extra.endswith("-torch"):
                    module = "torch"
                if has_module(module):
                    backends[b] = bs
            d["backends"] = backends
        out.append(d)
    return out


def classifier_info(classifier_id: str) -> dict[str, Any]:
    spec = get_spec(classifier_id)
    return spec.to_dict()


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
    spec = get_spec(classifier_id)
    params = _normalize_classifier_params(classifier_id, dict(params or {}))
    runtime = runtime or ClassifierRuntime()

    chosen_backend: str
    if backend == "auto":
        chosen_backend = ""
        for b in spec.preferred_backends:
            if b not in spec.backends:
                continue
            bs = spec.backends[b]
            if bs.required_extra is None:
                chosen_backend = b
                break
            module = "sklearn" if bs.required_extra == "sklearn" else bs.required_extra
            if bs.required_extra and bs.required_extra.endswith("-torch"):
                module = "torch"
            if has_module(module):
                chosen_backend = b
                break
        if not chosen_backend:
            # no backend available, raise based on first preferred backend
            first = spec.preferred_backends[0] if spec.preferred_backends else "unknown"
            if first in spec.backends and spec.backends[first].required_extra:
                raise OptionalDependencyError(
                    extra=str(spec.backends[first].required_extra),
                    feature=f"supervised:{classifier_id}",
                )
            raise UnknownBackendError(classifier_id, "auto")
    else:
        chosen_backend = backend

    bs = get_backend_spec(classifier_id, chosen_backend)

    # runtime injection (do not override explicit params)
    if "seed" not in params and runtime.seed is not None:
        params["seed"] = int(runtime.seed)
    if "n_jobs" not in params and runtime.n_jobs is not None:
        params["n_jobs"] = int(runtime.n_jobs)

    cls = _load_object(bs.factory)
    return cls(**params)
