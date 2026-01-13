from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

from modssc.transductive.base import MethodInfo, TransductiveMethod

"""Method registry for transductive node classification.

This registry stores *import strings* rather than importing methods eagerly.
This keeps optional heavyweight dependencies (e.g. torch) out of the core
import path.
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
    """Register built-in methods shipped with ModSSC.

    This function is idempotent and safe to call multiple times.
    """
    # Classic diffusion
    register_method(
        "label_propagation",
        "modssc.transductive.methods.classic.label_propagation:LabelPropagationMethod",
    )
    register_method(
        "label_spreading",
        "modssc.transductive.methods.classic.label_spreading:LabelSpreadingMethod",
    )
    register_method(
        "laplace_learning",
        "modssc.transductive.methods.classic.laplace_learning:LaplaceLearningMethod",
    )
    register_method(
        "lazy_random_walk",
        "modssc.transductive.methods.classic.lazy_random_walk:LazyRandomWalkMethod",
    )
    register_method(
        "dynamic_label_propagation",
        "modssc.transductive.methods.classic.dynamic_label_propagation:DynamicLabelPropagationMethod",
    )

    # Wave 2 (graph / PDE)
    register_method(
        "graph_mincuts",
        "modssc.transductive.methods.classic.graph_mincuts:GraphMincutsMethod",
    )
    register_method("tsvm", "modssc.transductive.methods.classic.tsvm:TSVMMethod")
    register_method(
        "poisson_learning",
        "modssc.transductive.methods.pde.poisson_learning:PoissonLearningMethod",
    )
    register_method(
        "poisson_mbo",
        "modssc.transductive.methods.pde.poisson_mbo:PoissonMBOMethod",
    )
    register_method(
        "p_laplace_learning",
        "modssc.transductive.methods.pde.p_laplace_learning:PLaplaceLearningMethod",
    )

    # GNN / embeddings (torch-only, no PyG)
    register_method("chebnet", "modssc.transductive.methods.gnn.chebnet:ChebNetMethod")
    register_method("planetoid", "modssc.transductive.methods.gnn.planetoid:PlanetoidMethod")
    register_method("gcn", "modssc.transductive.methods.gnn.gcn:GCNMethod")
    register_method("graphsage", "modssc.transductive.methods.gnn.graphsage:GraphSAGEMethod")
    register_method("gat", "modssc.transductive.methods.gnn.gat:GATMethod")
    register_method("sgc", "modssc.transductive.methods.gnn.sgc:SGCMethod")
    register_method("appnp", "modssc.transductive.methods.gnn.appnp:APPNPMethod")
    register_method("h_gcn", "modssc.transductive.methods.gnn.h_gcn:HGCNMethod")
    register_method("n_gcn", "modssc.transductive.methods.gnn.n_gcn:NGCNMethod")
    register_method("graphhop", "modssc.transductive.methods.gnn.graphhop:GraphHopMethod")
    register_method("grafn", "modssc.transductive.methods.gnn.grafn:GraFNMethod")
    register_method("gcnii", "modssc.transductive.methods.gnn.gcnii:GCNIIMethod")
    register_method("grand", "modssc.transductive.methods.gnn.grand:GRANDMethod")


def available_methods(*, available_only: bool = True) -> list[str]:
    register_builtin_methods()
    methods = sorted(_REGISTRY.keys())
    if not available_only:
        return methods
    return [m for m in methods if _REGISTRY[m].status != "planned"]


def get_method_class(method_id: str) -> type[TransductiveMethod]:
    register_builtin_methods()
    if method_id not in _REGISTRY:
        raise KeyError(f"Unknown method_id: {method_id!r}. Available: {available_methods()}")
    ref = _REGISTRY[method_id]
    mod_name, cls_name = ref.import_path.split(":")
    module = import_module(mod_name)
    return getattr(module, cls_name)


def get_method_info(method_id: str) -> MethodInfo:
    """Return the :class:`~modssc.transductive.base.MethodInfo` for a method."""
    cls = get_method_class(method_id)
    info = getattr(cls, "info", None)
    if not isinstance(info, MethodInfo):
        raise TypeError(f"Method class {cls} must expose a class attribute `info: MethodInfo`")
    return info


def _debug_registry() -> dict[str, Any]:
    """Internal helper for tests."""
    register_builtin_methods()
    return {k: v.import_path for k, v in _REGISTRY.items()}
