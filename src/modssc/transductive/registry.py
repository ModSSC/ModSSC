from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from modssc.registry_utils import (
    make_available_methods,
    make_debug_registry,
    make_get_method_class,
    make_get_method_info,
    make_register_method,
)
from modssc.transductive.base import MethodInfo

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
register_method = make_register_method(registry=_REGISTRY, ref_factory=MethodRef)


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
_debug_registry = make_debug_registry(
    registry=_REGISTRY,
    ensure_builtins=register_builtin_methods,
)
