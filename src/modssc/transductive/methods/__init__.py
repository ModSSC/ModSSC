"""Transductive methods.

This subpackage contains algorithm implementations operating on a fixed graph
to propagate labels or learned representations over all nodes.
"""

from __future__ import annotations

from modssc.utils.imports import load_object

_EXPORTS = {
    "DynamicLabelPropagationSpec": (
        "modssc.transductive.methods.classic.dynamic_label_propagation:DynamicLabelPropagationSpec"
    ),
    "dynamic_label_propagation": (
        "modssc.transductive.methods.classic.dynamic_label_propagation:dynamic_label_propagation"
    ),
    "GraphMincutsSpec": "modssc.transductive.methods.classic.graph_mincuts:GraphMincutsSpec",
    "graph_mincuts": "modssc.transductive.methods.classic.graph_mincuts:graph_mincuts",
    "LabelPropagationSpec": (
        "modssc.transductive.methods.classic.label_propagation:LabelPropagationSpec"
    ),
    "label_propagation": (
        "modssc.transductive.methods.classic.label_propagation:label_propagation"
    ),
    "LabelSpreadingSpec": (
        "modssc.transductive.methods.classic.label_spreading:LabelSpreadingSpec"
    ),
    "label_spreading": "modssc.transductive.methods.classic.label_spreading:label_spreading",
    "LaplaceLearningSpec": (
        "modssc.transductive.methods.classic.laplace_learning:LaplaceLearningSpec"
    ),
    "laplace_learning": "modssc.transductive.methods.classic.laplace_learning:laplace_learning",
    "LazyRandomWalkSpec": (
        "modssc.transductive.methods.classic.lazy_random_walk:LazyRandomWalkSpec"
    ),
    "lazy_random_walk": ("modssc.transductive.methods.classic.lazy_random_walk:lazy_random_walk"),
    "PLaplaceLearningSpec": (
        "modssc.transductive.methods.pde.p_laplace_learning:PLaplaceLearningSpec"
    ),
    "p_laplace_learning": ("modssc.transductive.methods.pde.p_laplace_learning:p_laplace_learning"),
    "PoissonLearningSpec": ("modssc.transductive.methods.pde.poisson_learning:PoissonLearningSpec"),
    "poisson_learning": "modssc.transductive.methods.pde.poisson_learning:poisson_learning",
    "PoissonMBOSpec": "modssc.transductive.methods.pde.poisson_mbo:PoissonMBOSpec",
    "poisson_mbo": "modssc.transductive.methods.pde.poisson_mbo:poisson_mbo",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        import_path = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return load_object(import_path)
