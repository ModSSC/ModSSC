from __future__ import annotations

from .classic.dynamic_label_propagation import (
    DynamicLabelPropagationSpec,
    dynamic_label_propagation,
)
from .classic.graph_mincuts import GraphMincutsSpec, graph_mincuts
from .classic.label_propagation import LabelPropagationSpec, label_propagation
from .classic.label_spreading import LabelSpreadingSpec, label_spreading
from .classic.laplace_learning import LaplaceLearningSpec, laplace_learning
from .classic.lazy_random_walk import LazyRandomWalkSpec, lazy_random_walk
from .pde.p_laplace_learning import PLaplaceLearningSpec, p_laplace_learning
from .pde.poisson_learning import PoissonLearningSpec, poisson_learning
from .pde.poisson_mbo import PoissonMBOSpec, poisson_mbo

"""Transductive methods.

This subpackage contains algorithm implementations that operate on a fixed graph
and propagate labels (or learned representations) over all nodes.

Only lightweight, dependency-minimal methods are placed here. Heavier models
(GNNs, transformers) should live in dedicated subpackages with optional extras.
"""

__all__ = [
    "GraphMincutsSpec",
    "graph_mincuts",
    "DynamicLabelPropagationSpec",
    "dynamic_label_propagation",
    "LaplaceLearningSpec",
    "laplace_learning",
    "LazyRandomWalkSpec",
    "lazy_random_walk",
    "LabelPropagationSpec",
    "label_propagation",
    "LabelSpreadingSpec",
    "label_spreading",
    "PoissonLearningSpec",
    "poisson_learning",
    "PoissonMBOSpec",
    "poisson_mbo",
    "PLaplaceLearningSpec",
    "p_laplace_learning",
]
