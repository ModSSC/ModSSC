from __future__ import annotations

from .classic.graph_mincuts import GraphMincutsSpec, graph_mincuts
from .classic.label_propagation import LabelPropagationSpec, label_propagation
from .classic.label_spreading import LabelSpreadingSpec, label_spreading
from .pde.poisson_learning import PoissonLearningSpec, poisson_learning

"""Transductive methods.

This subpackage contains algorithm implementations that operate on a fixed graph
and propagate labels (or learned representations) over all nodes.

Only lightweight, dependency-minimal methods are placed here. Heavier models
(GNNs, transformers) should live in dedicated subpackages with optional extras.
"""

__all__ = [
    "GraphMincutsSpec",
    "graph_mincuts",
    "LabelPropagationSpec",
    "label_propagation",
    "LabelSpreadingSpec",
    "label_spreading",
    "PoissonLearningSpec",
    "poisson_learning",
]
