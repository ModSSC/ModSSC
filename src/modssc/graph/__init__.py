"""Graph utilities for ModSSC.

This package provides:

- Graph construction: build a similarity graph from feature vectors
  (kNN, epsilon-ball, and anchor graphs).
- Graph featurization: derive tabular views from a graph
  (attribute, diffusion, and structural embeddings).
- Cache and fingerprints for reproducibility.

The graph representation is backend-agnostic. Optional backends exist (sklearn, faiss).
"""

from __future__ import annotations

from .artifacts import DatasetViews, GraphArtifact, NodeDataset
from .construction.api import build_graph
from .featurization.api import graph_to_views
from .specs import GraphBuilderSpec, GraphFeaturizerSpec, GraphWeightsSpec

__all__ = [
    "DatasetViews",
    "GraphArtifact",
    "NodeDataset",
    "GraphBuilderSpec",
    "GraphFeaturizerSpec",
    "GraphWeightsSpec",
    "build_graph",
    "graph_to_views",
]
