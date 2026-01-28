from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore


@dataclass
class SparseAdjacencyStep:
    """Prepare graph data as a sparse adjacency list for Inductive GNNs (Tabula Rasa)."""

    max_nodes: int = 20000
    batch_size: int = 1024
    embed_epochs: int = 5
    embed_lr: float = 0.01
    undirected: bool = True
    device: str = "auto"

    def transform(self, store: ArtifactStore, *, rng: Any) -> dict[str, Any]:
        # Consume the transductive full graph structure
        edge_index_raw = store.require("graph.edge_index")
        x_raw = store.get("features.X")
        if x_raw is None:
            x_raw = store.require("raw.X")
        if store.has("graph.edge_weight"):
            edge_weight_raw = store.require("graph.edge_weight")
        else:
            edge_weight_raw = None

        edge_index = to_numpy(edge_index_raw)
        x = to_numpy(x_raw)

        # In a real inductive splitting scenario (e.g. PPI dataset),
        # the splitting logic should have already produced separate graphs for train/test.
        # But ModSSC "inductive" on citation networks usually just masks nodes.
        # Here we pass the dict structure that our GraphSAGE classifier expects.
        # This effectively leaks the whole graph structure into the classifier input,
        # but pure inductive methods will stick to the 'train' mask for updates.

        structured_x = {
            "x": x,
            "edge_index": edge_index,
        }
        if edge_weight_raw is not None:
            structured_x["edge_weight"] = to_numpy(edge_weight_raw)

        # We overwrite features.X with this dictionary.
        # Downstream methods must be able to handle X being a dict (or we need a custom adapter)
        return {"features.X": structured_x}
