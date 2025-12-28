from __future__ import annotations

from modssc.data_loader.types import DatasetSpec

# See https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html?highlight=cora for details.

GRAPH_CATALOG: dict[str, DatasetSpec] = {
    "cora": DatasetSpec(
        key="cora",
        provider="pyg",
        uri="pyg:Planetoid/Cora",
        modality="graph",
        task="node_classification",
        description="Cora Planetoid citation graph (torch_geometric). Includes official masks.",
        required_extra="graph",
        source_kwargs={},
    ),
    "citeseer": DatasetSpec(
        key="citeseer",
        provider="pyg",
        uri="pyg:Planetoid/CiteSeer",
        modality="graph",
        task="node_classification",
        description="CiteSeer Planetoid citation graph (torch_geometric). Includes official masks.",
        required_extra="graph",
        source_kwargs={},
    ),
    "pubmed": DatasetSpec(
        key="pubmed",
        provider="pyg",
        uri="pyg:Planetoid/PubMed",
        modality="graph",
        task="node_classification",
        description="PubMed Planetoid citation graph (torch_geometric). Includes official masks.",
        required_extra="graph",
        source_kwargs={},
    ),
}
