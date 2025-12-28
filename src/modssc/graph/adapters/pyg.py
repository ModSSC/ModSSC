from __future__ import annotations

from typing import Any

import numpy as np

from ..artifacts import NodeDataset
from ..optional import optional_import


def to_pyg_data(dataset: NodeDataset) -> Any:
    """Convert a NodeDataset into a PyTorch Geometric Data object."""
    torch = optional_import("torch", extra="graph")
    pyg_data = optional_import("torch_geometric.data", extra="graph")

    edge_index = torch.as_tensor(np.asarray(dataset.graph.edge_index, dtype=np.int64))
    x = torch.as_tensor(np.asarray(dataset.X))
    y = torch.as_tensor(np.asarray(dataset.y))

    data = pyg_data.Data(x=x, edge_index=edge_index, y=y)

    if dataset.graph.edge_weight is not None:
        data.edge_weight = torch.as_tensor(np.asarray(dataset.graph.edge_weight, dtype=np.float32))

    # masks
    for name, mask in dataset.masks.items():
        setattr(data, f"{name}_mask", torch.as_tensor(np.asarray(mask, dtype=bool)))

    return data
