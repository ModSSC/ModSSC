from __future__ import annotations

import logging
from typing import Any

import numpy as np

from modssc.supervised.backends.torch.common import TorchSupportsProbaClassifierBase
from modssc.supervised.base import FitResult
from modssc.supervised.optional import optional_import
from modssc.supervised.utils import seed_everything

logger = logging.getLogger(__name__)


def _torch_geometric():
    torch = optional_import(
        "torch", extra="supervised-torch-geometric", feature="supervised:graphsage_inductive"
    )
    try:
        import importlib

        import torch_geometric

        nn_mod = importlib.import_module("torch_geometric.nn")
        if not hasattr(nn_mod, "SAGEConv"):
            raise ImportError("torch_geometric is required for GraphSAGE")
    except ImportError as e:
        raise ImportError("torch_geometric is required for GraphSAGE") from e
    return torch, torch_geometric


def _resolve_activation(name: str, torch):
    key = str(name).lower()
    if key == "relu":
        return torch.nn.ReLU()
    if key == "gelu":
        return torch.nn.GELU()
    if key == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"Unknown activation: {name!r}")


def _normalize_hidden_sizes(hidden_sizes: Any) -> tuple[int, ...] | None:
    if hidden_sizes is None:
        return None
    if isinstance(hidden_sizes, int):
        return (int(hidden_sizes),)
    if isinstance(hidden_sizes, (list, tuple)):
        return tuple(int(h) for h in hidden_sizes)
    raise ValueError("hidden_sizes must be an int or a sequence of ints.")


def _coerce_graph_input(X: Any, torch, *, allow_feature_only: bool) -> tuple[Any, Any]:
    if isinstance(X, dict):
        missing = {"x", "edge_index"} - set(X)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"TorchGraphSAGEClassifier requires X to define {missing_keys}.")
        x_feat = torch.as_tensor(X["x"], dtype=torch.float32)
        edge_index = torch.as_tensor(X["edge_index"], dtype=torch.long)
    elif allow_feature_only and hasattr(X, "shape"):
        x_feat = torch.as_tensor(X, dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        raise ValueError(
            "TorchGraphSAGEClassifier requires a dictionary with 'x' and 'edge_index' keys as X."
        )

    if int(x_feat.ndim) != 2:
        raise ValueError("Node features must be a 2D array.")
    if int(edge_index.ndim) != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("edge_index must have shape (2, num_edges).")
    return x_feat, edge_index


class TorchGraphSAGEClassifier(TorchSupportsProbaClassifierBase):
    """Inductive GraphSAGE classifier for graph-structured inputs."""

    classifier_id = "graphsage_inductive"
    backend = "torch"

    def __init__(
        self,
        *,
        hidden_channels: int | None = None,
        hidden_sizes: list[int] | None = None,
        num_layers: int | None = None,
        activation: str = "relu",
        dropout: float = 0.5,
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
        batch_size: int = 512,
        max_epochs: int = 100,
        num_neighbors: list[int] | None = None,  # e.g. [25, 10]
        seed: int | None = 0,
        n_jobs: int | None = None,
    ):
        super().__init__(seed=seed, n_jobs=n_jobs)
        resolved_hidden = hidden_channels
        resolved_layers = num_layers
        resolved_hidden_sizes = _normalize_hidden_sizes(hidden_sizes)
        if resolved_hidden_sizes:
            for h in resolved_hidden_sizes:
                if h <= 0:
                    raise ValueError("hidden_sizes must be positive.")
            if resolved_layers is not None and resolved_layers != len(resolved_hidden_sizes) + 1:
                raise ValueError(
                    "num_layers must equal len(hidden_sizes) + 1 when hidden_sizes is provided."
                )
            resolved_hidden = int(resolved_hidden_sizes[0])
            resolved_layers = len(resolved_hidden_sizes) + 1
        if resolved_hidden is None:
            resolved_hidden = 128
        if resolved_layers is None:
            resolved_layers = 2

        self.hidden_sizes = resolved_hidden_sizes
        self.hidden_channels = int(resolved_hidden)
        self.num_layers = int(resolved_layers)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.num_neighbors = num_neighbors or [15, 10]

        self._model: Any | None = None
        self._classes_t: Any | None = None

    def fit(self, X: Any, y: Any, **kwargs) -> FitResult:
        torch, _ = _torch_geometric()
        seed_value = None if self.seed is None else int(self.seed)
        if seed_value is not None:
            seed_everything(seed_value, deterministic=True)
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv

        x_feat, edge_index = _coerce_graph_input(X, torch, allow_feature_only=False)

        data = Data(x=x_feat, edge_index=edge_index)
        if "edge_weight" in X:
            data.edge_weight = torch.as_tensor(X["edge_weight"], dtype=torch.float32)

        y_tensor = torch.as_tensor(y, dtype=torch.long)
        if int(y_tensor.ndim) != 1:
            raise ValueError("y must be a 1D array of node labels.")
        if int(y_tensor.shape[0]) != int(x_feat.shape[0]):
            raise ValueError("X['x'] and y must contain the same number of nodes.")
        data.y = y_tensor

        num_classes = int(y_tensor.max().item()) + 1

        device = "cuda" if torch.cuda.is_available() and self.n_jobs != 0 else "cpu"

        data = data.to(device)
        classes = torch.arange(num_classes, device=device, dtype=torch.long)
        self._classes_t = classes
        self.classes_ = classes.detach().cpu().numpy()

        activation = _resolve_activation(self.activation, torch)

        class GNN(torch.nn.Module):
            def __init__(self, layer_sizes, dropout, activation):
                super().__init__()
                self.convs = torch.nn.ModuleList()
                for in_channels, out_channels in zip(
                    layer_sizes[:-1], layer_sizes[1:], strict=False
                ):
                    self.convs.append(SAGEConv(in_channels, out_channels))

                self.dropout = dropout
                self.activation = activation

            def forward(self, x, edge_index):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = self.activation(x)
                        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
                return x

        if self.hidden_sizes:
            layer_sizes = [data.num_features, *self.hidden_sizes, num_classes]
        else:
            layer_sizes = (
                [data.num_features] + [self.hidden_channels] * (self.num_layers - 1) + [num_classes]
            )

        self._model = GNN(layer_sizes=layer_sizes, dropout=self.dropout, activation=activation).to(
            device
        )

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        # Train on the provided subgraph in full-batch mode.
        self._model.train()
        for _epoch in range(self.max_epochs):
            optimizer.zero_grad()
            out = self._model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

        return FitResult(
            n_samples=int(data.x.shape[0]),
            n_features=int(data.x.shape[1]),
            n_classes=num_classes,
        )

    def predict(self, X: Any) -> Any:
        probs = self.predict_proba(X)
        if hasattr(probs, "cpu"):  # Tensor
            idx = probs.argmax(dim=1)
            if self._classes_t is not None:
                return self._classes_t[idx]
            return idx
        idx = probs.argmax(axis=1)
        classes = getattr(self, "classes_", None)
        if classes is not None:
            return np.asarray(classes)[idx]
        return idx

    def predict_proba(self, X: Any) -> Any:
        torch, _ = _torch_geometric()

        if not isinstance(X, dict) and not hasattr(X, "shape"):
            raise ValueError("Invalid input X")
        x_feat, edge_index = _coerce_graph_input(X, torch, allow_feature_only=True)

        device = next(self._model.parameters()).device
        x_feat = x_feat.to(device)
        edge_index = edge_index.to(device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x_feat, edge_index)
            probs = torch.softmax(logits, dim=1)

        return probs
