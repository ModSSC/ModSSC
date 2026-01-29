from __future__ import annotations

import logging
from typing import Any

from modssc.supervised.base import BaseSupervisedClassifier, FitResult
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


class TorchGraphSAGEClassifier(BaseSupervisedClassifier):
    """Inductive GraphSAGE classifier using neighbor sampling (Tabula Rasa context)."""

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
        self._data: Any | None = None  # We need to hold reference to graph structure for inference

    @property
    def supports_proba(self) -> bool:
        return True

    def fit(self, X: Any, y: Any, **kwargs) -> FitResult:
        # X here is expected to be a PyG Data object or similar dictionary containing adj details
        # However, following ModSSC inductive contract, X might be node features and we need graph structure from context?
        # BUT: Inductive methods in ModSSC usually receive just X (features).
        # To handle graphs in inductive mode properly without breaking API, X should probably be passed as a dictionary containing 'x', 'edge_index', etc.
        # OR we rely on the fact that for inductive graph benchmarks, we usually pass the whole graph structure implicitly.

        torch, pyg = _torch_geometric()
        seed_value = None if self.seed is None else int(self.seed)
        if seed_value is not None:
            seed_everything(seed_value, deterministic=True)
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv

        # HACK: ModSSC inductive API usually passes X as numpy array of features.
        # For GraphSAGE, we need the edge_index.
        # We assume X contains 'graph_data' key if it comes from our custom preprocess,
        # OR we might need to change how data is passed.
        # Let's assume for this specific component that X is a dict with keys: x, edge_index, (optional edge_weight)
        # If X is just array, we can't do graph convolution unless we assume it's fully connected (pointless) or structure is implicit.

        if not isinstance(X, dict):
            # Fallback/Error: If we just get features, we can't do graph learning.
            # But wait, maybe we can assume X is the Data object itself if passed correctly?
            raise ValueError(
                "TorchGraphSAGEClassifier requires a dictionary with 'x' and 'edge_index' keys as X."
            )

        x_feat = torch.as_tensor(X["x"], dtype=torch.float32)
        edge_index = torch.as_tensor(X["edge_index"], dtype=torch.long)

        # y contains labels only for the training nodes (because of slice in method_inductive.py)
        # This is tricky: NeighborLoader needs `data` (whole graph) and `input_nodes` (indices).
        # We need to reconstruction the context.
        # Ideally, `fit` receives the full graph structure + mask/indices.
        # Since I cannot easily change the signature of fit(X, y), I'll assume X contains the GLOBAL graph data
        # and GLOBAL indices for the training set are implied or passed somehow.

        # SIMPLIFICATION FOR BENCHMARK:
        # We assume X has: 'x' (all nodes), 'edge_index' (all edges), 'train_mask' (boolean mask for current X subset)
        # Actually, standard ModSSC inductive cuts rows. slicing X rows breaks edge_index.
        # So we MUST pass the full graph object in X, and `y` tells us which rows are labeled? No, y is just labels.

        # New strategy: X is a PyG Data object representing the LOCAL subgraph available for training (inductive setting).
        # If we are strictly inductive, the training graph is disjoint from test graph.
        # So providing the whole training subgraph is correct.

        data = Data(x=x_feat, edge_index=edge_index)
        if "edge_weight" in X:
            data.edge_weight = torch.as_tensor(X["edge_weight"], dtype=torch.float32)

        # y corresponds to all nodes in this subgraph X?
        # If X is the training set features (sliced), then yes.
        # But we need edge_index within this slice.
        # Preprocess must ensure 'edge_index' is re-indexed to [0, len(X)].

        y_tensor = torch.as_tensor(y, dtype=torch.long)
        data.y = y_tensor  # This assumes X covers exactly the labeled nodes + context nodes?
        # In semi-supervised, we might have X_l and X_u. Inductive fit gets X_l + X_u usually?
        # Standard fit(X, y) in sklearn gets only Labeled data.
        # This breaks SSL. But `method_inductive` usually calls fit on labeled data only for the supervised baseline,
        # OR the SSL method handles the split.

        # If this is fulfilling the "classifier" role inside a SSL wrapper (e.g. SelfTraining),
        # fit is called with Labeled data Only.
        # This is problematic for GNNs: we need neighbors (unlabeled nodes) to convolve!
        # PURE INDUCTIVE GNN requires that even labeled nodes have access to their local structure.

        # Implementation Detail:
        # We'll treat X as a mini-graph. If it's just isolated nodes (because we sliced only labeled rows), GAGE becomes MLP.
        # For this to work "right", the upstream logic must pass a SubGraph including neighbors.
        # Assuming for now X includes 'edge_index' connecting the rows of X.

        num_classes = int(y_tensor.max().item()) + 1

        device = "cuda" if torch.cuda.is_available() and self.n_jobs != 0 else "cpu"

        data = data.to(device)

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

        # Full batch training on the provided subgraph for simplicity
        # (NeighborLoader is better for large graphs, but let's start simple for "Tabula Rasa" proof)
        self._model.train()
        for _epoch in range(self.max_epochs):
            optimizer.zero_grad()
            out = self._model(data.x, data.edge_index)
            loss = criterion(out, data.y)  # Assumes all nodes in X are labeled y
            loss.backward()
            optimizer.step()

        self._data = data  # Store structure for transductive inference if needed, but here we expect X at predict time too
        return FitResult(
            n_samples=int(data.x.shape[0]),
            n_features=int(data.x.shape[1]),
            n_classes=num_classes,
        )

    def predict(self, X: Any) -> Any:
        # X must be dict with x and edge_index
        probs = self.predict_proba(X)
        if hasattr(probs, "cpu"):  # Tensor
            return probs.argmax(dim=1)
        return probs.argmax(axis=1)

    def predict_proba(self, X: Any) -> Any:
        torch, _ = _torch_geometric()

        if not isinstance(X, dict):
            # If we receive plain array, we treat it as nodes without edges (MLP fallback behavior of SAGE with empty edge_index)
            if hasattr(X, "shape"):  # numpy or tensor
                x_feat = torch.as_tensor(X, dtype=torch.float32)
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                raise ValueError("Invalid input X")
        else:
            x_feat = torch.as_tensor(X["x"], dtype=torch.float32)
            edge_index = torch.as_tensor(X["edge_index"], dtype=torch.long)

        device = next(self._model.parameters()).device
        x_feat = x_feat.to(device)
        edge_index = edge_index.to(device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x_feat, edge_index)
            probs = torch.softmax(logits, dim=1)

        return probs
