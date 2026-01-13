from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.device import resolve_device_name
from modssc.graph.featurization.node2vec import (
    _build_adjacency,
    _random_walks_node2vec,
    _sample_negatives,
    _walk_pairs,
)
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.optional import require
from modssc.preprocess.store import ArtifactStore


@dataclass
class GraphNode2VecStep:
    """Learn node2vec embeddings from graph edges and store as features.X."""

    embedding_dim: int = 16
    num_walks: int = 2
    walk_length: int = 5
    window_size: int = 2
    p: float = 1.0
    q: float = 1.0
    num_negative: int = 2
    batch_size: int = 256
    embed_epochs: int = 1
    embed_lr: float = 0.01
    undirected: bool = True
    device: str = "cpu"

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        edge_index = store.get("graph.edge_index")
        if edge_index is None:
            raise PreprocessValidationError("graph.node2vec requires graph.edge_index")

        y = store.get("raw.y")
        X = store.get("raw.X")
        if y is not None:
            n_nodes = int(np.asarray(y).shape[0])
        elif X is not None and hasattr(X, "shape"):
            n_nodes = int(np.asarray(X).shape[0])
        else:
            raise PreprocessValidationError(
                "graph.node2vec requires raw.y or raw.X to infer n_nodes"
            )

        edge_index_np = np.asarray(edge_index)
        if edge_index_np.ndim != 2:
            raise PreprocessValidationError("graph.edge_index must be a 2D array")
        if edge_index_np.shape[0] != 2 and edge_index_np.shape[1] == 2:
            edge_index_np = edge_index_np.T
        if edge_index_np.shape[0] != 2:
            raise PreprocessValidationError("graph.edge_index must have shape (2, E)")

        if int(self.embedding_dim) <= 0:
            raise PreprocessValidationError("embedding_dim must be > 0")
        if int(self.num_walks) <= 0:
            raise PreprocessValidationError("num_walks must be > 0")
        if int(self.walk_length) <= 1:
            raise PreprocessValidationError("walk_length must be > 1")
        if int(self.window_size) <= 0:
            raise PreprocessValidationError("window_size must be > 0")
        if int(self.num_negative) <= 0:
            raise PreprocessValidationError("num_negative must be > 0")
        if int(self.batch_size) <= 0:
            raise PreprocessValidationError("batch_size must be > 0")
        if int(self.embed_epochs) <= 0:
            raise PreprocessValidationError("embed_epochs must be > 0")

        seed = int(rng.integers(0, 1 << 31))
        np_rng = np.random.default_rng(seed)

        adj = _build_adjacency(edge_index_np, n_nodes=n_nodes, undirected=bool(self.undirected))
        walks = _random_walks_node2vec(
            adj,
            num_walks=int(self.num_walks),
            walk_length=int(self.walk_length),
            p=float(self.p),
            q=float(self.q),
            seed=seed,
        )
        centers, contexts = _walk_pairs(walks, window_size=int(self.window_size))
        if centers.size == 0:
            raise PreprocessValidationError("graph.node2vec could not generate training pairs")

        deg = np.asarray([len(neigh) for neigh in adj], dtype=np.float64)
        deg = np.maximum(deg, 1.0)
        dist = deg**0.75
        dist = dist / dist.sum()

        torch = require(module="torch", extra="transductive-torch", purpose="graph.node2vec")
        dev_name = resolve_device_name(self.device, torch=torch) or "cpu"
        device = torch.device(dev_name)
        emb = torch.nn.Embedding(n_nodes, int(self.embedding_dim)).to(device)
        ctx = torch.nn.Embedding(n_nodes, int(self.embedding_dim)).to(device)
        optimizer = torch.optim.Adam(
            list(emb.parameters()) + list(ctx.parameters()), lr=float(self.embed_lr)
        )

        centers_t = torch.as_tensor(centers, dtype=torch.long)
        contexts_t = torch.as_tensor(contexts, dtype=torch.long)

        n_pairs = int(centers_t.shape[0])
        order = np.arange(n_pairs, dtype=np.int64)

        for _epoch in range(int(self.embed_epochs)):
            np_rng.shuffle(order)
            for i in range(0, n_pairs, int(self.batch_size)):
                idx = order[i : i + int(self.batch_size)]
                c = centers_t[idx].to(device)
                pos = contexts_t[idx].to(device)

                neg_np = _sample_negatives(
                    np_rng,
                    num_nodes=n_nodes,
                    batch_size=len(idx),
                    num_neg=int(self.num_negative),
                    dist=dist,
                )
                neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)

                v = emb(c)
                u_pos = ctx(pos)
                pos_score = (v * u_pos).sum(dim=-1)

                u_neg = ctx(neg)
                neg_score = (v.unsqueeze(1) * u_neg).sum(dim=-1)

                loss = (
                    -torch.nn.functional.logsigmoid(pos_score)
                    - torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        out = emb.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        return {"features.X": out}
