from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.optional import optional_import

from .common import normalize_device_name, prepare_data_cached, torch, train_fullbatch

logger = logging.getLogger(__name__)


def _build_adjacency(edge_index: np.ndarray, *, n_nodes: int, undirected: bool) -> list[list[int]]:
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)

    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for s, d in zip(src.tolist(), dst.tolist(), strict=True):
        if 0 <= s < n_nodes and 0 <= d < n_nodes and s != d:
            adj[s].append(d)
            if undirected:
                adj[d].append(s)

    # Remove duplicates for stability (helps p/q logic).
    adj = [sorted(set(neigh)) for neigh in adj]
    return adj


def _random_walks_node2vec(
    adj: list[list[int]],
    *,
    num_walks: int,
    walk_length: int,
    p: float,
    q: float,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    n_nodes = len(adj)

    # For membership checks when p/q != 1
    neigh_sets = [set(neigh) for neigh in adj]

    walks: list[list[int]] = []
    for start in range(n_nodes):
        if not adj[start]:
            continue
        for _ in range(num_walks):
            walk = [start]
            prev = -1
            cur = start
            for _step in range(walk_length - 1):
                neigh = adj[cur]
                if not neigh:
                    break
                if prev == -1 or (p == 1.0 and q == 1.0):
                    nxt = neigh[int(rng.integers(0, len(neigh)))]
                else:
                    # node2vec biased transition probabilities
                    probs = []
                    for x in neigh:
                        if x == prev:
                            probs.append(1.0 / max(p, 1e-12))
                        elif x in neigh_sets[prev]:
                            probs.append(1.0)
                        else:
                            probs.append(1.0 / max(q, 1e-12))
                    probs = np.asarray(probs, dtype=np.float64)
                    probs = probs / probs.sum()
                    nxt = int(rng.choice(neigh, p=probs))
                walk.append(nxt)
                prev, cur = cur, nxt
            walks.append(walk)
    return walks


def _walk_pairs(walks: list[list[int]], *, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    centers: list[int] = []
    contexts: list[int] = []

    for w in walks:
        L = len(w)
        for i in range(L):
            center = w[i]
            j0 = max(0, i - window_size)
            j1 = min(L, i + window_size + 1)
            for j in range(j0, j1):
                if j == i:
                    continue
                centers.append(center)
                contexts.append(w[j])

    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


def _sample_negatives(
    rng: np.random.Generator, *, num_nodes: int, batch_size: int, num_neg: int, dist: np.ndarray
) -> np.ndarray:
    return rng.choice(num_nodes, size=(batch_size, num_neg), replace=True, p=dist).astype(np.int64)


@dataclass(frozen=True)
class Node2VecSpec:
    """Hyperparameters for node2vec + linear classifier."""

    embedding_dim: int = 128
    num_walks: int = 10
    walk_length: int = 40
    window_size: int = 5
    p: float = 1.0
    q: float = 1.0
    num_negative: int = 5
    batch_size: int = 1024
    embed_epochs: int = 1
    embed_lr: float = 0.01
    classifier_lr: float = 0.1
    classifier_weight_decay: float = 0.0
    classifier_max_epochs: int = 200
    classifier_patience: int = 50
    undirected: bool = True


class Node2VecMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="node2vec",
        name="node2vec",
        year=2016,
        family="embedding",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="node2vec: Scalable Feature Learning for Networks",
        paper_pdf="https://arxiv.org/abs/1607.00653",
        official_code="https://github.com/aditya-grover/node2vec",
    )

    def __init__(self, spec: Node2VecSpec | None = None) -> None:
        self.spec = spec or Node2VecSpec()
        self._device: Any | None = None
        self._emb: Any | None = None
        self._clf: Any | None = None
        self._n_nodes: int | None = None
        self._n_classes: int | None = None
        self._prep_cache: dict[str, Any] = {}

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> Node2VecMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        self._device = normalize_device_name(device)
        prep = prepare_data_cached(
            data,
            device="cpu",  # random walks on CPU (numpy)
            add_self_loops=False,
            norm_mode="rw",
            cache=self._prep_cache,
        )
        n_nodes = prep.n_nodes
        self._n_nodes = n_nodes
        self._n_classes = prep.n_classes
        train_count = int(prep.train_mask.sum())
        val_count = int(prep.val_mask.sum()) if prep.val_mask is not None else None
        logger.info(
            "Node2Vec sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            train_count,
            val_count if val_count is not None else "none",
        )

        # Build adjacency and random walks
        edge_index_np = prep.edge_index.detach().cpu().numpy()
        adj = _build_adjacency(edge_index_np, n_nodes=n_nodes, undirected=self.spec.undirected)
        walks = _random_walks_node2vec(
            adj,
            num_walks=self.spec.num_walks,
            walk_length=self.spec.walk_length,
            p=self.spec.p,
            q=self.spec.q,
            seed=seed,
        )
        centers, contexts = _walk_pairs(walks, window_size=self.spec.window_size)
        if centers.size == 0:
            raise ValueError("node2vec: no training pairs could be generated (graph too sparse?)")

        # Negative sampling distribution ~ degree^0.75
        deg = np.asarray([len(neigh) for neigh in adj], dtype=np.float64)
        deg = np.maximum(deg, 1.0)
        dist = deg**0.75
        dist = dist / dist.sum()

        # Train embeddings (torch)
        torch_device = torch.device(self._device)
        emb = torch.nn.Embedding(n_nodes, self.spec.embedding_dim).to(torch_device)
        ctx = torch.nn.Embedding(n_nodes, self.spec.embedding_dim).to(torch_device)
        optimizer = torch.optim.Adam(
            list(emb.parameters()) + list(ctx.parameters()), lr=self.spec.embed_lr
        )

        rng = np.random.default_rng(seed)
        n_pairs = centers.shape[0]
        order = np.arange(n_pairs)

        centers_t = torch.as_tensor(centers, dtype=torch.long)
        contexts_t = torch.as_tensor(contexts, dtype=torch.long)

        for _epoch in range(self.spec.embed_epochs):
            rng.shuffle(order)
            for i in range(0, n_pairs, self.spec.batch_size):
                idx = order[i : i + self.spec.batch_size]
                c = centers_t[idx].to(torch_device)
                pos = contexts_t[idx].to(torch_device)

                # negatives on CPU then to device
                neg_np = _sample_negatives(
                    rng,
                    num_nodes=n_nodes,
                    batch_size=len(idx),
                    num_neg=self.spec.num_negative,
                    dist=dist,
                )
                neg = torch.as_tensor(neg_np, dtype=torch.long, device=torch_device)

                v = emb(c)  # (B, D)
                u_pos = ctx(pos)  # (B, D)
                pos_score = (v * u_pos).sum(dim=-1)  # (B,)

                u_neg = ctx(neg)  # (B, K, D)
                neg_score = (v.unsqueeze(1) * u_neg).sum(dim=-1)  # (B, K)

                loss_pos = -torch.nn.functional.logsigmoid(pos_score)
                loss_neg = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
                loss = (loss_pos + loss_neg).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save embeddings (use 'emb' weights only)
        self._emb = emb.weight.detach().clone()

        # Train classifier on embeddings (full-batch)
        X_emb = self._emb.to(torch_device)
        y = prep.y.to(torch_device)
        train_mask = prep.train_mask.to(torch_device)
        val_mask = prep.val_mask.to(torch_device) if prep.val_mask is not None else None

        clf = torch.nn.Linear(self.spec.embedding_dim, prep.n_classes).to(torch_device)
        self._clf = clf

        train_fullbatch(
            model=clf,
            forward_fn=lambda: clf(X_emb),
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            lr=self.spec.classifier_lr,
            weight_decay=self.spec.classifier_weight_decay,
            max_epochs=self.spec.classifier_max_epochs,
            patience=self.spec.classifier_patience,
            seed=seed,
        )
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if (
            self._emb is None
            or self._clf is None
            or self._n_nodes is None
            or self._n_classes is None
        ):
            raise RuntimeError("Node2VecMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=False,
            norm_mode="rw",
            cache=self._prep_cache,
        )
        if prep.n_nodes != self._n_nodes:
            raise ValueError(
                f"node2vec was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}"
            )

        self._clf.eval()
        with torch.no_grad():
            logits = self._clf(self._emb.to(torch.device(self._device or "cpu")))
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
