from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

# Re-use the node2vec walk machinery (Planetoid uses a related context objective).
from modssc.graph.featurization.node2vec import (
    _build_adjacency,
    _random_walks_node2vec,
    _sample_negatives,
    _walk_pairs,
)
from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.optional import optional_import

from .common import normalize_device_name, prepare_data_cached, torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanetoidSpec:
    """A pragmatic Planetoid-style baseline.

    The original Planetoid (Yang et al.) combines a context prediction
    objective with a supervised classification objective. This implementation
    follows that spirit using a node2vec-style skip-gram objective plus a
    linear classifier trained jointly.

    Notes
    -----
    - This is intended as a *baseline* implementation without external deps.
    - It is full-batch for the supervised part and mini-batch for the context part.
    """

    embedding_dim: int = 128
    num_walks: int = 10
    walk_length: int = 40
    window_size: int = 5
    p: float = 1.0
    q: float = 1.0
    num_negative: int = 5
    batch_size: int = 1024
    sup_batch_size: int = 256
    epochs: int = 1
    lr: float = 0.01
    weight_decay: float = 0.0
    lambda_sup: float = 1.0
    undirected: bool = True


class PlanetoidMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="planetoid",
        name="Planetoid",
        year=2016,
        family="embedding",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Revisiting Semi-Supervised Learning with Graph Embeddings",
        paper_pdf="https://arxiv.org/abs/1603.08861",
        official_code="https://github.com/kimiyoung/planetoid",
    )

    def __init__(self, spec: PlanetoidSpec | None = None) -> None:
        self.spec = spec or PlanetoidSpec()
        self._device: Any | None = None
        self._emb: Any | None = None
        self._clf: Any | None = None
        self._n_nodes: int | None = None
        self._n_classes: int | None = None
        self._prep_cache: dict[str, Any] = {}

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> PlanetoidMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        optional_import("torch", extra="transductive-torch")

        self._device = normalize_device_name(device)

        # Walk generation on CPU
        prep = prepare_data_cached(
            data,
            device="cpu",
            add_self_loops=False,
            norm_mode="rw",
            cache=self._prep_cache,
        )
        n_nodes = prep.n_nodes
        self._n_nodes = n_nodes
        self._n_classes = prep.n_classes
        val_count = int(prep.val_mask.sum()) if prep.val_mask is not None else None
        logger.info(
            "Planetoid sizes: n_nodes=%s n_classes=%s train=%s val=%s",
            prep.n_nodes,
            prep.n_classes,
            int(prep.train_mask.sum()),
            val_count if val_count is not None else "none",
        )

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
            raise ValueError("planetoid: no training pairs could be generated (graph too sparse?)")

        # Negative sampling distribution
        deg = np.asarray([len(neigh) for neigh in adj], dtype=np.float64)
        deg = np.maximum(deg, 1.0)
        dist = deg**0.75
        dist = dist / dist.sum()

        torch_device = torch.device(self._device)
        emb = torch.nn.Embedding(n_nodes, self.spec.embedding_dim).to(torch_device)
        ctx = torch.nn.Embedding(n_nodes, self.spec.embedding_dim).to(torch_device)
        clf = torch.nn.Linear(self.spec.embedding_dim, prep.n_classes).to(torch_device)

        params = list(emb.parameters()) + list(ctx.parameters()) + list(clf.parameters())
        optimizer = torch.optim.Adam(params, lr=self.spec.lr, weight_decay=self.spec.weight_decay)

        rng = np.random.default_rng(seed)
        n_pairs = centers.shape[0]
        order = np.arange(n_pairs)

        centers_t = torch.as_tensor(centers, dtype=torch.long)
        contexts_t = torch.as_tensor(contexts, dtype=torch.long)

        y = prep.y.to(torch_device)
        train_idx = torch.where(prep.train_mask.to(torch_device))[0]
        if train_idx.numel() == 0:
            raise ValueError("planetoid: train_mask is empty")

        for _epoch in range(self.spec.epochs):
            rng.shuffle(order)
            for i in range(0, n_pairs, self.spec.batch_size):
                idx = order[i : i + self.spec.batch_size]
                c = centers_t[idx].to(torch_device)
                pos = contexts_t[idx].to(torch_device)

                neg_np = _sample_negatives(
                    rng,
                    num_nodes=n_nodes,
                    batch_size=len(idx),
                    num_neg=self.spec.num_negative,
                    dist=dist,
                )
                neg = torch.as_tensor(neg_np, dtype=torch.long, device=torch_device)

                v = emb(c)
                u_pos = ctx(pos)
                pos_score = (v * u_pos).sum(dim=-1)

                u_neg = ctx(neg)
                neg_score = (v.unsqueeze(1) * u_neg).sum(dim=-1)

                unsup_loss = (
                    -torch.nn.functional.logsigmoid(pos_score)
                    - torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
                ).mean()

                # supervised loss on a mini-batch of labeled nodes
                sup_sel = train_idx
                if train_idx.numel() > self.spec.sup_batch_size:
                    sel = torch.randperm(train_idx.numel(), device=torch_device)[
                        : self.spec.sup_batch_size
                    ]
                    sup_sel = train_idx[sel]

                logits = clf(emb(sup_sel))
                sup_loss = torch.nn.functional.cross_entropy(logits, y[sup_sel])

                loss = unsup_loss + self.spec.lambda_sup * sup_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._emb = emb.weight.detach().clone()
        self._clf = clf
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if self._emb is None or self._clf is None or self._n_nodes is None:
            raise RuntimeError("PlanetoidMethod is not fitted yet. Call fit() first.")

        prep = prepare_data_cached(
            data,
            device=self._device or "cpu",
            add_self_loops=False,
            norm_mode="rw",
            cache=self._prep_cache,
        )
        if prep.n_nodes != self._n_nodes:
            raise ValueError(
                f"Planetoid was fitted on n={self._n_nodes} nodes, got n={prep.n_nodes}"
            )

        self._clf.eval()
        with torch.no_grad():
            logits = self._clf(self._emb.to(torch.device(self._device or "cpu")))
            proba = torch.softmax(logits, dim=1)
        return proba.detach().cpu().numpy()
