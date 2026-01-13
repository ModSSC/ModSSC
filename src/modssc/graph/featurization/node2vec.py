from __future__ import annotations

import numpy as np


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
