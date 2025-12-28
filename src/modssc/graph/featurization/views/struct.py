from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...optional import optional_import

StructMethod = Literal["deepwalk", "node2vec"]


@dataclass(frozen=True)
class StructParams:
    method: StructMethod = "deepwalk"
    dim: int = 64
    walk_length: int = 40
    num_walks_per_node: int = 10
    window_size: int = 5
    p: float = 1.0
    q: float = 1.0
    max_dense_nodes: int = 2000  # use dense PPMI + numpy SVD under this threshold


def _derive_seed(seed: int, salt: int) -> int:
    x = (int(seed) + 0x9E3779B97F4A7C15 + int(salt)) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 30) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31) & 0xFFFFFFFFFFFFFFFF
    return int(x & 0xFFFFFFFF)


def _build_neighbors(
    edge_index: np.ndarray, *, n_nodes: int, with_sets: bool = True
) -> tuple[list[np.ndarray], list[set[int]] | None]:
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    neigh: list[list[int]] = [[] for _ in range(int(n_nodes))]
    for s, t in zip(src, dst, strict=True):
        s_i = int(s)
        t_i = int(t)
        if 0 <= s_i < n_nodes and 0 <= t_i < n_nodes:
            neigh[s_i].append(t_i)
    neigh_arr = [np.asarray(v, dtype=np.int64) for v in neigh]
    if not with_sets:
        return neigh_arr, None
    neigh_set = [set(v) for v in neigh]
    return neigh_arr, neigh_set


def _iter_random_walks(
    *,
    neighbors: list[np.ndarray],
    neighbor_sets: list[set[int]] | None,
    params: StructParams,
    seed: int,
) -> Iterator[np.ndarray]:
    if params.method != "deepwalk" and neighbor_sets is None:
        raise ValueError("neighbor_sets required for node2vec walks")
    n_nodes = len(neighbors)
    for start in range(n_nodes):
        if params.num_walks_per_node <= 0:
            break
        for w in range(int(params.num_walks_per_node)):
            rng = np.random.default_rng(_derive_seed(seed, start * 1000003 + w))
            walk = np.empty(int(params.walk_length), dtype=np.int64)
            walk[0] = start
            prev = -1
            cur = start
            for t in range(1, int(params.walk_length)):
                nbrs = neighbors[cur]
                if nbrs.size == 0:
                    walk[t] = cur
                    prev, cur = cur, cur
                    continue
                if params.method == "deepwalk" or prev < 0:
                    nxt = int(rng.choice(nbrs))
                else:
                    # node2vec biased transition
                    weights = np.empty(nbrs.size, dtype=np.float32)
                    prev_set = neighbor_sets[prev]
                    for i, v in enumerate(nbrs):
                        v_i = int(v)
                        if v_i == prev:
                            weights[i] = 1.0 / float(params.p)
                        elif v_i in prev_set:
                            weights[i] = 1.0
                        else:
                            weights[i] = 1.0 / float(params.q)
                    weights_sum = float(weights.sum())
                    if weights_sum <= 0:
                        nxt = int(rng.choice(nbrs))
                    else:
                        probs = weights / weights_sum
                        nxt = int(rng.choice(nbrs, p=probs))
                walk[t] = nxt
                prev, cur = cur, nxt
            yield walk


def _random_walks(
    *,
    neighbors: list[np.ndarray],
    neighbor_sets: list[set[int]] | None,
    params: StructParams,
    seed: int,
) -> list[np.ndarray]:
    return list(
        _iter_random_walks(
            neighbors=neighbors,
            neighbor_sets=neighbor_sets,
            params=params,
            seed=seed,
        )
    )


def _cooccurrence_counts(
    walks: Iterable[np.ndarray], *, window_size: int, n_nodes: int | None = None
) -> tuple[dict[tuple[int, int], int], np.ndarray, np.ndarray, int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    total = 0
    w = int(window_size)

    if n_nodes is None:
        row_sum: dict[int, int] = defaultdict(int)
        col_sum: dict[int, int] = defaultdict(int)
        for walk in walks:
            L = int(walk.size)
            for i in range(L):
                u = int(walk[i])
                lo = max(0, i - w)
                hi = min(L, i + w + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    v = int(walk[j])
                    counts[(u, v)] += 1
                    row_sum[u] += 1
                    col_sum[v] += 1
                    total += 1

        if total == 0:
            return {}, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), 0

        max_node = max(max(u, v) for (u, v) in counts)
        n = max_node + 1
        row_arr = np.zeros(n, dtype=np.int64)
        col_arr = np.zeros(n, dtype=np.int64)
        for k, v in row_sum.items():
            row_arr[k] = int(v)
        for k, v in col_sum.items():
            col_arr[k] = int(v)

        return counts, row_arr, col_arr, int(total)

    row_sum = np.zeros(int(n_nodes), dtype=np.int64)
    col_sum = np.zeros(int(n_nodes), dtype=np.int64)
    for walk in walks:
        L = int(walk.size)
        for i in range(L):
            u = int(walk[i])
            lo = max(0, i - w)
            hi = min(L, i + w + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                v = int(walk[j])
                if 0 <= u < n_nodes and 0 <= v < n_nodes:
                    counts[(u, v)] += 1
                    row_sum[u] += 1
                    col_sum[v] += 1
                    total += 1

    if total == 0:
        return {}, row_sum, col_sum, 0

    return counts, row_sum, col_sum, int(total)


def _ppmi_matrix_dense(
    *,
    n_nodes: int,
    counts: dict[tuple[int, int], int],
    row_sum: np.ndarray,
    col_sum: np.ndarray,
    total: int,
) -> np.ndarray:
    M = np.zeros((int(n_nodes), int(n_nodes)), dtype=np.float32)
    if total <= 0:
        return M
    tot = float(total)
    n_items = len(counts)
    if not n_items:
        return M

    u = np.fromiter((k[0] for k in counts), dtype=np.int64, count=n_items)
    v = np.fromiter((k[1] for k in counts), dtype=np.int64, count=n_items)
    c = np.fromiter((val for val in counts.values()), dtype=np.float64, count=n_items)

    mask = (u < int(n_nodes)) & (v < int(n_nodes))
    if not mask.any():
        return M

    ru = np.zeros(n_items, dtype=np.float64)
    cv = np.zeros(n_items, dtype=np.float64)
    ru_mask = u < row_sum.size
    cv_mask = v < col_sum.size
    if ru_mask.any():
        ru[ru_mask] = row_sum[u[ru_mask]]
    if cv_mask.any():
        cv[cv_mask] = col_sum[v[cv_mask]]

    mask &= (ru > 0) & (cv > 0)
    if not mask.any():
        return M

    pmi = np.zeros(n_items, dtype=np.float64)
    pmi[mask] = np.log((c[mask] * tot) / (ru[mask] * cv[mask]))
    mask &= pmi > 0
    if mask.any():
        M[u[mask], v[mask]] = pmi[mask].astype(np.float32, copy=False)
    return M


def struct_embeddings(
    *,
    edge_index: np.ndarray,
    n_nodes: int,
    params: StructParams,
    seed: int,
) -> np.ndarray:
    """Compute structural embeddings from the graph only."""
    if int(n_nodes) <= 0:
        return np.zeros((0, int(params.dim)), dtype=np.float32)

    use_sets = params.method == "node2vec"
    neighbors, neighbor_sets = _build_neighbors(
        edge_index, n_nodes=int(n_nodes), with_sets=use_sets
    )
    counts, row_sum, col_sum, total = _cooccurrence_counts(
        _iter_random_walks(
            neighbors=neighbors, neighbor_sets=neighbor_sets, params=params, seed=int(seed)
        ),
        window_size=int(params.window_size),
        n_nodes=int(n_nodes),
    )

    if int(n_nodes) <= int(params.max_dense_nodes):
        M = _ppmi_matrix_dense(
            n_nodes=int(n_nodes), counts=counts, row_sum=row_sum, col_sum=col_sum, total=total
        )
        # deterministic dense SVD
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        dim = int(min(int(params.dim), U.shape[1]))
        emb = (U[:, :dim] * S[:dim]).astype(np.float32)
        if dim < int(params.dim):
            out = np.zeros((int(n_nodes), int(params.dim)), dtype=np.float32)
            out[:, :dim] = emb
            return out
        return emb

    # large graphs require optional dependencies
    scipy_sparse = optional_import("scipy.sparse", extra="graph")
    sklearn_decomp = optional_import("sklearn.decomposition", extra="graph")
    TruncatedSVD = sklearn_decomp.TruncatedSVD

    tot = float(total) if total > 0 else 1.0
    n_items = len(counts)
    if n_items:
        u = np.fromiter((k[0] for k in counts), dtype=np.int64, count=n_items)
        v = np.fromiter((k[1] for k in counts), dtype=np.int64, count=n_items)
        c = np.fromiter((val for val in counts.values()), dtype=np.float64, count=n_items)

        mask = (u < int(n_nodes)) & (v < int(n_nodes))
        if mask.any():
            ru = np.zeros(n_items, dtype=np.float64)
            cv = np.zeros(n_items, dtype=np.float64)
            ru_mask = u < row_sum.size
            cv_mask = v < col_sum.size
            if ru_mask.any():
                ru[ru_mask] = row_sum[u[ru_mask]]
            if cv_mask.any():
                cv[cv_mask] = col_sum[v[cv_mask]]

            mask &= (ru > 0) & (cv > 0)
            if mask.any():
                pmi = np.zeros(n_items, dtype=np.float64)
                pmi[mask] = np.log((c[mask] * tot) / (ru[mask] * cv[mask]))
                mask &= pmi > 0
                rows = u[mask].astype(np.int64, copy=False)
                cols = v[mask].astype(np.int64, copy=False)
                data = pmi[mask].astype(np.float32, copy=False)
            else:
                rows = np.asarray([], dtype=np.int64)
                cols = np.asarray([], dtype=np.int64)
                data = np.asarray([], dtype=np.float32)
        else:
            rows = np.asarray([], dtype=np.int64)
            cols = np.asarray([], dtype=np.int64)
            data = np.asarray([], dtype=np.float32)
    else:
        rows = np.asarray([], dtype=np.int64)
        cols = np.asarray([], dtype=np.int64)
        data = np.asarray([], dtype=np.float32)

    M = scipy_sparse.csr_matrix((data, (rows, cols)), shape=(int(n_nodes), int(n_nodes)))
    svd = TruncatedSVD(n_components=int(params.dim), random_state=int(seed))
    emb = svd.fit_transform(M).astype(np.float32)
    return emb
