from __future__ import annotations

from typing import Literal

import numpy as np

SymmetrizeMode = Literal["none", "or", "mutual"]


def symmetrize_edges(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    mode: SymmetrizeMode,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Symmetrize a directed edge list.

    Representation
    -------------
    The returned edge list is directed. For an undirected pair (i, j), it will contain
    both (i -> j) and (j -> i).

    Weight aggregation
    ------------------
    If both directions exist, the undirected weight is the average of the two.
    If only one direction exists and mode is "or", the existing weight is used for both
    directions.
    """
    if mode == "none":
        return edge_index, edge_weight

    if mode not in ("or", "mutual"):
        raise ValueError(f"Unknown symmetrization mode: {mode!r}")

    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    w = np.asarray(edge_weight, dtype=np.float32) if edge_weight is not None else None
    if w is not None and w.shape[0] != src.shape[0]:
        m = min(int(src.shape[0]), int(dst.shape[0]), int(w.shape[0]))
        src = src[:m]
        dst = dst[:m]
        w = w[:m]

    if src.size == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.float32) if edge_weight is not None else None,
        )

    order = np.lexsort((dst, src))
    src_s = src[order]
    dst_s = dst[order]
    w_s = w[order] if w is not None else None

    change = (src_s[1:] != src_s[:-1]) | (dst_s[1:] != dst_s[:-1])
    starts = np.concatenate(([0], np.nonzero(change)[0] + 1))
    dir_src = src_s[starts]
    dir_dst = dst_s[starts]
    dir_w = np.maximum.reduceat(w_s, starts) if w_s is not None else None

    loop_mask = dir_src == dir_dst
    loops_src = dir_src[loop_mask]
    loops_dst = dir_dst[loop_mask]
    loops_w = dir_w[loop_mask] if dir_w is not None else None

    nl_mask = ~loop_mask
    dir_src = dir_src[nl_mask]
    dir_dst = dir_dst[nl_mask]
    dir_w = dir_w[nl_mask] if dir_w is not None else None

    if dir_src.size:
        a = np.minimum(dir_src, dir_dst)
        b = np.maximum(dir_src, dir_dst)
        forward = dir_src == a

        order2 = np.lexsort((b, a))
        a_s = a[order2]
        b_s = b[order2]
        forward_s = forward[order2]
        w_s2 = dir_w[order2] if dir_w is not None else None

        change2 = (a_s[1:] != a_s[:-1]) | (b_s[1:] != b_s[:-1])
        starts2 = np.concatenate(([0], np.nonzero(change2)[0] + 1))

        group_id = np.empty(a_s.shape[0], dtype=np.int64)
        group_id[0] = 0
        if group_id.size > 1:
            group_id[1:] = np.cumsum(change2)
        n_groups = int(group_id[-1]) + 1

        has_fwd = np.zeros(n_groups, dtype=bool)
        has_bwd = np.zeros(n_groups, dtype=bool)
        has_fwd[group_id[forward_s]] = True
        has_bwd[group_id[~forward_s]] = True

        keep = has_fwd & has_bwd if mode == "mutual" else has_fwd | has_bwd

        a_group = a_s[starts2]
        b_group = b_s[starts2]
        a_keep = a_group[keep]
        b_keep = b_group[keep]

        if w_s2 is not None:
            w_fwd = np.full(n_groups, np.nan, dtype=np.float32)
            w_bwd = np.full(n_groups, np.nan, dtype=np.float32)
            w_fwd[group_id[forward_s]] = w_s2[forward_s]
            w_bwd[group_id[~forward_s]] = w_s2[~forward_s]
            if mode == "mutual":
                w_pair = 0.5 * (w_fwd + w_bwd)
            else:
                both = has_fwd & has_bwd
                w_pair = np.where(both, 0.5 * (w_fwd + w_bwd), np.where(has_fwd, w_fwd, w_bwd))
            w_keep = w_pair[keep].astype(np.float32, copy=False)
            out_w = np.concatenate([w_keep, w_keep])
        else:
            out_w = None

        out_src = np.concatenate([a_keep, b_keep])
        out_dst = np.concatenate([b_keep, a_keep])
    else:
        out_src = np.asarray([], dtype=np.int64)
        out_dst = np.asarray([], dtype=np.int64)
        out_w = np.asarray([], dtype=np.float32) if edge_weight is not None else None

    if loops_src.size:
        if out_src.size:
            out_src = np.concatenate([out_src, loops_src])
            out_dst = np.concatenate([out_dst, loops_dst])
            if out_w is not None and loops_w is not None:
                out_w = np.concatenate([out_w, loops_w.astype(np.float32, copy=False)])
        else:
            out_src = loops_src.astype(np.int64, copy=False)
            out_dst = loops_dst.astype(np.int64, copy=False)
            if out_w is not None and loops_w is not None:
                out_w = loops_w.astype(np.float32, copy=False)

    ei = np.vstack([out_src.astype(np.int64, copy=False), out_dst.astype(np.int64, copy=False)])
    ew = out_w if edge_weight is not None else None

    # sanity: clip nodes
    if ei.size:
        ei[0] = np.clip(ei[0], 0, n_nodes - 1)
        ei[1] = np.clip(ei[1], 0, n_nodes - 1)

    return ei, ew
