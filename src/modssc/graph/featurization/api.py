from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from ..artifacts import DatasetViews, NodeDataset
from ..cache import ViewsCache
from ..fingerprint import fingerprint_dict
from ..specs import GraphFeaturizerSpec
from ..validation import validate_featurizer_spec, validate_view_matrix
from .views.attr import attr_view
from .views.diffusion import diffusion_view
from .views.struct import StructParams, struct_embeddings

logger = logging.getLogger(__name__)


def _views_fingerprint(*, graph_fingerprint: str, spec: GraphFeaturizerSpec, seed: int) -> str:
    payload = {
        "graph_fingerprint": graph_fingerprint,
        "spec": spec.to_dict(),
        "seed": int(seed),
    }
    return fingerprint_dict(payload)


def graph_to_views(
    dataset: NodeDataset,
    *,
    spec: GraphFeaturizerSpec,
    seed: int = 0,
    cache: bool | None = None,
    cache_dir: str | Path | None = None,
) -> DatasetViews:
    """Compute one or more views from a (graph, X) dataset."""
    start = perf_counter()
    validate_featurizer_spec(spec)

    graph_fp = str(dataset.graph.meta.get("fingerprint", "")) if dataset.graph.meta else ""
    if not graph_fp:
        # fallback: fingerprint of graph structure not available
        graph_fp = fingerprint_dict(
            {
                "n_nodes": int(dataset.graph.n_nodes),
                "edge_index": dataset.graph.edge_index[
                    :2, : min(1000, dataset.graph.edge_index.shape[1])
                ].tolist(),
            }
        )

    views_fp = _views_fingerprint(graph_fingerprint=graph_fp, spec=spec, seed=int(seed))

    cache_enabled = bool(spec.cache) if cache is None else bool(cache)
    cache_store = ViewsCache(
        root=Path(cache_dir) if cache_dir is not None else ViewsCache.default().root
    )

    if cache_enabled and cache_store.exists(views_fp):
        cached, _ = cache_store.load(views_fp, y=np.asarray(dataset.y), masks=dataset.masks)
        logger.info(
            "Graph views cached: fingerprint=%s views=%s duration_s=%.3f",
            views_fp,
            list(spec.views),
            perf_counter() - start,
        )
        return cached

    views: dict[str, np.ndarray] = {}
    for name in spec.views:
        step_start = perf_counter()
        if name == "attr":
            views["attr"] = attr_view(dataset.X)
        elif name == "diffusion":
            views["diffusion"] = diffusion_view(
                X=np.asarray(dataset.X),
                n_nodes=int(dataset.graph.n_nodes),
                edge_index=np.asarray(dataset.graph.edge_index),
                edge_weight=(
                    np.asarray(dataset.graph.edge_weight)
                    if dataset.graph.edge_weight is not None
                    else None
                ),
                steps=int(spec.diffusion_steps),
                alpha=float(spec.diffusion_alpha),
            )
        elif name == "struct":
            sp = StructParams(
                method=spec.struct_method,
                dim=int(spec.struct_dim),
                walk_length=int(spec.walk_length),
                num_walks_per_node=int(spec.num_walks_per_node),
                window_size=int(spec.window_size),
                p=float(spec.p),
                q=float(spec.q),
            )
            views["struct"] = struct_embeddings(
                edge_index=dataset.graph.edge_index,
                n_nodes=int(dataset.graph.n_nodes),
                params=sp,
                seed=int(seed),
            )
        else:
            raise ValueError(f"Unknown view: {name!r}")
        logger.debug("Graph view built: name=%s duration_s=%.3f", name, perf_counter() - step_start)

    # validate output
    for k, v in views.items():
        validate_view_matrix(v, n_nodes=int(dataset.graph.n_nodes), name=k)

    out = DatasetViews(
        views=views,
        y=np.asarray(dataset.y),
        masks=dataset.masks,
        meta={
            "fingerprint": views_fp,
            "graph_fingerprint": graph_fp,
            "spec_fingerprint": fingerprint_dict(spec.to_dict()),
            "seed": int(seed),
        },
    )

    if cache_enabled:
        manifest = {
            "fingerprint": views_fp,
            "graph_fingerprint": graph_fp,
            "spec": spec.to_dict(),
            "spec_fingerprint": out.meta.get("spec_fingerprint"),
            "seed": int(seed),
        }
        cache_store.save(fingerprint=views_fp, views=out, manifest=manifest)

    logger.info(
        "Graph views done: fingerprint=%s views=%s duration_s=%.3f",
        views_fp,
        list(spec.views),
        perf_counter() - start,
    )
    return out
