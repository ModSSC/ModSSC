from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from modssc.cli._utils import DatasetCacheOption, ensure_mapping, exit_with_error, load_yaml_or_json
from modssc.graph import GraphBuilderSpec, GraphFeaturizerSpec, build_graph, graph_to_views
from modssc.graph.artifacts import NodeDataset
from modssc.graph.cache import GraphCache, ViewsCache
from modssc.graph.errors import GraphError, GraphValidationError
from modssc.logging import LogLevelOption, add_log_level_callback, configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Graph utilities.")
add_log_level_callback(app)

cache_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Graph cache operations.")
views_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Graph-derived views.")

app.add_typer(cache_app, name="cache")
app.add_typer(views_app, name="views")

logger = logging.getLogger(__name__)


def _extract_xy(ds: Any) -> tuple[Any, Any | None]:
    """Best-effort extraction of (X, y) from a dataset object."""
    # common patterns across bricks
    if hasattr(ds, "X"):
        X = ds.X
        y = getattr(ds, "y", None)
        return X, y
    if hasattr(ds, "features") and hasattr(ds.features, "X"):
        X = ds.features.X
        y = getattr(ds, "y", None)
        if y is None and hasattr(ds.features, "y"):
            y = ds.features.y
        return X, y
    if hasattr(ds, "train") and hasattr(ds.train, "X"):
        X = ds.train.X
        y = getattr(ds.train, "y", None)
        return X, y
    exit_with_error("Could not extract X from dataset. Expected ds.X, ds.features.X or ds.train.X")


def _load_dataset(key: str, *, cache_dir: Path | None = None):
    from modssc.data_loader import load_dataset
    from modssc.data_loader.errors import DataLoaderError

    try:
        return load_dataset(key, cache_dir=cache_dir)
    except DataLoaderError as exc:
        exit_with_error(str(exc))


def _load_graph_spec(path: Path) -> GraphBuilderSpec:
    data = load_yaml_or_json(path, label="graph spec")
    data = ensure_mapping(data, message="Graph spec must be a mapping at the root")

    allowed = {
        "scheme",
        "metric",
        "k",
        "radius",
        "symmetrize",
        "weights",
        "normalize",
        "self_loops",
        "backend",
        "chunk_size",
        "feature_field",
        "n_anchors",
        "anchors_k",
        "anchors_method",
        "candidate_limit",
        "faiss_exact",
        "faiss_hnsw_m",
        "faiss_ef_search",
        "faiss_ef_construction",
    }
    unknown = set(data.keys()) - allowed
    if unknown:
        keys = ", ".join(sorted(unknown))
        exit_with_error(f"Unknown keys in graph spec: {keys}")
    weights = data.get("weights", {})
    if weights is not None:
        if not isinstance(weights, dict):
            exit_with_error("weights must be a mapping")
        unknown_weights = set(weights.keys()) - {"kind", "sigma"}
        if unknown_weights:
            keys = ", ".join(sorted(unknown_weights))
            exit_with_error(f"Unknown keys in graph spec weights: {keys}")
    try:
        spec = GraphBuilderSpec.from_dict(data)
        spec.validate()
    except GraphValidationError as exc:
        exit_with_error(str(exc))
    return spec


@app.command("build")
def build_cmd(
    dataset: Annotated[
        str, typer.Option("--dataset", help="Dataset key (from modssc data_loader).")
    ],
    spec_path: Annotated[
        Path | None,
        typer.Option(
            "--spec",
            exists=True,
            dir_okay=False,
            help="Graph spec file (YAML/JSON).",
        ),
    ] = None,
    scheme: Annotated[str, typer.Option(help="Graph scheme: knn, epsilon, anchor.")] = "knn",
    metric: Annotated[str, typer.Option(help="Distance metric: cosine, euclidean.")] = "cosine",
    k: Annotated[int, typer.Option(help="k for knn/anchor schemes.")] = 30,
    radius: Annotated[float, typer.Option(help="Radius for epsilon scheme.")] = 0.5,
    backend: Annotated[str, typer.Option(help="Backend: auto, numpy, sklearn, faiss.")] = "auto",
    chunk_size: Annotated[
        int, typer.Option(help="Chunk size for numpy brute-force backends.")
    ] = 512,
    # anchor options
    n_anchors: Annotated[
        int | None, typer.Option(help="Number of anchors for anchor scheme (default: sqrt(n)).")
    ] = None,
    anchors_k: Annotated[int, typer.Option(help="Nearest anchors per node (anchor scheme).")] = 5,
    candidate_limit: Annotated[
        int, typer.Option(help="Max candidates per node (anchor scheme).")
    ] = 1000,
    anchors_method: Annotated[
        str, typer.Option(help="Anchor selection: random, kmeans.")
    ] = "random",
    # faiss options
    faiss_exact: Annotated[
        bool, typer.Option(help="Use exact FAISS index instead of HNSW.")
    ] = False,
    faiss_hnsw_m: Annotated[int, typer.Option(help="FAISS HNSW M parameter.")] = 32,
    faiss_ef_search: Annotated[int, typer.Option(help="FAISS HNSW efSearch parameter.")] = 64,
    faiss_ef_construction: Annotated[
        int, typer.Option(help="FAISS HNSW efConstruction parameter.")
    ] = 200,
    # misc
    seed: Annotated[int, typer.Option(help="Seed for deterministic components.")] = 0,
    cache: Annotated[bool, typer.Option(help="Use disk cache.")] = True,
    cache_dir: Annotated[str | None, typer.Option(help="Override cache directory.")] = None,
    dataset_cache_dir: DatasetCacheOption = None,
    edge_shard_size: Annotated[
        int | None, typer.Option(help="Store edges in shards of this size.")
    ] = None,
    resume: Annotated[
        bool, typer.Option(help="Resume numpy chunk computations from cache workdir.")
    ] = True,
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    ds = _load_dataset(dataset, cache_dir=dataset_cache_dir)
    X, _ = _extract_xy(ds)

    if spec_path is not None:
        spec = _load_graph_spec(spec_path)
    else:
        spec = GraphBuilderSpec(
            scheme=scheme,  # type: ignore[arg-type]
            metric=metric,  # type: ignore[arg-type]
            k=int(k),
            radius=float(radius),
            backend=backend,  # type: ignore[arg-type]
            chunk_size=int(chunk_size),
            n_anchors=n_anchors,
            anchors_k=int(anchors_k),
            candidate_limit=int(candidate_limit),
            anchors_method=anchors_method,  # type: ignore[arg-type]
            faiss_exact=bool(faiss_exact),
            faiss_hnsw_m=int(faiss_hnsw_m),
            faiss_ef_search=int(faiss_ef_search),
            faiss_ef_construction=int(faiss_ef_construction),
        )
        try:
            spec.validate()
        except GraphValidationError as exc:
            exit_with_error(str(exc))

    try:
        g = build_graph(
            X,
            spec=spec,
            seed=int(seed),
            cache=bool(cache),
            cache_dir=cache_dir,
            edge_shard_size=edge_shard_size,
            resume=bool(resume),
        )
    except GraphError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Graph build failed for dataset %s", dataset)
        exit_with_error(str(exc))

    typer.echo(
        json.dumps(
            {
                "fingerprint": g.meta.get("fingerprint"),
                "n_nodes": g.n_nodes,
                "n_edges": int(g.edge_index.shape[1]),
            },
            indent=2,
        )
    )


@views_app.command("build")
def views_build_cmd(
    dataset: Annotated[
        str, typer.Option("--dataset", help="Dataset key (from modssc data_loader).")
    ],
    views: Annotated[
        list[str] | None,
        typer.Option(help="Views to compute (repeatable): attr, diffusion, struct."),
    ] = None,
    diffusion_steps: Annotated[int, typer.Option(help="Diffusion steps.")] = 5,
    diffusion_alpha: Annotated[float, typer.Option(help="Diffusion alpha.")] = 0.1,
    struct_method: Annotated[
        str, typer.Option(help="Structural embedding method: deepwalk, node2vec.")
    ] = "deepwalk",
    struct_dim: Annotated[int, typer.Option(help="Structural embedding dimension.")] = 64,
    walk_length: Annotated[int, typer.Option(help="Walk length for struct embeddings.")] = 40,
    num_walks_per_node: Annotated[
        int, typer.Option(help="Walks per node for struct embeddings.")
    ] = 10,
    window_size: Annotated[int, typer.Option(help="Context window for struct embeddings.")] = 5,
    p: Annotated[float, typer.Option(help="Node2Vec return parameter p.")] = 1.0,
    q: Annotated[float, typer.Option(help="Node2Vec in-out parameter q.")] = 1.0,
    seed: Annotated[int, typer.Option(help="Seed for deterministic components.")] = 0,
    cache: Annotated[bool, typer.Option(help="Use disk cache.")] = True,
    cache_dir: Annotated[str | None, typer.Option(help="Override views cache directory.")] = None,
    dataset_cache_dir: DatasetCacheOption = None,
    # graph spec options (so the command is self-contained)
    scheme: Annotated[str, typer.Option(help="Graph scheme: knn, epsilon, anchor.")] = "knn",
    metric: Annotated[str, typer.Option(help="Distance metric: cosine, euclidean.")] = "cosine",
    k_graph: Annotated[int, typer.Option(help="k for knn/anchor schemes.")] = 30,
    radius: Annotated[float, typer.Option(help="Radius for epsilon scheme.")] = 0.5,
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    if views is None:
        views = ["attr"]
    ds = _load_dataset(dataset, cache_dir=dataset_cache_dir)
    X, y = _extract_xy(ds)
    if y is None:
        import numpy as np

        y = np.zeros((int(X.shape[0]),), dtype=np.int64)

    gspec = GraphBuilderSpec(
        scheme=scheme,  # type: ignore[arg-type]
        metric=metric,  # type: ignore[arg-type]
        k=int(k_graph),
        radius=float(radius),
    )
    try:
        g = build_graph(X, spec=gspec, seed=int(seed), cache=True)
    except GraphError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Graph build failed for dataset %s", dataset)
        exit_with_error(str(exc))

    spec = GraphFeaturizerSpec(
        views=tuple(views),  # type: ignore[arg-type]
        diffusion_steps=int(diffusion_steps),
        diffusion_alpha=float(diffusion_alpha),
        struct_method=struct_method,  # type: ignore[arg-type]
        struct_dim=int(struct_dim),
        walk_length=int(walk_length),
        num_walks_per_node=int(num_walks_per_node),
        window_size=int(window_size),
        p=float(p),
        q=float(q),
        cache=bool(cache),
    )

    nd = NodeDataset(X=X, y=y, graph=g, masks={})
    try:
        res = graph_to_views(nd, spec=spec, seed=int(seed), cache_dir=cache_dir)
    except GraphError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Graph views build failed for dataset %s", dataset)
        exit_with_error(str(exc))

    typer.echo(
        json.dumps(
            {"fingerprint": res.meta.get("fingerprint"), "views": list(res.views.keys())}, indent=2
        )
    )


@cache_app.command("ls")
def cache_ls_cmd(log_level: LogLevelOption = None) -> None:
    if log_level is not None:
        configure_logging(log_level)
    store = GraphCache.default()
    for fp in store.list():
        typer.echo(fp)


@cache_app.command("purge")
def cache_purge_cmd(log_level: LogLevelOption = None) -> None:
    if log_level is not None:
        configure_logging(log_level)
    store = GraphCache.default()
    n = store.purge()
    typer.echo(f"Purged {n} cached graphs.")


@views_app.command("cache-ls")
def views_cache_ls_cmd(log_level: LogLevelOption = None) -> None:
    if log_level is not None:
        configure_logging(log_level)
    store = ViewsCache.default()
    for fp in store.list():
        typer.echo(fp)
