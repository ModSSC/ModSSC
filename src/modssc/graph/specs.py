from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .errors import GraphValidationError

# -----------------------------
# Public spec types
# -----------------------------

Metric = Literal["cosine", "euclidean"]
Scheme = Literal["knn", "epsilon", "anchor"]
Symmetrize = Literal["none", "or", "mutual"]
Normalize = Literal["none", "rw", "sym"]
Backend = Literal["auto", "numpy", "sklearn", "faiss"]

WeightKind = Literal["binary", "heat", "cosine"]

AnchorMethod = Literal["random", "kmeans"]

ViewName = Literal["attr", "diffusion", "struct"]
StructMethod = Literal["deepwalk", "node2vec"]


@dataclass(frozen=True)
class GraphWeightsSpec:
    """Specification for edge weights.

    Parameters
    ----------
    kind:
        - "binary": all edges weight 1
        - "heat": exp(-d^2/(2*sigma^2))
        - "cosine": convert cosine distances into similarities (1 - d)
    sigma:
        Used only for kind="heat".
    """

    kind: WeightKind = "binary"
    sigma: float | None = None

    def validate(self, *, metric: Metric) -> None:
        if self.kind not in ("binary", "heat", "cosine"):
            raise GraphValidationError(f"Unknown weight kind: {self.kind!r}")
        if self.kind == "heat":
            sigma = float(self.sigma or 0.0)
            if sigma <= 0:
                raise GraphValidationError("sigma must be > 0 for heat weights")
        if self.kind == "cosine" and metric != "cosine":
            raise GraphValidationError("cosine weights require metric='cosine'")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "sigma": self.sigma}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphWeightsSpec:
        return cls(kind=str(d.get("kind", "binary")), sigma=d.get("sigma"))


@dataclass(frozen=True)
class GraphBuilderSpec:
    """Graph construction specification.

    Notes
    -----
    This spec is designed to be serializable (via :meth:`to_dict`) and stable,
    so that it can be fingerprinted for reproducibility.

    Adds:
    - anchor scheme (approximate kNN via anchors)
    - faiss backend (optional)
    - chunk_size knob (for chunked numpy computations and resumable work dirs)
    """

    # main knobs
    scheme: Scheme = "knn"
    metric: Metric = "cosine"

    # scheme parameters
    k: int | None = 30
    radius: float | None = None  # epsilon

    # post-processing
    symmetrize: Symmetrize = "mutual"
    weights: GraphWeightsSpec = GraphWeightsSpec("heat", sigma=0.5)
    normalize: Normalize = "rw"
    self_loops: bool = True

    # backend selection
    backend: Backend = "auto"
    chunk_size: int = 512

    # where to read features from (when using higher-level orchestration)
    feature_field: str = "features.X"

    # anchor scheme
    n_anchors: int | None = None
    anchors_k: int = 5
    anchors_method: AnchorMethod = "random"
    candidate_limit: int = 1000

    # faiss backend (optional dependency)
    faiss_exact: bool = False
    faiss_hnsw_m: int = 32
    faiss_ef_search: int = 64
    faiss_ef_construction: int = 200

    def validate(self) -> None:
        if self.metric not in ("cosine", "euclidean"):
            raise GraphValidationError(f"Unknown metric: {self.metric!r}")

        if self.scheme == "knn":
            if self.k is None or int(self.k) <= 0:
                raise GraphValidationError("k must be a positive integer for knn scheme")
        elif self.scheme == "epsilon":
            if self.radius is None or float(self.radius) <= 0:
                raise GraphValidationError("radius must be > 0 for epsilon scheme")
        elif self.scheme == "anchor":
            if self.k is None or int(self.k) <= 0:
                raise GraphValidationError(
                    "k must be a positive integer for anchor scheme (final neighbors)"
                )
            if int(self.anchors_k) <= 0:
                raise GraphValidationError("anchors_k must be a positive integer")
            if self.n_anchors is not None and int(self.n_anchors) <= 0:
                raise GraphValidationError("n_anchors must be a positive integer when provided")
            if int(self.candidate_limit) <= 0:
                raise GraphValidationError("candidate_limit must be > 0")
            if self.anchors_method not in ("random", "kmeans"):
                raise GraphValidationError(f"Unknown anchors_method: {self.anchors_method!r}")
        else:
            raise GraphValidationError(f"Unknown scheme: {self.scheme!r}")

        if self.symmetrize not in ("none", "or", "mutual"):
            raise GraphValidationError(f"Unknown symmetrize mode: {self.symmetrize!r}")
        if self.normalize not in ("none", "rw", "sym"):
            raise GraphValidationError(f"Unknown normalize mode: {self.normalize!r}")

        if self.backend not in ("auto", "numpy", "sklearn", "faiss"):
            raise GraphValidationError(f"Unknown backend: {self.backend!r}")

        if int(self.chunk_size) <= 0:
            raise GraphValidationError("chunk_size must be > 0")

        # backend-specific constraints
        if self.backend == "faiss" and self.scheme == "epsilon":
            raise GraphValidationError("faiss backend does not support epsilon scheme")

        if int(self.faiss_hnsw_m) <= 0:
            raise GraphValidationError("faiss_hnsw_m must be > 0")
        if int(self.faiss_ef_search) <= 0:
            raise GraphValidationError("faiss_ef_search must be > 0")
        if int(self.faiss_ef_construction) <= 0:
            raise GraphValidationError("faiss_ef_construction must be > 0")

        self.weights.validate(metric=self.metric)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "metric": self.metric,
            "k": self.k,
            "radius": self.radius,
            "symmetrize": self.symmetrize,
            "weights": self.weights.to_dict(),
            "normalize": self.normalize,
            "self_loops": self.self_loops,
            "backend": self.backend,
            "chunk_size": int(self.chunk_size),
            "feature_field": self.feature_field,
            "n_anchors": self.n_anchors,
            "anchors_k": int(self.anchors_k),
            "anchors_method": self.anchors_method,
            "candidate_limit": int(self.candidate_limit),
            "faiss_exact": bool(self.faiss_exact),
            "faiss_hnsw_m": int(self.faiss_hnsw_m),
            "faiss_ef_search": int(self.faiss_ef_search),
            "faiss_ef_construction": int(self.faiss_ef_construction),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphBuilderSpec:
        # keep backward compatibility: missing keys fall back to legacy defaults
        return cls(
            scheme=str(d.get("scheme", "knn")),  # type: ignore[arg-type]
            metric=str(d.get("metric", "cosine")),  # type: ignore[arg-type]
            k=d.get("k"),
            radius=d.get("radius"),
            symmetrize=str(d.get("symmetrize", "mutual")),  # type: ignore[arg-type]
            weights=GraphWeightsSpec.from_dict(dict(d.get("weights", {}))),
            normalize=str(d.get("normalize", "rw")),  # type: ignore[arg-type]
            self_loops=bool(d.get("self_loops", True)),
            backend=str(d.get("backend", "auto")),  # type: ignore[arg-type]
            chunk_size=int(d.get("chunk_size", 512)),
            feature_field=str(d.get("feature_field", "features.X")),
            n_anchors=d.get("n_anchors"),
            anchors_k=int(d.get("anchors_k", 5)),
            anchors_method=str(d.get("anchors_method", "random")),  # type: ignore[arg-type]
            candidate_limit=int(d.get("candidate_limit", 1000)),
            faiss_exact=bool(d.get("faiss_exact", False)),
            faiss_hnsw_m=int(d.get("faiss_hnsw_m", 32)),
            faiss_ef_search=int(d.get("faiss_ef_search", 64)),
            faiss_ef_construction=int(d.get("faiss_ef_construction", 200)),
        )


@dataclass(frozen=True)
class GraphFeaturizerSpec:
    """Featurization spec to produce inductive views from a graph.

    Views
    -----
    attr:
        returns the original attribute matrix X
    diffusion:
        returns a simple diffusion of attributes over the graph
    struct:
        returns structural embeddings (DeepWalk/Node2Vec-style) computed from the graph
        only (X is ignored).

    Notes
    -----
    - The struct view is deterministic given the seed.
    - For large graphs, struct view may require optional dependencies.
    """

    views: tuple[ViewName, ...] = ("attr",)

    # diffusion
    diffusion_steps: int = 5
    diffusion_alpha: float = 0.1

    # struct
    struct_method: StructMethod = "deepwalk"
    struct_dim: int = 64
    walk_length: int = 40
    num_walks_per_node: int = 10
    window_size: int = 5
    p: float = 1.0
    q: float = 1.0

    cache: bool = True

    def validate(self) -> None:
        if self.diffusion_steps < 0:
            raise GraphValidationError("diffusion_steps must be >= 0")
        if not (0.0 <= float(self.diffusion_alpha) <= 1.0):
            raise GraphValidationError("diffusion_alpha must be in [0, 1]")

        if not self.views:
            raise GraphValidationError("views cannot be empty")
        for v in self.views:
            if v not in ("attr", "diffusion", "struct"):
                raise GraphValidationError(f"Unknown view: {v!r}")

        if self.struct_method not in ("deepwalk", "node2vec"):
            raise GraphValidationError(f"Unknown struct_method: {self.struct_method!r}")
        if int(self.struct_dim) <= 0:
            raise GraphValidationError("struct_dim must be > 0")
        if int(self.walk_length) <= 1:
            raise GraphValidationError("walk_length must be > 1")
        if int(self.num_walks_per_node) <= 0:
            raise GraphValidationError("num_walks_per_node must be > 0")
        if int(self.window_size) <= 0:
            raise GraphValidationError("window_size must be > 0")
        if float(self.p) <= 0:
            raise GraphValidationError("p must be > 0")
        if float(self.q) <= 0:
            raise GraphValidationError("q must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "views": list(self.views),
            "diffusion_steps": int(self.diffusion_steps),
            "diffusion_alpha": float(self.diffusion_alpha),
            "struct_method": self.struct_method,
            "struct_dim": int(self.struct_dim),
            "walk_length": int(self.walk_length),
            "num_walks_per_node": int(self.num_walks_per_node),
            "window_size": int(self.window_size),
            "p": float(self.p),
            "q": float(self.q),
            "cache": bool(self.cache),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphFeaturizerSpec:
        views = tuple(d.get("views", ["attr"]))
        return cls(
            views=views,  # type: ignore[arg-type]
            diffusion_steps=int(d.get("diffusion_steps", 5)),
            diffusion_alpha=float(d.get("diffusion_alpha", 0.1)),
            struct_method=str(d.get("struct_method", "deepwalk")),  # type: ignore[arg-type]
            struct_dim=int(d.get("struct_dim", 64)),
            walk_length=int(d.get("walk_length", 40)),
            num_walks_per_node=int(d.get("num_walks_per_node", 10)),
            window_size=int(d.get("window_size", 5)),
            p=float(d.get("p", 1.0)),
            q=float(d.get("q", 1.0)),
            cache=bool(d.get("cache", True)),
        )
