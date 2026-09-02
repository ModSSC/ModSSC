from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import GraphValidationError

# -----------------------------
# Public spec types
# -----------------------------

Metric = Literal["cosine", "euclidean"]
Scheme = Literal["knn", "epsilon", "anchor"]
Symmetrize = Literal["none", "or", "mutual", "mean", "sum"]
DiagonalPolicy = Literal["preserve", "zero"]
Normalize = Literal["none", "rw", "sym"]
Backend = Literal["auto", "numpy", "sklearn", "faiss", "annoy", "torch", "precomputed"]
EdgeWeightDType = Literal["float32", "float64"]

WeightKind = Literal["binary", "heat", "cosine", "knn_gaussian"]

AnchorMethod = Literal["random", "kmeans"]

ViewName = Literal["attr", "diffusion", "struct"]
StructMethod = Literal["deepwalk", "node2vec"]

_BACKEND_REQUIRED_EXTRAS: dict[str, str] = {
    "sklearn": "sklearn",
    "faiss": "graph-faiss",
    "annoy": "graph-annoy",
    "torch": "inductive-torch",
}


def graph_backend_required_extra(backend: str) -> str | None:
    """Return the optional extra used by one already-resolved graph backend."""

    if backend not in {"numpy", "sklearn", "faiss", "annoy", "torch", "precomputed"}:
        raise GraphValidationError(f"graph backend must be resolved, got {backend!r}")
    return _BACKEND_REQUIRED_EXTRAS.get(backend)


@dataclass(frozen=True)
class GraphWeightsSpec:
    """Specification for edge weights.

    Parameters
    ----------
    kind:
        - "binary": all edges weight 1
        - "heat": exp(-d^2/(2*sigma^2))
        - "cosine": convert cosine distances into similarities (1 - d)
        - "knn_gaussian": local-scale local scale
          exp(-4*d_ij^2/d_k(x_i)^2), for KNN graphs only
    sigma:
        Used only for kind="heat".
    """

    kind: WeightKind = "binary"
    sigma: float | None = None

    def validate(self, *, metric: Metric) -> None:
        if self.kind not in ("binary", "heat", "cosine", "knn_gaussian"):
            raise GraphValidationError(f"Unknown weight kind: {self.kind!r}")
        if self.kind == "heat":
            sigma = float(self.sigma or 0.0)
            if sigma <= 0:
                raise GraphValidationError("sigma must be > 0 for heat weights")
        if self.kind == "cosine" and metric != "cosine":
            raise GraphValidationError("cosine weights require metric='cosine'")
        if self.kind == "knn_gaussian" and metric != "euclidean":
            raise GraphValidationError("knn_gaussian weights require metric='euclidean'")

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

    # Features added after the original public constructor. Keep them
    # keyword-only so all historical positional calls remain valid.
    include_self_in_knn: bool = field(default=False, kw_only=True)
    edge_weight_dtype: EdgeWeightDType = field(default="float32", kw_only=True)
    diagonal_policy: DiagonalPolicy = field(default="preserve", kw_only=True)
    precomputed_path: str | None = field(default=None, kw_only=True)
    precomputed_sha256: str | None = field(default=None, kw_only=True)
    annoy_n_trees: int = field(default=10, kw_only=True)
    annoy_search_k: int = field(default=-1, kw_only=True)
    annoy_query_k: int | None = field(default=None, kw_only=True)
    annoy_rerank: bool = field(default=False, kw_only=True)

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

        if self.symmetrize not in ("none", "or", "mutual", "mean", "sum"):
            raise GraphValidationError(f"Unknown symmetrize mode: {self.symmetrize!r}")
        if self.diagonal_policy not in ("preserve", "zero"):
            raise GraphValidationError(f"Unknown diagonal_policy: {self.diagonal_policy!r}")
        if self.self_loops and self.diagonal_policy == "zero":
            raise GraphValidationError(
                "self_loops=True conflicts with diagonal_policy='zero'; "
                "set self_loops=False when zeroing the graph diagonal"
            )
        if self.normalize not in ("none", "rw", "sym"):
            raise GraphValidationError(f"Unknown normalize mode: {self.normalize!r}")
        if self.edge_weight_dtype not in ("float32", "float64"):
            raise GraphValidationError(f"Unknown edge_weight_dtype: {self.edge_weight_dtype!r}")

        if self.backend not in (
            "auto",
            "numpy",
            "sklearn",
            "faiss",
            "annoy",
            "torch",
            "precomputed",
        ):
            raise GraphValidationError(f"Unknown backend: {self.backend!r}")

        if int(self.chunk_size) <= 0:
            raise GraphValidationError("chunk_size must be > 0")

        # backend-specific constraints
        if self.backend == "faiss" and self.scheme == "epsilon":
            raise GraphValidationError("faiss backend does not support epsilon scheme")
        if self.backend == "annoy":
            if self.scheme != "knn":
                raise GraphValidationError("annoy backend supports only knn scheme")
            if self.metric != "euclidean":
                raise GraphValidationError("annoy backend currently requires metric='euclidean'")
        if self.backend == "torch" and self.scheme != "knn":
            raise GraphValidationError("torch backend currently supports only knn scheme")
        if self.backend == "precomputed":
            if self.scheme != "knn":
                raise GraphValidationError("precomputed backend supports only knn scheme")
            if not self.precomputed_path:
                raise GraphValidationError("precomputed_path is required for precomputed backend")
            digest = self.precomputed_sha256 or ""
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise GraphValidationError("precomputed_sha256 must be a lowercase SHA-256 digest")

        if int(self.faiss_hnsw_m) <= 0:
            raise GraphValidationError("faiss_hnsw_m must be > 0")
        if int(self.faiss_ef_search) <= 0:
            raise GraphValidationError("faiss_ef_search must be > 0")
        if int(self.faiss_ef_construction) <= 0:
            raise GraphValidationError("faiss_ef_construction must be > 0")
        if int(self.annoy_n_trees) <= 0:
            raise GraphValidationError("annoy_n_trees must be > 0")
        if int(self.annoy_search_k) != -1 and int(self.annoy_search_k) <= 0:
            raise GraphValidationError("annoy_search_k must be -1 or > 0")
        if self.annoy_query_k is not None:
            minimum_query_k = int(self.k or 0) + (0 if self.include_self_in_knn else 1)
            if int(self.annoy_query_k) < minimum_query_k:
                raise GraphValidationError(
                    "annoy_query_k must retrieve at least k candidates plus self when self "
                    "is excluded"
                )

        self.weights.validate(metric=self.metric)
        if self.weights.kind == "knn_gaussian" and self.scheme != "knn":
            raise GraphValidationError("knn_gaussian weights require scheme='knn'")

    def to_dict(self) -> dict[str, Any]:
        result = {
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
        # Keep historical fingerprints stable for every existing graph card.
        # The new fields become part of the fingerprint only when activated.
        if self.include_self_in_knn:
            result["include_self_in_knn"] = True
        if self.edge_weight_dtype != "float32":
            result["edge_weight_dtype"] = self.edge_weight_dtype
        if self.diagonal_policy != "preserve":
            result["diagonal_policy"] = self.diagonal_policy
        if self.precomputed_path is not None:
            result["precomputed_path"] = self.precomputed_path
        if self.precomputed_sha256 is not None:
            result["precomputed_sha256"] = self.precomputed_sha256
        if self.backend == "annoy":
            result["annoy_n_trees"] = int(self.annoy_n_trees)
            result["annoy_search_k"] = int(self.annoy_search_k)
            result["annoy_query_k"] = (
                None if self.annoy_query_k is None else int(self.annoy_query_k)
            )
            result["annoy_rerank"] = bool(self.annoy_rerank)
        return result

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return the semantic graph spec independent of artifact location."""

        result = self.to_dict()
        if self.precomputed_sha256 is not None:
            result.pop("precomputed_path", None)
        return result

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
            include_self_in_knn=bool(d.get("include_self_in_knn", False)),
            edge_weight_dtype=str(d.get("edge_weight_dtype", "float32")),  # type: ignore[arg-type]
            diagonal_policy=str(d.get("diagonal_policy", "preserve")),  # type: ignore[arg-type]
            backend=str(d.get("backend", "auto")),  # type: ignore[arg-type]
            chunk_size=int(d.get("chunk_size", 512)),
            precomputed_path=(
                None if d.get("precomputed_path") is None else str(d["precomputed_path"])
            ),
            precomputed_sha256=(
                None if d.get("precomputed_sha256") is None else str(d["precomputed_sha256"])
            ),
            feature_field=str(d.get("feature_field", "features.X")),
            n_anchors=d.get("n_anchors"),
            anchors_k=int(d.get("anchors_k", 5)),
            anchors_method=str(d.get("anchors_method", "random")),  # type: ignore[arg-type]
            candidate_limit=int(d.get("candidate_limit", 1000)),
            faiss_exact=bool(d.get("faiss_exact", False)),
            faiss_hnsw_m=int(d.get("faiss_hnsw_m", 32)),
            faiss_ef_search=int(d.get("faiss_ef_search", 64)),
            faiss_ef_construction=int(d.get("faiss_ef_construction", 200)),
            annoy_n_trees=int(d.get("annoy_n_trees", 10)),
            annoy_search_k=int(d.get("annoy_search_k", -1)),
            annoy_query_k=(None if d.get("annoy_query_k") is None else int(d["annoy_query_k"])),
            annoy_rerank=bool(d.get("annoy_rerank", False)),
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
