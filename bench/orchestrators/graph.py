"""YAML adapter for native graph materialization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modssc.graph import build_graph_from_preprocess, summarize_graph
from modssc.graph.artifacts import GraphArtifact
from modssc.graph.specs import GraphBuilderSpec
from modssc.preprocess.types import PreprocessResult

from ..utils.resources import resolve_graph_spec_resources


def build(
    pre: PreprocessResult,
    *,
    spec_dict: Mapping[str, Any],
    seed: int,
    dataset_fingerprint: str | None,
    cache: bool,
    require_cache_hit: bool,
    cache_dir: str | None,
    include_test: bool,
    expected_fingerprint: str | None = None,
    expected_preprocess_fingerprint: str | None = None,
    resource_root: Path | None = None,
) -> GraphArtifact:
    runtime_spec = (
        resolve_graph_spec_resources(spec_dict, resource_root=resource_root)
        if resource_root is not None
        else dict(spec_dict)
    )
    return build_graph_from_preprocess(
        pre,
        spec=GraphBuilderSpec.from_dict(runtime_spec),
        seed=seed,
        dataset_fingerprint=dataset_fingerprint,
        cache=cache,
        require_cache_hit=require_cache_hit,
        cache_dir=Path(cache_dir).expanduser().resolve() if cache_dir else None,
        include_test=include_test,
        expected_fingerprint=expected_fingerprint,
        expected_preprocess_fingerprint=expected_preprocess_fingerprint,
    )


__all__ = ["build", "summarize_graph"]
