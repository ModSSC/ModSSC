from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from modssc.preprocess.store import ArtifactStore

StepKind = Literal["transform", "fittable", "featurizer"]


@dataclass(frozen=True)
class StepSpec:
    """Declarative step specification for lazy loading and documentation."""

    step_id: str
    import_path: str
    kind: StepKind
    description: str = ""
    required_extra: str | None = None
    modalities: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedStep:
    step_id: str
    params: Mapping[str, Any]
    index: int
    spec: StepSpec


@dataclass(frozen=True)
class SkippedStep:
    step_id: str
    reason: str
    index: int


@dataclass(frozen=True)
class ResolvedPlan:
    """Plan after applying conditional logic for a specific dataset."""

    steps: tuple[ResolvedStep, ...]
    skipped: tuple[SkippedStep, ...] = ()
    fingerprint: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """Declarative model specification for pretrained encoders used in featurizer steps."""

    model_id: str
    import_path: str
    modality: str
    description: str = ""
    required_extra: str | None = None
    homepage: str | None = None
    license: str | None = None
    citation: str | None = None
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessResult:
    """Return type of the main preprocess() API."""

    dataset: Any
    plan: ResolvedPlan
    preprocess_fingerprint: str
    train_artifacts: ArtifactStore
    test_artifacts: ArtifactStore | None = None
    cache_dir: str | None = None
