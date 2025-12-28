from __future__ import annotations

import importlib
from typing import Any

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.types import ModelSpec


def _load_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(f"Invalid import path: {import_path!r}")
    module_name, qualname = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


BUILTIN_MODELS: tuple[ModelSpec, ...] = (
    # Stubs (no external downloads, used for tests and offline runs)
    ModelSpec(
        model_id="stub:text",
        import_path="modssc.preprocess.models_backends.stub:StubEncoder",
        modality="text",
        description="Deterministic stub text encoder (offline).",
        required_extra=None,
        default_kwargs={"dim": 8},
    ),
    ModelSpec(
        model_id="stub:vision",
        import_path="modssc.preprocess.models_backends.stub:StubEncoder",
        modality="vision",
        description="Deterministic stub vision encoder (offline).",
        required_extra=None,
        default_kwargs={"dim": 8},
    ),
    ModelSpec(
        model_id="stub:audio",
        import_path="modssc.preprocess.models_backends.stub:StubEncoder",
        modality="audio",
        description="Deterministic stub audio encoder (offline).",
        required_extra=None,
        default_kwargs={"dim": 8},
    ),
    # Real models (optional dependencies)
    ModelSpec(
        model_id="st:all-MiniLM-L6-v2",
        import_path="modssc.preprocess.models_backends.sentence_transformers:SentenceTransformerEncoder",
        modality="text",
        description="SentenceTransformer all-MiniLM-L6-v2 (dense text embeddings).",
        required_extra="preprocess-text",
        homepage="https://www.sbert.net/",
        default_kwargs={"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
    ),
    ModelSpec(
        model_id="openclip:ViT-B-32/openai",
        import_path="modssc.preprocess.models_backends.open_clip:OpenClipEncoder",
        modality="vision",
        description="OpenCLIP ViT-B-32 pretrained on OpenAI weights.",
        required_extra="preprocess-vision",
        default_kwargs={"model_name": "ViT-B-32", "pretrained": "openai"},
    ),
    ModelSpec(
        model_id="wav2vec2:base",
        import_path="modssc.preprocess.models_backends.torchaudio_wav2vec2:Wav2Vec2Encoder",
        modality="audio",
        description="torchaudio wav2vec2 base pipeline (mean pooled).",
        required_extra="preprocess-audio",
        default_kwargs={"bundle": "WAV2VEC2_BASE"},
    ),
)

_MODELS: dict[str, ModelSpec] = {m.model_id: m for m in BUILTIN_MODELS}


def available_models(*, modality: str | None = None) -> list[str]:
    if modality is None:
        return sorted(_MODELS.keys())
    return sorted([k for k, s in _MODELS.items() if s.modality == modality])


def model_spec(model_id: str) -> ModelSpec:
    try:
        return _MODELS[model_id]
    except KeyError as e:
        raise PreprocessValidationError(f"Unknown model id: {model_id!r}") from e


def model_info(model_id: str) -> dict[str, Any]:
    s = model_spec(model_id)
    return {
        "id": s.model_id,
        "modality": s.modality,
        "import_path": s.import_path,
        "required_extra": s.required_extra,
        "description": s.description,
        "homepage": s.homepage,
        "license": s.license,
        "citation": s.citation,
        "default_kwargs": dict(s.default_kwargs),
    }


def load_encoder(model_id: str, **kwargs: Any) -> Any:
    spec = model_spec(model_id)
    cls_obj = _load_object(spec.import_path)
    merged = dict(spec.default_kwargs)
    merged.update(kwargs)
    return cls_obj(**merged)
