from __future__ import annotations

import os
from pathlib import Path

_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}


def env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def resolve_model_cache_root() -> Path | None:
    value = os.environ.get("MODSSC_MODEL_CACHE_ROOT")
    if value:
        return _resolve_path(value)

    cache_root = os.environ.get("MODSSC_CACHE_ROOT")
    if cache_root:
        return _resolve_path(cache_root) / "models"
    return None


def resolve_hf_home() -> Path | None:
    value = os.environ.get("HF_HOME")
    if value:
        return _resolve_path(value)

    root = resolve_model_cache_root()
    if root is not None:
        return root / "hf"
    return None


def resolve_sentence_transformers_cache() -> str | None:
    value = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    if value:
        return str(_resolve_path(value))

    hf_home = resolve_hf_home()
    if hf_home is not None:
        return str(hf_home / "sentence_transformers")

    value = os.environ.get("TRANSFORMERS_CACHE")
    if value:
        return str(_resolve_path(value))
    return None


def resolve_hf_local_files_only() -> bool:
    return any(
        env_flag(name)
        for name in ("MODSSC_HF_LOCAL_FILES_ONLY", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def resolve_openclip_cache_dir() -> str | None:
    for env_name in ("MODSSC_OPENCLIP_CACHE_DIR", "OPENCLIP_CACHE_DIR"):
        value = os.environ.get(env_name)
        if value:
            return str(_resolve_path(value))

    root = resolve_model_cache_root()
    if root is not None:
        return str(root / "open_clip")
    return None


def resolve_torch_home() -> Path | None:
    for env_name in ("TORCH_HOME", "MODSSC_TORCH_HOME"):
        value = os.environ.get(env_name)
        if value:
            return _resolve_path(value)

    root = resolve_model_cache_root()
    if root is not None:
        return root / "torch"
    return None


def ensure_torch_home() -> Path | None:
    torch_home = resolve_torch_home()
    if torch_home is not None:
        os.environ.setdefault("TORCH_HOME", str(torch_home))
    return torch_home


def torchaudio_download_kwargs() -> dict[str, str | bool]:
    kwargs: dict[str, str | bool] = {"progress": False}
    torch_home = ensure_torch_home()
    if torch_home is not None:
        kwargs["model_dir"] = str(torch_home / "hub" / "checkpoints")
    return kwargs
