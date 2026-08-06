from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from modssc.preprocess.models import model_info

_TORCH_CHECKPOINT_NAMES = {
    "torchvision:resnet18": "resnet18-f37072fd.pth",
    "wav2vec2:base": "wav2vec2_fairseq_base_ls960.pth",
}


class ModelArtifactError(RuntimeError):
    """Raised when an offline model artifact cannot be locked or verified."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_artifact_lock_sha256(lock: Mapping[str, Any]) -> str:
    """Return the canonical digest of a model artifact lock."""

    return _canonical_sha256(lock)


def _model_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.startswith("model_id") and isinstance(item, str):
                found.add(item)
            found.update(_model_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_model_ids(item))
    return found


def discover_model_ids(config_roots: Sequence[Path]) -> list[str]:
    """Discover model identifiers referenced by YAML configurations."""

    model_ids: set[str] = set()
    for root in sorted({path.resolve() for path in config_roots}, key=str):
        paths = [root] if root.is_file() else sorted(root.rglob("*.yaml"))
        for path in paths:
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                raise ModelArtifactError(f"cannot inspect model ids in {path}") from exc
            model_ids.update(_model_ids(raw))
    return sorted(model_ids)


def _cache_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    value = os.environ.get("MODSSC_MODEL_CACHE_ROOT")
    if not value:
        raise ModelArtifactError(
            "external models require --model-cache-root or MODSSC_MODEL_CACHE_ROOT"
        )
    return Path(value).expanduser().resolve()


def _logical_files(paths: Iterable[Path], *, logical_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        if not path.is_file():
            raise ModelArtifactError(f"model artifact is missing: {path}")
        try:
            logical_path = path.relative_to(logical_root).as_posix()
        except ValueError as exc:
            raise ModelArtifactError(f"model artifact escapes its cache root: {path}") from exc
        if logical_path in seen:
            continue
        seen.add(logical_path)
        records.append(
            {
                "path": logical_path,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ModelArtifactError(f"no model files found below {logical_root}")
    return records


def _ordered_roots(
    cache_root: Path, *, environment_name: str | None, relative_roots: Sequence[str]
) -> list[Path]:
    candidates: list[Path] = []
    if environment_name and os.environ.get(environment_name):
        candidates.append(Path(os.environ[environment_name]).expanduser().resolve())
    candidates.extend(cache_root / relative for relative in relative_roots)
    candidates.append(cache_root)
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _hf_snapshot(
    model_name: str,
    cache_root: Path,
    *,
    sentence_transformer: bool = False,
) -> tuple[Path, str | None]:
    repo_dir_name = "models--" + model_name.replace("/", "--")
    roots = _ordered_roots(
        cache_root,
        environment_name="SENTENCE_TRANSFORMERS_HOME" if sentence_transformer else None,
        relative_roots=("hf/sentence_transformers", "hf/hub")
        if sentence_transformer
        else ("hf/hub",),
    )
    for root in roots:
        candidates = sorted(
            {
                path
                for path in root.rglob(repo_dir_name)
                if path.is_dir() and path.name == repo_dir_name
            },
            key=str,
        )
        valid: list[tuple[Path, str | None]] = []
        for repo_dir in candidates:
            snapshots = repo_dir / "snapshots"
            revision: str | None = None
            ref = repo_dir / "refs" / "main"
            if ref.is_file():
                revision = ref.read_text(encoding="utf-8").strip()
                selected = snapshots / revision
                if selected.is_dir():
                    valid.append((selected, revision))
                    continue
            snapshot_dirs = (
                sorted(path for path in snapshots.iterdir() if path.is_dir())
                if snapshots.is_dir()
                else []
            )
            if len(snapshot_dirs) == 1:
                valid.append((snapshot_dirs[0], snapshot_dirs[0].name))
        if len(valid) == 1:
            return valid[0]
        if len(valid) > 1:
            raise ModelArtifactError(
                f"multiple active offline snapshots for {model_name!r} below {root}"
            )

        # Older sentence-transformers caches use a directly materialized model
        # directory instead of the Hugging Face snapshots layout.
        leaf = model_name.rsplit("/", 1)[-1]
        direct = sorted(
            {
                path
                for path in root.rglob(leaf)
                if path.is_dir() and path.name == leaf and path.parent.name != "snapshots"
            },
            key=str,
        )
        if len(direct) == 1:
            return direct[0], None
        if len(direct) > 1:
            raise ModelArtifactError(
                f"multiple direct offline model directories for {model_name!r} below {root}"
            )
    raise ModelArtifactError(f"no active offline snapshot found for {model_name!r}")


def _files_below(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and ".locks" not in path.parts and not path.name.endswith(".lock")
    ]


def _torch_checkpoint_name(model_id: str, info: Mapping[str, Any]) -> str:
    fallback = _TORCH_CHECKPOINT_NAMES.get(model_id)
    try:
        if model_id.startswith("torchvision:"):
            import torchvision

            name = str(info["default_kwargs"]["model_name"])
            weights = torchvision.models.get_model_weights(name).DEFAULT
            resolved = Path(urlparse(str(weights.url)).path).name
            if resolved:
                return resolved
        if model_id.startswith("wav2vec2:"):
            import torchaudio

            bundle_name = str(info["default_kwargs"]["bundle"])
            bundle = getattr(torchaudio.pipelines, bundle_name)
            source = getattr(bundle, "_path", None)
            resolved = Path(urlparse(str(source)).path).name if source else ""
            if resolved:
                return resolved
    except Exception:
        # Importing optional multimedia packages can fail on a login node.  The
        # built-in fallback is still exact for ModSSC's currently registered
        # model ids and is verified by the content digest below.
        pass
    if fallback is None:
        raise ModelArtifactError(f"cannot determine the checkpoint name for {model_id!r}")
    return fallback


def _single_checkpoint(model_id: str, info: Mapping[str, Any], cache_root: Path) -> Path:
    expected = _torch_checkpoint_name(model_id, info)
    roots = _ordered_roots(
        cache_root,
        environment_name="TORCH_HOME",
        relative_roots=("torch",),
    )
    for root in roots:
        exact = sorted(
            path for path in root.rglob(expected) if path.is_file() and ".locks" not in path.parts
        )
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ModelArtifactError(
                f"multiple cached checkpoints {expected!r} for {model_id!r} below {root}"
            )
    raise ModelArtifactError(f"cached checkpoint {expected!r} for {model_id!r} is missing")


def _openclip_artifact(
    info: Mapping[str, Any], cache_root: Path
) -> tuple[list[Path], Path, str | None]:
    kwargs = info["default_kwargs"]
    model_name = str(kwargs["model_name"])
    pretrained = str(kwargs["pretrained"])
    expected: str | None = None
    revision: str | None = None
    try:
        import open_clip

        cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
        url = cfg.get("url") if isinstance(cfg, Mapping) else None
        if isinstance(url, str) and url:
            expected = Path(urlparse(url).path).name
        hf_hub = cfg.get("hf_hub") if isinstance(cfg, Mapping) else None
        if isinstance(hf_hub, str) and hf_hub:
            snapshot, revision = _hf_snapshot(hf_hub, cache_root)
            files = _files_below(snapshot)
            if files:
                return files, snapshot, revision
    except Exception:
        pass
    roots = _ordered_roots(
        cache_root,
        environment_name="MODSSC_OPENCLIP_CACHE_DIR",
        relative_roots=("open_clip",),
    )
    candidates = sorted(
        {
            path
            for root in roots
            if root.is_dir()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".bin", ".pt", ".pth", ".safetensors"}
            and (
                path.name == expected
                if expected is not None
                else re.search(r"vit[-_]?b[-_]?32", path.name, flags=re.IGNORECASE)
            )
        },
        key=str,
    )
    if len(candidates) != 1:
        raise ModelArtifactError(
            f"expected exactly one OpenCLIP checkpoint for {model_name}/{pretrained}, "
            f"found {len(candidates)}"
        )
    return [candidates[0]], candidates[0].parent, revision


def _lock_model(model_id: str, cache_root: Path | None) -> dict[str, Any]:
    info = model_info(model_id)
    provider = model_id.split(":", 1)[0]
    common: dict[str, Any] = {
        "model_id": model_id,
        "provider": provider,
        "implementation": info["import_path"],
        "default_kwargs": info["default_kwargs"],
    }
    if provider == "stub":
        return {**common, "artifact_free": True, "revision": None, "files": []}
    if cache_root is None:
        raise ModelArtifactError(f"external model {model_id!r} has no model cache root")
    if provider == "st":
        model_name = str(info["default_kwargs"]["model_name"])
        snapshot, revision = _hf_snapshot(
            model_name,
            cache_root,
            sentence_transformer=True,
        )
        return {
            **common,
            "artifact_free": False,
            "revision": revision,
            "files": _logical_files(_files_below(snapshot), logical_root=snapshot),
        }
    if provider in {"torchvision", "wav2vec2"}:
        checkpoint = _single_checkpoint(model_id, info, cache_root)
        return {
            **common,
            "artifact_free": False,
            "revision": None,
            "files": _logical_files([checkpoint], logical_root=checkpoint.parent),
        }
    if provider == "openclip":
        files, logical_root, revision = _openclip_artifact(info, cache_root)
        return {
            **common,
            "artifact_free": False,
            "revision": revision,
            "files": _logical_files(files, logical_root=logical_root),
        }
    raise ModelArtifactError(f"unsupported external model provider for {model_id!r}")


def build_model_artifact_lock(
    model_ids: Iterable[str], *, model_cache_root: Path | None = None
) -> dict[str, Any]:
    """Hash every file needed by the requested offline model identifiers."""

    ordered = sorted(set(model_ids))
    external = [model_id for model_id in ordered if not model_id.startswith("stub:")]
    cache_root = _cache_root(model_cache_root) if external else None
    models = [_lock_model(model_id, cache_root) for model_id in ordered]
    return {"schema_version": 1, "models": models}


def _model_map(lock: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if lock.get("schema_version") != 1 or not isinstance(lock.get("models"), list):
        raise ModelArtifactError("invalid model artifact lock schema")
    mapped: dict[str, Mapping[str, Any]] = {}
    for entry in lock["models"]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("model_id"), str):
            raise ModelArtifactError("invalid model entry in artifact lock")
        model_id = str(entry["model_id"])
        if model_id in mapped:
            raise ModelArtifactError(f"duplicate model artifact lock entry: {model_id}")
        mapped[model_id] = entry
    return mapped


def verify_model_artifact_lock(
    lock: Mapping[str, Any],
    required_model_ids: Iterable[str],
    *,
    model_cache_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Re-hash required models and return cheap post-preflight attestations."""

    locked = _model_map(lock)
    required = sorted(set(required_model_ids))
    missing = sorted(set(required) - locked.keys())
    if missing:
        raise ModelArtifactError(f"model artifact lock is missing: {missing}")
    external = [model_id for model_id in required if not model_id.startswith("stub:")]
    cache_root = _cache_root(model_cache_root) if external else None
    attestations: list[dict[str, Any]] = []
    for model_id in required:
        actual = _lock_model(model_id, cache_root)
        if actual != locked[model_id]:
            raise ModelArtifactError(f"cached artifacts differ for {model_id!r}")
        if actual["artifact_free"]:
            continue
        # Locate the same files once more without loading the model.  These
        # absolute paths are deliberately kept out of the immutable lock, but
        # let each task cheaply detect mutations after preflight using stat(2).
        info = model_info(model_id)
        provider = str(actual["provider"])
        if provider == "st":
            snapshot, _revision = _hf_snapshot(
                str(info["default_kwargs"]["model_name"]),
                cache_root,
                sentence_transformer=True,
            )
            paths = _files_below(snapshot)
            logical_root = snapshot
        elif provider in {"torchvision", "wav2vec2"}:
            paths = [_single_checkpoint(model_id, info, cache_root)]
            logical_root = paths[0].parent
        else:
            paths, logical_root, _revision = _openclip_artifact(info, cache_root)
        expected_files = {record["path"]: record for record in actual["files"]}
        for path in sorted(paths, key=str):
            logical_path = path.relative_to(logical_root).as_posix()
            record = expected_files.get(logical_path)
            if record is None:
                raise ModelArtifactError(f"unlocked model file encountered: {path}")
            stat = path.stat()
            attestations.append(
                {
                    "model_id": model_id,
                    "path": str(path.absolute()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "sha256": record["sha256"],
                }
            )
    return attestations


def verify_model_artifact_attestations(attestations: Any) -> None:
    """Cheaply reject model files changed since the successful preflight."""

    if not isinstance(attestations, list):
        raise ModelArtifactError("preflight model artifact attestations are missing")
    for record in attestations:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ModelArtifactError("invalid model artifact attestation")
        path = Path(str(record["path"]))
        try:
            stat = path.stat()
        except OSError as exc:
            raise ModelArtifactError(f"preflight model artifact is missing: {path}") from exc
        observed = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }
        if any(record.get(key) != value for key, value in observed.items()):
            raise ModelArtifactError(f"model artifact changed after preflight: {path}")


__all__ = [
    "ModelArtifactError",
    "build_model_artifact_lock",
    "discover_model_ids",
    "model_artifact_lock_sha256",
    "verify_model_artifact_attestations",
    "verify_model_artifact_lock",
]
