from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from platformdirs import user_cache_dir

from modssc.preprocess.errors import OptionalDependencyError, PreprocessCacheError
from modssc.preprocess.fingerprint import stable_json_dumps
from modssc.runtime.device import mps_is_available
from modssc.runtime.paths import default_local_cache_subdir
from modssc.utils.io import atomic_write_text, resolve_relative_path

OBJECT_JSON_MAX_ITEMS = int(
    os.environ.get("MODSSC_PREPROCESS_CACHE_OBJECT_JSON_MAX_ITEMS", "10000")
)
MMAP_THRESHOLD_BYTES = int(
    os.environ.get("MODSSC_PREPROCESS_CACHE_MMAP_THRESHOLD", str(64 * 1024 * 1024))
)

CACHE_ENV = "MODSSC_PREPROCESS_CACHE_DIR"
CACHE_ROOT_ENV = "MODSSC_CACHE_ROOT"
logger = logging.getLogger(__name__)

_CACHE_SCHEMA_VERSION = 2
_CACHE_KIND = "modssc.preprocess.step"
_MANIFEST_FILENAME = "manifest.json"
_GENERATION_MANIFEST_FILENAME = "generation.json"
_GENERATIONS_DIRNAME = "generations"
_LOCK_FILENAME = ".cache.lock"


def default_cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()

    root_override = os.environ.get(CACHE_ROOT_ENV)
    if root_override:
        return Path(root_override).expanduser().resolve() / "preprocess"

    local = default_local_cache_subdir("preprocess")
    if local is not None:
        return local

    return Path(user_cache_dir("modssc")) / "preprocess"


_WINDOWS_INVALID_CHARS = set('<>:"/\\|?*')


def _safe_path_component(value: str) -> str:
    if os.name != "nt":
        return value
    cleaned = []
    for ch in value:
        if ord(ch) < 32 or ch in _WINDOWS_INVALID_CHARS:
            cleaned.append("_")
        else:
            cleaned.append(ch)
    safe = "".join(cleaned).replace("..", ".").rstrip(" .")
    if safe in {"", ".", ".."}:
        return "_"
    return safe


def _safe_name(key: str) -> str:
    name = key.replace("/", "_").replace("..", ".").replace(".", "__")
    return _safe_path_component(name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _signed_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {str(key): value for key, value in payload.items() if key != "manifest_sha256"}
    digest = hashlib.sha256(stable_json_dumps(unsigned).encode("utf-8")).hexdigest()
    return {**unsigned, "manifest_sha256": digest}


def _verify_signed_manifest(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
        raise PreprocessCacheError(f"Unsupported {label} schema version")
    if payload.get("cache_kind") != _CACHE_KIND:
        raise PreprocessCacheError(f"Invalid {label} cache kind")
    expected = payload.get("manifest_sha256")
    if not _is_sha256(expected):
        raise PreprocessCacheError(f"Invalid {label} manifest digest")
    unsigned = {str(key): value for key, value in payload.items() if key != "manifest_sha256"}
    observed = hashlib.sha256(stable_json_dumps(unsigned).encode("utf-8")).hexdigest()
    if observed != expected:
        raise PreprocessCacheError(f"{label.capitalize()} manifest digest differs")
    return dict(payload)


def _fsync_directory(path: Path) -> None:
    """Durably publish directory entries where the host filesystem supports it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    with contextlib.suppress(OSError):
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


@contextmanager
def _exclusive_cache_lock(path: Path) -> Iterator[None]:
    """Serialize publishers for one cache key across local processes.

    ``flock`` is advisory, which is sufficient because every native publisher
    participates. Immutable generations and an atomic pointer keep readers safe
    without making them wait for a long-running preprocessing step.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _require_torch():
    import importlib

    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as e:
        raise OptionalDependencyError(
            extra="inductive-torch", purpose="preprocess cache torch IO"
        ) from e


def _is_torch_tensor(obj: Any) -> bool:
    mod = getattr(obj.__class__, "__module__", "")
    if not mod.startswith("torch"):
        return False
    return hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "device")


def _is_scipy_sparse(obj: Any) -> bool:
    mod = getattr(obj.__class__, "__module__", "")
    return mod.startswith("scipy.sparse")


def _save_json(path: Path, payload: Any, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    import json

    path_json = path.with_suffix(".json")
    atomic_write_text(path_json, json.dumps(payload, ensure_ascii=False))
    desc: dict[str, Any] = {"type": "json", "path": path_json.name}
    if extra:
        desc.update(extra)
    return desc


def _save_value(path: Path, value: Any) -> dict[str, Any] | None:
    """Save a value to disk and return a small descriptor for the manifest."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if int(value.size) > OBJECT_JSON_MAX_ITEMS:
                np.save(path.with_suffix(".npy"), value, allow_pickle=True)
                return {
                    "type": "npy",
                    "path": path.with_suffix(".npy").name,
                    "allow_pickle": True,
                }
            try:
                payload = value.tolist()
            except TypeError:
                return None
            return _save_json(path, payload, extra={"format": "ndarray", "dtype": "object"})
        np.save(path.with_suffix(".npy"), value, allow_pickle=False)
        return {"type": "npy", "path": path.with_suffix(".npy").name}

    if _is_torch_tensor(value):
        arr = value
        if hasattr(arr, "detach"):
            arr = arr.detach()
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        try:
            arr_np = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
            np.save(path.with_suffix(".npy"), arr_np, allow_pickle=False)
            return {
                "type": "torch_npy",
                "path": path.with_suffix(".npy").name,
                "dtype": str(getattr(value, "dtype", "")),
                "device": str(getattr(value, "device", "")),
            }
        except Exception:
            torch = _require_torch()
            torch.save(value, path.with_suffix(".pt"))
            return {
                "type": "torch_pt",
                "path": path.with_suffix(".pt").name,
                "dtype": str(getattr(value, "dtype", "")),
                "device": str(getattr(value, "device", "")),
            }

    # Support simple JSON-serializable payloads (lists of strings, small dicts, etc.).
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _save_json(path, value)
    if isinstance(value, (list, tuple, dict)):
        try:
            payload = value
            if isinstance(value, tuple):
                payload = list(value)
            return _save_json(path, payload)
        except TypeError:
            return None

    if _is_scipy_sparse(value):
        # Lazy import to keep base install light.
        try:
            from scipy import sparse  # type: ignore
        except ModuleNotFoundError as e:
            raise OptionalDependencyError(
                extra="preprocess-sklearn", purpose="scipy sparse IO"
            ) from e
        sparse.save_npz(path.with_suffix(".npz"), value)
        return {"type": "npz", "path": path.with_suffix(".npz").name}

    return None


def _load_value(path: Path, desc: dict[str, Any]) -> Any:
    t = desc.get("type")
    rel = desc.get("path")
    if not isinstance(rel, str):
        raise PreprocessCacheError("Invalid cache manifest entry: missing 'path'")
    try:
        fp = resolve_relative_path(path, rel, purpose="preprocess cache value path")
    except ValueError as e:
        raise PreprocessCacheError(str(e)) from e
    if t == "npy":
        allow_pickle = bool(desc.get("allow_pickle", False))
        mmap_mode = None
        if not allow_pickle:
            with contextlib.suppress(OSError):
                if fp.stat().st_size >= MMAP_THRESHOLD_BYTES:
                    mmap_mode = "r"
        try:
            return np.load(fp, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        except ValueError as e:
            raise PreprocessCacheError(f"Failed to load cached array: {fp}") from e
    if t == "torch_npy":
        torch = _require_torch()

        mmap_mode = None
        with contextlib.suppress(OSError):
            if fp.stat().st_size >= MMAP_THRESHOLD_BYTES:
                mmap_mode = "r"

        arr = np.load(fp, allow_pickle=False, mmap_mode=mmap_mode)
        dtype_str = str(desc.get("dtype") or "")
        dtype_name = dtype_str.split(".", 1)[-1] if dtype_str else ""
        dtype = getattr(torch, dtype_name, None) if dtype_name else None
        device_str = str(desc.get("device") or "")
        device = _resolve_cache_device(torch, device_str)
        return torch.as_tensor(arr, device=device, dtype=dtype)
    if t == "torch_pt":
        torch = _require_torch()
        try:
            obj = torch.load(fp, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(fp, map_location="cpu")
        device_str = str(desc.get("device") or "")
        if device_str:
            return obj.to(_resolve_cache_device(torch, device_str))
        return obj
    if t == "json":
        import json

        payload = json.loads(fp.read_text(encoding="utf-8"))
        if desc.get("format") == "ndarray":
            return np.asarray(payload, dtype=object)
        return payload
    if t == "npz":
        try:
            from scipy import sparse  # type: ignore
        except ModuleNotFoundError as e:
            raise OptionalDependencyError(
                extra="preprocess-sklearn", purpose="scipy sparse IO"
            ) from e
        return sparse.load_npz(fp)
    raise PreprocessCacheError(f"Unsupported cached value type: {t!r}")


def _value_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dimension) for dimension in shape]
    except (TypeError, ValueError):
        return None


def _value_dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def _authenticate_descriptor(
    directory: Path,
    descriptor: Mapping[str, Any],
    *,
    value: Any,
) -> dict[str, Any]:
    relative = descriptor.get("path")
    if not isinstance(relative, str):
        raise PreprocessCacheError("Invalid cache manifest entry: missing 'path'")
    try:
        file_path = resolve_relative_path(
            directory, relative, purpose="preprocess cache value path"
        )
    except ValueError as exc:
        raise PreprocessCacheError(str(exc)) from exc
    if not file_path.is_file() or file_path.is_symlink():
        raise PreprocessCacheError(f"Cached value is missing or is not a regular file: {file_path}")
    _fsync_file(file_path)
    result = dict(descriptor)
    result["sha256"] = _sha256_file(file_path)
    result["size_bytes"] = int(file_path.stat().st_size)
    result["dtype"] = _value_dtype(value)
    result["shape"] = _value_shape(value)
    return result


def _verified_descriptor_path(directory: Path, descriptor: Mapping[str, Any]) -> Path:
    relative = descriptor.get("path")
    if not isinstance(relative, str):
        raise PreprocessCacheError("Invalid cache manifest entry: missing 'path'")
    try:
        file_path = resolve_relative_path(
            directory, relative, purpose="preprocess cache value path"
        )
    except ValueError as exc:
        raise PreprocessCacheError(str(exc)) from exc
    if not file_path.is_file() or file_path.is_symlink():
        raise PreprocessCacheError(f"Cached value is missing or is not a regular file: {file_path}")

    expected_sha256 = descriptor.get("sha256")
    size_bytes = descriptor.get("size_bytes")
    if not _is_sha256(expected_sha256):
        raise PreprocessCacheError("Invalid cached value sha256")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise PreprocessCacheError("Invalid cached value size_bytes")

    before = file_path.stat()
    if int(before.st_size) != size_bytes:
        raise PreprocessCacheError(f"Cached value size differs: {file_path}")
    if _sha256_file(file_path) != expected_sha256:
        raise PreprocessCacheError(f"Cached value sha256 differs: {file_path}")
    after = file_path.stat()
    before_state = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_state = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_state != after_state:
        raise PreprocessCacheError(f"Cached value changed while hashing: {file_path}")
    return file_path


def _verify_loaded_value(value: Any, descriptor: Mapping[str, Any], *, key: str) -> None:
    expected_dtype = descriptor.get("dtype")
    expected_shape = descriptor.get("shape")
    if expected_dtype is not None and _value_dtype(value) != expected_dtype:
        raise PreprocessCacheError(f"Cached value dtype differs for {key!r}")
    if expected_shape is not None:
        if not isinstance(expected_shape, list) or not all(
            isinstance(dimension, int) and not isinstance(dimension, bool)
            for dimension in expected_shape
        ):
            raise PreprocessCacheError(f"Invalid cached value shape for {key!r}")
        if _value_shape(value) != expected_shape:
            raise PreprocessCacheError(f"Cached value shape differs for {key!r}")


def _resolve_cache_device(torch: Any, device_str: str) -> Any | None:
    if not device_str:
        return None
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if device_str.startswith("mps") and not mps_is_available(torch):
        return torch.device("cpu")
    return torch.device(device_str)


@dataclass
class CacheManager:
    root: Path
    dataset_fingerprint: str

    @classmethod
    def for_dataset(cls, dataset_fingerprint: str) -> CacheManager:
        root = default_cache_dir()
        return cls(root=root, dataset_fingerprint=dataset_fingerprint)

    def dataset_dir(self) -> Path:
        return self.root / _safe_path_component(self.dataset_fingerprint)

    def step_dir(self, step_fingerprint: str) -> Path:
        return self.dataset_dir() / "steps" / _safe_path_component(step_fingerprint)

    def split_dir(self, step_fingerprint: str, split: str) -> Path:
        step_root = self.step_dir(step_fingerprint)
        try:
            manifest = self._read_pointer(step_fingerprint)
            record = manifest["splits"][_safe_path_component(split)]
            generation = record["generation"]
            if isinstance(generation, str):
                return step_root / _GENERATIONS_DIRNAME / generation
        except (KeyError, PreprocessCacheError, TypeError):
            pass
        # Preserve the historical path helper for callers inspecting a miss.
        return step_root / _safe_path_component(split)

    def _manifest_path(self, step_fingerprint: str) -> Path:
        return self.step_dir(step_fingerprint) / _MANIFEST_FILENAME

    def _lock_path(self, step_fingerprint: str) -> Path:
        return self.step_dir(step_fingerprint) / _LOCK_FILENAME

    def _read_pointer(self, step_fingerprint: str) -> dict[str, Any]:
        manifest_path = self._manifest_path(step_fingerprint)
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PreprocessCacheError(f"Missing cache manifest for step {step_fingerprint}")
        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
            payload = json_load(manifest_text)
        except (OSError, UnicodeError) as exc:
            raise PreprocessCacheError("Invalid JSON manifest") from exc
        pointer = _verify_signed_manifest(payload, label="preprocess cache")
        if pointer.get("dataset_fingerprint") != self.dataset_fingerprint:
            raise PreprocessCacheError("Preprocess cache dataset fingerprint differs")
        if pointer.get("step_fingerprint") != step_fingerprint:
            raise PreprocessCacheError("Preprocess cache step fingerprint differs")
        splits = pointer.get("splits")
        if not isinstance(splits, dict):
            raise PreprocessCacheError("Invalid cache manifest structure")
        return pointer

    def _read_generation(self, *, step_fingerprint: str, split: str) -> tuple[Path, dict[str, Any]]:
        pointer = self._read_pointer(step_fingerprint)
        split_name = _safe_path_component(split)
        record = pointer["splits"].get(split_name)
        if not isinstance(record, dict):
            raise PreprocessCacheError(
                f"Missing cached outputs for step {step_fingerprint} split {split!r}"
            )
        generation = record.get("generation")
        expected_manifest_sha256 = record.get("generation_manifest_sha256")
        if not isinstance(generation, str) or not generation or Path(generation).name != generation:
            raise PreprocessCacheError("Invalid preprocess cache generation")
        if not _is_sha256(expected_manifest_sha256):
            raise PreprocessCacheError("Invalid preprocess generation manifest digest")
        generation_dir = self.step_dir(step_fingerprint) / _GENERATIONS_DIRNAME / generation
        if not generation_dir.is_dir() or generation_dir.is_symlink():
            raise PreprocessCacheError("Preprocess cache generation is missing")
        generation_path = generation_dir / _GENERATION_MANIFEST_FILENAME
        if not generation_path.is_file() or generation_path.is_symlink():
            raise PreprocessCacheError("Preprocess generation manifest is missing")
        try:
            generation_bytes = generation_path.read_bytes()
        except OSError as exc:
            raise PreprocessCacheError("Preprocess generation manifest is unreadable") from exc
        if hashlib.sha256(generation_bytes).hexdigest() != expected_manifest_sha256:
            raise PreprocessCacheError("Preprocess generation manifest file sha256 differs")
        try:
            payload = json_load(generation_bytes.decode("utf-8"))
        except UnicodeError as exc:
            raise PreprocessCacheError("Invalid preprocess generation manifest") from exc
        generation_manifest = _verify_signed_manifest(payload, label="preprocess generation")
        if generation_manifest.get("dataset_fingerprint") != self.dataset_fingerprint:
            raise PreprocessCacheError("Preprocess generation dataset fingerprint differs")
        if generation_manifest.get("step_fingerprint") != step_fingerprint:
            raise PreprocessCacheError("Preprocess generation step fingerprint differs")
        if generation_manifest.get("split") != split:
            raise PreprocessCacheError("Preprocess generation split differs")
        if not isinstance(generation_manifest.get("saved"), dict):
            raise PreprocessCacheError("Invalid cache manifest structure")
        return generation_dir, generation_manifest

    def has_step_outputs(self, step_fingerprint: str, *, split: str) -> bool:
        try:
            self._read_generation(step_fingerprint=step_fingerprint, split=split)
        except PreprocessCacheError:
            return False
        return True

    def save_step_outputs(
        self,
        *,
        step_fingerprint: str,
        split: str,
        produced: dict[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        step_root = self.step_dir(step_fingerprint)
        step_root.mkdir(parents=True, exist_ok=True)
        generations = step_root / _GENERATIONS_DIRNAME
        generations.mkdir(parents=True, exist_ok=True)
        split_name = _safe_path_component(split)

        with _exclusive_cache_lock(self._lock_path(step_fingerprint)):
            staging = generations / f".staging-{uuid.uuid4().hex}"
            staging.mkdir()
            try:
                saved: dict[str, dict[str, Any]] = {}
                for key, value in produced.items():
                    name = _safe_name(key)
                    descriptor = _save_value(staging / name, value)
                    if descriptor is not None:
                        saved[key] = _authenticate_descriptor(staging, descriptor, value=value)

                metadata = {str(key): value for key, value in manifest.items() if key != "saved"}
                generation_payload = _signed_manifest(
                    {
                        "schema_version": _CACHE_SCHEMA_VERSION,
                        "cache_kind": _CACHE_KIND,
                        "dataset_fingerprint": self.dataset_fingerprint,
                        "step_fingerprint": step_fingerprint,
                        "split": split,
                        "metadata": metadata,
                        "saved": saved,
                    }
                )
                generation_text = stable_json_dumps(generation_payload)
                generation_manifest_path = staging / _GENERATION_MANIFEST_FILENAME
                atomic_write_text(generation_manifest_path, generation_text)
                generation_manifest_file_sha256 = _sha256_file(generation_manifest_path)

                try:
                    _, current = self._read_generation(
                        step_fingerprint=step_fingerprint, split=split
                    )
                    # A signed manifest is insufficient if an immutable value was
                    # altered out of band. Validate every byte before treating the
                    # generation as an existing deterministic publication.
                    self.load_step_outputs(step_fingerprint=step_fingerprint, split=split)
                except PreprocessCacheError:
                    current = None
                if current is not None:
                    comparable = {
                        key: value for key, value in current.items() if key != "manifest_sha256"
                    }
                    candidate = {
                        key: value
                        for key, value in generation_payload.items()
                        if key != "manifest_sha256"
                    }
                    if comparable == candidate:
                        return
                    raise PreprocessCacheError(
                        "Different outputs were produced for the same preprocess cache key"
                    )

                generation_name = (
                    f"{split_name}-{generation_payload['manifest_sha256']}-{uuid.uuid4().hex}"
                )
                generation_dir = generations / generation_name
                _fsync_directory(staging)
                os.replace(staging, generation_dir)
                _fsync_directory(generations)

                try:
                    pointer = self._read_pointer(step_fingerprint)
                    split_records = dict(pointer["splits"])
                except PreprocessCacheError as error:
                    if self._manifest_path(step_fingerprint).exists():
                        logger.warning(
                            "Resetting corrupt preprocess manifest: step=%s split=%s error=%s",
                            step_fingerprint,
                            split,
                            error,
                        )
                    split_records = {}
                split_records[split_name] = {
                    "generation": generation_name,
                    "generation_manifest_sha256": generation_manifest_file_sha256,
                }
                pointer_payload = _signed_manifest(
                    {
                        "schema_version": _CACHE_SCHEMA_VERSION,
                        "cache_kind": _CACHE_KIND,
                        "dataset_fingerprint": self.dataset_fingerprint,
                        "step_fingerprint": step_fingerprint,
                        "splits": split_records,
                    }
                )
                atomic_write_text(
                    self._manifest_path(step_fingerprint), stable_json_dumps(pointer_payload)
                )
                _fsync_directory(step_root)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)

    def load_step_outputs(self, *, step_fingerprint: str, split: str) -> dict[str, Any]:
        generation_dir, generation = self._read_generation(
            step_fingerprint=step_fingerprint, split=split
        )
        saved = generation["saved"]
        out: dict[str, Any] = {}
        for key, desc in saved.items():
            if not isinstance(desc, dict):
                raise PreprocessCacheError("Invalid cache manifest structure")
            file_path = _verified_descriptor_path(generation_dir, desc)
            value = _load_value(generation_dir, desc)
            reverified_path = _verified_descriptor_path(generation_dir, desc)
            if reverified_path != file_path:
                raise PreprocessCacheError(f"Cached value path changed while loading: {file_path}")
            _verify_loaded_value(value, desc, key=str(key))
            out[str(key)] = value
        return out


def json_load(text: str) -> dict[str, Any]:
    import json

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise PreprocessCacheError("Invalid JSON manifest") from e
    if not isinstance(obj, dict):
        raise PreprocessCacheError("Invalid JSON manifest (expected object)")
    return obj
