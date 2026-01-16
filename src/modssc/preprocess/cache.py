from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from platformdirs import user_cache_dir

from modssc.device import mps_is_available
from modssc.preprocess.errors import OptionalDependencyError, PreprocessCacheError
from modssc.preprocess.fingerprint import stable_json_dumps

OBJECT_JSON_MAX_ITEMS = int(
    os.environ.get("MODSSC_PREPROCESS_CACHE_OBJECT_JSON_MAX_ITEMS", "10000")
)
MMAP_THRESHOLD_BYTES = int(
    os.environ.get("MODSSC_PREPROCESS_CACHE_MMAP_THRESHOLD", str(64 * 1024 * 1024))
)


def default_cache_dir() -> Path:
    override = os.environ.get("MODSSC_PREPROCESS_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()

    # Heuristic: if running in a dev repo (pyproject.toml exists in parents),
    # default to a local "cache" folder at the repo root.
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent / "cache" / "preprocess"

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
    path_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
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
    fp = path / rel
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
        return self.step_dir(step_fingerprint) / _safe_path_component(split)

    def has_step_outputs(self, step_fingerprint: str, *, split: str) -> bool:
        return (self.step_dir(step_fingerprint) / "manifest.json").exists() and self.split_dir(
            step_fingerprint, split
        ).exists()

    def save_step_outputs(
        self,
        *,
        step_fingerprint: str,
        split: str,
        produced: dict[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        d = self.split_dir(step_fingerprint, split)
        d.mkdir(parents=True, exist_ok=True)

        saved: dict[str, dict[str, Any]] = {}
        for key, value in produced.items():
            name = _safe_name(key)
            desc = _save_value(d / name, value)
            if desc is not None:
                saved[key] = desc

        step_root = self.step_dir(step_fingerprint)
        step_root.mkdir(parents=True, exist_ok=True)
        manifest_path = step_root / "manifest.json"

        payload = json_load(manifest_path.read_text()) if manifest_path.exists() else dict(manifest)

        # Keep metadata fresh (but never drop already-written saved entries).
        for k, v in manifest.items():
            if k != "saved":
                payload[k] = v

        payload.setdefault("saved", {})
        payload["saved"][split] = saved
        manifest_path.write_text(stable_json_dumps(payload))

    def load_step_outputs(self, *, step_fingerprint: str, split: str) -> dict[str, Any]:
        step_root = self.step_dir(step_fingerprint)
        manifest_path = step_root / "manifest.json"
        if not manifest_path.exists():
            raise PreprocessCacheError(f"Missing cache manifest for step {step_fingerprint}")

        manifest = manifest_path.read_text()
        data = json_load(manifest)
        saved = data.get("saved", {}).get(split, {})
        if not isinstance(saved, dict):
            raise PreprocessCacheError("Invalid cache manifest structure")
        out: dict[str, Any] = {}
        split_path = self.split_dir(step_fingerprint, split)
        for key, desc in saved.items():
            if isinstance(desc, dict):
                out[str(key)] = _load_value(split_path, desc)
        return out


def json_load(text: str) -> dict[str, Any]:
    import json

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise PreprocessCacheError("Invalid JSON manifest (expected object)")
    return obj
