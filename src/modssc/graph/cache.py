from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from platformdirs import user_cache_dir

from modssc.runtime.paths import default_local_cache_subdir

from .artifacts import DatasetViews, GraphArtifact

GRAPH_CACHE_ENV = "MODSSC_GRAPH_CACHE_DIR"
GRAPH_VIEWS_CACHE_ENV = "MODSSC_GRAPH_VIEWS_CACHE_DIR"
CACHE_ROOT_ENV = "MODSSC_CACHE_ROOT"

# v2 is the first self-authenticated schema. Pre-v2/legacy entries are
# intentionally cache misses and load fail-closed; they must be rebuilt.
CACHE_MANIFEST_SCHEMA_VERSION = 2
GRAPH_IMPLEMENTATION_IDENTITY_SCHEMA_VERSION = 1

_CACHE_MANIFEST_KEY = "_cache"
_MANIFEST_FILENAME = "manifest.json"
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_WINDOWS_RESERVED_COMPONENTS = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class GraphCacheError(RuntimeError):
    """Raised when a graph cache entry is missing or corrupted."""


def default_cache_dir() -> Path:
    override = os.environ.get(GRAPH_CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()

    root_override = os.environ.get(CACHE_ROOT_ENV)
    if root_override:
        return Path(root_override).expanduser().resolve() / "graph"

    local = default_local_cache_subdir("graph")
    if local is not None:
        return local

    return Path(user_cache_dir("modssc")) / "graph"


def default_views_cache_dir() -> Path:
    override = os.environ.get(GRAPH_VIEWS_CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()

    graph_override = os.environ.get(GRAPH_CACHE_ENV)
    if graph_override:
        return Path(graph_override).expanduser().resolve().parent / "graph_views"

    root_override = os.environ.get(CACHE_ROOT_ENV)
    if root_override:
        return Path(root_override).expanduser().resolve() / "graph_views"

    local = default_local_cache_subdir("graph_views")
    if local is not None:
        return local
    return default_cache_dir().parent / "graph_views"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cache_path_component(fingerprint: str) -> str:
    """Map a logical fingerprint to one portable, bounded path component."""

    raw = str(fingerprint)
    is_portable = (
        0 < len(raw) <= 120
        and _SAFE_PATH_COMPONENT.fullmatch(raw) is not None
        and raw not in {".", ".."}
        and not raw.endswith(".")
        and raw.casefold() not in _WINDOWS_RESERVED_COMPONENTS
    )
    if is_portable:
        return raw
    return f"sha256-{_sha256_bytes(raw.encode('utf-8'))}"


def _manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the complete manifest, excluding only its embedded digest field."""

    authenticated = dict(payload)
    envelope = authenticated.get(_CACHE_MANIFEST_KEY)
    if isinstance(envelope, Mapping):
        clean_envelope = dict(envelope)
        clean_envelope.pop("manifest_sha256", None)
        authenticated[_CACHE_MANIFEST_KEY] = clean_envelope
    return _sha256_bytes(_canonical_json_bytes(authenticated))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        try:
            stream = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise
        with stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def _safe_read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise GraphCacheError(f"Refusing symlinked cache manifest: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except FileNotFoundError as exc:
        raise GraphCacheError(f"Missing cached manifest: {path}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise GraphCacheError(f"Invalid json payload in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GraphCacheError(f"Invalid json payload in {path}")
    return data


def _ensure_windows_lock_byte(lock_stream: Any) -> None:
    """Create the byte locked by msvcrt exactly once, then rewind to it."""

    lock_stream.seek(0)
    if os.fstat(lock_stream.fileno()).st_size == 0:
        lock_stream.write(b"\0")
        lock_stream.flush()
    lock_stream.seek(0)


@contextmanager
def _entry_lock(root: Path, fingerprint: str, *, shared: bool = False) -> Iterator[None]:
    """Coordinate processes for one fingerprint (shared readers on POSIX)."""

    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock_name = _sha256_bytes(str(fingerprint).encode("utf-8"))
    lock_path = locks / f"{lock_name}.lock"
    lock_stream = lock_path.open("a+b")
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt

            _ensure_windows_lock_byte(lock_stream)
            msvcrt.locking(lock_stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(lock_stream.fileno(), mode)
        yield
    finally:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt

            lock_stream.seek(0)
            with contextlib.suppress(OSError):
                msvcrt.locking(lock_stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            with contextlib.suppress(OSError):
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@lru_cache(maxsize=1)
def _graph_source_sha256() -> str:
    """Hash installed graph implementation sources, independent of checkout path."""

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        relative = source.relative_to(root).as_posix()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(_sha256_bytes(source.read_bytes()).encode("ascii"))
    return digest.hexdigest()


def graph_implementation_identity(*, component: str, backend: str | None = None) -> dict[str, Any]:
    """Return the implementation/runtime identity used in graph cache keys."""

    dependencies: dict[str, str | None] = {"numpy": np.__version__}
    if backend == "sklearn" or component == "views":
        dependencies["scikit-learn"] = _distribution_version("scikit-learn")
    if backend == "faiss":
        dependencies["faiss"] = _distribution_version("faiss-cpu", "faiss-gpu")
    if backend == "annoy":
        dependencies["annoy"] = _distribution_version("annoy")
    if backend == "torch":
        dependencies["torch"] = _distribution_version("torch")
    if component == "views":
        dependencies["scipy"] = _distribution_version("scipy")

    return {
        "schema_version": GRAPH_IMPLEMENTATION_IDENTITY_SCHEMA_VERSION,
        "component": str(component),
        "backend": backend,
        "source_sha256": _graph_source_sha256(),
        "dependencies": dependencies,
    }


def _update_framed(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _hash_array_bytes(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    if contiguous.nbytes == 0:
        return
    raw = memoryview(contiguous).cast("B")
    chunk_size = 8 * 1024 * 1024
    for offset in range(0, len(raw), chunk_size):
        digest.update(raw[offset : offset + chunk_size])


def _array_descriptor(array: Any) -> dict[str, Any]:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise GraphCacheError("Object arrays cannot be stored in an authenticated cache")

    digest = hashlib.sha256()
    _update_framed(digest, "dtype", value.dtype.str.encode("ascii"))
    _update_framed(digest, "shape", _canonical_json_bytes(list(value.shape)))
    _hash_array_bytes(digest, value)
    return {
        "sha256": digest.hexdigest(),
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "nbytes": int(value.nbytes),
    }


def array_content_sha256(array: Any) -> str:
    """Return a full logical-content commitment for a dense array."""

    return str(_array_descriptor(array)["sha256"])


def graph_content_sha256(graph: GraphArtifact) -> str:
    payload: dict[str, Any] = {
        "n_nodes": int(graph.n_nodes),
        "directed": bool(graph.directed),
        "edge_index": _array_descriptor(graph.edge_index),
        "edge_weight": (
            _array_descriptor(graph.edge_weight) if graph.edge_weight is not None else None
        ),
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def views_content_sha256(views: Mapping[str, Any]) -> str:
    payload = {name: _array_descriptor(views[name]) for name in sorted(str(key) for key in views)}
    return _sha256_bytes(_canonical_json_bytes(payload))


def _file_descriptor(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise GraphCacheError(f"Refusing symlinked cache file: {path}")
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise GraphCacheError(f"Unable to authenticate cached file {path}: {exc}") from exc
    before_state = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_state = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_state != after_state:
        raise GraphCacheError(f"Cached file changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "size": int(after.st_size)}


def _build_authenticated_manifest(
    *,
    kind: str,
    fingerprint: str,
    payload: Mapping[str, Any],
    entry: Path,
    files: list[str],
    arrays: Mapping[str, Any],
    content_sha256: str,
) -> dict[str, Any]:
    if _CACHE_MANIFEST_KEY in payload:
        raise GraphCacheError(f"{_CACHE_MANIFEST_KEY!r} is reserved for cache authentication")
    if "fingerprint" in payload and str(payload["fingerprint"]) != str(fingerprint):
        raise GraphCacheError("Manifest fingerprint differs from the cache key")

    file_descriptors = {name: _file_descriptor(entry / name) for name in sorted(files)}
    envelope: dict[str, Any] = {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "kind": str(kind),
        "fingerprint": str(fingerprint),
        "cache_implementation": graph_implementation_identity(component="cache"),
        "files": file_descriptors,
        "arrays": dict(arrays),
        "content_sha256": str(content_sha256),
    }
    manifest = dict(payload)
    manifest["fingerprint"] = str(fingerprint)
    manifest[_CACHE_MANIFEST_KEY] = envelope
    envelope["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


def _validate_authenticated_manifest(
    *,
    entry: Path,
    fingerprint: str,
    kind: str,
    verify_file_hashes: bool,
) -> dict[str, Any]:
    manifest_path = entry / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise GraphCacheError(f"Missing cached {kind} manifest: {manifest_path}")
    manifest = _safe_read_json(manifest_path)
    envelope = manifest.get(_CACHE_MANIFEST_KEY)
    if not isinstance(envelope, dict):
        raise GraphCacheError(
            f"Legacy {kind} cache manifest is unauthenticated and must be rebuilt"
        )
    if envelope.get("schema_version") != CACHE_MANIFEST_SCHEMA_VERSION:
        raise GraphCacheError(f"Unsupported {kind} cache manifest schema")
    if envelope.get("kind") != kind:
        raise GraphCacheError(f"Cache kind mismatch: expected {kind!r}")
    if str(manifest.get("fingerprint", "")) != str(fingerprint):
        raise GraphCacheError("Manifest fingerprint differs from requested cache key")
    if str(envelope.get("fingerprint", "")) != str(fingerprint):
        raise GraphCacheError("Authenticated fingerprint differs from requested cache key")
    supplied_sha = envelope.get("manifest_sha256")
    if not isinstance(supplied_sha, str) or supplied_sha != _manifest_sha256(manifest):
        raise GraphCacheError("Cache manifest authentication failed")

    files = envelope.get("files")
    if not isinstance(files, dict) or not files:
        raise GraphCacheError("Cache manifest has no authenticated files")
    declared_names = set(files)
    actual_names = {path.name for path in entry.iterdir() if path.name != _MANIFEST_FILENAME}
    if actual_names != declared_names:
        missing = sorted(declared_names - actual_names)
        unexpected = sorted(actual_names - declared_names)
        raise GraphCacheError(
            f"Cache file set differs from manifest; missing={missing}, unexpected={unexpected}"
        )

    for name, expected in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise GraphCacheError(f"Unsafe cached file name in manifest: {name!r}")
        if not isinstance(expected, dict):
            raise GraphCacheError(f"Invalid file descriptor for {name!r}")
        path = entry / name
        if not path.is_file() or path.is_symlink():
            raise GraphCacheError(f"Missing or unsafe cached file: {path}")
        try:
            size = int(path.stat().st_size)
        except OSError as exc:
            raise GraphCacheError(f"Unable to stat cached file: {path}") from exc
        if size != expected.get("size"):
            raise GraphCacheError(f"Cached file size mismatch: {path}")
        if verify_file_hashes and _file_descriptor(path).get("sha256") != expected.get("sha256"):
            raise GraphCacheError(f"Cached file SHA-256 mismatch: {path}")
    return manifest


def _validate_expected_manifest(
    manifest: Mapping[str, Any], expected_manifest: Mapping[str, Any] | None
) -> None:
    if expected_manifest is None:
        return
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise GraphCacheError(
                f"Cached manifest field {field!r} differs from the requested specification"
            )


def _staging_dir(root: Path, fingerprint: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    component = _cache_path_component(fingerprint)
    path = root / f".{component}.staging-{os.getpid()}-{token}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def _publish_staged_entry(
    *,
    root: Path,
    fingerprint: str,
    kind: str,
    staging: Path,
    overwrite: bool,
) -> Path:
    component = _cache_path_component(fingerprint)
    destination = root / component
    _validate_authenticated_manifest(
        entry=staging,
        fingerprint=fingerprint,
        kind=kind,
        verify_file_hashes=True,
    )

    backup: Path | None = None
    with _entry_lock(root, fingerprint):
        if destination.exists():
            current: dict[str, Any] | None = None
            with contextlib.suppress(GraphCacheError):
                current = _validate_authenticated_manifest(
                    entry=destination,
                    fingerprint=fingerprint,
                    kind=kind,
                    verify_file_hashes=True,
                )
            if current is not None:
                staged = _safe_read_json(staging / _MANIFEST_FILENAME)
                current_content = current[_CACHE_MANIFEST_KEY].get("content_sha256")
                staged_content = staged[_CACHE_MANIFEST_KEY].get("content_sha256")
                if current_content == staged_content:
                    identity_fields = (
                        "dataset_fingerprint",
                        "preprocess_fingerprint",
                        "graph_fingerprint",
                        "graph_content_fingerprint",
                        "features_fingerprint",
                        "spec_fingerprint",
                        "seed",
                        "producer_identity",
                        "resolved_backend",
                    )
                    for field in identity_fields:
                        if current.get(field) != staged.get(field):
                            raise GraphCacheError(
                                "Cache fingerprint collision: authenticated identity field "
                                f"{field!r} differs for {fingerprint}"
                            )
                    shutil.rmtree(staging)
                    return destination
                raise GraphCacheError(
                    "Cache fingerprint collision or non-deterministic output: "
                    f"{fingerprint} already names different authenticated content"
                )
            if not overwrite:
                raise GraphCacheError(f"Cache entry already exists and is invalid: {destination}")
            backup = root / f".{component}.replaced-{os.getpid()}-{uuid.uuid4().hex}"
            os.replace(destination, backup)

        try:
            os.replace(staging, destination)
            _fsync_directory(root)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        else:
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
            return destination


class _FingerprintCacheOps:
    root: Path
    cache_kind: ClassVar[str]

    def entry_dir(self, fingerprint: str) -> Path:
        return self.root / _cache_path_component(fingerprint)

    def work_dir(self, fingerprint: str) -> Path:
        """Return resumable scratch space outside the immutable published entry."""

        return self.root / ".work" / _cache_path_component(fingerprint)

    def exists(self, fingerprint: str) -> bool:
        entry = self.entry_dir(fingerprint)
        try:
            with _entry_lock(self.root, fingerprint, shared=True):
                _validate_authenticated_manifest(
                    entry=entry,
                    fingerprint=fingerprint,
                    kind=self.cache_kind,
                    verify_file_hashes=True,
                )
        except GraphCacheError:
            return False
        return True

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        fingerprints: list[str] = []
        for path in self.root.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            logical_name = path.name
            with contextlib.suppress(GraphCacheError):
                manifest = _safe_read_json(path / _MANIFEST_FILENAME)
                value = manifest.get("fingerprint")
                if isinstance(value, str):
                    logical_name = value
            fingerprints.append(logical_name)
        return sorted(fingerprints)


@dataclass(frozen=True)
class GraphCache(_FingerprintCacheOps):
    """Authenticated, immutable-once-valid disk cache for constructed graphs.

    Manifest schema v2 deliberately invalidates unauthenticated legacy entries.
    Logical fingerprints remain in the manifest while unsafe path characters are
    mapped to a portable SHA-256 directory component.
    """

    root: Path
    edge_shard_size: int | None = None
    cache_kind: ClassVar[str] = "graph"

    @classmethod
    def default(cls) -> GraphCache:
        return cls(root=default_cache_dir())

    def _clear_entry_dir(self, directory: Path) -> None:
        """Compatibility helper; publishing never clears a live valid entry in place."""

        if not directory.exists():
            return
        for path in directory.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

    def save(
        self,
        *,
        fingerprint: str,
        graph: GraphArtifact,
        manifest: dict[str, Any],
        overwrite: bool = True,
    ) -> Path:
        staging = _staging_dir(self.root, fingerprint)
        try:
            edge_index = np.asarray(graph.edge_index, dtype=np.int64)
            weight_dtype = (
                np.float64 if graph.meta.get("edge_weight_dtype") == "float64" else np.float32
            )
            edge_weight = (
                np.asarray(graph.edge_weight, dtype=weight_dtype)
                if graph.edge_weight is not None
                else None
            )
            edge_count = int(edge_index.shape[1])
            shard_size = int(self.edge_shard_size) if self.edge_shard_size else 0
            files: list[str] = []

            if shard_size and shard_size < edge_count:
                n_shards = int((edge_count + shard_size - 1) // shard_size)
                for index in range(n_shards):
                    start = index * shard_size
                    stop = min(edge_count, (index + 1) * shard_size)
                    name = f"edges_{index:04d}.npz"
                    values: dict[str, np.ndarray] = {"edge_index": edge_index[:, start:stop]}
                    if edge_weight is not None:
                        values["edge_weight"] = edge_weight[start:stop]
                    np.savez_compressed(staging / name, **values)
                    _fsync_file(staging / name)
                    files.append(name)
                storage: dict[str, Any] = {
                    "edge": {
                        "kind": "sharded",
                        "num_shards": n_shards,
                        "shard_size": shard_size,
                        "files": files,
                    }
                }
            else:
                np.save(staging / "edge_index.npy", edge_index, allow_pickle=False)
                _fsync_file(staging / "edge_index.npy")
                files.append("edge_index.npy")
                if edge_weight is not None:
                    np.save(staging / "edge_weight.npy", edge_weight, allow_pickle=False)
                    _fsync_file(staging / "edge_weight.npy")
                    files.append("edge_weight.npy")
                storage = {
                    "edge": {
                        "kind": "single",
                        "files": files,
                    }
                }

            content_sha = graph_content_sha256(
                GraphArtifact(
                    n_nodes=int(graph.n_nodes),
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    directed=bool(graph.directed),
                    meta=dict(graph.meta),
                )
            )
            graph_meta_fingerprint = graph.meta.get("fingerprint")
            if graph_meta_fingerprint is not None and str(graph_meta_fingerprint) != str(
                fingerprint
            ):
                raise GraphCacheError("Graph metadata fingerprint differs from the cache key")
            producer_identity = (
                manifest.get("producer_identity")
                or graph.meta.get("producer_identity")
                or graph_implementation_identity(
                    component="construction",
                    backend=(
                        str(manifest["resolved_backend"])
                        if manifest.get("resolved_backend") is not None
                        else None
                    ),
                )
            )
            meta = dict(graph.meta)
            meta["graph_content_sha256"] = content_sha
            # The fallback above always supplies a native producer identity.
            meta["producer_identity"] = producer_identity

            payload = {**manifest, **graph.to_dict()}
            payload["meta"] = meta
            payload["graph_content_sha256"] = content_sha
            payload["producer_identity"] = producer_identity
            payload["_storage"] = storage
            arrays = {
                "edge_index": _array_descriptor(edge_index),
                "edge_weight": (
                    _array_descriptor(edge_weight) if edge_weight is not None else None
                ),
            }
            authenticated = _build_authenticated_manifest(
                kind=self.cache_kind,
                fingerprint=fingerprint,
                payload=payload,
                entry=staging,
                files=files,
                arrays=arrays,
                content_sha256=content_sha,
            )
            _safe_write_json(staging / _MANIFEST_FILENAME, authenticated)
            _fsync_directory(staging)
            return _publish_staged_entry(
                root=self.root,
                fingerprint=fingerprint,
                kind=self.cache_kind,
                staging=staging,
                overwrite=overwrite,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _load_edges_single(
        self,
        directory: Path,
        *,
        weight_dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        edge_index_path = directory / "edge_index.npy"
        if not edge_index_path.exists():
            raise GraphCacheError("Missing cached edge_index.npy")
        try:
            edge_index = np.load(edge_index_path, allow_pickle=False)
        except Exception as exc:
            raise GraphCacheError("Corrupted cached edge_index.npy") from exc

        edge_weight_path = directory / "edge_weight.npy"
        edge_weight = None
        if edge_weight_path.exists():
            try:
                edge_weight = np.load(edge_weight_path, allow_pickle=False)
            except Exception as exc:
                raise GraphCacheError("Corrupted cached edge_weight.npy") from exc
        return np.asarray(edge_index, dtype=np.int64), (
            np.asarray(edge_weight, dtype=weight_dtype) if edge_weight is not None else None
        )

    def _load_edges_sharded(
        self,
        directory: Path,
        *,
        num_shards: int,
        weight_dtype: type[np.float32] | type[np.float64] = np.float32,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        indexes: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        has_weight: bool | None = None
        for index in range(int(num_shards)):
            shard_path = directory / f"edges_{index:04d}.npz"
            if not shard_path.exists():
                raise GraphCacheError(f"Missing edge shard: {shard_path}")
            try:
                with np.load(shard_path, allow_pickle=False) as archive:
                    if "edge_index" not in archive:
                        raise GraphCacheError(f"Shard missing edge_index: {shard_path}")
                    edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
                    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                        raise GraphCacheError(f"Invalid edge_index in shard: {shard_path}")
                    indexes.append(edge_index)
                    present = "edge_weight" in archive
                    if has_weight is None:
                        has_weight = present
                    elif has_weight != present:
                        raise GraphCacheError(f"Inconsistent edge_weight in shard: {shard_path}")
                    if present:
                        weight = np.asarray(archive["edge_weight"], dtype=weight_dtype)
                        if weight.ndim != 1 or weight.shape[0] != edge_index.shape[1]:
                            raise GraphCacheError(f"Invalid edge_weight in shard: {shard_path}")
                        weights.append(weight)
            except GraphCacheError:
                raise
            except Exception as exc:
                raise GraphCacheError(f"Corrupted edge shard: {shard_path}") from exc

        edge_index = (
            np.concatenate(indexes, axis=1) if indexes else np.zeros((2, 0), dtype=np.int64)
        )
        edge_weight = (
            np.concatenate(weights)
            if has_weight and weights
            else (np.zeros((0,), dtype=weight_dtype) if has_weight else None)
        )
        return edge_index, edge_weight

    def load(
        self,
        fingerprint: str,
        *,
        expected_manifest: Mapping[str, Any] | None = None,
    ) -> tuple[GraphArtifact, dict[str, Any]]:
        directory = self.entry_dir(fingerprint)
        with _entry_lock(self.root, fingerprint, shared=True):
            manifest = _validate_authenticated_manifest(
                entry=directory,
                fingerprint=fingerprint,
                kind=self.cache_kind,
                verify_file_hashes=True,
            )
            _validate_expected_manifest(manifest, expected_manifest)
            envelope = manifest[_CACHE_MANIFEST_KEY]
            arrays = envelope.get("arrays")
            if not isinstance(arrays, dict):
                raise GraphCacheError("Graph cache has no authenticated array descriptors")

            meta = dict(manifest.get("meta", {}))
            meta_fingerprint = meta.get("fingerprint")
            if meta_fingerprint is not None and str(meta_fingerprint) != str(fingerprint):
                raise GraphCacheError("Cached graph metadata fingerprint differs from cache key")
            weight_dtype = np.float64 if meta.get("edge_weight_dtype") == "float64" else np.float32
            storage = manifest.get("_storage")
            if not isinstance(storage, dict) or not isinstance(storage.get("edge"), dict):
                raise GraphCacheError("Invalid graph cache storage descriptor")
            edge_storage = storage["edge"]
            kind = edge_storage.get("kind")
            if kind == "sharded":
                num_shards = edge_storage.get("num_shards")
                if not isinstance(num_shards, int) or num_shards < 0:
                    raise GraphCacheError("Invalid graph cache shard count")
                edge_index, edge_weight = self._load_edges_sharded(
                    directory,
                    num_shards=num_shards,
                    weight_dtype=weight_dtype,
                )
            elif kind == "single":
                edge_index, edge_weight = self._load_edges_single(
                    directory, weight_dtype=weight_dtype
                )
            else:
                raise GraphCacheError(f"Unknown graph cache storage kind: {kind!r}")

            if _array_descriptor(edge_index) != arrays.get("edge_index"):
                raise GraphCacheError("Cached edge_index logical content differs from manifest")
            expected_weight = arrays.get("edge_weight")
            actual_weight = _array_descriptor(edge_weight) if edge_weight is not None else None
            if actual_weight != expected_weight:
                raise GraphCacheError("Cached edge_weight logical content differs from manifest")

            try:
                n_nodes = int(manifest["n_nodes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GraphCacheError("Invalid cached graph n_nodes") from exc
            directed = bool(manifest.get("directed", False))
            if int(manifest.get("n_edges", -1)) != int(edge_index.shape[1]):
                raise GraphCacheError("Cached graph n_edges differs from stored arrays")
            if bool(manifest.get("has_edge_weight")) != (edge_weight is not None):
                raise GraphCacheError("Cached graph weight presence differs from manifest")
            if edge_weight is not None and not np.isfinite(edge_weight).all():
                raise GraphCacheError("Cached graph contains non-finite edge weights")

            graph = GraphArtifact(
                n_nodes=n_nodes,
                edge_index=edge_index,
                edge_weight=edge_weight,
                directed=directed,
                meta=meta,
            )
            content_sha = graph_content_sha256(graph)
            if content_sha != envelope.get("content_sha256"):
                raise GraphCacheError("Cached graph content commitment differs from manifest")
            if manifest.get("graph_content_sha256") != content_sha:
                raise GraphCacheError("Cached graph content fingerprint differs from manifest")
            post_load_manifest = _validate_authenticated_manifest(
                entry=directory,
                fingerprint=fingerprint,
                kind=self.cache_kind,
                verify_file_hashes=True,
            )
            if post_load_manifest != manifest:
                raise GraphCacheError("Graph cache state changed during deserialization")
            return graph, manifest

    def purge(self) -> int:
        if not self.root.exists():
            return 0
        count = 0
        for path in self.root.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                shutil.rmtree(path, ignore_errors=True)
                count += 1
        return count


@dataclass(frozen=True)
class ViewsCache(_FingerprintCacheOps):
    """Authenticated v2 disk cache for graph-derived views.

    Legacy entries are controlled misses; v2 entries use portable digest-backed
    paths without changing the logical fingerprint exposed by the API.
    """

    root: Path
    cache_kind: ClassVar[str] = "views"

    @classmethod
    def default(cls) -> ViewsCache:
        return cls(root=default_views_cache_dir())

    def save(
        self,
        *,
        fingerprint: str,
        views: DatasetViews,
        manifest: dict[str, Any],
        overwrite: bool = True,
    ) -> Path:
        staging = _staging_dir(self.root, fingerprint)
        try:
            arrays = {name: np.asarray(value) for name, value in views.views.items()}
            if any(value.dtype.hasobject for value in arrays.values()):
                raise GraphCacheError(
                    "Object arrays cannot be stored in an authenticated views cache"
                )
            np.savez_compressed(staging / "views.npz", **arrays)
            _fsync_file(staging / "views.npz")

            content_sha = views_content_sha256(arrays)
            views_meta_fingerprint = views.meta.get("fingerprint")
            if views_meta_fingerprint is not None and str(views_meta_fingerprint) != str(
                fingerprint
            ):
                raise GraphCacheError("Views metadata fingerprint differs from the cache key")
            producer_identity = (
                manifest.get("producer_identity")
                or views.meta.get("producer_identity")
                or graph_implementation_identity(component="views")
            )
            meta = dict(views.meta)
            meta["views_content_sha256"] = content_sha
            # The fallback above always supplies a native producer identity.
            meta["producer_identity"] = producer_identity

            payload = dict(manifest)
            payload["meta"] = meta
            payload["n_nodes"] = int(np.asarray(views.y).shape[0])
            payload["view_names"] = sorted(arrays)
            payload["view_shapes"] = {name: list(value.shape) for name, value in arrays.items()}
            payload["view_dtypes"] = {name: value.dtype.str for name, value in arrays.items()}
            payload["views_content_sha256"] = content_sha
            payload["producer_identity"] = producer_identity
            descriptors = {name: _array_descriptor(value) for name, value in sorted(arrays.items())}
            authenticated = _build_authenticated_manifest(
                kind=self.cache_kind,
                fingerprint=fingerprint,
                payload=payload,
                entry=staging,
                files=["views.npz"],
                arrays=descriptors,
                content_sha256=content_sha,
            )
            _safe_write_json(staging / _MANIFEST_FILENAME, authenticated)
            _fsync_directory(staging)
            return _publish_staged_entry(
                root=self.root,
                fingerprint=fingerprint,
                kind=self.cache_kind,
                staging=staging,
                overwrite=overwrite,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def load(
        self,
        fingerprint: str,
        *,
        y: np.ndarray,
        masks: dict[str, np.ndarray],
        expected_manifest: Mapping[str, Any] | None = None,
    ) -> tuple[DatasetViews, dict[str, Any]]:
        directory = self.entry_dir(fingerprint)
        with _entry_lock(self.root, fingerprint, shared=True):
            manifest = _validate_authenticated_manifest(
                entry=directory,
                fingerprint=fingerprint,
                kind=self.cache_kind,
                verify_file_hashes=True,
            )
            _validate_expected_manifest(manifest, expected_manifest)
            envelope = manifest[_CACHE_MANIFEST_KEY]
            descriptors = envelope.get("arrays")
            if not isinstance(descriptors, dict):
                raise GraphCacheError("Views cache has no authenticated array descriptors")

            try:
                with np.load(directory / "views.npz", allow_pickle=False) as archive:
                    views_dict = {name: np.asarray(archive[name]) for name in archive.files}
            except Exception as exc:
                raise GraphCacheError("Corrupted cached views.npz") from exc

            expected_names = manifest.get("view_names")
            if not isinstance(expected_names, list) or sorted(views_dict) != sorted(
                str(name) for name in expected_names
            ):
                raise GraphCacheError("Cached view names differ from manifest")
            actual_descriptors = {
                name: _array_descriptor(value) for name, value in sorted(views_dict.items())
            }
            if actual_descriptors != descriptors:
                raise GraphCacheError("Cached view logical content differs from manifest")
            content_sha = views_content_sha256(views_dict)
            if content_sha != envelope.get("content_sha256"):
                raise GraphCacheError("Cached views content commitment differs from manifest")
            if manifest.get("views_content_sha256") != content_sha:
                raise GraphCacheError("Cached views content fingerprint differs from manifest")

            y_array = np.asarray(y)
            if int(manifest.get("n_nodes", -1)) != int(y_array.shape[0]):
                raise GraphCacheError("Cached views n_nodes differs from supplied labels")
            meta = dict(manifest.get("meta", {}))
            meta_fingerprint = meta.get("fingerprint")
            if meta_fingerprint is not None and str(meta_fingerprint) != str(fingerprint):
                raise GraphCacheError("Cached views metadata fingerprint differs from cache key")
            views = DatasetViews(
                views=views_dict,
                y=y_array,
                masks=dict(masks),
                meta=meta,
            )
            post_load_manifest = _validate_authenticated_manifest(
                entry=directory,
                fingerprint=fingerprint,
                kind=self.cache_kind,
                verify_file_hashes=True,
            )
            if post_load_manifest != manifest:
                raise GraphCacheError("Views cache state changed during deserialization")
            return views, manifest
