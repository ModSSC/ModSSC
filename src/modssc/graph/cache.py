from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from platformdirs import user_cache_dir

from .artifacts import DatasetViews, GraphArtifact


class GraphCacheError(RuntimeError):
    """Raised when a graph cache entry is missing or corrupted."""


def default_cache_dir() -> Path:
    return Path(user_cache_dir("modssc")) / "graphs"


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise GraphCacheError(f"Invalid json payload in {path}: {e}") from e
    if not isinstance(data, dict):
        raise GraphCacheError(f"Invalid json payload in {path}")
    return data


@dataclass(frozen=True)
class GraphCache:
    """Disk cache for constructed graphs.

    Optional sharded edge storage:
    - for large graphs, storing a single `edge_index.npy` may be unwieldy
    - if `edge_shard_size` is set, edges are stored in multiple compressed `.npz` shards

    The manifest is always written last, so incomplete runs leave an entry without
    `manifest.json` (and are therefore ignored by `exists`/`load`).
    """

    root: Path
    edge_shard_size: int | None = None

    @classmethod
    def default(cls) -> GraphCache:
        return cls(root=default_cache_dir())

    def entry_dir(self, fingerprint: str) -> Path:
        return self.root / str(fingerprint)

    def exists(self, fingerprint: str) -> bool:
        return (self.entry_dir(fingerprint) / "manifest.json").exists()

    def _clear_entry_dir(self, d: Path) -> None:
        if not d.exists():
            return
        # Only remove files we create, keep directory (for resumable work dirs).
        for p in d.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                with contextlib.suppress(FileNotFoundError):
                    p.unlink()

    def save(
        self,
        *,
        fingerprint: str,
        graph: GraphArtifact,
        manifest: dict[str, Any],
        overwrite: bool = True,
    ) -> Path:
        d = self.entry_dir(fingerprint)
        d.mkdir(parents=True, exist_ok=True)

        if overwrite:
            self._clear_entry_dir(d)

        E = int(graph.edge_index.shape[1])
        shard_size = int(self.edge_shard_size) if self.edge_shard_size else 0

        storage: dict[str, Any]
        if shard_size and shard_size < E:
            n_shards = int((E + shard_size - 1) // shard_size)
            for i in range(n_shards):
                s = i * shard_size
                e = min(E, (i + 1) * shard_size)
                shard_path = d / f"edges_{i:04d}.npz"
                payload = {"edge_index": np.asarray(graph.edge_index[:, s:e], dtype=np.int64)}
                if graph.edge_weight is not None:
                    payload["edge_weight"] = np.asarray(graph.edge_weight[s:e], dtype=np.float32)
                np.savez_compressed(shard_path, **payload)

            storage = {
                "edge": {"kind": "sharded", "num_shards": n_shards, "shard_size": shard_size}
            }
        else:
            np.save(d / "edge_index.npy", np.asarray(graph.edge_index, dtype=np.int64))
            if graph.edge_weight is not None:
                np.save(d / "edge_weight.npy", np.asarray(graph.edge_weight, dtype=np.float32))
            storage = {"edge": {"kind": "single"}}

        payload = {**manifest, **graph.to_dict(), "_storage": storage}
        _safe_write_json(d / "manifest.json", payload)
        return d

    def _load_edges_single(self, d: Path) -> tuple[np.ndarray, np.ndarray | None]:
        try:
            edge_index = np.load(d / "edge_index.npy", allow_pickle=False)
        except Exception as e:
            raise GraphCacheError("Missing cached edge_index.npy") from e

        edge_weight_path = d / "edge_weight.npy"
        edge_weight = None
        if edge_weight_path.exists():
            try:
                edge_weight = np.load(edge_weight_path, allow_pickle=False)
            except Exception as e:
                raise GraphCacheError("Corrupted cached edge_weight.npy") from e

        return np.asarray(edge_index, dtype=np.int64), (
            np.asarray(edge_weight, dtype=np.float32) if edge_weight is not None else None
        )

    def _load_edges_sharded(
        self, d: Path, *, num_shards: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        shard_lengths: list[int] = []
        has_w: bool | None = None

        for i in range(int(num_shards)):
            shard_path = d / f"edges_{i:04d}.npz"
            if not shard_path.exists():
                raise GraphCacheError(f"Missing edge shard: {shard_path}")
            with np.load(shard_path, allow_pickle=False) as npz:
                if "edge_index" not in npz:
                    raise GraphCacheError(f"Shard missing edge_index: {shard_path}")
                idx = np.asarray(npz["edge_index"], dtype=np.int64)
                shard_lengths.append(int(idx.shape[1]))
                present = "edge_weight" in npz
                if has_w is None:
                    has_w = present
                elif has_w != present:
                    raise GraphCacheError(f"Inconsistent edge_weight in shard: {shard_path}")

        total = int(sum(shard_lengths))
        if total == 0:
            return np.zeros((2, 0), dtype=np.int64), (
                np.zeros((0,), dtype=np.float32) if has_w else None
            )

        edge_index = np.empty((2, total), dtype=np.int64)
        edge_weight = np.empty((total,), dtype=np.float32) if has_w else None

        offset = 0
        for i, length in enumerate(shard_lengths):
            if length == 0:
                continue
            shard_path = d / f"edges_{i:04d}.npz"
            with np.load(shard_path, allow_pickle=False) as npz:
                if "edge_index" not in npz:
                    raise GraphCacheError(f"Shard missing edge_index: {shard_path}")
                idx = np.asarray(npz["edge_index"], dtype=np.int64)
                edge_index[:, offset : offset + length] = idx
                if has_w:
                    if "edge_weight" not in npz:
                        raise GraphCacheError(f"Shard missing edge_weight: {shard_path}")
                    w = np.asarray(npz["edge_weight"], dtype=np.float32)
                    edge_weight[offset : offset + length] = w
            offset += length

        return edge_index, edge_weight

    def load(self, fingerprint: str) -> tuple[GraphArtifact, dict[str, Any]]:
        d = self.entry_dir(fingerprint)
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            raise GraphCacheError(f"Missing cached graph manifest: {manifest_path}")

        manifest = _safe_read_json(manifest_path)

        storage = manifest.get("_storage", {}).get("edge", {"kind": "single"})
        kind = storage.get("kind", "single")
        if kind == "sharded":
            edge_index, edge_weight = self._load_edges_sharded(
                d, num_shards=int(storage["num_shards"])
            )
        else:
            edge_index, edge_weight = self._load_edges_single(d)

        n_nodes = int(manifest.get("n_nodes"))
        directed = bool(manifest.get("directed", False))
        meta = dict(manifest.get("meta", {}))

        graph = GraphArtifact(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            directed=directed,
            meta=meta,
        )
        return graph, manifest

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def purge(self) -> int:
        if not self.root.exists():
            return 0
        n = 0
        for p in self.root.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                n += 1
        return n


@dataclass(frozen=True)
class ViewsCache:
    root: Path

    @classmethod
    def default(cls) -> ViewsCache:
        return cls(root=default_cache_dir().parent / "graph_views")

    def entry_dir(self, fingerprint: str) -> Path:
        return self.root / str(fingerprint)

    def exists(self, fingerprint: str) -> bool:
        return (self.entry_dir(fingerprint) / "manifest.json").exists()

    def save(
        self,
        *,
        fingerprint: str,
        views: DatasetViews,
        manifest: dict[str, Any],
        overwrite: bool = True,
    ) -> Path:
        d = self.entry_dir(fingerprint)
        d.mkdir(parents=True, exist_ok=True)
        if overwrite and d.exists():
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()

        # store views as a single compressed file
        arrays = {k: np.asarray(v) for k, v in views.views.items()}
        np.savez_compressed(d / "views.npz", **arrays)

        payload = dict(manifest)
        payload["meta"] = dict(views.meta)
        payload["n_nodes"] = int(np.asarray(views.y).shape[0])
        payload["view_names"] = sorted(list(views.views.keys()))
        payload["view_shapes"] = {k: list(np.asarray(v).shape) for k, v in views.views.items()}
        _safe_write_json(d / "manifest.json", payload)
        return d

    def load(
        self,
        fingerprint: str,
        *,
        y: np.ndarray,
        masks: dict[str, np.ndarray],
    ) -> tuple[DatasetViews, dict[str, Any]]:
        d = self.entry_dir(fingerprint)
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            raise GraphCacheError(f"Missing cached views manifest: {manifest_path}")

        manifest = _safe_read_json(manifest_path)
        views_path = d / "views.npz"
        if not views_path.exists():
            raise GraphCacheError("Missing cached views.npz")

        with np.load(views_path, allow_pickle=False) as npz:
            views_dict = {k: np.asarray(npz[k]) for k in npz.files}

        meta = dict(manifest.get("meta", {}))
        views = DatasetViews(views=views_dict, y=np.asarray(y), masks=dict(masks), meta=meta)
        return views, manifest

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])
