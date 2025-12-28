from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from modssc.sampling.result import SamplingResult


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def save_split(result: SamplingResult, out_dir: Path, *, overwrite: bool = False) -> Path:
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already exists, pass overwrite=True")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema_version": int(result.schema_version),
        "created_at": result.created_at,
        "dataset_fingerprint": result.dataset_fingerprint,
        "split_fingerprint": result.split_fingerprint,
        "plan": dict(result.plan),
        "refs": dict(result.refs),
        "stats": dict(result.stats),
        "mode": "graph" if result.is_graph() else "inductive",
    }
    _atomic_write_text(out_dir / "split.json", json.dumps(meta, indent=2, sort_keys=True) + "\n")

    arrays: dict[str, np.ndarray] = {}
    for k, v in result.indices.items():
        arrays[f"idx__{k}"] = np.asarray(v, dtype=np.int64)
    for k, v in result.masks.items():
        arrays[f"mask__{k}"] = np.asarray(v, dtype=bool)

    np.savez_compressed(out_dir / "arrays.npz", **arrays)
    return out_dir


def load_split(dir_path: Path) -> SamplingResult:
    dir_path = dir_path.expanduser().resolve()
    meta = json.loads((dir_path / "split.json").read_text(encoding="utf-8"))
    npz = np.load(dir_path / "arrays.npz", allow_pickle=False)

    indices: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for k in npz.files:
        if k.startswith("idx__"):
            indices[k[len("idx__") :]] = np.asarray(npz[k], dtype=np.int64)
        elif k.startswith("mask__"):
            masks[k[len("mask__") :]] = np.asarray(npz[k], dtype=bool)

    return SamplingResult(
        schema_version=int(meta["schema_version"]),
        created_at=str(meta["created_at"]),
        dataset_fingerprint=str(meta["dataset_fingerprint"]),
        split_fingerprint=str(meta["split_fingerprint"]),
        plan=meta["plan"],
        indices=indices,
        refs=meta.get("refs", {}),
        masks=masks,
        stats=meta.get("stats", {}),
    )
