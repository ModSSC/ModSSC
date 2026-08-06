"""Fail when a release artifact leaks operational or audit-only material."""

from __future__ import annotations

import argparse
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

FORBIDDEN_PARTS = {
    "__pycache__",
    "hpc",
    "provenance",
    "slurm",
    "tools",
    "weka",
}
FORBIDDEN_SUFFIXES = {".class", ".jar", ".java", ".pyc", ".pyo"}
FORBIDDEN_FRAGMENTS = {
    "graphlearningold-04bece45/",
    "match_reference/sources/",
}
REQUIRED_PATHS = {
    "bench/assets/calder2020/protocol_inputs/graph/mnist-vae-knn30.npz",
    "bench/assets/cifar10_paper_splits/fixmatch-google-cifar10-250-seeds1-5.npz",
    "bench/assets/dataset-integrity-registry.json",
    "bench/campaigns/locks/dcl-vote-zhou-goldman-2004-v1/selected-partitions.json",
    "bench/reproduce.py",
}


def _normalized_names(names: Iterable[str], *, strip_root: bool) -> set[str]:
    normalized: set[str] = set()
    for name in names:
        parts = PurePosixPath(name).parts
        if strip_root and parts:
            parts = parts[1:]
        if parts:
            normalized.add(PurePosixPath(*parts).as_posix())
    return normalized


def _assert_safe(names: set[str], *, artifact: Path) -> None:
    forbidden: list[str] = []
    for name in names:
        path = PurePosixPath(name)
        lowered = name.lower()
        if (
            FORBIDDEN_PARTS.intersection(part.lower() for part in path.parts)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS)
        ):
            forbidden.append(name)
    if forbidden:
        raise AssertionError(f"forbidden entries in {artifact.name}: {sorted(forbidden)}")
    missing = REQUIRED_PATHS - names
    if missing:
        raise AssertionError(f"missing entries in {artifact.name}: {sorted(missing)}")


def audit_wheel(wheel: Path) -> None:
    with ZipFile(wheel) as archive:
        names = _normalized_names(archive.namelist(), strip_root=False)
        _assert_safe(names, artifact=wheel)
        entry_points = next(
            (name for name in names if name.endswith(".dist-info/entry_points.txt")),
            None,
        )
        if entry_points is None:
            raise AssertionError(f"missing entry_points.txt in {wheel.name}")
        scripts = archive.read(entry_points).decode("utf-8")
        if "modssc-reproduce = bench.reproduce:main" not in scripts:
            raise AssertionError(f"missing modssc-reproduce entry point in {wheel.name}")


def audit_sdist(sdist: Path) -> None:
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = _normalized_names(archive.getnames(), strip_root=True)
    _assert_safe(names, artifact=sdist)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    args = parser.parse_args()
    audit_wheel(args.wheel)
    audit_sdist(args.sdist)


if __name__ == "__main__":
    main()
