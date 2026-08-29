"""Fail when a release artifact leaks operational or audit-only material."""

from __future__ import annotations

import argparse
import configparser
import io
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

FORBIDDEN_PARTS = {
    "__pycache__",
    "hpc",
    "modssc_cache",
    "provenance",
    "slurm",
    "tools",
    "uv.lock",
    "weka",
}
FORBIDDEN_SUFFIXES = {".class", ".jar", ".java", ".pyc", ".pyo"}
FORBIDDEN_FRAGMENTS = {
    "bench/assets/",
    "bench/campaign/",
    "bench/campaigns/",
    "bench/configs/reproductions/resources/",
    "graphlearningold-04bece45/",
    "match_reference/sources/",
    "tests/bench/test_hpc_",
    "tests/bench/test_match_continuation_controller.py",
    "tests/bench/test_public_hpc_portability.py",
}
REQUIRED_PATHS = {"bench/main.py"}


def _required_paths() -> set[str]:
    repo_root = Path(__file__).resolve().parents[2]
    cards_root = repo_root / "bench" / "configs" / "reproductions"
    cards = {
        path.relative_to(repo_root).as_posix()
        for path in cards_root.rglob("*.yaml")
        if path.is_file()
    }
    if not cards:
        raise AssertionError("no reproduction cards found in the source tree")
    return REQUIRED_PATHS | cards


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
    missing = _required_paths() - names
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
        parser = configparser.ConfigParser()
        parser.read_file(io.StringIO(scripts))
        console_scripts = dict(parser.items("console_scripts"))
        if console_scripts.get("modssc-bench") != "bench.main:main":
            raise AssertionError(f"missing modssc-bench entry point in {wheel.name}")
        benchmark_runners = {
            name: target for name, target in console_scripts.items() if target.startswith("bench.")
        }
        if benchmark_runners != {"modssc-bench": "bench.main:main"}:
            raise AssertionError(
                f"unexpected benchmark runners in {wheel.name}: {benchmark_runners}"
            )


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
