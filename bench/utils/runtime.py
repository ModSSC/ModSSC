from __future__ import annotations

import hashlib
import os
import platform
import stat
import subprocess
import sys
from functools import cache
from importlib import import_module, metadata
from pathlib import Path
from typing import Any


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


@cache
def _distribution_fingerprint(name: str) -> str | None:
    """Hash the installed ModSSC and bench payload independently of Git."""

    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    files = distribution.files
    if files is None:
        return None
    selected = [
        entry
        for entry in files
        if str(entry).split("/", 1)[0] in {"bench", "modssc"}
        and "__pycache__" not in entry.parts
        and entry.suffix != ".pyc"
    ]
    if not selected:
        return None
    hasher = hashlib.sha256(b"modssc-installed-payload-v1\0")
    for entry in sorted(selected, key=str):
        path = Path(distribution.locate_file(entry))
        if not path.is_file():
            return None
        content = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                content.update(chunk)
        _update_framed(hasher, b"path", str(entry).encode("utf-8"))
        _update_framed(hasher, b"content_sha256", content.digest())
    return hasher.hexdigest()


def _git_output(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(repo_root),
        stderr=subprocess.DEVNULL,
    )


def _git_repo_root(repo_root: Path | None = None) -> Path | None:
    if repo_root is None:
        repo_root = Path.cwd()
    try:
        out = _git_output(repo_root, "rev-parse", "--show-toplevel")
    except Exception:
        return None
    root = os.fsdecode(out.rstrip(b"\r\n"))
    return Path(root) if root else None


def _git_sha(repo_root: Path | None = None) -> str | None:
    root = _git_repo_root(repo_root)
    if root is None:
        return None
    try:
        out = _git_output(root, "rev-parse", "HEAD")
    except Exception:
        return None
    sha = out.decode("ascii", errors="replace").strip()
    return sha or None


def _update_framed(hasher: Any, label: bytes, payload: bytes) -> None:
    hasher.update(len(label).to_bytes(4, "big"))
    hasher.update(label)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _untracked_file_fingerprint(repo_root: Path, relative_path: bytes) -> bytes:
    path = repo_root / os.fsdecode(relative_path)
    file_stat = path.lstat()

    if stat.S_ISLNK(file_stat.st_mode):
        kind = b"symlink"
        content_hash = hashlib.sha256(os.fsencode(os.readlink(path))).digest()
    elif stat.S_ISREG(file_stat.st_mode):
        kind = b"executable" if file_stat.st_mode & stat.S_IXUSR else b"regular"
        content_hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                content_hasher.update(chunk)
        content_hash = content_hasher.digest()
    else:
        kind = f"special:{stat.S_IFMT(file_stat.st_mode):o}".encode("ascii")
        content_hash = b""

    hasher = hashlib.sha256()
    _update_framed(hasher, b"path", relative_path)
    _update_framed(hasher, b"kind", kind)
    _update_framed(hasher, b"content_sha256", content_hash)
    return hasher.digest()


def _git_provenance(
    repo_root: Path | None = None,
) -> tuple[str | None, bool | None, str | None]:
    """Return commit, dirty state, and an opaque worktree-diff fingerprint.

    The fingerprint covers staged and unstaged tracked diffs plus the paths, Git-like
    modes, and content hashes of non-ignored untracked files. Only the final digest is
    returned; file names and diff contents are never persisted in the run report.
    """

    root = _git_repo_root(repo_root)
    if root is None:
        return None, None, None

    try:
        sha_output = _git_output(root, "rev-parse", "HEAD")
        sha = sha_output.decode("ascii", errors="replace").strip() or None
    except Exception:
        # An initialized repository without a first commit still has useful worktree
        # provenance, even though it has no HEAD SHA yet.
        sha = None

    try:
        status = _git_output(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        staged_diff = _git_output(
            root,
            "diff",
            "--cached",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=none",
            "--",
        )
        unstaged_diff = _git_output(
            root,
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=none",
            "--",
        )
        untracked = _git_output(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )

        hasher = hashlib.sha256(b"modssc-git-worktree-v1\0")
        _update_framed(hasher, b"status", status)
        _update_framed(hasher, b"staged_diff", staged_diff)
        _update_framed(hasher, b"unstaged_diff", unstaged_diff)
        for relative_path in sorted(path for path in untracked.split(b"\0") if path):
            _update_framed(
                hasher,
                b"untracked_file",
                _untracked_file_fingerprint(root, relative_path),
            )
    except Exception:
        # Git metadata or a file can change while provenance is being collected. A
        # nullable pair reports that provenance was unavailable without blocking a run.
        return sha, None, None

    return sha, bool(status), hasher.hexdigest()


def collect_runtime_versions(*, repo_root: Path | None = None) -> dict[str, Any]:
    git_sha, git_dirty, git_diff_sha256 = _git_provenance(repo_root)
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "modssc": _pkg_version("modssc"),
        "distribution_sha256": _distribution_fingerprint("modssc"),
        "numpy": _pkg_version("numpy"),
        "scikit_learn": _pkg_version("scikit-learn"),
        "torch": _pkg_version("torch"),
        "torch_geometric": _pkg_version("torch-geometric"),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256,
        "executable": sys.executable,
    }

    try:
        torch = import_module("torch")
    except Exception:
        out["cuda"] = None
        out["cudnn"] = None
        return out

    out["cuda"] = getattr(torch.version, "cuda", None)
    try:
        cudnn = getattr(getattr(torch.backends, "cudnn", None), "version", None)
        out["cudnn"] = int(cudnn()) if callable(cudnn) else None
    except Exception:
        out["cudnn"] = None
    return out
