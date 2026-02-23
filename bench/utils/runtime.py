from __future__ import annotations

import platform
import subprocess
import sys
from importlib import import_module, metadata
from pathlib import Path
from typing import Any


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def _git_sha(repo_root: Path | None = None) -> str | None:
    if repo_root is None:
        repo_root = Path.cwd()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    sha = str(out).strip()
    return sha or None


def collect_runtime_versions(*, repo_root: Path | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "modssc": _pkg_version("modssc"),
        "numpy": _pkg_version("numpy"),
        "scikit_learn": _pkg_version("scikit-learn"),
        "torch": _pkg_version("torch"),
        "torch_geometric": _pkg_version("torch-geometric"),
        "git_sha": _git_sha(repo_root),
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
