"""CLI smoke test.

Goal
- Run a few CLI commands via `python -m modssc ...`
- Helps validate entrypoints and discover available features.

Expected
- Runs after: pip install modssc
"""

from __future__ import annotations

import subprocess
import sys


def run_cli(*args: str) -> int:
    cmd = [sys.executable, "-m", "modssc", *args]
    res = subprocess.run(cmd, text=True)
    return int(res.returncode)


def main() -> None:
    rc = 0
    rc |= run_cli("--help")
    rc |= run_cli("doctor")
    rc |= run_cli("datasets", "list")
    rc |= run_cli("sampling", "--help")
    rc |= run_cli("preprocess", "steps", "list")
    rc |= run_cli("inductive", "methods", "list")
    rc |= run_cli("transductive", "methods", "list")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
