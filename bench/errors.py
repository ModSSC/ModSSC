from __future__ import annotations

import re
from dataclasses import dataclass

_CODE_RE = re.compile(r"^(E_[A-Z0-9_]+):")
_VALID_CODE_RE = re.compile(r"^E_[A-Z0-9_]+$")


@dataclass(frozen=True)
class BenchRuntimeError(RuntimeError):
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", str(self.code))
        object.__setattr__(self, "message", str(self.message))
        RuntimeError.__init__(self, f"{self.code}: {self.message}")


def extract_error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    # Native bricks own their failure taxonomy.  The runner records that code
    # verbatim instead of collapsing every non-bench failure to a generic one.
    if isinstance(code, str) and _VALID_CODE_RE.fullmatch(code):
        return code
    msg = str(exc)
    match = _CODE_RE.match(msg)
    if match:
        return str(match.group(1))
    return "E_BENCH_RUNTIME"
