from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

BackendId = Literal["auto", "numpy", "sklearn", "torch", "faiss"]


@dataclass(frozen=True)
class ClassifierRuntime:
    """Runtime options that should not change the mathematical definition.

    Examples: random seed, number of CPU workers, device choice.
    """

    seed: int | None = 0
    n_jobs: int | None = None
    device: str | None = None  # ex: "cpu", "cuda", "mps"
    deterministic: bool = True


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    factory: str  # import path "module:callable"
    required_extra: str | None = None
    supports_gpu: bool = False
    notes: str = ""


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    description: str
    backends: dict[str, BackendSpec]
    preferred_backends: tuple[str, ...] = ("sklearn", "numpy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "preferred_backends": list(self.preferred_backends),
            "backends": {k: vars(v) for k, v in self.backends.items()},
        }
