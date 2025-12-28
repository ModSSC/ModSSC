from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArtifactStore:
    """A minimal container for named artifacts produced by preprocessing.

    Keys are flat strings. Recommended convention: dotted namespaces, for example:
    - raw.X, raw.y
    - features.X
    - tokens.input_ids
    - graph.edge_index
    """

    data: dict[str, Any] = field(default_factory=dict)

    def has(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def require(self, key: str) -> Any:
        if key not in self.data:
            raise KeyError(f"Missing required artifact: {key!r}")
        return self.data[key]

    def keys(self) -> list[str]:
        return sorted(self.data.keys())

    def copy(self) -> ArtifactStore:
        return ArtifactStore(dict(self.data))

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data
