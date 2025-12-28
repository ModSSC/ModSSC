from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils.hashing import derive_seed
from .utils.io import dump_yaml, write_json


@dataclass
class RunContext:
    name: str
    seed: int
    output_dir: Path
    run_dir: Path
    started_at: str
    fail_fast: bool = True
    config_path: Path | None = None

    @classmethod
    def from_run_config(
        cls,
        *,
        name: str,
        seed: int,
        output_dir: str | Path,
        config_path: Path | None,
        fail_fast: bool,
    ) -> RunContext:
        out_root = Path(output_dir).expanduser().resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = out_root / f"{name}-{timestamp}"
        started_at = datetime.now(timezone.utc).isoformat()
        return cls(
            name=name,
            seed=int(seed),
            output_dir=out_root,
            run_dir=run_dir,
            started_at=started_at,
            fail_fast=bool(fail_fast),
            config_path=config_path,
        )

    def seed_for(self, label: str, override: int | None = None) -> int:
        return int(override) if override is not None else int(derive_seed(self.seed, label))

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def write_config_copy(self, data: dict[str, Any]) -> None:
        dump_yaml(data, self.run_dir / "config.yaml")

    def write_json(self, name: str, data: Any) -> None:
        write_json(self.run_dir / name, data)
