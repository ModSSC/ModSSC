"""Command-line entry points for optional HPC operations.

These commands intentionally live outside :mod:`bench`: allocation accounting,
site profiles, and scheduler-facing reports are deployment concerns rather than
part of a scientific replication protocol.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from bench.campaign.errors import CampaignError

from .daily_report import generate_daily_report
from .preflight import run_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.hpc",
        description="Optional allocation and resource operations for ModSSC campaigns",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    daily = subparsers.add_parser(
        "daily-report", help="summarize observed resource consumption and retries"
    )
    daily.add_argument("--manifest", type=Path, required=True)
    daily.add_argument("--meta", type=Path, default=None)
    daily.add_argument("--reconcile", type=Path, required=True)
    daily.add_argument("--output-dir", type=Path, required=True)
    daily.add_argument("--resource-catalog", type=Path, default=None)
    daily.add_argument("--allocation", type=Path, default=None)
    daily.add_argument("--explained-oom-task-id", action="append", default=[])

    preflight = subparsers.add_parser(
        "preflight", help="verify runtime resources and an allocation reserve"
    )
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--allocation", type=Path, required=True)
    preflight.add_argument("--site", type=Path, action="append", required=True)
    preflight.add_argument("--repo-root", type=Path, default=Path.cwd())
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--runtime-estimates", type=Path, default=None)
    preflight.add_argument("--environment-lock-sha256", default=None)
    preflight.add_argument("--environment-manifest", type=Path, default=None)
    preflight.add_argument("--dataset-cache-dir", type=Path, default=None)
    preflight.add_argument("--model-cache-root", type=Path, default=None)
    preflight.add_argument("--max-allocation-age-hours", type=float, default=24.0)
    preflight.add_argument(
        "--require-architecture", choices=("CPU", "A100", "V100", "H100"), default=None
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "daily-report":
            result = generate_daily_report(
                args.manifest,
                meta_path=args.meta,
                reconcile_path=args.reconcile,
                output_dir=args.output_dir,
                resource_catalog_path=args.resource_catalog,
                allocation_path=args.allocation,
                explained_oom_task_ids=args.explained_oom_task_id,
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return 0
        result = run_preflight(
            args.manifest,
            allocation_path=args.allocation,
            site_paths=args.site,
            repo_root=args.repo_root,
            output_path=args.output,
            runtime_estimates_path=args.runtime_estimates,
            environment_lock_sha256=args.environment_lock_sha256,
            environment_manifest_path=args.environment_manifest,
            dataset_cache_dir=args.dataset_cache_dir,
            model_cache_root=args.model_cache_root,
            require_architecture=args.require_architecture,
            max_allocation_age_hours=args.max_allocation_age_hours,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0 if result.status == "pass" else 2
    except CampaignError as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
