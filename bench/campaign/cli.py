from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from .checkpoint import PLANNED_CONTINUATION_EXIT_CODE
from .dataset_lock import create_dataset_lock
from .dcl_diagnostics import evaluate_dcl_diagnostics
from .dcl_partition_selection import select_dcl_vote_partitions
from .errors import CampaignError, TaskLockedError
from .executor import execute_task
from .generate import generate_campaign
from .paper_acceptance import evaluate_paper_campaign
from .reconcile import reconcile_campaign
from .scientific_gates import ARTICLE10_METHODS, evaluate_gate, load_gate_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.campaign",
        description="Generate, execute, and reconcile immutable ModSSC campaigns",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="build an immutable task manifest")
    generate.add_argument("--spec", type=Path, required=True)
    generate.add_argument("--repo-root", type=Path, default=Path.cwd())
    generate.add_argument("--output-dir", type=Path, required=True)

    lock_datasets = subparsers.add_parser(
        "lock-datasets",
        help="hash offline datasets into a standardized lock or paper observations",
    )
    lock_datasets.add_argument("--spec", type=Path, required=True)
    lock_datasets.add_argument("--repo-root", type=Path, default=Path.cwd())
    lock_datasets.add_argument("--output", type=Path, required=True)
    lock_datasets.add_argument("--dataset-cache-dir", type=Path, default=None)
    lock_datasets.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing lock after explicit operator review",
    )

    run_task = subparsers.add_parser("run-task", help="execute one manifest task")
    run_task.add_argument("--manifest", type=Path, required=True)
    run_task.add_argument("--meta", type=Path, default=None)
    selector = run_task.add_mutually_exclusive_group(required=True)
    selector.add_argument("--index", type=int)
    selector.add_argument("--task-id")
    run_task.add_argument("--repo-root", type=Path, default=Path.cwd())
    run_task.add_argument("--result-root", type=Path, required=True)
    run_task.add_argument("--work-root", type=Path, required=True)
    run_task.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="persistent authenticated checkpoint root for cooperative task continuation",
    )
    run_task.add_argument("--site-id", required=True)
    run_task.add_argument("--environment-lock-sha256", default=None)
    run_task.add_argument("--environment-manifest", type=Path, default=None)
    run_task.add_argument("--preflight-report", type=Path, default=None)
    run_task.add_argument("--reclaim-stale-lock-after-hours", type=float, default=None)
    run_task.add_argument(
        "--scientific-gates",
        type=Path,
        default=None,
        help="gate registry; defaults to bench/campaigns/scientific-gates.yaml when present",
    )

    reconcile = subparsers.add_parser("reconcile", help="audit results and emit retries")
    reconcile.add_argument("--manifest", type=Path, required=True)
    reconcile.add_argument("--meta", type=Path, default=None)
    reconcile.add_argument("--result-root", type=Path, action="append", required=True)
    reconcile.add_argument("--output-dir", type=Path, required=True)
    reconcile.add_argument("--stale-after-hours", type=float, default=120.0)
    reconcile.add_argument("--no-retry", action="store_true")

    gates = subparsers.add_parser(
        "gate-status", help="check whether scientific conformity gates permit execution"
    )
    gates.add_argument("--registry", type=Path, required=True)
    gates.add_argument("--campaign-id", required=True)
    gates.add_argument("--track", choices=("paper", "standardized"), required=True)
    gates.add_argument("--method", choices=ARTICLE10_METHODS, default=None)

    acceptance = subparsers.add_parser(
        "evaluate-paper", help="evaluate completed paper protocols without inflating claims"
    )
    acceptance.add_argument("--manifest", type=Path, required=True)
    acceptance.add_argument("--meta", type=Path, default=None)
    acceptance.add_argument("--reconcile", type=Path, required=True)
    acceptance.add_argument("--acceptance", type=Path, required=True)
    acceptance.add_argument("--scientific-gates", type=Path, required=True)
    acceptance.add_argument("--output-dir", type=Path, required=True)

    dcl_diagnostics = subparsers.add_parser(
        "evaluate-dcl-diagnostics",
        help=(
            "evaluate DCL v2 control-integrity, numerical-equivalence, "
            "confidence, and dynamics gates without a paper claim"
        ),
    )
    dcl_diagnostics.add_argument("--manifest", type=Path, required=True)
    dcl_diagnostics.add_argument("--meta", type=Path, default=None)
    dcl_diagnostics.add_argument("--reconcile", type=Path, required=True)
    dcl_diagnostics.add_argument("--acceptance", type=Path, required=True)
    dcl_diagnostics.add_argument("--output-dir", type=Path, required=True)

    dcl_selection = subparsers.add_parser(
        "select-dcl-vote-partitions",
        help="lock the first 20 eligible DCL Vote screening partitions without test metrics",
    )
    dcl_selection.add_argument("--manifest", type=Path, required=True)
    dcl_selection.add_argument("--meta", type=Path, default=None)
    dcl_selection.add_argument("--reconcile", type=Path, required=True)
    dcl_selection.add_argument("--output", type=Path, required=True)
    dcl_selection.add_argument("--protocol-id", default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_campaign(
                args.spec,
                repo_root=args.repo_root,
                output_dir=args.output_dir,
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return 0
        if args.command == "lock-datasets":
            result = create_dataset_lock(
                args.spec,
                repo_root=args.repo_root,
                output_path=args.output,
                dataset_cache_dir=args.dataset_cache_dir,
                overwrite=args.overwrite,
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return 0
        if args.command == "run-task":
            if (
                args.reclaim_stale_lock_after_hours is not None
                and args.reclaim_stale_lock_after_hours <= 0
            ):
                raise CampaignError(
                    "E_CAMPAIGN_CLI", "--reclaim-stale-lock-after-hours must be greater than zero"
                )
            result = execute_task(
                args.manifest,
                meta_path=args.meta,
                repo_root=args.repo_root,
                result_root=args.result_root,
                work_root=args.work_root,
                site_id=args.site_id,
                index=args.index,
                task_id=args.task_id,
                environment_lock_sha256=args.environment_lock_sha256,
                environment_manifest_path=args.environment_manifest,
                preflight_report_path=args.preflight_report,
                reclaim_stale_lock_after=(
                    timedelta(hours=args.reclaim_stale_lock_after_hours)
                    if args.reclaim_stale_lock_after_hours is not None
                    else None
                ),
                gate_registry_path=args.scientific_gates,
                checkpoint_root=args.checkpoint_root,
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return PLANNED_CONTINUATION_EXIT_CODE if result.status == "continuation" else 0
        if args.command == "gate-status":
            registry = load_gate_registry(args.registry)
            methods = ARTICLE10_METHODS if args.method is None else (args.method,)
            decisions = [
                evaluate_gate(
                    registry,
                    campaign_id=args.campaign_id,
                    track=args.track,
                    method_id=method_id,
                )
                for method_id in methods
            ]
            allowed = all(decision.allowed for decision in decisions)
            print(
                json.dumps(
                    {
                        "allowed": allowed,
                        "registry_id": registry.registry_id,
                        "decisions": [decision.to_dict() for decision in decisions],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if allowed else 1
        if args.command == "evaluate-paper":
            result = evaluate_paper_campaign(
                args.manifest,
                meta_path=args.meta,
                reconcile_path=args.reconcile,
                acceptance_path=args.acceptance,
                gate_registry_path=args.scientific_gates,
                output_dir=args.output_dir,
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate-dcl-diagnostics":
            result = evaluate_dcl_diagnostics(
                args.manifest,
                meta_path=args.meta,
                reconcile_path=args.reconcile,
                acceptance_path=args.acceptance,
                output_dir=args.output_dir,
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return 0 if result.status == "passed" else 1
        if args.command == "select-dcl-vote-partitions":
            result = select_dcl_vote_partitions(
                args.manifest,
                meta_path=args.meta,
                reconcile_path=args.reconcile,
                output_path=args.output,
                protocol_id=args.protocol_id,
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return 0
        if args.command != "reconcile":
            raise CampaignError("E_CAMPAIGN_CLI", f"unsupported command: {args.command}")
        if args.stale_after_hours <= 0:
            raise CampaignError("E_CAMPAIGN_CLI", "--stale-after-hours must be greater than zero")
        result = reconcile_campaign(
            args.manifest,
            meta_path=args.meta,
            result_roots=args.result_root,
            output_dir=args.output_dir,
            stale_after=timedelta(hours=args.stale_after_hours),
            emit_retry=not args.no_retry,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        if result.status == "complete":
            return 0
        if result.status == "invalid":
            return 2
        return 1
    except TaskLockedError as exc:
        print(str(exc), file=sys.stderr)
        return 75
    except CampaignError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
