from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.partition_selection_schema import (
    DCL_PARTITION_SELECTION_KIND,
    PARTITION_SELECTION_TASK_FIELDS,
)
from modssc.sampling.plan import SamplingPlan

from ..utils.hashing import hash_any
from .errors import CampaignError
from .manifest import sha256_file

DCL_METHOD_ID = "democratic_co_learning"
DCL_METHOD_PROFILE = "paper:zhou-goldman-2004-vote-table3"
DCL_DIAGNOSTIC_METHOD_PROFILE = "paper:zhou2004-vote-diagnostic-v2"
DCL_DATASET_ID = "vote"
DCL_PAPER_PROTOCOL_ID = "zhou-goldman-2004-vote-table3"
DCL_SCREENING_PROTOCOL_ID = "zhou-goldman-2004-vote-table3-partition-screening"
DCL_DIAGNOSTIC_CONTROL_PROTOCOLS = {
    "zhou-goldman-2004-vote-table3-control-naive-bayes-v2": "learner_0",
    "zhou-goldman-2004-vote-table3-control-c45-v2": "learner_1",
    "zhou-goldman-2004-vote-table3-control-3nn-v2": "learner_2",
    "zhou-goldman-2004-vote-table3-control-combining-only-v2": "combining_only",
}
DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS = {
    "zhou-goldman-2004-vote-table2-confidence-resub-wald-v2": (
        "training_accuracy",
        "wald",
    ),
    "zhou-goldman-2004-vote-table2-confidence-10fold-wald-v2": (
        "kfold_oof",
        "wald",
    ),
    "zhou-goldman-2004-vote-table2-confidence-10fold-wilson-v2": (
        "kfold_oof",
        "wilson",
    ),
    "zhou-goldman-2004-vote-table2-confidence-10fold-clopper-pearson-v2": (
        "kfold_oof",
        "clopper_pearson",
    ),
}
DCL_DIAGNOSTIC_PROTOCOL_IDS = frozenset(
    {*DCL_DIAGNOSTIC_CONTROL_PROTOCOLS, *DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS}
)
DCL_SELECTION_COUNT = 20
DCL_SELECTION_FILE_SHA256 = "5f586b2ab21bd6c2b0e058ab9d588ec1fc04b41b7d93e5a125d0a5f2ea1b36fb"
DCL_SELECTION_PAYLOAD_SHA256 = "69c7e8fb0b2f2066c53cc9ab3b33fdef7e88d0cd30cf640d20cf88be2407cc6d"
DCL_SELECTION_ARTIFACT_URI = (
    "evidence://modssc/historical/dcl-vote-zhou-goldman-2004-v1/selection/"
    f"{DCL_SELECTION_PAYLOAD_SHA256}"
)
DCL_SOURCE_URI = "evidence://historical/dcl-vote-zhou-goldman-2004-v1/raw-v1"
DCL_SOURCE_ARTIFACT_SHA256 = {
    "selected-partitions.json": "efa80d397d70dd6d9679d6414a99069a1ef7578a7a28ab865a65eccd9e075043",
    "source/manifest.jsonl": "08c2d658c8dd3ba821439bb3f2694dcb8ea46dec316c331fafb92e6d6b3be123",
    "source/manifest.meta.json": "7ced19045325778bb1db6121b582aada81cb315443e4adb151f854d4cdbd8a6e",
    "source/reconcile.json": "c13e2c65a353e5c530e748076c30cb671065312459dd3e665f0ef2e8ba3cf7a1",
}


def is_dcl_vote_partition_replay_identity(
    *,
    track: str,
    method_id: str,
    method_profile: str,
    dataset_id: str,
    protocol_id: str | None,
) -> bool:
    """Return whether a task may replay the immutable Vote v1 partitions."""

    if track != "paper" or method_id != DCL_METHOD_ID or dataset_id != DCL_DATASET_ID:
        return False
    if method_profile == DCL_METHOD_PROFILE:
        return protocol_id == DCL_PAPER_PROTOCOL_ID
    if method_profile == DCL_DIAGNOSTIC_METHOD_PROFILE:
        return protocol_id in DCL_DIAGNOSTIC_PROTOCOL_IDS
    return False


_SELECTION_FIELDS = {
    "artifact_uri",
    "candidate_count",
    "claim_eligible",
    "cutoff_seed",
    "dataset_id",
    "evaluated_candidate_count",
    "evaluated_candidates",
    "kind",
    "method_id",
    "method_profile",
    "protocol_id",
    "rejected",
    "rejected_count",
    "required_selection_count",
    "schema_version",
    "selected",
    "selected_count",
    "selection_rule",
    "provenance",
}
_ENTRY_FIELDS = {
    "converged",
    "decision",
    "evaluation_rank",
    "n_iter",
    "pseudo_labels_added_total",
    "run_json_sha256",
    "seed",
    "selection_rank",
    "split_arrays_sha256",
    "split_fingerprint",
    "split_json_sha256",
    "split_manifest_sha256",
    "task_id",
    "task_row_sha256",
}
_RULE_FIELDS = {
    "diagnostic_path",
    "operator",
    "order_by",
    "required_converged",
    "required_n_iter_lt",
    "test_information_used",
    "unresolved_prefix_policy",
    "value",
}
_PROVENANCE_FIELDS = {
    "artifact_sha256",
    "derivation",
    "environment_lock_sha256",
    "expected_dataset_content_sha256",
    "expected_dataset_fingerprint",
    "expected_git_diff_sha256",
    "expected_git_sha",
    "selection_payload_sha256",
    "source_uri",
    "verification_scope",
}


def _invalid(message: str) -> CampaignError:
    return CampaignError("E_CAMPAIGN_PARTITION_SELECTION_INVALID", message)


def _digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise _invalid(f"{field} must be a lowercase SHA-256 digest")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _invalid(f"{field} must be a positive integer")
    return int(value)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _invalid(f"cannot read {label}: {path}") from exc
    if not isinstance(raw, dict):
        raise _invalid(f"{label} must be a JSON object")
    return raw


def resolve_repo_path(repo_root: Path, value: str, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{label} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise _invalid(f"{label} must be a repository-relative path")
    root = repo_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise _invalid(f"{label} escapes the repository") from exc
    return resolved


@dataclass(frozen=True)
class DCLSelectedPartition:
    seed: int
    selection_rank: int
    evaluation_rank: int
    source_task_id: str
    source_task_row_sha256: str
    run_json_sha256: str
    split_fingerprint: str
    split_manifest_sha256: str
    split_json_sha256: str
    split_arrays_sha256: str


@dataclass(frozen=True)
class DCLPartitionSelectionLock:
    path: Path
    sha256: str
    artifact_uri: str
    claim_eligible: bool
    source_uri: str
    environment_lock_sha256: str
    dataset_fingerprint: str
    dataset_content_sha256: str
    source_artifact_sha256: dict[str, str]
    selected: tuple[DCLSelectedPartition, ...]

    def by_seed(self) -> dict[int, DCLSelectedPartition]:
        return {entry.seed: entry for entry in self.selected}


@dataclass(frozen=True)
class VerifiedDCLPartitionReplay:
    selection: DCLPartitionSelectionLock
    entry: DCLSelectedPartition
    replay_dir: Path
    manifest: dict[str, Any]
    split_metadata: dict[str, Any]


def load_dcl_partition_selection(
    path: Path,
    *,
    expected_sha256: str,
    expected_dataset_fingerprint: str | None = None,
    expected_dataset_content_sha256: str | None = None,
) -> DCLPartitionSelectionLock:
    """Load the public, non-claimable descriptor for the frozen DCL splits.

    The descriptor authenticates its selection rows and the public replay
    bytes.  It records, but deliberately cannot revalidate, the external raw
    screening bundle whose content digests are retained for private audit.
    """

    path = path.resolve()
    expected_digest = _digest(expected_sha256, field="partition_selection.sha256")
    if not path.is_file():
        raise _invalid(f"partition selection lock is missing: {path}")
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_MISMATCH",
            f"partition selection digest differs: expected {expected_digest}, got {actual_digest}",
        )
    raw = _read_json(path, label="partition selection lock")
    if set(raw) != _SELECTION_FIELDS:
        raise _invalid("partition selection lock has unexpected or missing top-level fields")
    expected_scalars: dict[str, Any] = {
        "schema_version": 2,
        "kind": DCL_PARTITION_SELECTION_KIND,
        "method_id": DCL_METHOD_ID,
        "method_profile": DCL_METHOD_PROFILE,
        "dataset_id": DCL_DATASET_ID,
        "protocol_id": DCL_SCREENING_PROTOCOL_ID,
        "artifact_uri": DCL_SELECTION_ARTIFACT_URI,
        "claim_eligible": False,
        "candidate_count": 100,
        "evaluated_candidate_count": DCL_SELECTION_COUNT,
        "selected_count": DCL_SELECTION_COUNT,
        "required_selection_count": DCL_SELECTION_COUNT,
        "rejected_count": 0,
        "cutoff_seed": DCL_SELECTION_COUNT,
    }
    for field, expected in expected_scalars.items():
        if raw.get(field) != expected:
            raise _invalid(f"partition selection {field} must equal {expected!r}")
    if raw.get("rejected") != []:
        raise _invalid("partition selection must not contain rejected evaluated candidates")

    rule = raw.get("selection_rule")
    if not isinstance(rule, Mapping) or set(rule) != _RULE_FIELDS:
        raise _invalid("partition selection rule is incomplete or has unexpected fields")
    expected_rule = {
        "diagnostic_path": "artifacts.method.diagnostics.pseudo_labels_added_total",
        "operator": "gt",
        "order_by": "manifest_seed_ascending",
        "required_converged": True,
        "required_n_iter_lt": 20,
        "test_information_used": False,
        "unresolved_prefix_policy": "fail_closed",
        "value": 0,
    }
    if dict(rule) != expected_rule:
        raise _invalid("partition selection rule differs from the preregistered fail-closed rule")

    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != _PROVENANCE_FIELDS:
        raise _invalid("partition selection provenance descriptor is incomplete")
    expected_provenance_scalars = {
        "derivation": "public-split-replay-descriptor-v1",
        "selection_payload_sha256": DCL_SELECTION_PAYLOAD_SHA256,
        "source_uri": DCL_SOURCE_URI,
        "verification_scope": "public-split-replay-only",
    }
    for field, expected in expected_provenance_scalars.items():
        if provenance.get(field) != expected:
            raise _invalid(f"partition selection provenance {field} must equal {expected!r}")
    environment_lock_sha256 = _digest(
        provenance.get("environment_lock_sha256"),
        field="provenance.environment_lock_sha256",
    )
    _digest(
        provenance.get("expected_git_diff_sha256"),
        field="provenance.expected_git_diff_sha256",
    )
    source_git_sha = _string(
        provenance.get("expected_git_sha"),
        field="provenance.expected_git_sha",
    )
    if len(source_git_sha) != 40 or any(char not in "0123456789abcdef" for char in source_git_sha):
        raise _invalid("provenance.expected_git_sha must be a lowercase Git SHA")
    artifact_sha256 = provenance.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or dict(artifact_sha256) != (
        DCL_SOURCE_ARTIFACT_SHA256
    ):
        raise _invalid("partition selection private source artifact digests differ")
    source_digests = {
        name: _digest(value, field=f"provenance.artifact_sha256.{name}")
        for name, value in artifact_sha256.items()
    }
    dataset_fingerprint = _digest(
        provenance.get("expected_dataset_fingerprint"),
        field="provenance.expected_dataset_fingerprint",
    )
    dataset_content_sha256 = _digest(
        provenance.get("expected_dataset_content_sha256"),
        field="provenance.expected_dataset_content_sha256",
    )
    if (
        expected_dataset_fingerprint is not None
        and dataset_fingerprint != expected_dataset_fingerprint
    ):
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_MISMATCH",
            "partition selection dataset fingerprint differs from the paper cell",
        )
    if (
        expected_dataset_content_sha256 is not None
        and dataset_content_sha256 != expected_dataset_content_sha256
    ):
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_MISMATCH",
            "partition selection dataset content digest differs from the paper cell",
        )

    selected_raw = raw.get("selected")
    evaluated_raw = raw.get("evaluated_candidates")
    if (
        not isinstance(selected_raw, list)
        or len(selected_raw) != DCL_SELECTION_COUNT
        or not isinstance(evaluated_raw, list)
        or evaluated_raw != selected_raw
    ):
        raise _invalid("selected and evaluated candidates must be the same ordered 20 rows")
    selection_payload_sha256 = hash_any({"selection_rule": dict(rule), "selected": selected_raw})
    if selection_payload_sha256 != DCL_SELECTION_PAYLOAD_SHA256:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_MISMATCH",
            "partition selection payload digest differs from its logical content address",
        )

    selected: list[DCLSelectedPartition] = []
    seen_seeds: set[int] = set()
    seen_task_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for rank, raw_entry in enumerate(selected_raw, start=1):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != _ENTRY_FIELDS:
            raise _invalid(f"selected[{rank - 1}] has unexpected or missing fields")
        seed = _positive_int(raw_entry.get("seed"), field=f"selected[{rank - 1}].seed")
        selection_rank = _positive_int(
            raw_entry.get("selection_rank"),
            field=f"selected[{rank - 1}].selection_rank",
        )
        evaluation_rank = _positive_int(
            raw_entry.get("evaluation_rank"),
            field=f"selected[{rank - 1}].evaluation_rank",
        )
        if seed != rank or selection_rank != rank or evaluation_rank != rank:
            raise _invalid(
                "selected rows must be ordered by unique seeds and contiguous ranks 1..20"
            )
        if (
            raw_entry.get("decision") != "accepted"
            or raw_entry.get("converged") is not True
            or _positive_int(
                raw_entry.get("pseudo_labels_added_total"),
                field=f"selected[{rank - 1}].pseudo_labels_added_total",
            )
            <= 0
        ):
            raise _invalid(f"selected row {rank} does not satisfy the acceptance diagnostic")
        n_iter = _positive_int(raw_entry.get("n_iter"), field=f"selected[{rank - 1}].n_iter")
        if n_iter >= 20:
            raise _invalid(f"selected row {rank} hit the safety iteration cap")

        source_task_id = _digest(raw_entry.get("task_id"), field=f"selected[{rank - 1}].task_id")
        source_task_row_sha256 = _digest(
            raw_entry.get("task_row_sha256"),
            field=f"selected[{rank - 1}].task_row_sha256",
        )
        run_json_sha256 = _digest(
            raw_entry.get("run_json_sha256"),
            field=f"selected[{rank - 1}].run_json_sha256",
        )
        split_fingerprint = _digest(
            raw_entry.get("split_fingerprint"),
            field=f"selected[{rank - 1}].split_fingerprint",
        )
        if (
            seed in seen_seeds
            or source_task_id in seen_task_ids
            or split_fingerprint in seen_fingerprints
        ):
            raise _invalid("selected seeds, source task ids, and split fingerprints must be unique")
        seen_seeds.add(seed)
        seen_task_ids.add(source_task_id)
        seen_fingerprints.add(split_fingerprint)
        selected.append(
            DCLSelectedPartition(
                seed=seed,
                selection_rank=selection_rank,
                evaluation_rank=evaluation_rank,
                source_task_id=source_task_id,
                source_task_row_sha256=source_task_row_sha256,
                run_json_sha256=run_json_sha256,
                split_fingerprint=split_fingerprint,
                split_manifest_sha256=_digest(
                    raw_entry.get("split_manifest_sha256"),
                    field=f"selected[{rank - 1}].split_manifest_sha256",
                ),
                split_json_sha256=_digest(
                    raw_entry.get("split_json_sha256"),
                    field=f"selected[{rank - 1}].split_json_sha256",
                ),
                split_arrays_sha256=_digest(
                    raw_entry.get("split_arrays_sha256"),
                    field=f"selected[{rank - 1}].split_arrays_sha256",
                ),
            )
        )

    return DCLPartitionSelectionLock(
        path=path,
        sha256=actual_digest,
        artifact_uri=DCL_SELECTION_ARTIFACT_URI,
        claim_eligible=False,
        source_uri=DCL_SOURCE_URI,
        environment_lock_sha256=environment_lock_sha256,
        dataset_fingerprint=dataset_fingerprint,
        dataset_content_sha256=dataset_content_sha256,
        source_artifact_sha256=source_digests,
        selected=tuple(selected),
    )


def build_task_partition_selection(
    *,
    selection_path: str,
    lock: DCLPartitionSelectionLock,
    entry: DCLSelectedPartition,
    replay_path: str,
) -> dict[str, Any]:
    return {
        "kind": DCL_PARTITION_SELECTION_KIND,
        "selection_path": selection_path,
        "selection_sha256": lock.sha256,
        "selection_rank": entry.selection_rank,
        "source_task_id": entry.source_task_id,
        "source_task_row_sha256": entry.source_task_row_sha256,
        "replay_path": replay_path,
        "split_fingerprint": entry.split_fingerprint,
        "split_manifest_sha256": entry.split_manifest_sha256,
        "split_json_sha256": entry.split_json_sha256,
        "split_arrays_sha256": entry.split_arrays_sha256,
    }


def verify_dcl_partition_replay(
    evidence: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_dataset_fingerprint: str,
    expected_plan: Mapping[str, Any],
) -> VerifiedDCLPartitionReplay:
    if set(evidence) != PARTITION_SELECTION_TASK_FIELDS:
        raise _invalid("partition replay evidence has unexpected or missing fields")
    if evidence.get("kind") != DCL_PARTITION_SELECTION_KIND:
        raise _invalid("partition replay kind differs")
    selection_path = Path(_string(evidence.get("selection_path"), field="selection_path"))
    replay_dir = Path(_string(evidence.get("replay_path"), field="replay_path"))
    lock = load_dcl_partition_selection(
        selection_path,
        expected_sha256=_digest(evidence.get("selection_sha256"), field="selection_sha256"),
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    rank = _positive_int(evidence.get("selection_rank"), field="selection_rank")
    matches = [entry for entry in lock.selected if entry.selection_rank == rank]
    if len(matches) != 1:
        raise _invalid(f"selection rank {rank} does not resolve to exactly one row")
    entry = matches[0]
    expected_fields = {
        "source_task_id": entry.source_task_id,
        "source_task_row_sha256": entry.source_task_row_sha256,
        "split_fingerprint": entry.split_fingerprint,
        "split_manifest_sha256": entry.split_manifest_sha256,
        "split_json_sha256": entry.split_json_sha256,
        "split_arrays_sha256": entry.split_arrays_sha256,
    }
    if entry.seed != expected_seed:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_MISMATCH",
            f"selected partition seed {entry.seed} differs from task seed {expected_seed}",
        )
    for field, expected in expected_fields.items():
        if evidence.get(field) != expected:
            raise CampaignError(
                "E_CAMPAIGN_PARTITION_SELECTION_MISMATCH",
                f"partition replay {field} differs from the signed selection row",
            )
    if not replay_dir.is_dir():
        raise _invalid(f"partition replay directory is missing: {replay_dir}")
    expected_names = {"MANIFEST.json", "split.json", "arrays.npz"}
    if {path.name for path in replay_dir.iterdir()} != expected_names:
        raise _invalid("partition replay directory must contain exactly the three signed files")
    paths = {
        "manifest": replay_dir / "MANIFEST.json",
        "split_json": replay_dir / "split.json",
        "split_arrays": replay_dir / "arrays.npz",
    }
    for label, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise _invalid(f"partition replay {label} is missing: {path}")
    actual_digests = {
        "split_manifest_sha256": sha256_file(paths["manifest"]),
        "split_json_sha256": sha256_file(paths["split_json"]),
        "split_arrays_sha256": sha256_file(paths["split_arrays"]),
    }
    for field, actual in actual_digests.items():
        if actual != expected_fields[field]:
            raise CampaignError(
                "E_CAMPAIGN_PARTITION_REPLAY_MISMATCH",
                f"partition replay {field} differs from the signed selection row",
            )

    manifest = _read_json(paths["manifest"], label="partition replay manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != "modssc.sampling.storage.v1"
        or manifest.get("dataset_fingerprint") != expected_dataset_fingerprint
        or manifest.get("split_fingerprint") != entry.split_fingerprint
    ):
        raise _invalid("partition replay manifest identity differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise _invalid("partition replay manifest files table is missing")
    for name, digest in (
        ("split.json", entry.split_json_sha256),
        ("arrays.npz", entry.split_arrays_sha256),
    ):
        record = files.get(name)
        if not isinstance(record, Mapping) or record.get("sha256") != digest:
            raise _invalid(f"partition replay manifest digest differs for {name}")

    split_metadata = _read_json(paths["split_json"], label="partition replay split metadata")
    if (
        split_metadata.get("schema_version") != 1
        or split_metadata.get("dataset_fingerprint") != expected_dataset_fingerprint
        or split_metadata.get("split_fingerprint") != entry.split_fingerprint
    ):
        raise _invalid("partition replay split identity differs")
    raw_plan = split_metadata.get("plan")
    if not isinstance(raw_plan, Mapping):
        raise _invalid("partition replay split plan is missing")
    try:
        replay_plan = SamplingPlan.from_dict(dict(raw_plan)).as_dict()
        configured_plan = SamplingPlan.from_dict(dict(expected_plan)).as_dict()
    except (TypeError, ValueError) as exc:
        raise _invalid("partition replay contains an invalid sampling plan") from exc
    if replay_plan != configured_plan:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_REPLAY_MISMATCH",
            "partition replay plan differs from the effective configuration",
        )
    historical_split_fingerprint = hash_any(
        {
            "schema_version": int(split_metadata["schema_version"]),
            "dataset_fingerprint": expected_dataset_fingerprint,
            # Keep the signed historical representation here.  Normalizing it
            # through SamplingPlan would erase explicit legacy defaults and
            # therefore change the already-published schema-v1 fingerprint.
            "plan": dict(raw_plan),
            "seed": int(expected_seed),
        }
    )
    if historical_split_fingerprint != entry.split_fingerprint:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_REPLAY_MISMATCH",
            "partition replay split fingerprint differs from its signed historical plan",
        )
    return VerifiedDCLPartitionReplay(
        selection=lock,
        entry=entry,
        replay_dir=replay_dir,
        manifest=manifest,
        split_metadata=split_metadata,
    )
