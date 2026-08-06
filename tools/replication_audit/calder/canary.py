from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bench.campaign.errors import CampaignError
from bench.campaign.executor import validate_result_directory
from bench.campaign.manifest import load_manifest
from bench.campaign.protocols.calder.official import (
    OFFICIAL_COMMIT,
    OFFICIAL_KNN_SHA256,
    OFFICIAL_LABELS_SHA256,
    OFFICIAL_PERMUTATIONS_SHA256,
    OFFICIAL_REPOSITORY,
    OFFICIAL_RESULTS_SHA256,
    OFFICIAL_SOURCE_SHA256,
    PERMUTATIONS_ARTIFACT_SHA256,
    CalderOfficialArtifactError,
    verify_calder_official_assets,
)
from tools.replication_audit.calder.artifacts import (
    EFFECTIVE_CONFIG_KIND,
    CalderArtifactError,
    verify_calder_artifact_lock,
    write_immutable_json,
)
from tools.replication_audit.calder.campaigns import (
    CANARY_CAMPAIGN_ID,
    PRODUCTION_CAMPAIGN_ID,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_CONFIG_SHA256 as _SOURCE_REPLAY_CONFIG_SHA256,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_HISTORY_DESCRIPTOR_RELATIVE as _SOURCE_REPLAY_HISTORY_DESCRIPTOR_RELATIVE,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_HISTORY_DESCRIPTOR_SHA256 as _SOURCE_REPLAY_HISTORY_DESCRIPTOR_SHA256,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_KIND as _SOURCE_REPLAY_KIND,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_LABELED_INDICES_SHA256 as _SOURCE_REPLAY_LABELED_INDICES_SHA256,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_MODSSC_GIT_SHA as _SOURCE_REPLAY_MODSSC_GIT_SHA,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_MODSSC_MODULE as _SOURCE_REPLAY_MODSSC_MODULE,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_OFFICIAL_PATH as _SOURCE_REPLAY_OFFICIAL_PATH,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_ORACLE_RELATIVE as _SOURCE_REPLAY_ORACLE_RELATIVE,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_ORACLE_SHA256 as _SOURCE_REPLAY_ORACLE_SHA256,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_PERMUTATION_ROW_SHA256 as _SOURCE_REPLAY_PERMUTATION_ROW_SHA256,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_PREDICTION_SHA256 as _SOURCE_REPLAY_PREDICTION_SHA256,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_SCORE_SHA256 as _SOURCE_REPLAY_SCORE_SHA256,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_SPLIT_FINGERPRINT as _SOURCE_REPLAY_SPLIT_FINGERPRINT,
)

ACCEPTANCE_KIND = "modssc.calder2020-mnist-table1-canary-acceptance"
ACCEPTANCE_SCHEMA_VERSION = 3
_EXPECTED_IDENTITIES = {
    ("laplace_learning", 1),
    ("laplace_learning", 5),
    ("poisson_learning", 1),
    ("poisson_learning", 5),
}
_EXPECTED_PRODUCTION_IDENTITIES = {
    (method, budget)
    for method in ("laplace_learning", "poisson_learning")
    for budget in range(1, 6)
}
_SHA256_LENGTH = 64
_ARCHIVE_TOLERANCE_PERCENT = 0.0050000001
_ARCHIVE_PRECISION_DECIMALS = 2
_SCOPED_NODE_ALLOWANCE = {("laplace_learning", 5, 0): 1}
_PREPARED_PERMUTATIONS_SHA256 = "8740039403c6e287e24f0cb0a9013011c9ffc552dedc06ae6bd2ab00b3af1fb3"


class CalderCanaryError(RuntimeError):
    """Raised when Calder canary evidence is invalid or cannot authorize production."""


@dataclass(frozen=True)
class CalderCanaryReport:
    status: str
    output_path: str
    campaign_id: str
    comparison_count: int
    passed_count: int
    acceptance_sha256: str


@dataclass(frozen=True)
class _ProductionEvidence:
    spec_sha256: str
    artifact_lock_sha256: str
    effective_manifest_path: str
    effective_manifest_sha256: str
    effective_manifest_lock_sha256: str
    effective_config_sha256: dict[str, str]
    source_replay_oracle_path: str
    source_replay_oracle_sha256: str
    official: dict[str, str]
    dataset: dict[str, str]
    graph: dict[str, str]

    def audit_payload(self) -> dict[str, Any]:
        return {
            "production_spec_sha256": self.spec_sha256,
            "artifact_lock_sha256": self.artifact_lock_sha256,
            "effective_manifest": {
                "path": self.effective_manifest_path,
                "sha256": self.effective_manifest_sha256,
                "lock_sha256": self.effective_manifest_lock_sha256,
            },
            "effective_configs": dict(self.effective_config_sha256),
            "source_replay_oracle": {
                "path": self.source_replay_oracle_path,
                "sha256": self.source_replay_oracle_sha256,
            },
            "official": dict(self.official),
            "dataset": dict(self.dataset),
            "graph": dict(self.graph),
        }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalderCanaryError(f"{label} must be a mapping")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CalderCanaryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise CalderCanaryError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        if resolved.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        else:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CalderCanaryError(f"cannot read {label}: {candidate}") from exc
    if not isinstance(raw, dict):
        raise CalderCanaryError(f"{label} root must be a mapping")
    return raw


def _repo_generated_file(repo_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise CalderCanaryError(f"{label} must be a repository-relative path")
    candidate = repo_root / value
    generated_hint = (repo_root / "bench" / "generated").resolve(strict=False)
    try:
        generated = (repo_root / "bench" / "generated").resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(generated)
    except (OSError, ValueError) as exc:
        raise CalderCanaryError(f"{label} must be below {generated_hint}") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise CalderCanaryError(f"{label} must be a regular non-symlink file")
    return resolved


def _budget_from_task(task: Any) -> int:
    protocol = task.protocol_id
    if not isinstance(protocol, str):
        raise CalderCanaryError(f"canary task has no protocol_id: {task.task_id}")
    prefix = f"calder-2020-mnist-table1-{task.method_id.removesuffix('_learning')}-"
    suffix = "-label-per-class"
    if not protocol.startswith(prefix) or not protocol.endswith(suffix):
        raise CalderCanaryError(f"canary protocol identity differs: {task.task_id}")
    raw_budget = protocol[len(prefix) : -len(suffix)]
    try:
        budget = int(raw_budget)
    except ValueError as exc:
        raise CalderCanaryError(f"canary budget is invalid: {task.task_id}") from exc
    return budget


def _validate_canary_tasks(
    *,
    repo_root: Path,
    meta: Mapping[str, Any],
    tasks: Sequence[Any],
) -> dict[tuple[str, int], Any]:
    if (
        meta.get("campaign_id") != CANARY_CAMPAIGN_ID
        or meta.get("task_count") != 4
        or len(tasks) != 4
    ):
        raise CalderCanaryError("canary manifest must contain exactly the four registered tasks")
    by_identity: dict[tuple[str, int], Any] = {}
    for task in tasks:
        budget = _budget_from_task(task)
        identity = (task.method_id, budget)
        config = repo_root / task.config_path
        try:
            config_resolved = config.resolve(strict=True)
            config_resolved.relative_to((repo_root / "bench" / "generated").resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise CalderCanaryError(
                f"canary effective config is outside bench/generated: {task.task_id}"
            ) from exc
        if config.is_symlink() or _sha256_file(config_resolved) != task.source_config_sha256:
            raise CalderCanaryError(f"canary effective config SHA-256 differs: {task.task_id}")
        if (
            identity in by_identity
            or task.campaign_id != CANARY_CAMPAIGN_ID
            or task.track != "paper"
            or task.seed != 0
            or task.required_seed_count != 1
            or task.dataset_id != "mnist"
            or task.assigned_site != "local-cpu"
            or task.resource_profile != "cpu_graph"
            or task.fidelity_status != "not_claimable"
            or task.label_budget != f"per_class:{budget}"
        ):
            raise CalderCanaryError(f"canary task contract differs: {task.task_id}")
        by_identity[identity] = task
    if set(by_identity) != _EXPECTED_IDENTITIES:
        raise CalderCanaryError("canary method/budget identities differ")
    if len({task.expected_git_sha for task in tasks}) != 1:
        raise CalderCanaryError("canary manifest mixes Git commits")
    if len({task.expected_git_diff_sha256 for task in tasks}) != 1:
        raise CalderCanaryError("canary manifest mixes worktree identities")
    if len({task.environment_lock_sha256 for task in tasks}) != 1:
        raise CalderCanaryError("canary manifest mixes environments")
    if len({task.expected_dataset_fingerprint for task in tasks}) != 1:
        raise CalderCanaryError("canary manifest mixes dataset fingerprints")
    if len({task.expected_dataset_content_sha256 for task in tasks}) != 1:
        raise CalderCanaryError("canary manifest mixes dataset content digests")
    return by_identity


def _archive_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {1: [], 5: []}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CalderCanaryError(f"cannot read archived GraphLearning results: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        columns = line.split(",")
        if len(columns) != 2:
            continue
        try:
            total_labels = int(columns[0])
            value = float(columns[1])
        except ValueError:
            continue
        if total_labels in {10, 50} and math.isfinite(value):
            rows[total_labels // 10].append({"line_number": line_number, "accuracy_percent": value})
    if any(len(values) != 100 for values in rows.values()):
        raise CalderCanaryError(
            f"archived GraphLearning results must contain 100 rows for budgets 1 and 5: {path}"
        )
    return rows


def _diagnostics_pass(method_id: str, payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    artifacts = _mapping(payload.get("artifacts"), label="run artifacts")
    method = _mapping(artifacts.get("method"), label="run method artifacts")
    diagnostics = _mapping(method.get("diagnostics"), label="run method diagnostics")
    failures: list[str] = []
    if diagnostics.get("converged") is not True:
        failures.append("converged")
    iterations = diagnostics.get("iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        failures.append("iterations")
    if method_id == "laplace_learning":
        if diagnostics.get("solver") != "calder2020_conjugate_gradient":
            failures.append("solver")
        residual = diagnostics.get("absolute_residual")
        if (
            isinstance(residual, bool)
            or not isinstance(residual, int | float)
            or not math.isfinite(float(residual))
            or not 0 <= float(residual) <= 1e-5
        ):
            failures.append("absolute_residual")
    else:
        if diagnostics.get("solver") != "paper_iteration":
            failures.append("solver")
        if diagnostics.get("decision_rule") != "paper_class_prior_correction":
            failures.append("decision_rule")
        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or not 50 <= iterations <= 1000
        ):
            failures.append("iterations_range")
        residual = diagnostics.get("mixing_residual")
        if (
            isinstance(residual, bool)
            or not isinstance(residual, int | float)
            or not math.isfinite(float(residual))
            or not 0 <= float(residual) <= 1 / 70_000
        ):
            failures.append("mixing_residual")
    return not failures, failures


def _metric(payload: Mapping[str, Any]) -> float:
    metrics = _mapping(payload.get("metrics"), label="run metrics")
    unlabeled = _mapping(metrics.get("unlabeled"), label="run unlabeled metrics")
    value = unlabeled.get("accuracy")
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise CalderCanaryError("run unlabeled accuracy must be finite and between zero and one")
    return float(value)


def _unlabeled_count(payload: Mapping[str, Any], *, budget: int) -> int:
    artifacts = _mapping(payload.get("artifacts"), label="run artifacts")
    sampling = _mapping(artifacts.get("sampling"), label="run sampling artifacts")
    stats = _mapping(sampling.get("stats"), label="run sampling statistics")
    unlabeled = _mapping(
        stats.get("train_unlabeled"),
        label="run unlabeled sampling statistics",
    )
    count = unlabeled.get("n")
    expected = 70_000 - budget * 10
    if isinstance(count, bool) or not isinstance(count, int) or count != expected:
        raise CalderCanaryError(
            f"run unlabeled sample count must equal the frozen protocol value {expected}"
        )
    return count


def _correct_count(accuracy: float, *, unlabeled_count: int) -> int:
    raw_count = accuracy * unlabeled_count
    count = round(raw_count)
    if not math.isclose(raw_count, count, rel_tol=0, abs_tol=1e-8):
        raise CalderCanaryError("run unlabeled accuracy is not a discrete correct-count proportion")
    return count


def _archive_compatible_correct_counts(
    accuracy_percent: float,
    *,
    unlabeled_count: int,
) -> list[int]:
    archived_text = f"{accuracy_percent:.{_ARCHIVE_PRECISION_DECIMALS}f}"
    center = accuracy_percent * unlabeled_count / 100
    rounding_radius = 0.01 * unlabeled_count / 100
    lower = max(0, math.floor(center - rounding_radius) - 2)
    upper = min(unlabeled_count, math.ceil(center + rounding_radius) + 2)
    compatible = [
        count
        for count in range(lower, upper + 1)
        if f"{100 * count / unlabeled_count:.{_ARCHIVE_PRECISION_DECIMALS}f}" == archived_text
    ]
    if not compatible:
        raise CalderCanaryError("archived GraphLearning accuracy has no compatible discrete count")
    return compatible


def _load_source_replay_oracle(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve(strict=True)
    path = (root / _SOURCE_REPLAY_ORACLE_RELATIVE).resolve(strict=True)
    if not path.is_relative_to(root):
        raise CalderCanaryError("Calder source-replay oracle escapes the repository")
    if _sha256_file(path) != _SOURCE_REPLAY_ORACLE_SHA256:
        raise CalderCanaryError("Calder source-replay oracle SHA-256 differs")
    oracle = dict(_read_mapping(path, label="Calder source-replay oracle"))
    oracle_sha256 = _require_sha256(
        oracle.get("oracle_sha256"),
        label="Calder source-replay oracle seal",
    )
    unsigned_oracle = dict(oracle)
    unsigned_oracle.pop("oracle_sha256")
    if _canonical_sha256(unsigned_oracle) != oracle_sha256:
        raise CalderCanaryError("Calder source-replay oracle seal differs")
    identity = _mapping(oracle.get("identity"), label="source-replay identity")
    bindings = _mapping(oracle.get("bindings"), label="source-replay bindings")
    archive = _mapping(oracle.get("archive"), label="source-replay archive evidence")
    replay = _mapping(oracle.get("replay"), label="source-replay evidence")
    protocol = _mapping(oracle.get("protocol"), label="source-replay protocol")
    prediction_sha256 = _mapping(
        replay.get("prediction_sha256"),
        label="source-replay prediction SHA-256 evidence",
    )
    iterations = _mapping(
        replay.get("iterations"),
        label="source-replay iteration evidence",
    )
    residual = _mapping(replay.get("residual"), label="source-replay residual evidence")
    score_sha256 = _mapping(
        replay.get("score_sha256"),
        label="source-replay score SHA-256 evidence",
    )
    official_source = _mapping(
        bindings.get("official_source"),
        label="source-replay official source",
    )
    modssc_source = _mapping(
        bindings.get("modssc_source"),
        label="source-replay ModSSC source",
    )
    if (
        oracle.get("schema_version") != 1
        or oracle.get("kind") != _SOURCE_REPLAY_KIND
        or identity
        != {
            "method_id": "laplace_learning",
            "budget_per_class": 5,
            "permutation": 0,
            "fixed_permutation_row": 4,
        }
        or bindings.get("official_commit") != OFFICIAL_COMMIT
        or bindings.get("official_graph_sha256") != OFFICIAL_KNN_SHA256
        or bindings.get("official_permutations_sha256") != OFFICIAL_PERMUTATIONS_SHA256
        or bindings.get("labels_sha256") != OFFICIAL_LABELS_SHA256
        or bindings.get("effective_config_sha256") != _SOURCE_REPLAY_CONFIG_SHA256
        or bindings.get("labeled_indices_permutation_order_sha256")
        != _SOURCE_REPLAY_PERMUTATION_ROW_SHA256
        or bindings.get("labeled_indices_sorted_sha256") != _SOURCE_REPLAY_LABELED_INDICES_SHA256
        or bindings.get("split_fingerprint") != _SOURCE_REPLAY_SPLIT_FINGERPRINT
        or archive.get("results_sha256") != OFFICIAL_RESULTS_SHA256["laplace_learning"]
        or official_source
        != {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "upstream_path": _SOURCE_REPLAY_OFFICIAL_PATH,
            "sha256": OFFICIAL_SOURCE_SHA256,
        }
        or modssc_source.get("module") != _SOURCE_REPLAY_MODSSC_MODULE
        or replay.get("unlabeled_count") != 69_950
        or replay.get("correct_count") != 48_269
        or not math.isclose(
            float(replay.get("accuracy", math.nan)),
            48_269 / 69_950,
            rel_tol=0,
            abs_tol=0,
        )
        or replay.get("prediction_count") != 70_000
        or replay.get("prediction_shape") != [70_000]
        or replay.get("prediction_byte_count") != 560_000
        or replay.get("prediction_encoding") != "numpy-int64-little-endian-c-order"
        or prediction_sha256
        != {
            "official_source": _SOURCE_REPLAY_PREDICTION_SHA256,
            "modssc": _SOURCE_REPLAY_PREDICTION_SHA256,
        }
        or replay.get("differing_predictions") != 0
        or replay.get("score_shape") != [70_000, 10]
        or replay.get("score_byte_count") != 5_600_000
        or replay.get("score_encoding") != "numpy-float64-little-endian-c-order"
        or score_sha256
        != {
            "official_source": _SOURCE_REPLAY_SCORE_SHA256,
            "modssc": _SOURCE_REPLAY_SCORE_SHA256,
        }
        or not math.isclose(
            float(replay.get("max_absolute_score_delta", math.nan)),
            0.0,
            rel_tol=0,
            abs_tol=0,
        )
        or iterations != {"official_source": 148, "modssc": 148}
        or residual
        != {
            "official_system": 9.842962973879334e-06,
            "modssc_recursive": 9.842962973957787e-06,
        }
        or protocol
        != {
            "archive_format": "%.2f",
            "cg_max_iter": 100_000,
            "cg_tol": 1e-05,
            "labeled_count": 50,
            "n_classes": 10,
            "n_nodes": 70_000,
            "solver": "calder2020_conjugate_gradient",
            "unlabeled_count": 69_950,
        }
        or archive.get("accuracy_percent_text") != "69.00"
        or archive.get("compatible_correct_count") != [48_263, 48_268]
        or archive.get("node_delta") != 1
    ):
        raise CalderCanaryError("Calder source-replay oracle contents differ")
    _require_sha256(
        official_source.get("sha256"),
        label=f"source-replay source SHA-256 for {official_source['upstream_path']}",
    )

    # This oracle authenticates the ModSSC implementation at its immutable
    # execution commit. Comparing it with today's implementation would make
    # honest historical evidence fail whenever profile dispatch evolves.
    _require_sha256(
        modssc_source.get("sha256"),
        label=f"source-replay source SHA-256 for {modssc_source['module']}",
    )
    history_path = (root / _SOURCE_REPLAY_HISTORY_DESCRIPTOR_RELATIVE).resolve(strict=True)
    if not history_path.is_relative_to(root):
        raise CalderCanaryError("Calder source history descriptor escapes the repository")
    if _sha256_file(history_path) != _SOURCE_REPLAY_HISTORY_DESCRIPTOR_SHA256:
        raise CalderCanaryError("Calder source history descriptor SHA-256 differs")
    history = _read_mapping(history_path, label="Calder source history descriptor")
    execution_commits = history.get("execution_commits")
    if (
        history.get("complete_committed_history") is not True
        or not isinstance(execution_commits, list)
        or _SOURCE_REPLAY_MODSSC_GIT_SHA not in execution_commits
    ):
        raise CalderCanaryError("Calder execution commit is absent from source history")
    for field in (
        "dataset_content_sha256",
        "dataset_fingerprint",
        "environment_lock_sha256",
        "graph_fingerprint",
        "labeled_indices_permutation_order_sha256",
        "labeled_indices_sorted_sha256",
        "split_fingerprint",
        "effective_config_sha256",
    ):
        _require_sha256(bindings.get(field), label=f"source-replay {field}")
    preprocess_fingerprint = bindings.get("preprocess_fingerprint")
    if not isinstance(preprocess_fingerprint, str) or not preprocess_fingerprint:
        raise CalderCanaryError("Calder source-replay preprocess fingerprint is invalid")
    permutations_path = (
        root / "bench/assets/calder2020/protocol_inputs/splits/"
        "mnist-table1-permutations.ragged-int64-v1.npz"
    ).resolve(strict=True)
    if _sha256_file(permutations_path) != PERMUTATIONS_ARTIFACT_SHA256:
        raise CalderCanaryError("Calder safe permutation artifact SHA-256 differs")
    try:
        with np.load(permutations_path, allow_pickle=False) as archive_data:
            offsets = np.asarray(archive_data["offsets"], dtype=np.int64)
            values = np.asarray(archive_data["values"], dtype=np.int64)
            row = np.ascontiguousarray(values[offsets[4] : offsets[5]], dtype="<i8")
    except (KeyError, OSError, ValueError) as exc:
        raise CalderCanaryError("cannot replay the Calder labeled-index evidence") from exc
    if (
        row.shape != (50,)
        or hashlib.sha256(row.tobytes(order="C")).hexdigest()
        != _SOURCE_REPLAY_PERMUTATION_ROW_SHA256
        or hashlib.sha256(np.sort(row).tobytes(order="C")).hexdigest()
        != _SOURCE_REPLAY_LABELED_INDICES_SHA256
    ):
        raise CalderCanaryError("Calder labeled-index replay evidence differs")
    return oracle


def _validate_source_replay_campaign_binding(
    oracle: Mapping[str, Any],
    *,
    environment_lock_sha256: str,
    dataset_fingerprint: str,
    dataset_content_sha256: str,
    production: _ProductionEvidence,
) -> None:
    bindings = _mapping(oracle.get("bindings"), label="source-replay bindings")
    if (
        bindings.get("environment_lock_sha256") != environment_lock_sha256
        or bindings.get("dataset_fingerprint") != dataset_fingerprint
        or bindings.get("dataset_content_sha256") != dataset_content_sha256
        or bindings.get("dataset_fingerprint") != production.dataset["prepared_fingerprint"]
        or bindings.get("dataset_content_sha256") != production.dataset["content_sha256"]
        or bindings.get("graph_fingerprint") != production.graph["fingerprint"]
        or bindings.get("preprocess_fingerprint") != production.graph["preprocess_fingerprint"]
        or bindings.get("official_commit") != production.official["commit"]
        or bindings.get("labels_sha256") != production.official["labels_sha256"]
        or bindings.get("official_permutations_sha256")
        != production.official["permutations_sha256"]
        or bindings.get("official_graph_sha256") != production.official["knn_sha256"]
    ):
        raise CalderCanaryError("Calder source-replay oracle is not bound to this campaign")


def _source_replay_exception_evidence(
    *,
    oracle: Mapping[str, Any],
    payload: Mapping[str, Any],
    run_path: Path,
    task: Any,
    accuracy: float,
    unlabeled_count: int,
    correct_count: int,
    matching_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    replay = _mapping(oracle.get("replay"), label="source-replay evidence")
    archive = _mapping(oracle.get("archive"), label="source-replay archive evidence")
    artifacts = _mapping(payload.get("artifacts"), label="run artifacts")
    method = _mapping(artifacts.get("method"), label="run method artifacts")
    diagnostics = _mapping(method.get("diagnostics"), label="run method diagnostics")
    sampling = _mapping(artifacts.get("sampling"), label="run sampling artifacts")
    sampling_replay = _mapping(
        sampling.get("replay"),
        label="run sampling replay evidence",
    )
    prediction = _mapping(
        diagnostics.get("prediction_evidence"),
        label="run prediction evidence",
    )
    scores = _mapping(
        diagnostics.get("score_evidence"),
        label="run score evidence",
    )
    bindings = _mapping(oracle.get("bindings"), label="source-replay bindings")
    replay_path = sampling_replay.get("path")
    replay_manifest = sampling_replay.get("manifest")
    replay_manifest_sha256 = sampling_replay.get("manifest_sha256")
    if (
        not isinstance(replay_path, str)
        or not replay_path
        or replay_path.startswith("/")
        or ".." in replay_path.split("/")
        or replay_manifest != "MANIFEST.json"
        or not isinstance(replay_manifest_sha256, str)
    ):
        return None
    sampling_dir = (run_path.parent / replay_path).resolve()
    if not sampling_dir.is_relative_to(run_path.parent.resolve()):
        return None
    manifest_path = sampling_dir / replay_manifest
    arrays_path = sampling_dir / "arrays.npz"
    try:
        if _sha256_file(manifest_path) != replay_manifest_sha256:
            return None
        with np.load(arrays_path, allow_pickle=False) as replay_arrays:
            labeled_indices = np.ascontiguousarray(
                replay_arrays["idx__train_labeled"],
                dtype="<i8",
            )
    except (KeyError, OSError, ValueError):
        return None
    labeled_indices_sha256 = hashlib.sha256(labeled_indices.tobytes(order="C")).hexdigest()
    archive_rows_match = any(
        f"{float(row['accuracy_percent']):.{_ARCHIVE_PRECISION_DECIMALS}f}"
        == archive["accuracy_percent_text"]
        and row.get("archive_compatible_correct_count") == archive["compatible_correct_count"]
        and row.get("node_delta") == archive["node_delta"]
        for row in matching_rows
    )
    if (
        not archive_rows_match
        or task.source_config_sha256 != bindings["effective_config_sha256"]
        or sampling.get("split_fingerprint") != bindings["split_fingerprint"]
        or labeled_indices.shape != (50,)
        or labeled_indices_sha256 != bindings["labeled_indices_sorted_sha256"]
        or unlabeled_count != replay["unlabeled_count"]
        or correct_count != replay["correct_count"]
        or not math.isclose(
            accuracy,
            float(replay["accuracy"]),
            rel_tol=0,
            abs_tol=0,
        )
        or diagnostics.get("iterations") != replay["iterations"]["modssc"]
        or not math.isclose(
            float(diagnostics.get("absolute_residual", math.nan)),
            float(replay["residual"]["modssc_recursive"]),
            rel_tol=0,
            abs_tol=0,
        )
        or prediction.get("encoding") != replay["prediction_encoding"]
        or prediction.get("shape") != replay["prediction_shape"]
        or prediction.get("count") != replay["prediction_count"]
        or prediction.get("byte_count") != replay["prediction_byte_count"]
        or prediction.get("sha256") != replay["prediction_sha256"]["modssc"]
        or scores.get("encoding") != replay["score_encoding"]
        or scores.get("shape") != replay["score_shape"]
        or scores.get("byte_count") != replay["score_byte_count"]
        or scores.get("sha256") != replay["score_sha256"]["modssc"]
    ):
        return None
    return {
        "oracle_sha256": _SOURCE_REPLAY_ORACLE_SHA256,
        "official_source_prediction_sha256": replay["prediction_sha256"]["official_source"],
        "modssc_prediction_sha256": prediction["sha256"],
        "prediction_count": prediction["count"],
        "prediction_shape": list(prediction["shape"]),
        "prediction_byte_count": prediction["byte_count"],
        "prediction_encoding": prediction["encoding"],
        "official_source_score_sha256": replay["score_sha256"]["official_source"],
        "modssc_score_sha256": scores["sha256"],
        "score_shape": list(scores["shape"]),
        "score_byte_count": scores["byte_count"],
        "score_encoding": scores["encoding"],
        "effective_config_sha256": task.source_config_sha256,
        "split_fingerprint": sampling["split_fingerprint"],
        "labeled_indices_sha256": labeled_indices_sha256,
        "differing_predictions": replay["differing_predictions"],
        "max_absolute_score_delta": replay["max_absolute_score_delta"],
        "iterations": dict(replay["iterations"]),
        "residual": dict(replay["residual"]),
        "archive_compatible_correct_count": list(archive["compatible_correct_count"]),
        "node_delta": archive["node_delta"],
    }


def _validate_reconcile(
    *,
    meta: Mapping[str, Any],
    tasks: Sequence[Any],
    reconcile: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if (
        reconcile.get("schema_version") != 1
        or reconcile.get("campaign_id") != CANARY_CAMPAIGN_ID
        or reconcile.get("manifest_sha256") != meta.get("manifest_sha256")
        or reconcile.get("status") != "complete"
        or reconcile.get("task_count") != 4
    ):
        raise CalderCanaryError("canary reconciliation is not a complete matching report")
    rows = reconcile.get("tasks")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise CalderCanaryError("canary reconciliation task rows are invalid")
    states = {str(row.get("task_id")): row for row in rows}
    if len(states) != 4 or set(states) != {task.task_id for task in tasks}:
        raise CalderCanaryError("canary reconciliation task set differs")
    if any(row.get("status") != "success" for row in states.values()):
        raise CalderCanaryError("all four canary tasks must be successful")
    return states


def _validate_production_spec(
    *,
    path: Path,
    repo_root: Path,
    tasks: Sequence[Any] | None = None,
) -> _ProductionEvidence:
    spec = _read_mapping(path, label="Calder production spec")
    code = _mapping(spec.get("code"), label="production spec code")
    expect = _mapping(spec.get("expect"), label="production spec expectations")
    evidence = _mapping(
        spec.get("calder_artifacts"),
        label="production spec Calder artifacts",
    )
    cells = spec.get("cells")
    task_sequence = list(tasks or [])
    git_sha = task_sequence[0].expected_git_sha if task_sequence else code.get("git_sha")
    git_diff_sha256 = (
        task_sequence[0].expected_git_diff_sha256 if task_sequence else code.get("git_diff_sha256")
    )
    environment_sha256 = (
        task_sequence[0].environment_lock_sha256
        if task_sequence
        else code.get("environment_lock_sha256")
    )
    dataset_evidence = _mapping(
        evidence.get("dataset"),
        label="production spec Calder dataset evidence",
    )
    dataset_fingerprint = dataset_evidence.get("prepared_fingerprint")
    dataset_content_sha256 = dataset_evidence.get("content_sha256")
    if task_sequence and (
        task_sequence[0].expected_dataset_fingerprint != dataset_fingerprint
        or task_sequence[0].expected_dataset_content_sha256 != dataset_content_sha256
    ):
        raise CalderCanaryError("production spec dataset evidence differs from the canary manifest")
    if (
        spec.get("schema_version") != 1
        or spec.get("campaign_id") != PRODUCTION_CAMPAIGN_ID
        or spec.get("track") != "paper"
        or spec.get("default_site") != "local-cpu"
        or code.get("git_sha") != git_sha
        or code.get("git_diff_sha256") != git_diff_sha256
        or code.get("environment_lock_sha256") != environment_sha256
        or code.get("require_clean") is not True
        or expect.get("config_count") != 10
        or expect.get("task_count") != 1000
        or expect.get("tasks_per_method") != {"laplace_learning": 500, "poisson_learning": 500}
        or expect.get("tasks_by_profile") != {"cpu_graph": 1000}
        or expect.get("tasks_by_site") != {"local-cpu": 1000}
        or not isinstance(cells, list)
        or len(cells) != 10
    ):
        raise CalderCanaryError("production spec identity or expected counts differ")

    artifact_lock_sha256 = _require_sha256(
        evidence.get("artifact_lock_sha256"),
        label="production spec artifact lock SHA-256",
    )
    oracle_evidence = _mapping(
        evidence.get("source_replay_oracle"),
        label="production spec source-replay oracle evidence",
    )
    if oracle_evidence != {
        "path": _SOURCE_REPLAY_ORACLE_RELATIVE.as_posix(),
        "sha256": _SOURCE_REPLAY_ORACLE_SHA256,
    }:
        raise CalderCanaryError("production spec source-replay oracle evidence differs")
    oracle_path = (repo_root / str(oracle_evidence["path"])).resolve(strict=True)
    if (
        not oracle_path.is_relative_to(repo_root)
        or oracle_path.is_symlink()
        or not oracle_path.is_file()
        or _sha256_file(oracle_path) != _SOURCE_REPLAY_ORACLE_SHA256
    ):
        raise CalderCanaryError("production spec source-replay oracle SHA-256 differs")
    manifest_evidence = _mapping(
        evidence.get("effective_manifest"),
        label="production spec effective MANIFEST evidence",
    )
    if set(manifest_evidence) != {"path", "sha256", "lock_sha256"}:
        raise CalderCanaryError("production spec effective MANIFEST evidence fields differ")
    manifest_path = _repo_generated_file(
        repo_root,
        manifest_evidence.get("path"),
        label="production spec effective MANIFEST",
    )
    manifest_sha256 = _require_sha256(
        manifest_evidence.get("sha256"),
        label="production spec effective MANIFEST file SHA-256",
    )
    manifest_lock_sha256 = _require_sha256(
        manifest_evidence.get("lock_sha256"),
        label="production spec effective MANIFEST seal",
    )
    if _sha256_file(manifest_path) != manifest_sha256:
        raise CalderCanaryError("effective MANIFEST file SHA-256 differs")
    manifest = _read_mapping(manifest_path, label="effective configuration MANIFEST")
    actual_manifest_seal = _require_sha256(
        manifest.get("lock_sha256"),
        label="effective configuration MANIFEST seal",
    )
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("lock_sha256")
    if (
        actual_manifest_seal != manifest_lock_sha256
        or _canonical_sha256(unsigned_manifest) != actual_manifest_seal
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != EFFECTIVE_CONFIG_KIND
        or manifest.get("artifact_lock_sha256") != artifact_lock_sha256
    ):
        raise CalderCanaryError("effective configuration MANIFEST identity differs")

    raw_config_evidence = evidence.get("effective_configs")
    if not isinstance(raw_config_evidence, Mapping) or len(raw_config_evidence) != 10:
        raise CalderCanaryError("production spec must bind exactly ten effective configurations")
    config_evidence: dict[str, str] = {}
    for config_path, digest in raw_config_evidence.items():
        if not isinstance(config_path, str) or not config_path:
            raise CalderCanaryError("effective configuration evidence path is invalid")
        config_evidence[config_path] = _require_sha256(
            digest,
            label=f"effective configuration SHA-256 for {config_path}",
        )
    manifest_records = manifest.get("configs")
    if not isinstance(manifest_records, list) or len(manifest_records) != 10:
        raise CalderCanaryError("effective configuration MANIFEST must contain exactly ten records")
    manifest_config_evidence: dict[str, str] = {}
    for record in manifest_records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "repo_path",
            "sha256",
        }:
            raise CalderCanaryError("effective configuration MANIFEST record is invalid")
        repo_path = record.get("repo_path")
        if not isinstance(repo_path, str) or repo_path in manifest_config_evidence:
            raise CalderCanaryError("effective configuration MANIFEST paths must be unique")
        manifest_config_evidence[repo_path] = _require_sha256(
            record.get("sha256"),
            label=f"effective configuration MANIFEST SHA-256 for {repo_path}",
        )
    if manifest_config_evidence != config_evidence:
        raise CalderCanaryError("effective configuration evidence differs from its MANIFEST")

    official = _mapping(
        evidence.get("official"),
        label="production spec official GraphLearning evidence",
    )
    if set(official) != {
        "commit",
        "labels_sha256",
        "permutations_sha256",
        "knn_sha256",
    }:
        raise CalderCanaryError("official GraphLearning evidence fields differ")
    official_evidence = {
        "commit": str(official.get("commit", "")),
        "labels_sha256": _require_sha256(
            official.get("labels_sha256"),
            label="official GraphLearning labels SHA-256",
        ),
        "permutations_sha256": _require_sha256(
            official.get("permutations_sha256"),
            label="official GraphLearning permutations SHA-256",
        ),
        "knn_sha256": _require_sha256(
            official.get("knn_sha256"),
            label="official GraphLearning kNN SHA-256",
        ),
    }
    if not official_evidence["commit"]:
        raise CalderCanaryError("official GraphLearning commit is missing")
    graph = _mapping(
        evidence.get("graph"),
        label="production spec Calder graph evidence",
    )
    if set(graph) != {"fingerprint", "preprocess_fingerprint"} or any(
        not isinstance(graph.get(key), str) or not graph[key]
        for key in ("fingerprint", "preprocess_fingerprint")
    ):
        raise CalderCanaryError("Calder graph evidence fields differ")
    graph_evidence = {
        "fingerprint": str(graph["fingerprint"]),
        "preprocess_fingerprint": str(graph["preprocess_fingerprint"]),
    }
    dataset_evidence_validated = {
        "prepared_fingerprint": _require_sha256(
            dataset_fingerprint,
            label="Calder prepared dataset fingerprint",
        ),
        "content_sha256": _require_sha256(
            dataset_content_sha256,
            label="Calder dataset content SHA-256",
        ),
    }

    config_identities: dict[tuple[str, int], tuple[str, str]] = {}
    for config_path, expected_digest in config_evidence.items():
        resolved = _repo_generated_file(
            repo_root,
            config_path,
            label=f"effective configuration {config_path}",
        )
        if _sha256_file(resolved) != expected_digest:
            raise CalderCanaryError(f"effective configuration SHA-256 differs: {config_path}")
        config = _read_mapping(resolved, label=f"effective configuration {config_path}")
        method = _mapping(config.get("method"), label=f"{config_path}.method")
        dataset = _mapping(config.get("dataset"), label=f"{config_path}.dataset")
        run = _mapping(config.get("run"), label=f"{config_path}.run")
        sampling = _mapping(config.get("sampling"), label=f"{config_path}.sampling")
        plan = _mapping(sampling.get("plan"), label=f"{config_path}.sampling.plan")
        labeling = _mapping(
            plan.get("labeling"),
            label=f"{config_path}.sampling.plan.labeling",
        )
        fixed = _mapping(
            labeling.get("fixed_indices_artifact"),
            label=f"{config_path}.fixed_indices_artifact",
        )
        graph_config = _mapping(config.get("graph"), label=f"{config_path}.graph")
        graph_spec = _mapping(
            graph_config.get("spec"),
            label=f"{config_path}.graph.spec",
        )
        method_id = method.get("id")
        budget = labeling.get("value")
        if (
            method_id not in {"laplace_learning", "poisson_learning"}
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget not in range(1, 6)
        ):
            raise CalderCanaryError(f"effective configuration method/budget differs: {config_path}")
        short_method = str(method_id).removesuffix("_learning")
        expected_profile = f"paper:calder2020-mnist-table1-{short_method}-{budget}-label-per-class"
        if (
            method.get("profile") != expected_profile
            or method.get("kind") != "transductive"
            or dataset.get("id") != "mnist"
            or dataset.get("download") is not False
            or run.get("seed") != 0
            or run.get("seeds") != list(range(100))
            or run.get("seeded_sections") != ["sampling"]
            or labeling.get("mode") != "per_class"
            or labeling.get("value") != budget
            or labeling.get("per_class") is not True
            or fixed.get("sha256") != _PREPARED_PERMUTATIONS_SHA256
            or fixed.get("source_sha256") != official_evidence["permutations_sha256"]
            or fixed.get("index_stride") != 5
            or fixed.get("index_offset") != budget - 1
            or fixed.get("expected_size") != budget * 10
            or fixed.get("expected_per_class") != budget
            or graph_config.get("require_cache_hit") is not True
            or graph_config.get("expected_fingerprint") != graph_evidence["fingerprint"]
            or graph_config.get("expected_preprocess_fingerprint")
            != graph_evidence["preprocess_fingerprint"]
            or graph_spec.get("backend") != "precomputed"
            or graph_spec.get("precomputed_sha256") != official_evidence["knn_sha256"]
        ):
            raise CalderCanaryError(
                f"effective configuration protocol identity differs: {config_path}"
            )
        identity = (str(method_id), budget)
        if identity in config_identities:
            raise CalderCanaryError(f"duplicate effective configuration identity: {identity}")
        config_identities[identity] = (config_path, expected_digest)
    if set(config_identities) != _EXPECTED_PRODUCTION_IDENTITIES:
        raise CalderCanaryError("effective configuration identities differ")

    identities: set[tuple[str, int]] = set()
    used_configs: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise CalderCanaryError("production spec cell is not a mapping")
        protocol = cell.get("protocol_id")
        config_path = cell.get("config")
        if not isinstance(config_path, str) or config_path not in config_evidence:
            raise CalderCanaryError("production spec config evidence differs")
        matching = [
            identity for identity, record in config_identities.items() if record[0] == config_path
        ]
        if len(matching) != 1:
            raise CalderCanaryError("production spec config identity is ambiguous")
        method, budget = matching[0]
        short_method = method.removesuffix("_learning")
        expected_protocol = f"calder-2020-mnist-table1-{short_method}-{budget}-label-per-class"
        identities.add((method, budget))
        used_configs.add(config_path)
        if (
            protocol != expected_protocol
            or cell.get("effective_config_sha256") != config_evidence[config_path]
            or cell.get("seeds") != "from_config"
            or cell.get("site") != "local-cpu"
            or cell.get("resource_profile") != "cpu_graph"
            or cell.get("fidelity_status") != "paper_matched"
            or cell.get("expected_dataset_fingerprint") != dataset_fingerprint
            or cell.get("expected_dataset_content_sha256") != dataset_content_sha256
        ):
            raise CalderCanaryError("production spec cell contract differs")
    if (
        identities != _EXPECTED_PRODUCTION_IDENTITIES
        or used_configs != set(config_evidence)
        or len(identities) != len(cells)
    ):
        raise CalderCanaryError("production spec method/budget cells differ")
    return _ProductionEvidence(
        spec_sha256=_sha256_file(path.resolve(strict=True)),
        artifact_lock_sha256=artifact_lock_sha256,
        effective_manifest_path=str(manifest_evidence["path"]),
        effective_manifest_sha256=manifest_sha256,
        effective_manifest_lock_sha256=manifest_lock_sha256,
        effective_config_sha256=config_evidence,
        source_replay_oracle_path=str(oracle_evidence["path"]),
        source_replay_oracle_sha256=str(oracle_evidence["sha256"]),
        official=official_evidence,
        dataset=dataset_evidence_validated,
        graph=graph_evidence,
    )


def _validate_canary_provenance(
    *,
    lock: Mapping[str, Any],
    tasks: Sequence[Any],
    production: _ProductionEvidence,
) -> None:
    if lock.get("lock_sha256") != production.artifact_lock_sha256:
        raise CalderCanaryError("artifact lock differs from the production specification")
    pins = _mapping(lock.get("pins"), label="Calder artifact lock pins")
    official = _mapping(
        lock.get("official_evidence"),
        label="Calder artifact lock official evidence",
    )
    if (
        pins.get("official_commit") != production.official["commit"]
        or pins.get("official_knn_sha256") != production.official["knn_sha256"]
        or pins.get("official_permutations_sha256") != production.official["permutations_sha256"]
        or official.get("labels_sha256") != production.official["labels_sha256"]
        or pins.get("graph_fingerprint") != production.graph["fingerprint"]
        or pins.get("preprocess_fingerprint") != production.graph["preprocess_fingerprint"]
    ):
        raise CalderCanaryError(
            "artifact lock scientific evidence differs from the production specification"
        )
    dataset = _mapping(lock.get("dataset"), label="Calder artifact lock dataset")
    content = _mapping(
        dataset.get("content_evidence"),
        label="Calder artifact lock dataset content evidence",
    )
    if (
        dataset.get("prepared_fingerprint") != production.dataset["prepared_fingerprint"]
        or content.get("content_sha256") != production.dataset["content_sha256"]
    ):
        raise CalderCanaryError(
            "artifact lock dataset evidence differs from the production specification"
        )
    for task in tasks:
        expected_digest = production.effective_config_sha256.get(task.config_path)
        if expected_digest is None or expected_digest != task.source_config_sha256:
            raise CalderCanaryError(
                f"canary task effective configuration is not production-bound: {task.task_id}"
            )


def _seal_acceptance(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("acceptance_sha256", None)
    sealed["acceptance_sha256"] = _canonical_sha256(sealed)
    return sealed


def _source_file_evidence(path: Path, *, label: str) -> dict[str, str]:
    candidate = path.expanduser()
    resolved = candidate.resolve(strict=True)
    if candidate.is_symlink() or not resolved.is_file():
        raise CalderCanaryError(f"{label} must be a regular non-symlink file")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
    }


def _manifest_meta_path(manifest_path: Path, meta_path: Path | None) -> Path:
    if meta_path is not None:
        return meta_path
    meta_name = (
        "manifest.meta.json"
        if manifest_path.name == "manifest.jsonl"
        else f"{manifest_path.stem}.meta.json"
    )
    return manifest_path.with_name(meta_name)


def validate_calder_canary(
    *,
    repo_root: Path,
    artifact_lock_path: Path,
    manifest_path: Path,
    reconcile_path: Path,
    production_spec_path: Path,
    output_path: Path,
    meta_path: Path | None = None,
) -> CalderCanaryReport:
    """Compare the four seed-0 runs with authenticated individual archive rows."""

    root = repo_root.expanduser().resolve(strict=True)
    lock = _read_mapping(artifact_lock_path, label="Calder artifact lock")
    try:
        verify_calder_artifact_lock(lock)
    except (CalderArtifactError, OSError, ValueError) as exc:
        raise CalderCanaryError(f"Calder artifact lock verification failed: {exc}") from exc
    artifacts = _mapping(lock.get("artifacts"), label="Calder artifact lock artifacts")
    official_inventory = _mapping(
        artifacts.get("protocol_inputs"),
        label="Calder protocol-input inventory",
    )
    official_root_value = official_inventory.get("root")
    if not isinstance(official_root_value, str) or not official_root_value:
        raise CalderCanaryError("Calder protocol-input root is missing")
    official_root = Path(official_root_value)
    try:
        verify_calder_official_assets(official_root)
    except (CalderOfficialArtifactError, OSError, ValueError) as exc:
        raise CalderCanaryError(f"Calder protocol-input verification failed: {exc}") from exc

    try:
        meta, tasks = load_manifest(manifest_path, meta_path=meta_path, verify_digest=True)
    except CampaignError as exc:
        raise CalderCanaryError(f"canary manifest verification failed: {exc}") from exc
    by_identity = _validate_canary_tasks(repo_root=root, meta=meta, tasks=tasks)
    production_evidence = _validate_production_spec(
        path=production_spec_path,
        repo_root=root,
        tasks=tasks,
    )
    _validate_canary_provenance(
        lock=lock,
        tasks=tasks,
        production=production_evidence,
    )
    source_replay_oracle = _load_source_replay_oracle(root)
    _validate_source_replay_campaign_binding(
        source_replay_oracle,
        environment_lock_sha256=tasks[0].environment_lock_sha256,
        dataset_fingerprint=tasks[0].expected_dataset_fingerprint,
        dataset_content_sha256=tasks[0].expected_dataset_content_sha256,
        production=production_evidence,
    )
    reconcile = _read_mapping(reconcile_path, label="Calder canary reconciliation")
    states = _validate_reconcile(meta=meta, tasks=tasks, reconcile=reconcile)

    archive_by_method: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for method_id, filename in {
        "laplace_learning": "mnist-vae-k10-laplace-accuracy.csv",
        "poisson_learning": "mnist-vae-k10-poisson-accuracy.csv",
    }.items():
        archive_path = official_root / "references" / filename
        if _sha256_file(archive_path) != OFFICIAL_RESULTS_SHA256[method_id]:
            raise CalderCanaryError(f"archived {method_id} result SHA-256 differs")
        archive_by_method[method_id] = _archive_rows(archive_path)

    comparisons: list[dict[str, Any]] = []
    for identity in sorted(by_identity):
        task = by_identity[identity]
        method_id, budget = identity
        state = states[task.task_id]
        result_dirs = state.get("result_dirs")
        paths = state.get("run_json_paths")
        digests = state.get("run_json_sha256")
        if (
            not isinstance(result_dirs, list)
            or len(result_dirs) != 1
            or not isinstance(paths, list)
            or len(paths) != 1
            or not isinstance(digests, list)
            or len(digests) != 1
        ):
            raise CalderCanaryError(f"canary result identity is incomplete: {task.task_id}")
        try:
            run_path, payload, run_sha256 = validate_result_directory(
                Path(str(result_dirs[0])),
                task,
            )
        except CampaignError as exc:
            raise CalderCanaryError(
                f"canary result validation failed for {task.task_id}: {exc}"
            ) from exc
        if run_path.resolve() != Path(str(paths[0])).resolve() or run_sha256 != digests[0]:
            raise CalderCanaryError(f"reconciled run identity differs: {task.task_id}")
        accuracy = _metric(payload)
        accuracy_percent = accuracy * 100
        unlabeled_count = _unlabeled_count(payload, budget=budget)
        correct_count = _correct_count(
            accuracy,
            unlabeled_count=unlabeled_count,
        )
        max_node_delta = _SCOPED_NODE_ALLOWANCE.get((method_id, budget, 0), 0)
        matches: list[dict[str, Any]] = []
        for row in archive_by_method[method_id][budget]:
            compatible = _archive_compatible_correct_counts(
                float(row["accuracy_percent"]),
                unlabeled_count=unlabeled_count,
            )
            node_delta = min(abs(correct_count - count) for count in compatible)
            if node_delta <= max_node_delta:
                matches.append(
                    {
                        **row,
                        "archive_compatible_correct_count": [
                            min(compatible),
                            max(compatible),
                        ],
                        "node_delta": node_delta,
                    }
                )
        if matches:
            smallest_node_delta = min(int(row["node_delta"]) for row in matches)
            matches = [row for row in matches if int(row["node_delta"]) == smallest_node_delta]
        else:
            smallest_node_delta = None
        exception_evidence = None
        if smallest_node_delta is not None and smallest_node_delta > 0:
            exception_evidence = _source_replay_exception_evidence(
                oracle=source_replay_oracle,
                payload=payload,
                run_path=run_path,
                task=task,
                accuracy=accuracy,
                unlabeled_count=unlabeled_count,
                correct_count=correct_count,
                matching_rows=matches,
            )
            if exception_evidence is None:
                matches = []
        diagnostics_ok, diagnostic_failures = _diagnostics_pass(method_id, payload)
        passed = bool(matches) and diagnostics_ok
        comparisons.append(
            {
                "task_id": task.task_id,
                "method_id": method_id,
                "budget_per_class": budget,
                "permutation": 0,
                "fixed_permutation_row": budget - 1,
                "run_json_sha256": run_sha256,
                "accuracy": accuracy,
                "accuracy_percent": accuracy_percent,
                "unlabeled_count": unlabeled_count,
                "correct_count": correct_count,
                "archive_precision_decimals": _ARCHIVE_PRECISION_DECIMALS,
                "archive_tolerance_percent": _ARCHIVE_TOLERANCE_PERCENT,
                "max_node_delta": max_node_delta,
                "numeric_environment_exception": exception_evidence,
                "matching_archived_rows": matches,
                "diagnostics_ok": diagnostics_ok,
                "diagnostic_failures": diagnostic_failures,
                "passed": passed,
            }
        )

    passed_count = sum(bool(comparison["passed"]) for comparison in comparisons)
    status = "passed" if passed_count == 4 else "failed"
    resolved_manifest_path = manifest_path.expanduser().resolve(strict=True)
    resolved_meta_path = _manifest_meta_path(
        resolved_manifest_path,
        None if meta_path is None else meta_path.expanduser(),
    )
    source_evidence = {
        "artifact_lock": _source_file_evidence(
            artifact_lock_path,
            label="Calder artifact lock source",
        ),
        "manifest": _source_file_evidence(
            resolved_manifest_path,
            label="Calder canary manifest source",
        ),
        "meta": _source_file_evidence(
            resolved_meta_path,
            label="Calder canary metadata source",
        ),
        "reconcile": _source_file_evidence(
            reconcile_path,
            label="Calder canary reconciliation source",
        ),
    }
    payload = _seal_acceptance(
        {
            "schema_version": ACCEPTANCE_SCHEMA_VERSION,
            "kind": ACCEPTANCE_KIND,
            "campaign_id": CANARY_CAMPAIGN_ID,
            "status": status,
            "comparison_basis": (
                "locked_permutation_0_discrete_correct_count_and_same-budget_"
                "membership_in_authenticated_graphlearning_archive"
            ),
            "archive_limitation": (
                "GraphLearningOld CSV rows contain label count and accuracy but no "
                "permutation identifier; the locked task proves permutation 0 and "
                "the numerical check uses the complete same-budget row multiset. "
                "Only Laplace budget 5 may differ by one classified node because "
                "an immutable oracle proves identical predictions from the "
                "authenticated source and ModSSC in the locked current environment."
            ),
            "manifest_sha256": meta["manifest_sha256"],
            "artifact_lock_sha256": lock["lock_sha256"],
            "production_spec_sha256": production_evidence.spec_sha256,
            "production_evidence": production_evidence.audit_payload(),
            "git_sha": tasks[0].expected_git_sha,
            "git_diff_sha256": tasks[0].expected_git_diff_sha256,
            "environment_lock_sha256": tasks[0].environment_lock_sha256,
            "dataset_fingerprint": tasks[0].expected_dataset_fingerprint,
            "dataset_content_sha256": tasks[0].expected_dataset_content_sha256,
            "official_results_sha256": dict(OFFICIAL_RESULTS_SHA256),
            "source_replay_oracle_sha256": _SOURCE_REPLAY_ORACLE_SHA256,
            "source_evidence": source_evidence,
            "comparisons": comparisons,
        }
    )
    try:
        write_immutable_json(output_path, payload)
    except CalderArtifactError as exc:
        raise CalderCanaryError(f"cannot publish immutable canary acceptance: {exc}") from exc
    return CalderCanaryReport(
        status=status,
        output_path=str(output_path.expanduser().resolve()),
        campaign_id=CANARY_CAMPAIGN_ID,
        comparison_count=4,
        passed_count=passed_count,
        acceptance_sha256=str(payload["acceptance_sha256"]),
    )


def _acceptance_source_paths(acceptance: Mapping[str, Any]) -> dict[str, Path]:
    source = _mapping(
        acceptance.get("source_evidence"),
        label="Calder canary source evidence",
    )
    expected = {"artifact_lock", "manifest", "meta", "reconcile"}
    if set(source) != expected:
        raise CalderCanaryError("Calder canary source evidence fields differ")
    paths: dict[str, Path] = {}
    for name in sorted(expected):
        evidence = _mapping(
            source.get(name),
            label=f"Calder canary {name} source evidence",
        )
        if set(evidence) != {"path", "sha256"}:
            raise CalderCanaryError(f"Calder canary {name} source evidence fields differ")
        raw_path = evidence.get("path")
        if not isinstance(raw_path, str) or not raw_path.startswith("/"):
            raise CalderCanaryError(f"Calder canary {name} source path must be absolute")
        candidate = Path(raw_path)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise CalderCanaryError(f"Calder canary {name} source is unavailable") from exc
        expected_sha256 = _require_sha256(
            evidence.get("sha256"),
            label=f"Calder canary {name} source SHA-256",
        )
        if (
            candidate.is_symlink()
            or not resolved.is_file()
            or _sha256_file(resolved) != expected_sha256
        ):
            raise CalderCanaryError(f"Calder canary {name} source SHA-256 differs")
        paths[name] = resolved
    return paths


def _validate_acceptance_document(acceptance: Mapping[str, Any]) -> None:
    digest = _require_sha256(
        acceptance.get("acceptance_sha256"),
        label="Calder canary acceptance seal",
    )
    unsigned = dict(acceptance)
    unsigned.pop("acceptance_sha256")
    if _canonical_sha256(unsigned) != digest:
        raise CalderCanaryError("Calder canary acceptance SHA-256 differs")
    if (
        acceptance.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION
        or acceptance.get("kind") != ACCEPTANCE_KIND
        or acceptance.get("campaign_id") != CANARY_CAMPAIGN_ID
        or acceptance.get("status") != "passed"
    ):
        raise CalderCanaryError("Calder canary acceptance does not authorize production")
    _acceptance_source_paths(acceptance)

    if acceptance.get("official_results_sha256") != dict(OFFICIAL_RESULTS_SHA256):
        raise CalderCanaryError("authenticated GraphLearning result evidence differs")
    if acceptance.get("source_replay_oracle_sha256") != _SOURCE_REPLAY_ORACLE_SHA256:
        raise CalderCanaryError("Calder source-replay oracle evidence differs")

    comparisons = acceptance.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 4:
        raise CalderCanaryError("Calder canary acceptance comparisons are incomplete")
    identities: set[tuple[str, int]] = set()
    task_ids: set[str] = set()
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise CalderCanaryError("Calder canary acceptance comparison is not a mapping")
        method_id = comparison.get("method_id")
        budget = comparison.get("budget_per_class")
        permutation = comparison.get("permutation")
        task_id = comparison.get("task_id")
        if (
            method_id not in {"laplace_learning", "poisson_learning"}
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or isinstance(permutation, bool)
            or not isinstance(permutation, int)
            or not isinstance(task_id, str)
            or not task_id
        ):
            raise CalderCanaryError("Calder canary acceptance comparison identity is invalid")
        identity = (str(method_id), int(budget))
        if identity in identities or task_id in task_ids:
            raise CalderCanaryError("Calder canary acceptance contains duplicate comparisons")
        identities.add(identity)
        task_ids.add(task_id)
        accuracy = comparison.get("accuracy")
        accuracy_percent = comparison.get("accuracy_percent")
        unlabeled_count = comparison.get("unlabeled_count")
        correct_count = comparison.get("correct_count")
        max_node_delta = comparison.get("max_node_delta")
        expected_unlabeled_count = 70_000 - int(budget) * 10
        expected_max_node_delta = _SCOPED_NODE_ALLOWANCE.get(
            (str(method_id), int(budget), int(permutation)),
            0,
        )
        if (
            comparison.get("passed") is not True
            or permutation != 0
            or comparison.get("fixed_permutation_row") != permutation * 5 + int(budget) - 1
            or comparison.get("diagnostics_ok") is not True
            or comparison.get("diagnostic_failures") != []
            or isinstance(accuracy, bool)
            or not isinstance(accuracy, int | float)
            or not math.isfinite(float(accuracy))
            or isinstance(accuracy_percent, bool)
            or not isinstance(accuracy_percent, int | float)
            or not math.isfinite(float(accuracy_percent))
            or not math.isclose(
                float(accuracy) * 100,
                float(accuracy_percent),
                rel_tol=0,
                abs_tol=1e-12,
            )
            or isinstance(unlabeled_count, bool)
            or not isinstance(unlabeled_count, int)
            or unlabeled_count != expected_unlabeled_count
            or isinstance(correct_count, bool)
            or not isinstance(correct_count, int)
            or correct_count
            != _correct_count(
                float(accuracy),
                unlabeled_count=expected_unlabeled_count,
            )
            or isinstance(max_node_delta, bool)
            or not isinstance(max_node_delta, int)
            or max_node_delta != expected_max_node_delta
            or comparison.get("archive_precision_decimals") != _ARCHIVE_PRECISION_DECIMALS
            or not math.isclose(
                float(comparison.get("archive_tolerance_percent", math.nan)),
                _ARCHIVE_TOLERANCE_PERCENT,
                rel_tol=0,
                abs_tol=0,
            )
        ):
            raise CalderCanaryError("Calder canary acceptance comparison evidence is invalid")
        matches = comparison.get("matching_archived_rows")
        if not isinstance(matches, list) or not matches:
            raise CalderCanaryError("Calder canary acceptance has no matching archived row")
        for row in matches:
            if (
                not isinstance(row, Mapping)
                or isinstance(row.get("line_number"), bool)
                or not isinstance(row.get("line_number"), int)
                or int(row["line_number"]) <= 0
                or isinstance(row.get("accuracy_percent"), bool)
                or not isinstance(row.get("accuracy_percent"), int | float)
                or not math.isfinite(float(row["accuracy_percent"]))
            ):
                raise CalderCanaryError("Calder canary archived-row evidence is invalid")
            compatible = _archive_compatible_correct_counts(
                float(row["accuracy_percent"]),
                unlabeled_count=expected_unlabeled_count,
            )
            node_delta = min(abs(int(correct_count) - count) for count in compatible)
            if (
                row.get("archive_compatible_correct_count") != [min(compatible), max(compatible)]
                or row.get("node_delta") != node_delta
                or node_delta > expected_max_node_delta
            ):
                raise CalderCanaryError("Calder canary archived-row discrete evidence is invalid")
        observed_node_delta = min(int(row["node_delta"]) for row in matches)
        if any(int(row["node_delta"]) != observed_node_delta for row in matches):
            raise CalderCanaryError("Calder canary archived-row evidence mixes node deltas")
        exception = comparison.get("numeric_environment_exception")
        if observed_node_delta == 0:
            if exception is not None:
                raise CalderCanaryError("Calder canary has an unnecessary numerical exception")
        elif (
            observed_node_delta != 1
            or identity != ("laplace_learning", 5)
            or not isinstance(exception, Mapping)
            or exception.get("oracle_sha256") != _SOURCE_REPLAY_ORACLE_SHA256
            or exception.get("official_source_prediction_sha256")
            != _SOURCE_REPLAY_PREDICTION_SHA256
            or exception.get("modssc_prediction_sha256") != _SOURCE_REPLAY_PREDICTION_SHA256
            or exception.get("prediction_count") != 70_000
            or exception.get("prediction_shape") != [70_000]
            or exception.get("prediction_byte_count") != 560_000
            or exception.get("prediction_encoding") != "numpy-int64-little-endian-c-order"
            or exception.get("official_source_score_sha256") != _SOURCE_REPLAY_SCORE_SHA256
            or exception.get("modssc_score_sha256") != _SOURCE_REPLAY_SCORE_SHA256
            or exception.get("score_shape") != [70_000, 10]
            or exception.get("score_byte_count") != 5_600_000
            or exception.get("score_encoding") != "numpy-float64-little-endian-c-order"
            or exception.get("effective_config_sha256") != _SOURCE_REPLAY_CONFIG_SHA256
            or exception.get("split_fingerprint") != _SOURCE_REPLAY_SPLIT_FINGERPRINT
            or exception.get("labeled_indices_sha256") != _SOURCE_REPLAY_LABELED_INDICES_SHA256
            or exception.get("differing_predictions") != 0
            or not math.isclose(
                float(exception.get("max_absolute_score_delta", math.nan)),
                0.0,
                rel_tol=0,
                abs_tol=0,
            )
            or exception.get("iterations") != {"official_source": 148, "modssc": 148}
            or exception.get("residual")
            != {
                "official_system": 9.842962973879334e-06,
                "modssc_recursive": 9.842962973957787e-06,
            }
            or exception.get("archive_compatible_correct_count") != [48_263, 48_268]
            or exception.get("node_delta") != 1
        ):
            raise CalderCanaryError("Calder canary numerical exception evidence is invalid")
        _require_sha256(
            comparison.get("run_json_sha256"),
            label=f"canary run JSON SHA-256 for {task_id}",
        )
    if identities != _EXPECTED_IDENTITIES:
        raise CalderCanaryError("Calder canary acceptance method/budget identities differ")


def verify_calder_canary_acceptance(
    acceptance_path: Path,
    *,
    repo_root: Path,
    production_spec_path: Path,
) -> dict[str, Any]:
    """Fail closed unless passing canaries bind the current spec and all inputs."""

    acceptance = _read_mapping(acceptance_path, label="Calder canary acceptance")
    _validate_acceptance_document(acceptance)

    root = repo_root.expanduser().resolve(strict=True)
    production = _validate_production_spec(
        path=production_spec_path,
        repo_root=root,
    )
    if (
        acceptance.get("production_spec_sha256") != production.spec_sha256
        or acceptance.get("artifact_lock_sha256") != production.artifact_lock_sha256
        or acceptance.get("production_evidence") != production.audit_payload()
    ):
        raise CalderCanaryError(
            "production spec differs or effective artifacts differ from the passing "
            "canary acceptance"
        )
    source_replay_oracle = _load_source_replay_oracle(root)
    _validate_source_replay_campaign_binding(
        source_replay_oracle,
        environment_lock_sha256=str(acceptance.get("environment_lock_sha256", "")),
        dataset_fingerprint=str(acceptance.get("dataset_fingerprint", "")),
        dataset_content_sha256=str(acceptance.get("dataset_content_sha256", "")),
        production=production,
    )
    source_paths = _acceptance_source_paths(acceptance)
    with tempfile.TemporaryDirectory(prefix="modssc-calder-replay-") as temporary:
        replay_path = Path(temporary) / "acceptance.json"
        replay_report = validate_calder_canary(
            repo_root=root,
            artifact_lock_path=source_paths["artifact_lock"],
            manifest_path=source_paths["manifest"],
            meta_path=source_paths["meta"],
            reconcile_path=source_paths["reconcile"],
            production_spec_path=production_spec_path,
            output_path=replay_path,
        )
        replayed = _read_mapping(
            Path(replay_report.output_path),
            label="replayed Calder canary acceptance",
        )
    if replay_report.status != "passed" or replayed != acceptance:
        raise CalderCanaryError(
            "Calder canary acceptance differs from its replayed source evidence"
        )
    return acceptance


def verify_embedded_calder_release_evidence(
    manifest_path: Path,
    *,
    manifest_meta: Mapping[str, Any],
    tasks: Sequence[Any],
) -> dict[str, Any]:
    """Verify the production campaign's copied canary proof against its tasks."""

    if manifest_meta.get("campaign_id") != PRODUCTION_CAMPAIGN_ID:
        raise CalderCanaryError(
            "embedded Calder release evidence is only valid for its production campaign"
        )
    release = _mapping(
        manifest_meta.get("release_evidence"),
        label="campaign release evidence metadata",
    )
    if set(release) != {
        "kind",
        "path",
        "file_sha256",
        "acceptance_sha256",
        "canary_manifest_sha256",
        "production_evidence",
    }:
        raise CalderCanaryError("campaign release evidence metadata fields differ")
    if release.get("kind") != ACCEPTANCE_KIND:
        raise CalderCanaryError("campaign release evidence kind differs")
    relative = release.get("path")
    if (
        not isinstance(relative, str)
        or relative != "release-evidence.json"
        or relative.startswith("/")
        or ".." in relative.split("/")
    ):
        raise CalderCanaryError("campaign release evidence path differs")
    campaign_dir = manifest_path.expanduser().resolve(strict=True).parent
    candidate = campaign_dir / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(campaign_dir)
    except (OSError, ValueError) as exc:
        raise CalderCanaryError(
            "campaign release evidence is outside the campaign directory"
        ) from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise CalderCanaryError("campaign release evidence must be a regular non-symlink file")
    expected_file_sha256 = _require_sha256(
        release.get("file_sha256"),
        label="campaign release evidence file SHA-256",
    )
    if _sha256_file(resolved) != expected_file_sha256:
        raise CalderCanaryError("campaign release evidence file SHA-256 differs")

    acceptance = _read_mapping(resolved, label="embedded Calder canary acceptance")
    _validate_acceptance_document(acceptance)
    if (
        release.get("acceptance_sha256") != acceptance.get("acceptance_sha256")
        or release.get("canary_manifest_sha256") != acceptance.get("manifest_sha256")
        or release.get("production_evidence") != acceptance.get("production_evidence")
    ):
        raise CalderCanaryError("campaign release metadata differs from the embedded acceptance")

    task_sequence = list(tasks)
    if len(task_sequence) != 1000 or any(
        task.campaign_id != PRODUCTION_CAMPAIGN_ID
        or task.track != "paper"
        or task.assigned_site != "local-cpu"
        or task.resource_profile != "cpu_graph"
        for task in task_sequence
    ):
        raise CalderCanaryError("Calder production task population differs")
    by_config: dict[str, list[Any]] = {}
    for task in task_sequence:
        by_config.setdefault(task.config_path, []).append(task)
    if len(by_config) != 10 or any(
        sorted(task.seed for task in config_tasks) != list(range(100))
        or len({task.source_config_sha256 for task in config_tasks}) != 1
        for config_tasks in by_config.values()
    ):
        raise CalderCanaryError("Calder production tasks do not contain 100 unique seeds per cell")
    production_evidence = _mapping(
        acceptance.get("production_evidence"),
        label="embedded production evidence",
    )
    config_evidence = _mapping(
        production_evidence.get("effective_configs"),
        label="embedded effective configuration evidence",
    )
    actual_configs = {
        config_path: config_tasks[0].source_config_sha256
        for config_path, config_tasks in by_config.items()
    }
    if dict(config_evidence) != actual_configs:
        raise CalderCanaryError(
            "embedded effective configuration evidence differs from the task manifest"
        )
    if (
        acceptance.get("production_spec_sha256") != manifest_meta.get("spec_sha256")
        or production_evidence.get("production_spec_sha256") != manifest_meta.get("spec_sha256")
        or production_evidence.get("artifact_lock_sha256") != acceptance.get("artifact_lock_sha256")
    ):
        raise CalderCanaryError(
            "embedded production/specification evidence differs from campaign metadata"
        )
    first = task_sequence[0]
    if (
        acceptance.get("git_sha") != first.expected_git_sha
        or acceptance.get("git_diff_sha256") != first.expected_git_diff_sha256
        or acceptance.get("environment_lock_sha256") != first.environment_lock_sha256
        or acceptance.get("dataset_fingerprint") != first.expected_dataset_fingerprint
        or acceptance.get("dataset_content_sha256") != first.expected_dataset_content_sha256
        or any(
            task.expected_git_sha != first.expected_git_sha
            or task.expected_git_diff_sha256 != first.expected_git_diff_sha256
            or task.environment_lock_sha256 != first.environment_lock_sha256
            or task.expected_dataset_fingerprint != first.expected_dataset_fingerprint
            or task.expected_dataset_content_sha256 != first.expected_dataset_content_sha256
            for task in task_sequence
        )
    ):
        raise CalderCanaryError(
            "embedded acceptance code, environment, or dataset identity differs"
        )
    return acceptance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.replication_audit.calder.canary",
        description="Validate Calder seed-0 canaries and gate the production specification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--repo-root", type=Path, required=True)
    validate.add_argument("--artifact-lock", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--meta", type=Path)
    validate.add_argument("--reconcile", type=Path, required=True)
    validate.add_argument("--production-spec", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-production")
    verify.add_argument("--acceptance", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--production-spec", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_calder_canary(
                repo_root=args.repo_root,
                artifact_lock_path=args.artifact_lock,
                manifest_path=args.manifest,
                meta_path=args.meta,
                reconcile_path=args.reconcile,
                production_spec_path=args.production_spec,
                output_path=args.output,
            )
            print(json.dumps(asdict(report), indent=2, sort_keys=True))
            return 0 if report.status == "passed" else 1
        acceptance = verify_calder_canary_acceptance(
            args.acceptance,
            repo_root=args.repo_root,
            production_spec_path=args.production_spec,
        )
    except CalderCanaryError as exc:
        parser.exit(2, f"calder-canary: {exc}\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "acceptance_sha256": acceptance["acceptance_sha256"],
                "production_spec_sha256": acceptance["production_spec_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACCEPTANCE_KIND",
    "CalderCanaryError",
    "CalderCanaryReport",
    "validate_calder_canary",
    "verify_calder_canary_acceptance",
    "verify_embedded_calder_release_evidence",
]
