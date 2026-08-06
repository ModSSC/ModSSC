from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from bench.schema import ExperimentConfig
from bench.seed_sweep import apply_global_seed
from bench.utils.hashing import derive_seed, hash_any
from bench.utils.io import load_yaml
from bench.utils.runtime import collect_runtime_versions
from modssc.sampling.plan import SamplingPlan

from .catalog import scientific_scope, validate_method_scope
from .dcl_partition_lock import (
    DCL_DATASET_ID,
    DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS,
    DCL_DIAGNOSTIC_CONTROL_PROTOCOLS,
    DCL_DIAGNOSTIC_METHOD_PROFILE,
    DCL_DIAGNOSTIC_PROTOCOL_IDS,
    DCL_METHOD_ID,
    DCL_METHOD_PROFILE,
    DCL_PAPER_PROTOCOL_ID,
    build_task_partition_selection,
    is_dcl_vote_partition_replay_identity,
    load_dcl_partition_selection,
    resolve_repo_path,
    verify_dcl_partition_replay,
)
from .errors import CampaignError
from .manifest import finalize_task_row, sha256_file, write_manifest
from .models import CampaignTask, GeneratedCampaign
from .scientific_gates import load_gate_registry
from .spec import load_spec, string_list

_DCL_V2_VOTE_FEATURE_SCHEMA = [{"type": "nominal", "values": ["n", "y"]} for _ in range(16)]
_DCL_V2_VOTE_SAMPLING_PLAN = {
    "split": {
        "kind": "holdout",
        "test_fraction": 0.4482758620689655,
        "val_fraction": 0.0,
        "stratify": False,
        "shuffle": True,
    },
    "labeling": {
        "mode": "count",
        "value": 40,
        "strategy": "random",
        "min_per_class": 0,
        "per_class": False,
        "fixed_indices": None,
    },
    "imbalance": {"kind": "none"},
    "policy": {
        "respect_official_test": True,
        "use_official_graph_masks": True,
        "allow_override_official": False,
    },
}
_DCL_V2_VOTE_PREPROCESS_PLAN = {
    "output_key": "features.X",
    "steps": [
        {"id": "labels.encode"},
        {"id": "core.copy_raw"},
        {"id": "core.to_numpy"},
    ],
}


def _dcl_v2_classifier_specs() -> list[dict[str, Any]]:
    common = {
        "feature_schema": _DCL_V2_VOTE_FEATURE_SCHEMA,
        "missing_values": ["?"],
    }
    return [
        {
            "classifier_id": "gaussian_nb",
            "classifier_backend": "numpy",
            "classifier_params": {"alpha": 1.0, "fit_prior": True, **common},
        },
        {
            "classifier_id": "decision_tree",
            "classifier_backend": "numpy",
            "classifier_params": {
                "min_num_obj": 2,
                "unpruned": True,
                "binary_splits": False,
                "feature_schema": _DCL_V2_VOTE_FEATURE_SCHEMA,
                "missing_values": ["?"],
            },
        },
        {
            "classifier_id": "knn",
            "classifier_backend": "numpy",
            "classifier_params": {
                "k": 3,
                "metric": "euclidean",
                "weights": "uniform",
                **common,
            },
        },
    ]


def _raise_dcl_v2_core_mismatch(field: str) -> None:
    raise CampaignError(
        "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
        f"DCL Vote v2 scientific core differs at {field}",
    )


def _validate_dcl_v2_scientific_core(cfg: ExperimentConfig) -> None:
    """Reject a diagnostic card whose paper-relevant protocol has drifted."""

    if (
        cfg.dataset.id != DCL_DATASET_ID
        or cfg.dataset.download is not False
        or cfg.dataset.options != {}
    ):
        _raise_dcl_v2_core_mismatch("dataset")
    if cfg.sampling.plan != _DCL_V2_VOTE_SAMPLING_PLAN or cfg.sampling.replay is not None:
        _raise_dcl_v2_core_mismatch("sampling")
    if cfg.preprocess.fit_on != "train" or cfg.preprocess.plan != _DCL_V2_VOTE_PREPROCESS_PLAN:
        _raise_dcl_v2_core_mismatch("preprocess")
    if (
        cfg.run.seeds != list(range(1, 21))
        or cfg.run.seeded_sections != ["sampling", "preprocess"]
        or cfg.run.benchmark_mode is not False
        or cfg.run.allow_custom_factories is not False
    ):
        _raise_dcl_v2_core_mismatch("run")
    if (
        cfg.method.kind != "inductive"
        or cfg.method.method_id != DCL_METHOD_ID
        or cfg.method.profile != DCL_DIAGNOSTIC_METHOD_PROFILE
        or cfg.method.model is not None
        or cfg.method.device.device != "cpu"
        or cfg.method.device.dtype != "float32"
        or cfg.method.device.resolved_device is not None
    ):
        _raise_dcl_v2_core_mismatch("method identity")
    variant_keys = {
        "confidence_estimator",
        "confidence_interval",
        "confidence_folds",
        "confidence_seed",
        "control_mode",
    }
    method_core = {
        key: value for key, value in cfg.method.params.items() if key not in variant_keys
    }
    if method_core != {
        "training_mode": "confidence_weighted",
        "max_iter": 20,
        "confidence_level": 0.95,
        "min_confidence": 0.5,
        "diagnostic_trace": True,
        "classifier_specs": _dcl_v2_classifier_specs(),
    }:
        _raise_dcl_v2_core_mismatch("method parameters")
    if cfg.evaluation.split_for_model_selection is not None or cfg.evaluation.metrics != [
        "accuracy",
        "macro_f1",
    ]:
        _raise_dcl_v2_core_mismatch("evaluation")
    if any(
        section is not None
        for section in (
            cfg.graph,
            cfg.views,
            cfg.augmentation,
            cfg.search,
            cfg.limits,
        )
    ):
        _raise_dcl_v2_core_mismatch("auxiliary sections")


class _AtomicCampaignDirectory:
    """Build a new campaign beside its destination and publish it once complete."""

    def __init__(self, output_dir: Path) -> None:
        self.destination = Path(output_dir).resolve(strict=False)
        self.lock_fd: int | None = None
        self.staging: Path | None = None

    def _close(self) -> None:
        if self.staging is not None:
            shutil.rmtree(self.staging, ignore_errors=True)
            self.staging = None
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self.lock_fd)
                self.lock_fd = None

    def __enter__(self) -> Path:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.destination.parent / f".{self.destination.name}.generate.lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self.lock_fd = os.open(lock_path, flags, 0o600)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.path.lexists(self.destination):
                raise CampaignError(
                    "E_CAMPAIGN_DESTINATION_EXISTS",
                    f"campaign destination already exists: {self.destination}",
                )
            self.staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.destination.name}.staging-",
                    dir=self.destination.parent,
                )
            )
            return self.staging
        except CampaignError:
            self._close()
            raise
        except (BlockingIOError, OSError) as exc:
            self._close()
            raise CampaignError(
                "E_CAMPAIGN_DESTINATION_BUSY",
                f"cannot exclusively lock campaign destination: {self.destination}",
            ) from exc

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        _ = exc, traceback
        try:
            if exc_type is None:
                if self.staging is None:
                    raise RuntimeError("campaign staging directory is unavailable")
                if os.path.lexists(self.destination):
                    raise CampaignError(
                        "E_CAMPAIGN_DESTINATION_EXISTS",
                        f"campaign destination appeared during generation: {self.destination}",
                    )
                os.rename(self.staging, self.destination)
                self.staging = None
        finally:
            self._close()
        return False


def _is_template_placeholder(value: Any) -> bool:
    return isinstance(value, str) and (value == "unlocked" or value.startswith("REPLACE_WITH_"))


def _template_paths(value: Any, *, prefix: str = "") -> list[str]:
    if _is_template_placeholder(value):
        return [prefix or "<root>"]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_template_paths(child, prefix=child_prefix))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_template_paths(child, prefix=f"{prefix}[{index}]"))
        return paths
    return []


def _reject_unpinned_production_tasks(
    spec: Mapping[str, Any], payloads: Sequence[Mapping[str, Any]], *, repo_root: Path
) -> None:
    code = spec.get("code")
    require_clean = code.get("require_clean", True) if isinstance(code, Mapping) else True
    if require_clean is not True:
        return
    for payload in payloads:
        for field in (
            "expected_git_sha",
            "environment_lock_sha256",
            "expected_dataset_fingerprint",
            "expected_dataset_content_sha256",
        ):
            value = payload.get(field)
            if not isinstance(value, str) or not value.strip() or _is_template_placeholder(value):
                raise CampaignError(
                    "E_CAMPAIGN_TEMPLATE_PLACEHOLDER",
                    f"production task does not contain an immutable value for {field}",
                )
    for config_path in sorted({str(payload["config_path"]) for payload in payloads}):
        placeholders = _template_paths(load_yaml(repo_root / config_path))
        if placeholders:
            raise CampaignError(
                "E_CAMPAIGN_TEMPLATE_PLACEHOLDER",
                f"production config {config_path} contains template values at {placeholders}",
            )


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise CampaignError(
            "E_CAMPAIGN_CONFIG_OUTSIDE_REPO", f"config is outside repository: {path}"
        ) from exc


def _resolve_code_identity(
    spec: Mapping[str, Any], *, repo_root: Path
) -> tuple[str, str | None, str]:
    raw_code = spec.get("code")
    if not isinstance(raw_code, Mapping):
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "code must be a mapping")
    versions = collect_runtime_versions(repo_root=repo_root)
    actual_sha = versions.get("git_sha")
    actual_dirty = versions.get("git_dirty")
    actual_diff = versions.get("git_diff_sha256")

    expected_sha = raw_code.get("git_sha") or actual_sha
    if not isinstance(expected_sha, str) or not expected_sha.strip():
        raise CampaignError("E_CAMPAIGN_CODE_UNPINNED", "code.git_sha must be pinned")
    if isinstance(actual_sha, str) and actual_sha != expected_sha:
        raise CampaignError(
            "E_CAMPAIGN_CODE_MISMATCH",
            f"spec expects Git {expected_sha}, current repository is {actual_sha}",
        )

    require_clean = raw_code.get("require_clean", True)
    if not isinstance(require_clean, bool):
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "code.require_clean must be boolean")
    if require_clean and actual_dirty is not False:
        raise CampaignError(
            "E_CAMPAIGN_DIRTY_WORKTREE",
            "production campaign generation requires a clean Git worktree",
        )

    explicit_diff = raw_code.get("git_diff_sha256")
    if explicit_diff is not None and not isinstance(explicit_diff, str):
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID", "code.git_diff_sha256 must be a string or null"
        )
    expected_diff = explicit_diff
    if expected_diff is None and actual_diff is not None:
        expected_diff = str(actual_diff)
    if explicit_diff is not None and actual_diff is not None and explicit_diff != actual_diff:
        raise CampaignError(
            "E_CAMPAIGN_CODE_MISMATCH", "code.git_diff_sha256 differs from the worktree"
        )

    lock_file = raw_code.get("environment_lock_file")
    lock_digest = raw_code.get("environment_lock_sha256")
    if lock_file is not None:
        if not isinstance(lock_file, str) or not lock_file:
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID", "code.environment_lock_file must be a path"
            )
        lock_path = (repo_root / lock_file).resolve()
        if not lock_path.is_file():
            raise CampaignError(
                "E_CAMPAIGN_ENVIRONMENT_UNPINNED", f"environment lock not found: {lock_file}"
            )
        computed = sha256_file(lock_path)
        if lock_digest is not None and lock_digest != computed:
            raise CampaignError(
                "E_CAMPAIGN_ENVIRONMENT_MISMATCH",
                "environment lock digest does not match the lock file",
            )
        lock_digest = computed
    if not isinstance(lock_digest, str) or not lock_digest.strip():
        raise CampaignError(
            "E_CAMPAIGN_ENVIRONMENT_UNPINNED",
            "pin code.environment_lock_sha256 or environment_lock_file",
        )
    return expected_sha, expected_diff, lock_digest


def _path_metadata(relative_path: str) -> tuple[str | None, str | None]:
    parts = Path(relative_path).parts
    regime = next((part for part in parts if part.startswith("R") and part[1:].isdigit()), None)
    modality = None
    for candidate in ("tabular", "vision", "text", "audio", "graph"):
        if candidate in parts:
            modality = candidate
            break
    return modality, regime


def _resolve_dataset_lock(
    selection: Mapping[str, Any], *, repo_root: Path
) -> tuple[dict[str, tuple[str, str | None]], str]:
    lock_value = selection.get("dataset_lock_file")
    if not isinstance(lock_value, str) or not lock_value.strip():
        raise CampaignError(
            "E_CAMPAIGN_DATASET_UNPINNED",
            "standardized campaigns require selection.dataset_lock_file",
        )
    lock_path = (repo_root / lock_value).resolve()
    try:
        lock_path.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignError(
            "E_CAMPAIGN_DATASET_UNPINNED", "dataset lock must be inside the repository"
        ) from exc
    if not lock_path.is_file():
        raise CampaignError("E_CAMPAIGN_DATASET_UNPINNED", f"dataset lock not found: {lock_value}")
    raw = load_yaml(lock_path)
    schema_version = raw.get("schema_version")
    if schema_version not in {1, 2} or not isinstance(raw.get("datasets"), Mapping):
        raise CampaignError(
            "E_CAMPAIGN_DATASET_UNPINNED", f"invalid dataset lock schema: {lock_value}"
        )
    fingerprints: dict[str, tuple[str, str | None]] = {}
    for raw_id, raw_entry in raw["datasets"].items():
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise CampaignError(
                "E_CAMPAIGN_DATASET_UNPINNED", "dataset lock identifiers must be strings"
            )
        if schema_version == 1:
            raw_fingerprint = raw_entry
            raw_content = None
        elif isinstance(raw_entry, Mapping):
            raw_fingerprint = raw_entry.get("fingerprint")
            raw_content = raw_entry.get("content_sha256")
        else:
            raw_fingerprint = None
            raw_content = None
        if not isinstance(raw_fingerprint, str) or not raw_fingerprint.strip():
            raise CampaignError(
                "E_CAMPAIGN_DATASET_UNPINNED",
                f"dataset lock fingerprint is missing for {raw_id}",
            )
        if schema_version == 2 and (not isinstance(raw_content, str) or not raw_content.strip()):
            raise CampaignError(
                "E_CAMPAIGN_DATASET_UNPINNED",
                f"dataset lock content digest is missing for {raw_id}",
            )
        fingerprints[raw_id] = (
            raw_fingerprint,
            raw_content if isinstance(raw_content, str) else None,
        )
    return fingerprints, sha256_file(lock_path)


def _rule_matches(
    rule: Mapping[str, Any],
    *,
    method_id: str,
    method_kind: str,
    device: str,
    modality: str | None,
    track: str,
) -> bool:
    dimensions: dict[str, str | None] = {
        "methods": method_id,
        "method_kinds": method_kind,
        "devices": device,
        "modalities": modality,
        "tracks": track,
    }
    for field, actual in dimensions.items():
        if field not in rule:
            continue
        allowed = rule[field]
        if not isinstance(allowed, list) or any(not isinstance(item, str) for item in allowed):
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID", f"profile rule {field} must be a list[str]"
            )
        if actual not in allowed:
            return False
    return True


def _resolve_profile(
    spec: Mapping[str, Any],
    *,
    method_id: str,
    method_kind: str,
    device: str,
    modality: str | None,
    track: str,
    explicit_profile: str | None,
    explicit_site: str | None,
) -> tuple[str, str]:
    if explicit_profile is not None:
        if not explicit_profile:
            raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "resource_profile cannot be empty")
        site = explicit_site or spec.get("default_site")
        if not isinstance(site, str) or not site:
            raise CampaignError(
                "E_CAMPAIGN_PROFILE_UNRESOLVED",
                f"no site assigned to explicit profile {explicit_profile}",
            )
        return explicit_profile, site

    rules = spec.get("profile_rules")
    if not isinstance(rules, list):
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "profile_rules must be a list")
    matches: list[Mapping[str, Any]] = []
    for raw_rule in rules:
        if not isinstance(raw_rule, Mapping):
            raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "profile rule must be a mapping")
        if _rule_matches(
            raw_rule,
            method_id=method_id,
            method_kind=method_kind,
            device=device,
            modality=modality,
            track=track,
        ):
            matches.append(raw_rule)
    if len(matches) != 1:
        raise CampaignError(
            "E_CAMPAIGN_PROFILE_UNRESOLVED",
            f"expected one profile rule for {method_id}/{device}/{modality}, found {len(matches)}",
        )
    profile = matches[0].get("profile")
    site = matches[0].get("site", spec.get("default_site"))
    if not isinstance(profile, str) or not profile:
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "profile rule needs profile")
    if not isinstance(site, str) or not site:
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "profile rule needs site")
    return profile, site


@lru_cache(maxsize=8)
def _cached_gate_policy_identity(
    registry_path: Path,
    mtime_ns: int,
    size_bytes: int,
) -> tuple[str, str]:
    # The stat values are part of the cache key so a policy changed during a
    # long-lived process is never authenticated with stale registry content.
    _ = mtime_ns, size_bytes
    registry = load_gate_registry(registry_path)
    return registry.registry_id, sha256_file(registry_path)


def _gate_policy_identity(registry_path: Path) -> tuple[str, str]:
    stat = registry_path.stat()
    return _cached_gate_policy_identity(registry_path, stat.st_mtime_ns, stat.st_size)


def _resolved_scientific_scope(spec: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    governance = scientific_scope(spec)
    registry_path = repo_root / "bench" / "campaigns" / "scientific-gates.yaml"
    if not registry_path.is_file():
        raise CampaignError(
            "E_SCIENTIFIC_GATE_SCHEMA",
            f"scientific gate registry not found: {registry_path}",
        )
    registry_id, digest = _gate_policy_identity(registry_path)
    if registry_id != governance["gate_policy_id"]:
        raise CampaignError(
            "E_SCIENTIFIC_GATE_SCHEMA",
            "scientific_scope.gate_policy_id differs from the tracked registry",
        )
    configured_digest = governance["gate_policy_sha256"]
    if configured_digest not in {"from_registry", digest}:
        raise CampaignError(
            "E_SCIENTIFIC_GATE_SCHEMA",
            "scientific_scope.gate_policy_sha256 differs from the tracked registry",
        )
    return {**governance, "gate_policy_sha256": digest}


def _task_payloads_for_config(
    *,
    spec: Mapping[str, Any],
    repo_root: Path,
    config_path: Path,
    seeds: Iterable[int],
    track: str,
    expected_git_sha: str,
    expected_git_diff_sha256: str | None,
    environment_lock_sha256: str,
    dataset_lock_sha256: str | None = None,
    protocol_id: str | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_dataset_content_sha256: str | None = None,
    fidelity_status: str | None = None,
    explicit_profile: str | None = None,
    explicit_site: str | None = None,
    model_seed_policy: str = "derived",
) -> list[dict[str, Any]]:
    raw = load_yaml(config_path)
    cfg = ExperimentConfig.from_dict(raw)
    relative = _repo_relative(config_path, repo_root)
    modality, regime = _path_metadata(relative)
    profile, site = _resolve_profile(
        spec,
        method_id=cfg.method.method_id,
        method_kind=cfg.method.kind,
        device=cfg.method.device.device,
        modality=modality,
        track=track,
        explicit_profile=explicit_profile,
        explicit_site=explicit_site,
    )
    governance = _resolved_scientific_scope(spec, repo_root=repo_root)
    validate_method_scope(
        method_id=cfg.method.method_id,
        claim_scope_id=str(governance["claim_scope_id"]),
    )
    source_digest = sha256_file(config_path)
    seed_values = list(seeds)
    if not seed_values:
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID",
            f"at least one seed is required for {relative}",
        )
    payloads: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for raw_seed in seed_values:
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID", f"seed for {relative} must be an integer"
            )
        seed = int(raw_seed)
        if seed in seen_seeds:
            raise CampaignError(
                "E_CAMPAIGN_DUPLICATE_TASK", f"duplicate seed {seed} for {relative}"
            )
        seen_seeds.add(seed)
        effective = apply_global_seed(
            raw,
            seed=seed,
            seeded_sections=cfg.run.seeded_sections,
        )
        effective_cfg = ExperimentConfig.from_dict(effective)
        dataset_block = effective.get("dataset", {})
        sampling_block = effective.get("sampling", {})
        dataset_request_sha256 = hash_any(dataset_block)
        split_request_sha256 = hash_any(
            {
                "dataset_request_sha256": dataset_request_sha256,
                "sampling": sampling_block,
            }
        )
        dataset_options = (
            dataset_block.get("options", {}) if isinstance(dataset_block, Mapping) else {}
        )
        configured_data_seed = (
            dataset_options.get("seed") if isinstance(dataset_options, Mapping) else None
        )
        data_seed = int(configured_data_seed) if isinstance(configured_data_seed, int) else seed
        configured_sampling_seed = (
            sampling_block.get("seed") if isinstance(sampling_block, Mapping) else None
        )
        sampling_seed = (
            int(configured_sampling_seed)
            if isinstance(configured_sampling_seed, int)
            and not isinstance(configured_sampling_seed, bool)
            else int(derive_seed(seed, "sampling"))
        )
        plan_raw = sampling_block.get("plan", {}) if isinstance(sampling_block, Mapping) else {}
        normalized_plan = SamplingPlan.from_dict(dict(plan_raw))
        sampling_component_seeds = normalized_plan.component_seeds.resolve(sampling_seed)
        split_seed = sampling_component_seeds["split"]
        expected_split_fingerprint = None
        if expected_dataset_fingerprint is not None:
            expected_split_fingerprint = hash_any(
                {
                    "schema_version": 1,
                    "dataset_fingerprint": expected_dataset_fingerprint,
                    "plan": normalized_plan.as_dict(),
                    "seed": sampling_seed,
                }
            )
        labeling = (
            sampling_block.get("plan", {}).get("labeling", {})
            if isinstance(sampling_block, Mapping)
            and isinstance(sampling_block.get("plan"), Mapping)
            else {}
        )
        mode = (
            labeling.get("mode", "unspecified") if isinstance(labeling, Mapping) else "unspecified"
        )
        value = (
            labeling.get("value", "unspecified") if isinstance(labeling, Mapping) else "unspecified"
        )
        payloads.append(
            {
                "campaign_id": str(spec["campaign_id"]),
                "track": track,
                "protocol_id": protocol_id,
                "config_path": relative,
                "source_config_sha256": source_digest,
                "method_profile": effective_cfg.method.profile,
                "label_budget": f"{mode}:{value}",
                "required_seed_count": len(seed_values),
                "seed": seed,
                "data_seed": data_seed,
                "split_seed": split_seed,
                "sampling_component_seeds": sampling_component_seeds,
                "model_seed": (
                    seed if model_seed_policy == "literal" else int(derive_seed(seed, "method"))
                ),
                "seeded_sections": effective_cfg.run.seeded_sections,
                "method_id": cfg.method.method_id,
                "method_kind": cfg.method.kind,
                "dataset_id": cfg.dataset.id,
                "modality": modality,
                "regime": regime,
                "resource_profile": profile,
                "assigned_site": site,
                "expected_git_sha": expected_git_sha,
                "expected_git_diff_sha256": expected_git_diff_sha256,
                "environment_lock_sha256": environment_lock_sha256,
                "dataset_lock_sha256": dataset_lock_sha256,
                "expected_dataset_fingerprint": expected_dataset_fingerprint,
                "expected_dataset_content_sha256": expected_dataset_content_sha256,
                "dataset_request_sha256": dataset_request_sha256,
                "split_request_sha256": split_request_sha256,
                "expected_split_fingerprint": expected_split_fingerprint,
                "fidelity_status": fidelity_status,
                **governance,
            }
        )
    return payloads


def _standardized_payloads(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_git_sha: str,
    expected_git_diff_sha256: str | None,
    environment_lock_sha256: str,
) -> list[dict[str, Any]]:
    selection = spec.get("selection")
    if not isinstance(selection, Mapping):
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "selection must be a mapping")
    dataset_locks, dataset_lock_sha256 = _resolve_dataset_lock(selection, repo_root=repo_root)
    root_value = selection.get("config_root")
    if not isinstance(root_value, str) or not root_value:
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "selection.config_root is required")
    methods = set(string_list(selection.get("methods"), field="selection.methods"))
    selected_filters: dict[str, set[str] | None] = {}
    for field in ("regimes", "modalities", "datasets"):
        raw_values = selection.get(field)
        if raw_values is None:
            selected_filters[field] = None
            continue
        values = string_list(raw_values, field=f"selection.{field}")
        if len(values) != len(set(values)):
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID",
                f"selection.{field} must contain unique values",
            )
        selected_filters[field] = set(values)

    raw_seeds = selection.get("seeds", "from_config")
    selected_seeds: list[int] | None
    if raw_seeds == "from_config":
        selected_seeds = None
    elif isinstance(raw_seeds, list) and raw_seeds:
        selected_seeds = []
        for seed in raw_seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise CampaignError(
                    "E_CAMPAIGN_SPEC_INVALID",
                    "standardized selection.seeds must contain only integers",
                )
            selected_seeds.append(seed)
        if len(selected_seeds) != len(set(selected_seeds)):
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID",
                "standardized selection.seeds must contain unique values",
            )
    else:
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID",
            "standardized selection.seeds must be 'from_config' or a non-empty list[int]",
        )
    config_root = (repo_root / root_value).resolve()
    if not config_root.is_dir():
        raise CampaignError(
            "E_CAMPAIGN_SPEC_INVALID", f"configuration root not found: {root_value}"
        )
    payloads: list[dict[str, Any]] = []
    found_methods: set[str] = set()
    for config_path in sorted(config_root.rglob("*.yaml")):
        if config_path.name == "regime_manifest.yaml":
            continue
        relative_parts = config_path.relative_to(config_root).parts
        if len(relative_parts) < 5 or relative_parts[2] not in methods:
            continue
        modality, regime = _path_metadata(_repo_relative(config_path, repo_root))
        selected_regimes = selected_filters["regimes"]
        if selected_regimes is not None and regime not in selected_regimes:
            continue
        selected_modalities = selected_filters["modalities"]
        if selected_modalities is not None and modality not in selected_modalities:
            continue
        raw = load_yaml(config_path)
        cfg = ExperimentConfig.from_dict(raw)
        if cfg.method.method_id not in methods:
            continue
        selected_datasets = selected_filters["datasets"]
        if selected_datasets is not None and cfg.dataset.id not in selected_datasets:
            continue
        dataset_lock = dataset_locks.get(cfg.dataset.id)
        if dataset_lock is None:
            raise CampaignError(
                "E_CAMPAIGN_DATASET_UNPINNED",
                f"dataset lock has no fingerprint for {cfg.dataset.id}",
            )
        expected_dataset_fingerprint, expected_dataset_content_sha256 = dataset_lock
        found_methods.add(cfg.method.method_id)
        seeds = selected_seeds if selected_seeds is not None else (cfg.run.seeds or [cfg.run.seed])
        payloads.extend(
            _task_payloads_for_config(
                spec=spec,
                repo_root=repo_root,
                config_path=config_path,
                seeds=seeds,
                track="standardized",
                expected_git_sha=expected_git_sha,
                expected_git_diff_sha256=expected_git_diff_sha256,
                environment_lock_sha256=environment_lock_sha256,
                dataset_lock_sha256=dataset_lock_sha256,
                expected_dataset_fingerprint=expected_dataset_fingerprint,
                expected_dataset_content_sha256=expected_dataset_content_sha256,
            )
        )
    missing = sorted(methods - found_methods)
    if missing:
        raise CampaignError(
            "E_CAMPAIGN_SELECTION_EMPTY", f"no configurations found for methods: {missing}"
        )
    return payloads


def _paper_payloads(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_git_sha: str,
    expected_git_diff_sha256: str | None,
    environment_lock_sha256: str,
) -> list[dict[str, Any]]:
    governance = _resolved_scientific_scope(spec, repo_root=repo_root)
    cells = spec.get("cells")
    if not isinstance(cells, list) or not cells:
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "paper campaign cells must be non-empty")
    payloads: list[dict[str, Any]] = []
    for raw_cell in cells:
        if not isinstance(raw_cell, Mapping):
            raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "paper cell must be a mapping")
        protocol_id = raw_cell.get("protocol_id")
        config_value = raw_cell.get("config")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "paper cell needs protocol_id")
        if not isinstance(config_value, str) or not config_value:
            raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "paper cell needs config")
        config_path = (repo_root / config_value).resolve()
        if not config_path.is_file():
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID", f"paper config not found: {config_value}"
            )
        cfg = ExperimentConfig.from_dict(load_yaml(config_path))
        if not cfg.method.profile.startswith("paper:"):
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID",
                f"paper config must use a paper:* method profile: {config_value}",
            )
        model_seed_policy = raw_cell.get("model_seed_policy", "derived")
        if model_seed_policy not in {"derived", "literal"}:
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID",
                f"paper cell has invalid model_seed_policy: {protocol_id}",
            )
        is_official_grand_profile = (
            cfg.method.method_id == "grand"
            and cfg.method.profile.startswith("paper:feng2020-cora-table1")
        )
        if is_official_grand_profile and model_seed_policy != "literal":
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID",
                "the GRAND Cora/Table 1 profile requires model_seed_policy=literal",
            )
        expected_dataset_fingerprint = raw_cell.get("expected_dataset_fingerprint")
        if (
            not isinstance(expected_dataset_fingerprint, str)
            or not expected_dataset_fingerprint.strip()
        ):
            raise CampaignError(
                "E_CAMPAIGN_DATASET_UNPINNED",
                f"paper cell must pin expected_dataset_fingerprint: {protocol_id}",
            )
        expected_dataset_content_sha256 = raw_cell.get("expected_dataset_content_sha256")
        if expected_dataset_content_sha256 is not None and (
            not isinstance(expected_dataset_content_sha256, str)
            or not expected_dataset_content_sha256.strip()
        ):
            raise CampaignError(
                "E_CAMPAIGN_DATASET_UNPINNED",
                f"paper cell has invalid expected_dataset_content_sha256: {protocol_id}",
            )
        fidelity_status = raw_cell.get("fidelity_status")
        if fidelity_status not in {"paper_matched", "paper_approx", "not_claimable"}:
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID",
                f"paper cell must declare a valid fidelity_status: {protocol_id}",
            )
        raw_seeds = raw_cell.get("seeds", "from_config")
        raw_partition_selection = raw_cell.get("partition_selection")
        canonical_dcl_vote_test_cell = (
            cfg.method.method_id == DCL_METHOD_ID
            and cfg.method.profile == DCL_METHOD_PROFILE
            and cfg.dataset.id == DCL_DATASET_ID
            and "test" in cfg.evaluation.report_splits
        )
        diagnostic_dcl_vote_cell = (
            cfg.method.method_id == DCL_METHOD_ID
            and cfg.method.profile == DCL_DIAGNOSTIC_METHOD_PROFILE
            and cfg.dataset.id == DCL_DATASET_ID
        )
        if canonical_dcl_vote_test_cell and protocol_id != DCL_PAPER_PROTOCOL_ID:
            raise CampaignError(
                "E_CAMPAIGN_PARTITION_SELECTION_REQUIRED",
                "DCL Vote test reporting requires the canonical paper protocol id",
            )
        if protocol_id == DCL_PAPER_PROTOCOL_ID and not canonical_dcl_vote_test_cell:
            raise CampaignError(
                "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                "the DCL Vote paper protocol id requires its exact method, profile, "
                "dataset, and test report",
            )
        if diagnostic_dcl_vote_cell:
            _validate_dcl_v2_scientific_core(cfg)
            expected_control_mode = DCL_DIAGNOSTIC_CONTROL_PROTOCOLS.get(protocol_id)
            expected_confidence = DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS.get(protocol_id)
            if expected_control_mode is None and expected_confidence is None:
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_REQUIRED",
                    "DCL Vote v2 diagnostics require a registered protocol id",
                )
            if cfg.method.params.get("diagnostic_trace") is not True:
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                    "DCL Vote v2 diagnostics require diagnostic_trace=true",
                )
            if expected_control_mode is not None:
                if cfg.evaluation.report_splits != ["test"]:
                    raise CampaignError(
                        "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                        "DCL Vote v2 Table 3 controls must report exactly the test split",
                    )
                if (
                    cfg.method.params.get("control_mode") != expected_control_mode
                    or cfg.method.params.get("confidence_estimator") != "training_accuracy"
                    or cfg.method.params.get("confidence_interval") != "wald"
                    or cfg.method.params.get("confidence_folds") != 10
                    or cfg.method.params.get("confidence_seed") != 0
                    or cfg.method.params.get("confidence_level") != 0.95
                ):
                    raise CampaignError(
                        "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                        "DCL Vote v2 control protocol and immutable parameters differ",
                    )
            else:
                assert expected_confidence is not None
                expected_estimator, expected_interval = expected_confidence
                if cfg.evaluation.report_splits != ["train_labeled"]:
                    raise CampaignError(
                        "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                        "DCL Vote v2 confidence diagnostics must report exactly train_labeled",
                    )
                if (
                    cfg.evaluation.split_for_model_selection is not None
                    or cfg.method.params.get("control_mode") != "dcl"
                    or cfg.method.params.get("confidence_estimator") != expected_estimator
                    or cfg.method.params.get("confidence_interval") != expected_interval
                    or cfg.method.params.get("confidence_folds") != 10
                    or cfg.method.params.get("confidence_seed") != 0
                    or cfg.method.params.get("confidence_level") != 0.95
                ):
                    raise CampaignError(
                        "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                        "DCL Vote v2 confidence protocol and method parameters differ",
                    )
        if protocol_id in DCL_DIAGNOSTIC_PROTOCOL_IDS and not diagnostic_dcl_vote_cell:
            raise CampaignError(
                "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                "DCL Vote v2 diagnostic protocol requires its exact method, profile, and dataset",
            )
        requires_partition_selection = is_dcl_vote_partition_replay_identity(
            track="paper",
            method_id=cfg.method.method_id,
            method_profile=cfg.method.profile,
            dataset_id=cfg.dataset.id,
            protocol_id=protocol_id,
        )
        partition_selection_by_seed: dict[int, dict[str, Any]] | None = None
        if requires_partition_selection:
            if raw_seeds != "from_partition_selection":
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_REQUIRED",
                    "DCL Vote locked runs must derive seeds from the frozen partition selection",
                )
            if not isinstance(raw_partition_selection, Mapping) or set(raw_partition_selection) != {
                "path",
                "sha256",
                "replay_root",
            }:
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_REQUIRED",
                    "DCL Vote locked runs require partition_selection path, "
                    "sha256, and replay_root",
                )
            selection_value = raw_partition_selection.get("path")
            selection_sha256 = raw_partition_selection.get("sha256")
            replay_root_value = raw_partition_selection.get("replay_root")
            if not isinstance(selection_value, str) or not isinstance(replay_root_value, str):
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                    "partition selection paths must be strings",
                )
            selection_path = resolve_repo_path(
                repo_root,
                selection_value,
                label="partition_selection.path",
            )
            replay_root = resolve_repo_path(
                repo_root,
                replay_root_value,
                label="partition_selection.replay_root",
            )
            if not isinstance(selection_sha256, str):
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                    "partition_selection.sha256 must be a SHA-256 digest",
                )
            lock = load_dcl_partition_selection(
                selection_path,
                expected_sha256=selection_sha256,
                expected_dataset_fingerprint=expected_dataset_fingerprint,
                expected_dataset_content_sha256=expected_dataset_content_sha256,
            )
            if bool(governance["claim_eligible"]) and not lock.claim_eligible:
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_PRIVATE_REQUIRED",
                    "claim-eligible DCL generation requires the authenticated private source "
                    "bundle; the public descriptor supports non-claimable replay only",
                )
            seeds = [entry.seed for entry in lock.selected]
            configured_seeds = cfg.run.seeds or [cfg.run.seed]
            if configured_seeds != seeds:
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_MISMATCH",
                    "DCL Vote configuration seeds differ from the frozen selected partitions",
                )
            selection_relative = _repo_relative(selection_path, repo_root)
            partition_selection_by_seed = {}
            raw_config = load_yaml(config_path)
            for entry in lock.selected:
                replay_dir = replay_root / f"seed-{entry.seed:03d}"
                replay_relative = _repo_relative(replay_dir, repo_root)
                evidence = build_task_partition_selection(
                    selection_path=selection_relative,
                    lock=lock,
                    entry=entry,
                    replay_path=replay_relative,
                )
                effective = apply_global_seed(
                    raw_config,
                    seed=entry.seed,
                    seeded_sections=cfg.run.seeded_sections,
                )
                sampling = effective.get("sampling")
                plan = sampling.get("plan") if isinstance(sampling, Mapping) else None
                if not isinstance(plan, Mapping):
                    raise CampaignError(
                        "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                        "effective sampling plan is missing",
                    )
                runtime_evidence = dict(evidence)
                runtime_evidence["selection_path"] = str(selection_path)
                runtime_evidence["replay_path"] = str(replay_dir)
                verify_dcl_partition_replay(
                    runtime_evidence,
                    expected_seed=entry.seed,
                    expected_dataset_fingerprint=expected_dataset_fingerprint,
                    expected_plan=plan,
                )
                partition_selection_by_seed[entry.seed] = evidence
        else:
            if raw_partition_selection is not None or raw_seeds == "from_partition_selection":
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                    "partition_selection is only valid for a registered DCL Vote locked cell",
                )
            if raw_seeds == "from_config":
                seeds = cfg.run.seeds or [cfg.run.seed]
            elif isinstance(raw_seeds, list):
                seeds = raw_seeds
            else:
                raise CampaignError(
                    "E_CAMPAIGN_SPEC_INVALID",
                    "paper cell seeds must be from_config, from_partition_selection, or list[int]",
                )
        cell_payloads = _task_payloads_for_config(
            spec=spec,
            repo_root=repo_root,
            config_path=config_path,
            seeds=seeds,
            track="paper",
            expected_git_sha=expected_git_sha,
            expected_git_diff_sha256=expected_git_diff_sha256,
            environment_lock_sha256=environment_lock_sha256,
            protocol_id=protocol_id,
            expected_dataset_fingerprint=expected_dataset_fingerprint,
            expected_dataset_content_sha256=expected_dataset_content_sha256,
            fidelity_status=str(fidelity_status),
            explicit_profile=raw_cell.get("resource_profile"),
            explicit_site=raw_cell.get("site"),
            model_seed_policy=str(model_seed_policy),
        )
        if partition_selection_by_seed is not None:
            for payload in cell_payloads:
                evidence = partition_selection_by_seed[int(payload["seed"])]
                # A locked replay carries the historical schema-v1 split
                # identity.  Its signed replay metadata has already been
                # checked for semantic equivalence with the effective plan
                # and for byte-level fingerprint integrity above.  Do not
                # replace that identity with a hash of today's normalized
                # plan serialization.
                payload["expected_split_fingerprint"] = evidence["split_fingerprint"]
                payload["partition_selection"] = evidence
        payloads.extend(cell_payloads)
    return payloads


def _check_expected_counts(spec: Mapping[str, Any], payloads: list[dict[str, Any]]) -> None:
    expected = spec.get("expect")
    if expected is None:
        return
    if not isinstance(expected, Mapping):
        raise CampaignError("E_CAMPAIGN_SPEC_INVALID", "expect must be a mapping")
    task_count = expected.get("task_count")
    if task_count is not None and task_count != len(payloads):
        raise CampaignError(
            "E_CAMPAIGN_EXPECTATION_FAILED",
            f"expected {task_count} tasks, generated {len(payloads)}",
        )
    config_count = expected.get("config_count")
    actual_config_count = len({payload["config_path"] for payload in payloads})
    if config_count is not None and config_count != actual_config_count:
        raise CampaignError(
            "E_CAMPAIGN_EXPECTATION_FAILED",
            f"expected {config_count} configs, found {actual_config_count}",
        )
    per_method = expected.get("tasks_per_method")
    if per_method is not None:
        counts = Counter(str(payload["method_id"]) for payload in payloads)
        if isinstance(per_method, int):
            mismatches = {key: value for key, value in counts.items() if value != per_method}
            if mismatches:
                raise CampaignError(
                    "E_CAMPAIGN_EXPECTATION_FAILED",
                    f"expected {per_method} tasks per method, got {dict(sorted(mismatches.items()))}",
                )
        elif isinstance(per_method, Mapping):
            expected_counts = {str(key): int(value) for key, value in per_method.items()}
            if dict(sorted(counts.items())) != dict(sorted(expected_counts.items())):
                raise CampaignError(
                    "E_CAMPAIGN_EXPECTATION_FAILED", "tasks_per_method does not match"
                )
        else:
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID", "expect.tasks_per_method has invalid type"
            )
    for expectation_field, payload_field in (
        ("tasks_by_profile", "resource_profile"),
        ("tasks_by_site", "assigned_site"),
    ):
        raw_expected_counts = expected.get(expectation_field)
        if raw_expected_counts is None:
            continue
        if not isinstance(raw_expected_counts, Mapping) or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in raw_expected_counts.items()
        ):
            raise CampaignError(
                "E_CAMPAIGN_SPEC_INVALID",
                f"expect.{expectation_field} must be a mapping of non-negative counts",
            )
        actual_counts = Counter(str(payload[payload_field]) for payload in payloads)
        expected_counts = {str(key): int(value) for key, value in raw_expected_counts.items()}
        if dict(sorted(actual_counts.items())) != dict(sorted(expected_counts.items())):
            raise CampaignError(
                "E_CAMPAIGN_EXPECTATION_FAILED",
                f"{expectation_field} does not match: expected "
                f"{dict(sorted(expected_counts.items()))}, got {dict(sorted(actual_counts.items()))}",
            )


def generate_campaign(
    spec_path: Path,
    *,
    repo_root: Path,
    output_dir: Path,
    _allow_template_placeholders: bool = False,
) -> GeneratedCampaign:
    repo_root = repo_root.resolve()
    spec_path = spec_path.resolve()
    output_dir = Path(output_dir).resolve(strict=False)
    spec = load_spec(spec_path)
    expected_sha, expected_diff, environment_digest = _resolve_code_identity(
        spec, repo_root=repo_root
    )
    if spec["track"] == "standardized":
        payloads = _standardized_payloads(
            spec,
            repo_root=repo_root,
            expected_git_sha=expected_sha,
            expected_git_diff_sha256=expected_diff,
            environment_lock_sha256=environment_digest,
        )
    else:
        payloads = _paper_payloads(
            spec,
            repo_root=repo_root,
            expected_git_sha=expected_sha,
            expected_git_diff_sha256=expected_diff,
            environment_lock_sha256=environment_digest,
        )

    if not _allow_template_placeholders:
        _reject_unpinned_production_tasks(spec, payloads, repo_root=repo_root)

    payloads.sort(key=lambda row: (str(row["config_path"]), int(row["seed"])))
    _check_expected_counts(spec, payloads)
    tasks: list[CampaignTask] = [
        finalize_task_row(payload, task_index=index) for index, payload in enumerate(payloads)
    ]
    with _AtomicCampaignDirectory(output_dir) as staging_dir:
        staged_manifest_path, staged_meta_path, meta = write_manifest(
            tasks,
            output_dir=staging_dir,
            campaign_id=str(spec["campaign_id"]),
            spec_sha256=sha256_file(spec_path),
            expected_git_sha=expected_sha,
            expected_git_diff_sha256=expected_diff,
            environment_lock_sha256=environment_digest,
            release_evidence=None,
        )

    manifest_path = output_dir / staged_manifest_path.relative_to(staging_dir)
    meta_path = output_dir / staged_meta_path.relative_to(staging_dir)

    return GeneratedCampaign(
        campaign_id=str(spec["campaign_id"]),
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        meta_path=str(meta_path),
        task_count=len(tasks),
        manifest_sha256=str(meta["manifest_sha256"]),
        counts_by_profile=dict(meta["counts_by_profile"]),
    )
