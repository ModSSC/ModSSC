from __future__ import annotations

import json
import math
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from bench.campaign.model_artifacts import (
    ModelArtifactError,
    model_artifact_lock_sha256,
    verify_model_artifact_lock,
)
from bench.orchestrators import sampling as sampling_orch
from bench.schema import ExperimentConfig
from bench.seed_sweep import apply_global_seed
from bench.utils.io import atomic_write_json, load_yaml
from modssc.data_loader import load_dataset, verify_dataset_content
from modssc.graph.cache import GraphCache
from modssc.inductive.registry import get_method_class as get_inductive_method_class
from modssc.preprocess.models import load_encoder
from modssc.preprocess.steps.core.vae import _default_vae_cache_dir, _safe_cache_component
from modssc.preprocess.steps.vision.aet import AetStep
from modssc.supervised.api import create_classifier
from modssc.supervised.types import ClassifierRuntime
from modssc.transductive.registry import get_method_class as get_transductive_method_class

from .dcl_partition_lock import (
    DCL_DATASET_ID,
    DCL_DIAGNOSTIC_METHOD_PROFILE,
    DCL_DIAGNOSTIC_PROTOCOL_IDS,
    DCL_METHOD_ID,
    DCL_METHOD_PROFILE,
    DCL_PAPER_PROTOCOL_ID,
    DCL_SCREENING_PROTOCOL_ID,
    is_dcl_vote_partition_replay_identity,
    resolve_repo_path,
    verify_dcl_partition_replay,
)
from .errors import CampaignError
from .manifest import load_manifest, sha256_file
from .models import CampaignTask
from .preflight_coverage import build_task_coverage
from .scientific_gates import discover_gate_registry, guard_task

_METHOD_KINDS = {
    "pseudo_label": "inductive",
    "tri_training": "inductive",
    "democratic_co_learning": "inductive",
    "fixmatch": "inductive",
    "flexmatch": "inductive",
    "free_match": "inductive",
    "softmatch": "inductive",
    "laplace_learning": "transductive",
    "poisson_learning": "transductive",
    "grand": "transductive",
    "co_training": "inductive",
}
_OFFLINE_FLAGS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "MODSSC_HF_LOCAL_FILES_ONLY",
)
_HISTORICAL_NUMPY_CLASSIFIERS = frozenset({"decision_tree", "gaussian_nb", "knn"})


@dataclass(frozen=True)
class PreflightReport:
    campaign_id: str
    status: str
    task_count: int
    report_path: str
    error_count: int


def _read_mapping(path: Path, *, code: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CampaignError(code, f"cannot read {path}") from exc
    if not isinstance(raw, dict):
        raise CampaignError(code, f"root must be a mapping: {path}")
    return raw


def load_resource_catalog(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    raw = _read_mapping(path, code="E_CAMPAIGN_RESOURCE_CATALOG_INVALID")
    entries = raw.get("resources")
    if raw.get("schema_version") != 1 or not isinstance(entries, list):
        raise CampaignError(
            "E_CAMPAIGN_RESOURCE_CATALOG_INVALID", "invalid resource catalog schema"
        )
    resources: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CampaignError(
                "E_CAMPAIGN_RESOURCE_CATALOG_INVALID", "resource entry must be a mapping"
            )
        site = entry.get("site_id")
        profile_id = entry.get("profile_id")
        if not isinstance(site, str) or not isinstance(profile_id, str):
            raise CampaignError(
                "E_CAMPAIGN_RESOURCE_CATALOG_INVALID", "resource identity is missing"
            )
        key = (site, profile_id)
        if key in resources:
            raise CampaignError(
                "E_CAMPAIGN_RESOURCE_CATALOG_INVALID", f"duplicate resource {site}.{profile_id}"
            )
        resources[key] = dict(entry)
    return resources


def _preflight_task_scope(
    tasks: Sequence[CampaignTask],
    resources: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    require_architecture: str | None,
) -> tuple[list[CampaignTask], str | None, list[str]]:
    if require_architecture is None:
        return list(tasks), None, []
    architecture = require_architecture.upper()
    if not architecture.strip():
        return [], architecture, [f"unsupported required architecture: {require_architecture}"]
    selected: list[CampaignTask] = []
    errors: list[str] = []
    for task in tasks:
        resource = resources.get((task.assigned_site, task.resource_profile))
        if resource is None:
            continue
        if str(resource.get("architecture", "")).upper() == architecture:
            selected.append(task)
    if not selected:
        errors.append(f"manifest contains no task assigned to architecture {architecture}")
    return selected, architecture, errors


def _is_cpu_calder_campaign(
    tasks: Sequence[CampaignTask],
    resources: Mapping[tuple[str, str], Mapping[str, Any]],
) -> bool:
    """Return whether every task uses a CPU-only Laplace/Poisson resource."""

    if not tasks or not {task.method_id for task in tasks} <= {
        "laplace_learning",
        "poisson_learning",
    }:
        return False
    for task in tasks:
        resource = resources.get((task.assigned_site, task.resource_profile))
        if resource is None:
            return False
        if (
            str(resource.get("architecture", "")).upper() != "CPU"
            or resource.get("accelerators_per_task") != 0
        ):
            return False
    return True


def _model_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.startswith("model_id") and isinstance(item, str):
                found.add(item)
            found.update(_model_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_model_ids(item))
    return found


def _sha256_matches(path: Path, expected: Any) -> bool:
    return isinstance(expected, str) and bool(expected) and sha256_file(path) == expected


def _frozen_vae_entries(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    preprocess = raw.get("preprocess")
    plan = preprocess.get("plan") if isinstance(preprocess, Mapping) else None
    steps = plan.get("steps") if isinstance(plan, Mapping) else None
    if not isinstance(steps, list):
        return []
    entries: list[Mapping[str, Any]] = []
    for step in steps:
        if not isinstance(step, Mapping) or step.get("id", step.get("step_id")) != "core.vae":
            continue
        params = step.get("params")
        if isinstance(params, Mapping) and params.get("require_cache_hit") is True:
            entries.append(params)
    return entries


def _frozen_aet_entries(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    preprocess = raw.get("preprocess")
    plan = preprocess.get("plan") if isinstance(preprocess, Mapping) else None
    steps = plan.get("steps") if isinstance(plan, Mapping) else None
    if not isinstance(steps, list):
        return []
    entries: list[Mapping[str, Any]] = []
    for step in steps:
        if not isinstance(step, Mapping) or step.get("id", step.get("step_id")) != "vision.aet":
            continue
        params = step.get("params")
        if isinstance(params, Mapping) and params.get("source") == "precomputed":
            entries.append(params)
    return entries


def _check_frozen_dependencies(
    config_tasks: Mapping[str, CampaignTask],
    configs: Mapping[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    vae_records: list[dict[str, Any]] = []
    aet_records: list[dict[str, Any]] = []
    graph_records: list[dict[str, Any]] = []
    seen_vae: set[tuple[str, ...]] = set()
    seen_aet: set[tuple[str, ...]] = set()
    seen_graph: set[tuple[str, ...]] = set()

    for config_path, task in config_tasks.items():
        raw = configs[config_path]
        for params in _frozen_vae_entries(raw):
            cache_key = params.get("cache_key")
            if not isinstance(cache_key, str) or not cache_key:
                errors.append(f"{config_path}: frozen VAE requires cache_key")
                continue
            expected_fingerprint = params.get("expected_model_fingerprint")
            if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
                errors.append(f"{config_path}: frozen VAE requires expected_model_fingerprint")
                continue
            explicit_root = params.get("model_cache_dir")
            root = (
                Path(explicit_root).expanduser().resolve()
                if isinstance(explicit_root, str) and explicit_root
                else _default_vae_cache_dir()
            )
            identity = (str(root), cache_key, expected_fingerprint)
            if identity in seen_vae:
                continue
            seen_vae.add(identity)
            prefix = f"{_safe_cache_component(cache_key)}-"
            candidates = (
                sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
                if root.is_dir()
                else []
            )
            valid: list[dict[str, Any]] = []
            for candidate in candidates:
                manifest_path = candidate / "manifest.json"
                model_path = candidate / "model.pt"
                state_path = candidate / "state.npz"
                try:
                    manifest = _read_mapping(
                        manifest_path, code="E_CAMPAIGN_FROZEN_ARTIFACT_INVALID"
                    )
                except CampaignError:
                    continue
                cache_info = manifest.get("cache")
                manifest_params = manifest.get("params")
                file_hashes = manifest.get("file_sha256")
                if not isinstance(cache_info, Mapping) or cache_info.get("cache_key") != cache_key:
                    continue
                if not isinstance(manifest_params, Mapping) or manifest_params.get(
                    "preset"
                ) != params.get("preset"):
                    continue
                if not isinstance(file_hashes, Mapping):
                    continue
                if not model_path.is_file() or not state_path.is_file():
                    continue
                if not _sha256_matches(model_path, file_hashes.get("model.pt")):
                    continue
                if not _sha256_matches(state_path, file_hashes.get("state.npz")):
                    continue
                fingerprint = manifest.get("fingerprint")
                if fingerprint == expected_fingerprint:
                    valid.append(
                        {
                            "dir": str(candidate),
                            "fingerprint": fingerprint,
                            "model_sha256": file_hashes["model.pt"],
                            "state_sha256": file_hashes["state.npz"],
                        }
                    )
            if len(valid) != 1:
                errors.append(
                    f"{config_path}: expected one verified frozen VAE for {cache_key}, "
                    f"found {len(valid)}"
                )
            vae_records.extend(valid)

        for params in _frozen_aet_entries(raw):
            expected_features = params.get("expected_features_sha256")
            expected_labels = params.get("expected_labels_sha256")
            if not isinstance(expected_features, str) or not expected_features:
                errors.append(f"{config_path}: frozen AET requires expected_features_sha256")
                continue
            if not isinstance(expected_labels, str) or not expected_labels:
                errors.append(f"{config_path}: frozen AET requires expected_labels_sha256")
                continue
            try:
                step = AetStep(**dict(params))
                step._validate_params()
                features_path = step._features_path()
                labels_path = step._labels_path()
            except Exception as exc:
                errors.append(f"{config_path}: invalid frozen AET parameters: {exc}")
                continue
            identity = (
                str(features_path),
                str(labels_path),
                str(step.expected_features_sha256),
                str(step.expected_labels_sha256),
            )
            if identity in seen_aet:
                continue
            seen_aet.add(identity)
            if not features_path.is_file() or not labels_path.is_file():
                errors.append(f"{config_path}: expected one verified frozen AET, found 0")
                continue
            features_sha256 = sha256_file(features_path)
            labels_sha256 = sha256_file(labels_path)
            if (
                features_sha256 != step.expected_features_sha256
                or labels_sha256 != step.expected_labels_sha256
            ):
                errors.append(f"{config_path}: expected one verified frozen AET, found 0")
                continue
            aet_records.append(
                {
                    "features_path": str(features_path),
                    "features_sha256": features_sha256,
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_sha256,
                }
            )

        graph_raw = raw.get("graph")
        if not isinstance(graph_raw, Mapping) or graph_raw.get("require_cache_hit") is not True:
            continue
        root_value = graph_raw.get("cache_dir")
        spec = graph_raw.get("spec")
        seed = graph_raw.get("seed")
        expected_fingerprint = graph_raw.get("expected_fingerprint")
        expected_preprocess_fingerprint = graph_raw.get("expected_preprocess_fingerprint")
        if not isinstance(root_value, str) or not root_value or not isinstance(spec, Mapping):
            errors.append(f"{config_path}: frozen graph cache_dir/spec is missing")
            continue
        if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
            errors.append(f"{config_path}: frozen graph requires expected_fingerprint")
            continue
        if (
            not isinstance(expected_preprocess_fingerprint, str)
            or not expected_preprocess_fingerprint
        ):
            errors.append(f"{config_path}: frozen graph requires expected_preprocess_fingerprint")
            continue
        root = Path(root_value).expanduser().resolve()
        dataset_fingerprint = task.expected_dataset_fingerprint
        if dataset_fingerprint is None:
            errors.append(f"{config_path}: frozen graph requires a dataset fingerprint")
            continue
        identity = (
            str(root),
            task.dataset_request_sha256,
            json.dumps(spec, sort_keys=True),
            expected_fingerprint,
            expected_preprocess_fingerprint,
        )
        if identity in seen_graph:
            continue
        seen_graph.add(identity)
        store = GraphCache(root=root)
        valid_graphs: list[dict[str, Any]] = []
        for fingerprint in store.list():
            if fingerprint != expected_fingerprint:
                continue
            try:
                graph, manifest = store.load(fingerprint)
            except Exception:
                continue
            if manifest.get("fingerprint") != fingerprint:
                continue
            if manifest.get("dataset_fingerprint") != dataset_fingerprint:
                continue
            if manifest.get("preprocess_fingerprint") != expected_preprocess_fingerprint:
                continue
            if manifest.get("spec") != dict(spec) or manifest.get("seed") != seed:
                continue
            if graph.meta.get("fingerprint") != fingerprint:
                continue
            if graph.meta.get("preprocess_fingerprint") != expected_preprocess_fingerprint:
                continue
            valid_graphs.append(
                {
                    "dir": str(store.entry_dir(fingerprint)),
                    "fingerprint": fingerprint,
                    "preprocess_fingerprint": manifest.get("preprocess_fingerprint"),
                }
            )
        if len(valid_graphs) != 1:
            errors.append(
                f"{config_path}: expected one verified frozen graph, found {len(valid_graphs)}"
            )
        graph_records.extend(valid_graphs)

    return errors, {"vae": vae_records, "aet": aet_records, "graphs": graph_records}


def _check_frozen_partition_replays(
    tasks: Sequence[CampaignTask],
    configs: Mapping[str, dict[str, Any]],
    *,
    repo_root: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    for task in tasks:
        canonical_dcl_vote_task = (
            task.track == "paper"
            and task.method_id == DCL_METHOD_ID
            and task.method_profile == DCL_METHOD_PROFILE
            and task.dataset_id == DCL_DATASET_ID
        )
        diagnostic_dcl_vote_task = (
            task.track == "paper"
            and task.method_id == DCL_METHOD_ID
            and task.method_profile == DCL_DIAGNOSTIC_METHOD_PROFILE
            and task.dataset_id == DCL_DATASET_ID
        )
        if task.protocol_id == DCL_PAPER_PROTOCOL_ID and not canonical_dcl_vote_task:
            errors.append(f"{task.task_id}: DCL Vote paper protocol has the wrong task identity")
            continue
        if canonical_dcl_vote_task and task.protocol_id not in {
            DCL_PAPER_PROTOCOL_ID,
            DCL_SCREENING_PROTOCOL_ID,
        }:
            errors.append(f"{task.task_id}: DCL Vote task has an unrecognized protocol id")
            continue
        if diagnostic_dcl_vote_task and task.protocol_id not in DCL_DIAGNOSTIC_PROTOCOL_IDS:
            errors.append(
                f"{task.task_id}: DCL Vote v2 diagnostic task has an unrecognized protocol id"
            )
            continue
        required = is_dcl_vote_partition_replay_identity(
            track=task.track,
            method_id=task.method_id,
            method_profile=task.method_profile,
            dataset_id=task.dataset_id,
            protocol_id=task.protocol_id,
        )
        if task.partition_selection is None:
            if required:
                errors.append(f"{task.task_id}: DCL Vote paper task has no partition selection")
            continue
        if not required:
            errors.append(f"{task.task_id}: partition selection is attached to an ineligible task")
            continue
        try:
            evidence = dict(task.partition_selection)
            evidence["selection_path"] = str(
                resolve_repo_path(
                    repo_root,
                    str(evidence["selection_path"]),
                    label="task.partition_selection.selection_path",
                )
            )
            evidence["replay_path"] = str(
                resolve_repo_path(
                    repo_root,
                    str(evidence["replay_path"]),
                    label="task.partition_selection.replay_path",
                )
            )
            raw = configs[task.config_path]
            effective = apply_global_seed(
                raw,
                seed=task.seed,
                seeded_sections=task.seeded_sections,
            )
            sampling = effective.get("sampling")
            plan = sampling.get("plan") if isinstance(sampling, Mapping) else None
            if not isinstance(plan, Mapping):
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                    "effective sampling plan is missing",
                )
            verified = verify_dcl_partition_replay(
                evidence,
                expected_seed=task.seed,
                expected_dataset_fingerprint=str(task.expected_dataset_fingerprint),
                expected_plan=plan,
            )
            if (
                verified.selection.dataset_content_sha256 != task.expected_dataset_content_sha256
                or verified.entry.split_fingerprint != task.expected_split_fingerprint
            ):
                raise CampaignError(
                    "E_CAMPAIGN_PARTITION_SELECTION_MISMATCH",
                    "partition replay differs from the task dataset or split identity",
                )
            records.append(
                {
                    "task_id": task.task_id,
                    "seed": task.seed,
                    "selection_sha256": verified.selection.sha256,
                    "selection_rank": verified.entry.selection_rank,
                    "source_task_id": verified.entry.source_task_id,
                    "source_artifact_sha256": dict(verified.selection.source_artifact_sha256),
                    "split_fingerprint": verified.entry.split_fingerprint,
                    "split_manifest_sha256": verified.entry.split_manifest_sha256,
                    "split_json_sha256": verified.entry.split_json_sha256,
                    "split_arrays_sha256": verified.entry.split_arrays_sha256,
                }
            )
        except Exception as exc:
            errors.append(f"{task.task_id}: {type(exc).__name__}: {exc}")
    return errors, records


def _classifier_specs(value: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        classifier_id = value.get("classifier_id")
        classifier_backend = value.get("classifier_backend")
        if classifier_id in _HISTORICAL_NUMPY_CLASSIFIERS and isinstance(classifier_backend, str):
            raw_params = value.get("classifier_params", {})
            if isinstance(raw_params, Mapping):
                specs.append(
                    {
                        "classifier_id": str(classifier_id),
                        "classifier_backend": classifier_backend,
                        "classifier_params": dict(raw_params),
                    }
                )
        for item in value.values():
            specs.extend(_classifier_specs(item))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            specs.extend(_classifier_specs(item))
    return specs


def _historical_classifier_specs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    method = value.get("method")
    if not isinstance(method, Mapping):
        return []
    profile = method.get("profile")
    if not (isinstance(profile, str) and profile.startswith("paper:")):
        return []
    return _classifier_specs(method.get("params", {}))


def _default_historical_classifier_checker(
    classifier_id: str,
    classifier_params: Mapping[str, Any],
) -> dict[str, str]:
    classifier = create_classifier(
        classifier_id,
        backend="numpy",
        params=dict(classifier_params),
        runtime=ClassifierRuntime(seed=0),
    )
    return {
        "classifier_id": classifier_id,
        "backend": str(classifier.backend),
        "implementation": f"{type(classifier).__module__}:{type(classifier).__qualname__}",
    }


def _default_versions() -> dict[str, str]:
    import sklearn

    versions = {
        "python": platform.python_version(),
        "scikit_learn": str(sklearn.__version__),
    }
    try:
        import torch
    except Exception:
        return versions
    versions["torch"] = str(torch.__version__)
    versions["cuda_available"] = str(bool(torch.cuda.is_available())).lower()
    if torch.cuda.is_available():
        versions["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    return versions


def _default_method_importer(kind: str, method_id: str) -> None:
    if kind == "inductive":
        get_inductive_method_class(method_id)
    else:
        get_transductive_method_class(method_id)


def _default_dataset_checker(
    raw: dict[str, Any], task: CampaignTask, *, rehash: bool = True
) -> dict[str, str | None]:
    effective = apply_global_seed(raw, seed=task.seed, seeded_sections=task.seeded_sections)
    cfg = ExperimentConfig.from_dict(effective)
    if cfg.dataset.download:
        raise ValueError(f"{task.config_path} permits dataset downloads")
    resolved_cache_dir = (
        Path(cfg.dataset.cache_dir).expanduser().resolve() if cfg.dataset.cache_dir else None
    )
    dataset = load_dataset(
        cfg.dataset.id,
        cache_dir=resolved_cache_dir,
        download=False,
        options=dict(cfg.dataset.options),
    )
    content = verify_dataset_content(
        cfg.dataset.id,
        cache_dir=resolved_cache_dir,
        options=dict(cfg.dataset.options),
        rehash=rehash,
    )
    dataset = sampling_orch.prepare_dataset(dataset, plan_dict=cfg.sampling.plan)
    fingerprint = None
    if isinstance(dataset.meta, Mapping):
        value = dataset.meta.get("dataset_fingerprint")
        fingerprint = str(value) if isinstance(value, str) and value else None
        attached_content = dataset.meta.get("dataset_content_sha256")
        if attached_content != content["content_sha256"]:
            raise ValueError("loaded dataset content evidence differs from the cache manifest")
    return {"fingerprint": fingerprint, **content}


def run_preflight(
    manifest_path: Path,
    *,
    repo_root: Path,
    output_path: Path,
    resources: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    authorization_created_at: datetime | None = None,
    authorization_expires_at: datetime | None = None,
    environment_lock_sha256: str | None = None,
    environment_manifest_path: Path | None = None,
    dataset_cache_dir: Path | None = None,
    model_cache_root: Path | None = None,
    require_architecture: str | None = None,
    max_authorization_age_hours: float = 24.0,
    now_provider: Callable[[], datetime] = lambda: datetime.now(UTC),
    version_provider: Callable[[], dict[str, str]] = _default_versions,
    method_importer: Callable[[str, str], None] = _default_method_importer,
    dataset_checker: Callable[
        [dict[str, Any], CampaignTask], str | None | Mapping[str, Any]
    ] = _default_dataset_checker,
    model_checker: Callable[[str], Any] = load_encoder,
    historical_classifier_checker: Callable[
        [str, Mapping[str, Any]], dict[str, str]
    ] = _default_historical_classifier_checker,
) -> PreflightReport:
    if (
        isinstance(max_authorization_age_hours, bool)
        or not isinstance(max_authorization_age_hours, int | float)
        or not math.isfinite(float(max_authorization_age_hours))
        or float(max_authorization_age_hours) <= 0.0
    ):
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID",
            "max_authorization_age_hours must be a finite positive number",
        )
    observed_at = now_provider()
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID", "preflight clock must include a timezone"
        )
    observed_at = observed_at.astimezone(UTC)
    validity = timedelta(hours=float(max_authorization_age_hours))
    created_at = authorization_created_at or observed_at
    expires_at = authorization_expires_at or (created_at + validity)
    if (
        created_at.tzinfo is None
        or created_at.utcoffset() is None
        or expires_at.tzinfo is None
        or expires_at.utcoffset() is None
    ):
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID",
            "authorization timestamps must include a timezone",
        )
    created_at = created_at.astimezone(UTC)
    expires_at = expires_at.astimezone(UTC)
    meta, tasks = load_manifest(manifest_path, verify_digest=True)
    normalized_resources = dict(resources or {})
    covered_tasks, required_architecture, coverage_errors = _preflight_task_scope(
        tasks,
        normalized_resources,
        require_architecture=require_architecture,
    )
    task_coverage = build_task_coverage(
        (task.task_id for task in covered_tasks),
        architecture=required_architecture,
    )
    checks: list[dict[str, Any]] = []

    def record(name: str, errors: list[str], **details: Any) -> None:
        checks.append(
            {"name": name, "status": "pass" if not errors else "fail", "errors": errors, **details}
        )

    methods_errors: list[str] = []
    unexpected = sorted(set(task.method_id for task in tasks) - _METHOD_KINDS.keys())
    if unexpected:
        methods_errors.append(f"methods outside the technical catalog: {unexpected}")
    kind_mismatches = sorted(
        {
            f"{task.method_id}:{task.method_kind}"
            for task in tasks
            if _METHOD_KINDS.get(task.method_id, task.method_kind) != task.method_kind
        }
    )
    if kind_mismatches:
        methods_errors.append(f"method kind mismatches: {kind_mismatches}")
    for method_id, kind in _METHOD_KINDS.items():
        try:
            method_importer(kind, method_id)
        except Exception as exc:  # pragma: no cover - concrete exception depends on optional stack
            methods_errors.append(f"{method_id}: {type(exc).__name__}: {exc}")
    record("method_imports", methods_errors, method_count=len(_METHOD_KINDS))
    record("task_coverage", coverage_errors, **task_coverage)

    gate_errors: list[str] = []
    try:
        gate_path = discover_gate_registry(repo_root, None)
        for task in covered_tasks:
            guard_task(task, gate_path)
    except CampaignError as exc:
        gate_errors.append(str(exc))
    record(
        "scientific_gate_policy",
        gate_errors,
        gate_policy_id=meta.get("gate_policy_id"),
        gate_policy_sha256=meta.get("gate_policy_sha256"),
    )

    version_errors: list[str] = []
    try:
        versions = version_provider()
    except Exception as exc:
        versions = {}
        version_errors.append(f"cannot collect versions: {type(exc).__name__}: {exc}")
    expected_versions = {"python": "3.12.13", "scikit_learn": "1.8"}
    if not _is_cpu_calder_campaign(tasks, normalized_resources):
        expected_versions["torch"] = "2.10"
    for name, expected in expected_versions.items():
        actual = versions.get(name)
        exact = name == "python"
        if not isinstance(actual, str) or (
            actual != expected if exact else not actual.startswith(expected)
        ):
            comparator = "=" if exact else " prefix="
            version_errors.append(f"{name}: expected{comparator}{expected}, got {actual!r}")
    if required_architecture is not None and required_architecture != "CPU":
        device_name = versions.get("cuda_device_name")
        if versions.get("cuda_available") != "true":
            version_errors.append("CUDA is not available in this preflight job")
        elif not isinstance(device_name, str) or required_architecture not in device_name.upper():
            version_errors.append(f"expected a {required_architecture} device, got {device_name!r}")
    record("runtime_versions", version_errors, actual=versions, expected=expected_versions)

    lock_errors: list[str] = []
    expected_locks = {task.environment_lock_sha256 for task in tasks}
    manifest_path = environment_manifest_path or (
        Path(os.environ["MODSSC_ENVIRONMENT_MANIFEST"])
        if os.environ.get("MODSSC_ENVIRONMENT_MANIFEST")
        else None
    )
    manifest_details: dict[str, Any] | None = None
    environment_manifest: dict[str, Any] | None = None
    actual_lock = environment_lock_sha256 or os.environ.get("MODSSC_ENVIRONMENT_LOCK_SHA256")
    scientific_campaign = any(task.claim_eligible for task in tasks)
    if manifest_path is not None:
        try:
            from bench.campaign.build_manifest import (
                collect_environment_identity,
                environment_identity_sha256,
                python_environment_identity,
                validate_environment_lock,
            )

            environment_manifest = _read_mapping(
                manifest_path, code="E_CAMPAIGN_ENVIRONMENT_MISMATCH"
            )
            locked_identity = environment_manifest.get("environment_lock")
            if not isinstance(locked_identity, dict):
                raise CampaignError(
                    "E_CAMPAIGN_ENVIRONMENT_MISMATCH",
                    "environment manifest has no immutable environment_lock payload",
                )
            validate_environment_lock(locked_identity)
            computed_digest = environment_identity_sha256(locked_identity)
            if environment_manifest.get("environment_lock_sha256") != computed_digest:
                raise CampaignError(
                    "E_CAMPAIGN_ENVIRONMENT_MISMATCH",
                    "environment manifest lock digest is invalid",
                )
            active_identity = collect_environment_identity()
            if python_environment_identity(active_identity) != python_environment_identity(
                locked_identity
            ):
                raise CampaignError(
                    "E_CAMPAIGN_ENVIRONMENT_MISMATCH",
                    "active environment differs from the build manifest",
                )
            actual_lock = computed_digest
            manifest_details = {
                "path": str(manifest_path.resolve()),
                "environment_lock_sha256": computed_digest,
                "manifest_sha256": sha256_file(manifest_path),
            }
        except Exception as exc:
            lock_errors.append(f"environment manifest: {type(exc).__name__}: {exc}")
    elif scientific_campaign:
        lock_errors.append("scientific preflight requires an environment build manifest")
    if len(expected_locks) != 1:
        lock_errors.append("manifest contains multiple environment lock digests")
    elif actual_lock not in expected_locks:
        lock_errors.append("active environment lock digest differs from the manifest")
    record(
        "environment_lock",
        lock_errors,
        active=actual_lock,
        expected=sorted(expected_locks),
        manifest=manifest_details,
    )

    build_manifest_errors: list[str] = []
    build_manifest_details: dict[str, Any] | None = None
    if scientific_campaign:
        if environment_manifest is None or manifest_path is None:
            build_manifest_errors.append("scientific preflight requires a readable build manifest")
        else:
            try:
                from bench.campaign.build_manifest import validate_build_manifest

                build_manifest_details = validate_build_manifest(
                    environment_manifest,
                    repo_root=repo_root,
                    expected_git_sha=str(meta["expected_git_sha"]),
                    expected_git_diff_sha256=meta.get("expected_git_diff_sha256"),
                )
                build_manifest_details = {
                    **build_manifest_details,
                    "path": str(manifest_path.resolve()),
                    "manifest_sha256": sha256_file(manifest_path),
                }
            except Exception as exc:
                build_manifest_errors.append(f"{type(exc).__name__}: {exc}")
    record("build_manifest", build_manifest_errors, manifest=build_manifest_details)

    offline_errors = [name for name in _OFFLINE_FLAGS if os.environ.get(name) != "1"]
    record("offline_mode", [f"{name} must equal 1" for name in offline_errors])

    resolved_dataset_cache = dataset_cache_dir or (
        Path(os.environ["MODSSC_DATASET_CACHE_DIR"])
        if os.environ.get("MODSSC_DATASET_CACHE_DIR")
        else None
    )
    resolved_model_cache = model_cache_root or (
        Path(os.environ["MODSSC_MODEL_CACHE_ROOT"])
        if os.environ.get("MODSSC_MODEL_CACHE_ROOT")
        else None
    )
    configs: dict[str, dict[str, Any]] = {}

    def config_for(task: CampaignTask) -> dict[str, Any]:
        raw = configs.get(task.config_path)
        if raw is None:
            config_path = repo_root / task.config_path
            if sha256_file(config_path) != task.source_config_sha256:
                raise CampaignError(
                    "E_CAMPAIGN_SOURCE_CONFIG_MISMATCH",
                    f"source configuration differs: {task.config_path}",
                )
            raw = load_yaml(config_path)
            configs[task.config_path] = raw
        return raw

    config_tasks: dict[str, CampaignTask] = {}
    config_errors: list[str] = []
    for task in covered_tasks:
        if task.config_path in config_tasks:
            continue
        try:
            config_for(task)
        except Exception as exc:
            config_errors.append(f"{task.config_path}: {type(exc).__name__}: {exc}")
            continue
        config_tasks[task.config_path] = task
    record(
        "source_configs",
        config_errors,
        config_count=len(config_tasks),
        expected_config_count=len({task.config_path for task in covered_tasks}),
    )

    required_models = {model_id for raw in configs.values() for model_id in _model_ids(raw)}
    external_models = sorted(
        model_id for model_id in required_models if not model_id.startswith("stub:")
    )
    cache_errors: list[str] = []
    if covered_tasks and (
        resolved_dataset_cache is None or not resolved_dataset_cache.expanduser().is_dir()
    ):
        cache_errors.append("dataset cache directory is missing")
    if external_models and (
        resolved_model_cache is None or not resolved_model_cache.expanduser().is_dir()
    ):
        cache_errors.append("model cache directory is missing")
    record(
        "cache_roots",
        cache_errors,
        dataset_required=bool(covered_tasks),
        model_required=bool(external_models),
        dataset_cache=None if resolved_dataset_cache is None else str(resolved_dataset_cache),
        model_cache=None if resolved_model_cache is None else str(resolved_model_cache),
    )

    dataset_errors: list[str] = []
    dataset_fingerprints: dict[str, str | None] = {}
    dataset_evidence: dict[str, dict[str, str | None]] = {}

    representative_tasks: dict[str, CampaignTask] = {}
    for task in covered_tasks:
        representative_tasks.setdefault(task.split_request_sha256, task)
    for split_request_sha, task in sorted(representative_tasks.items()):
        try:
            raw = config_for(task)
            if dataset_checker is _default_dataset_checker:
                checked = _default_dataset_checker(
                    raw,
                    task,
                    rehash=task.dataset_request_sha256 not in dataset_evidence,
                )
            else:
                checked = dataset_checker(raw, task)
            if isinstance(checked, Mapping):
                raw_fingerprint = checked.get("fingerprint")
                actual_fingerprint = (
                    str(raw_fingerprint)
                    if isinstance(raw_fingerprint, str) and raw_fingerprint
                    else None
                )
                evidence = {
                    key: str(value) if isinstance(value, str) and value else None
                    for key, value in checked.items()
                    if key
                    in {
                        "fingerprint",
                        "content_sha256",
                        "content_manifest_sha256",
                        "cache_state_sha256",
                        "cache_fingerprint",
                    }
                }
            else:
                actual_fingerprint = checked
                evidence = {"fingerprint": actual_fingerprint}
            dataset_fingerprints[split_request_sha] = actual_fingerprint
            previous_evidence = dataset_evidence.get(task.dataset_request_sha256)
            if previous_evidence is not None and any(
                previous_evidence.get(field) != evidence.get(field)
                for field in (
                    "content_sha256",
                    "content_manifest_sha256",
                    "cache_state_sha256",
                    "cache_fingerprint",
                )
            ):
                dataset_errors.append(
                    f"{task.dataset_id}: dataset content proof changed during preflight"
                )
            dataset_evidence[task.dataset_request_sha256] = evidence
            if (
                task.expected_dataset_fingerprint is not None
                and actual_fingerprint != task.expected_dataset_fingerprint
            ):
                dataset_errors.append(f"{task.dataset_id}: dataset fingerprint differs")
            if actual_fingerprint is None:
                dataset_errors.append(f"{task.dataset_id}: dataset fingerprint is missing")
            actual_content = evidence.get("content_sha256")
            if (
                task.expected_dataset_content_sha256 is not None
                and actual_content != task.expected_dataset_content_sha256
            ):
                dataset_errors.append(f"{task.dataset_id}: dataset content digest differs")
            if task.expected_dataset_content_sha256 is not None and not all(
                evidence.get(field)
                for field in (
                    "content_sha256",
                    "content_manifest_sha256",
                    "cache_state_sha256",
                    "cache_fingerprint",
                )
            ):
                dataset_errors.append(f"{task.dataset_id}: dataset content proof is incomplete")
        except Exception as exc:
            dataset_errors.append(f"{task.dataset_id}: {type(exc).__name__}: {exc}")
    record(
        "datasets",
        dataset_errors,
        dataset_count=len({task.dataset_id for task in covered_tasks}),
        request_count=len({task.dataset_request_sha256 for task in covered_tasks}),
        prepared_request_count=len(representative_tasks),
        fingerprints=dataset_fingerprints,
        evidence_by_request=dataset_evidence,
    )

    model_errors: list[str] = []
    for model_id in sorted(required_models):
        try:
            model_checker(model_id)
        except Exception as exc:
            model_errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
    record("offline_models", model_errors, models=sorted(required_models))

    model_lock_errors: list[str] = []
    model_attestations: list[dict[str, Any]] = []
    model_lock_digest: str | None = None
    if environment_manifest is None:
        if external_models:
            model_lock_errors.append(
                "external models require a verified environment model artifact lock"
            )
    else:
        locked_identity = environment_manifest.get("environment_lock")
        model_lock = (
            locked_identity.get("model_artifacts") if isinstance(locked_identity, Mapping) else None
        )
        if not isinstance(model_lock, Mapping):
            model_lock_errors.append("environment manifest has no model artifact lock")
        else:
            model_lock_digest = model_artifact_lock_sha256(model_lock)
            try:
                model_attestations = verify_model_artifact_lock(
                    model_lock,
                    required_models,
                    model_cache_root=resolved_model_cache,
                )
            except ModelArtifactError as exc:
                model_lock_errors.append(str(exc))
    record(
        "model_artifact_lock",
        model_lock_errors,
        required_models=sorted(required_models),
        external_models=external_models,
        model_artifacts_sha256=model_lock_digest,
        attestations=model_attestations,
    )
    historical_errors: list[str] = []
    historical_attestations: list[dict[str, str]] = []
    historical_specs = [
        (config_path, spec)
        for config_path, raw in configs.items()
        for spec in _historical_classifier_specs(raw)
    ]
    for config_path, spec in historical_specs:
        classifier_id = str(spec["classifier_id"])
        backend = str(spec["classifier_backend"])
        if backend != "numpy":
            historical_errors.append(
                f"{config_path}: historical classifier {classifier_id!r} "
                "must use the embedded NumPy backend"
            )
            continue
        try:
            attestation = historical_classifier_checker(
                classifier_id,
                spec["classifier_params"],
            )
        except Exception as exc:
            historical_errors.append(f"{config_path}: {classifier_id}: {type(exc).__name__}: {exc}")
        else:
            historical_attestations.append({"config_path": config_path, **attestation})
    record(
        "historical_numpy_backends",
        historical_errors,
        required=bool(historical_specs),
        attestations=historical_attestations,
    )

    frozen_errors, frozen_artifacts = _check_frozen_dependencies(config_tasks, configs)
    record("frozen_paper_artifacts", frozen_errors, **frozen_artifacts)
    partition_errors, partition_records = _check_frozen_partition_replays(
        covered_tasks,
        configs,
        repo_root=repo_root,
    )
    record(
        "frozen_partition_replays",
        partition_errors,
        count=len(partition_records),
        attestations=partition_records,
    )
    authorization_errors: list[str] = []
    if created_at > observed_at:
        authorization_errors.append("authorization creation time is in the future")
    elif observed_at >= expires_at:
        authorization_errors.append("authorization has expired")
    elif expires_at > created_at + validity:
        authorization_errors.append("authorization exceeds its maximum validity")
    record(
        "authorization_window",
        authorization_errors,
        created_at=created_at.isoformat(),
        observed_at=observed_at.isoformat(),
        max_age_hours=float(max_authorization_age_hours),
        expires_at=expires_at.isoformat(),
    )

    errors = [error for check in checks for error in check["errors"]]
    payload = {
        "schema_version": 1,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "max_authorization_age_hours": float(max_authorization_age_hours),
        "campaign_id": meta["campaign_id"],
        "manifest_sha256": meta["manifest_sha256"],
        "claim_scope_id": meta.get("claim_scope_id"),
        "campaign_stage": meta.get("campaign_stage"),
        "claim_eligible": meta.get("claim_eligible"),
        "gate_policy_id": meta.get("gate_policy_id"),
        "gate_policy_sha256": meta.get("gate_policy_sha256"),
        "task_count": len(tasks),
        "covered_task_count": len(covered_tasks),
        "task_coverage": task_coverage,
        "status": "pass" if not errors else "blocked",
        "required_architecture": required_architecture,
        "environment_lock_sha256": actual_lock,
        "environment_manifest_sha256": (
            None if manifest_details is None else manifest_details["manifest_sha256"]
        ),
        "build_manifest_sha256": (
            None if build_manifest_details is None else build_manifest_details["manifest_sha256"]
        ),
        "required_model_ids": sorted(required_models),
        "model_artifacts_sha256": model_lock_digest,
        "model_artifact_attestations": model_attestations,
        "checks": checks,
        "error_count": len(errors),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, payload)
    return PreflightReport(
        campaign_id=str(meta["campaign_id"]),
        status=str(payload["status"]),
        task_count=len(tasks),
        report_path=str(output_path),
        error_count=len(errors),
    )


__all__ = [
    "PreflightReport",
    "load_resource_catalog",
    "run_preflight",
]
