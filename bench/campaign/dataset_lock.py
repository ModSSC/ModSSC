from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bench.orchestrators import sampling as sampling_orch
from bench.schema import ExperimentConfig
from bench.seed_sweep import apply_global_seed
from bench.utils.hashing import hash_any
from bench.utils.io import load_yaml
from modssc.data_loader import load_dataset, verify_dataset_content
from modssc.utils.io import atomic_write_text

from .errors import CampaignError
from .spec import load_spec, string_list


@dataclass(frozen=True)
class DatasetLockResult:
    output_path: str
    dataset_count: int
    prepared_request_count: int
    datasets: dict[str, dict[str, str]]


@dataclass(frozen=True)
class PaperDatasetObservationResult:
    output_path: str
    protocol_count: int
    prepared_request_count: int
    protocols: dict[str, dict[str, Any]]


def _path_metadata(path: Path) -> tuple[str | None, str | None]:
    parts = path.parts
    regime = next((part for part in parts if part.startswith("R") and part[1:].isdigit()), None)
    modality = next(
        (
            candidate
            for candidate in ("tabular", "vision", "text", "audio", "graph")
            if candidate in parts
        ),
        None,
    )
    return modality, regime


def _selected_configs(spec: Mapping[str, Any], *, repo_root: Path) -> list[Path]:
    selection = spec.get("selection")
    if spec.get("track") != "standardized" or not isinstance(selection, Mapping):
        raise CampaignError(
            "E_CAMPAIGN_DATASET_LOCK_INVALID",
            "lock-datasets requires a standardized campaign spec",
        )
    root_value = selection.get("config_root")
    if not isinstance(root_value, str) or not root_value:
        raise CampaignError("E_CAMPAIGN_DATASET_LOCK_INVALID", "selection.config_root is required")
    config_root = (repo_root / root_value).resolve()
    if not config_root.is_dir():
        raise CampaignError(
            "E_CAMPAIGN_DATASET_LOCK_INVALID", f"configuration root not found: {root_value}"
        )
    methods = set(string_list(selection.get("methods"), field="selection.methods"))
    filters: dict[str, set[str] | None] = {}
    for field in ("regimes", "modalities", "datasets"):
        value = selection.get(field)
        filters[field] = (
            None if value is None else set(string_list(value, field=f"selection.{field}"))
        )

    selected: list[Path] = []
    for path in sorted(config_root.rglob("*.yaml")):
        if path.name == "regime_manifest.yaml":
            continue
        relative = path.relative_to(config_root)
        if len(relative.parts) < 5 or relative.parts[2] not in methods:
            continue
        modality, regime = _path_metadata(relative)
        if filters["modalities"] is not None and modality not in filters["modalities"]:
            continue
        if filters["regimes"] is not None and regime not in filters["regimes"]:
            continue
        cfg = ExperimentConfig.from_dict(load_yaml(path))
        if cfg.method.method_id not in methods:
            continue
        if filters["datasets"] is not None and cfg.dataset.id not in filters["datasets"]:
            continue
        selected.append(path)
    if not selected:
        raise CampaignError("E_CAMPAIGN_SELECTION_EMPTY", "no configurations selected")
    return selected


def _prepared_request_key(effective: Mapping[str, Any], cfg: ExperimentConfig) -> str:
    plan = cfg.sampling.plan
    merge = bool(
        plan.get("policy", {}).get("merge_official_splits", False)
        if isinstance(plan.get("policy"), Mapping)
        else False
    )
    return hash_any(
        {
            "dataset": effective.get("dataset", {}),
            "merge_official_splits": merge,
        }
    )


def _observe_prepared_requests(
    prepared_requests: Mapping[str, tuple[str, dict[str, Any]]],
    *,
    dataset_cache_dir: Path | None,
) -> dict[str, dict[str, str]]:
    observations: dict[str, dict[str, str]] = {}
    verified_requests: set[str] = set()
    for prepared_key, (dataset_id, effective) in sorted(prepared_requests.items()):
        cfg = ExperimentConfig.from_dict(effective)
        if cfg.dataset.download:
            raise CampaignError(
                "E_CAMPAIGN_DATASET_LOCK_INVALID",
                f"{dataset_id} permits downloads; lock creation is offline-only",
            )
        cache_dir = dataset_cache_dir or (
            Path(cfg.dataset.cache_dir).expanduser().resolve() if cfg.dataset.cache_dir else None
        )
        dataset_request = hash_any(effective.get("dataset", {}))
        dataset = load_dataset(
            dataset_id,
            cache_dir=cache_dir,
            download=False,
            options=dict(cfg.dataset.options),
        )
        prepared = sampling_orch.prepare_dataset(dataset, plan_dict=cfg.sampling.plan)
        evidence = verify_dataset_content(
            dataset_id,
            cache_dir=cache_dir,
            options=dict(cfg.dataset.options),
            rehash=dataset_request not in verified_requests,
        )
        verified_requests.add(dataset_request)
        logical = prepared.meta.get("dataset_fingerprint")
        if not isinstance(logical, str) or not logical:
            raise CampaignError(
                "E_CAMPAIGN_DATASET_LOCK_INVALID",
                f"{dataset_id} has no logical dataset fingerprint",
            )
        observations[prepared_key] = {
            "dataset_id": dataset_id,
            "fingerprint": logical,
            "content_sha256": evidence["content_sha256"],
        }
    return observations


def _paper_observations(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    output_path: Path,
    dataset_cache_dir: Path | None,
) -> PaperDatasetObservationResult:
    cells = spec.get("cells")
    if not isinstance(cells, list) or not cells:
        raise CampaignError("E_CAMPAIGN_DATASET_LOCK_INVALID", "paper cells are required")
    prepared_requests: dict[str, tuple[str, dict[str, Any]]] = {}
    protocol_keys: dict[str, set[str]] = {}
    protocol_dataset_requests: dict[str, set[str]] = {}
    protocol_split_requests: dict[str, set[str]] = {}
    protocol_dataset_ids: dict[str, set[str]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise CampaignError("E_CAMPAIGN_DATASET_LOCK_INVALID", "paper cell must be a mapping")
        protocol_id = cell.get("protocol_id")
        config_value = cell.get("config")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise CampaignError("E_CAMPAIGN_DATASET_LOCK_INVALID", "paper protocol_id is required")
        if protocol_id in protocol_keys:
            raise CampaignError(
                "E_CAMPAIGN_DATASET_LOCK_INVALID", f"duplicate paper protocol_id: {protocol_id}"
            )
        if not isinstance(config_value, str) or not config_value:
            raise CampaignError("E_CAMPAIGN_DATASET_LOCK_INVALID", "paper config is required")
        config_path = (repo_root / config_value).resolve()
        try:
            config_path.relative_to(repo_root)
        except ValueError as exc:
            raise CampaignError(
                "E_CAMPAIGN_DATASET_LOCK_INVALID", "paper config must be inside the repository"
            ) from exc
        if not config_path.is_file():
            raise CampaignError(
                "E_CAMPAIGN_DATASET_LOCK_INVALID", f"paper config not found: {config_value}"
            )
        raw = load_yaml(config_path)
        cfg = ExperimentConfig.from_dict(raw)
        raw_seeds = cell.get("seeds", "from_config")
        if raw_seeds == "from_config":
            seeds = cfg.run.seeds or [cfg.run.seed]
        elif (
            isinstance(raw_seeds, list)
            and raw_seeds
            and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in raw_seeds)
        ):
            seeds = raw_seeds
        else:
            raise CampaignError(
                "E_CAMPAIGN_DATASET_LOCK_INVALID",
                f"invalid seeds for paper protocol {protocol_id}",
            )
        protocol_keys[protocol_id] = set()
        protocol_dataset_requests[protocol_id] = set()
        protocol_split_requests[protocol_id] = set()
        protocol_dataset_ids[protocol_id] = set()
        for seed in seeds:
            effective = apply_global_seed(raw, seed=seed, seeded_sections=cfg.run.seeded_sections)
            effective_cfg = ExperimentConfig.from_dict(effective)
            dataset_block = effective.get("dataset", {})
            sampling_block = effective.get("sampling", {})
            dataset_request = hash_any(dataset_block)
            split_request = hash_any(
                {
                    "dataset_request_sha256": dataset_request,
                    "sampling": sampling_block,
                }
            )
            prepared_key = _prepared_request_key(effective, effective_cfg)
            prepared_requests.setdefault(prepared_key, (effective_cfg.dataset.id, effective))
            protocol_keys[protocol_id].add(prepared_key)
            protocol_dataset_requests[protocol_id].add(dataset_request)
            protocol_split_requests[protocol_id].add(split_request)
            protocol_dataset_ids[protocol_id].add(effective_cfg.dataset.id)

    observed = _observe_prepared_requests(
        prepared_requests,
        dataset_cache_dir=dataset_cache_dir,
    )
    protocols: dict[str, dict[str, Any]] = {}
    for protocol_id in sorted(protocol_keys):
        identities = [observed[key] for key in sorted(protocol_keys[protocol_id])]
        dataset_ids = {identity["dataset_id"] for identity in identities}
        fingerprints = {identity["fingerprint"] for identity in identities}
        contents = {identity["content_sha256"] for identity in identities}
        if (
            len(dataset_ids) != 1
            or len(fingerprints) != 1
            or len(contents) != 1
            or len(protocol_dataset_ids[protocol_id]) != 1
        ):
            raise CampaignError(
                "E_CAMPAIGN_DATASET_LOCK_DIVERGENCE",
                f"paper protocol {protocol_id} resolves to multiple dataset identities",
            )
        protocols[protocol_id] = {
            "dataset_id": next(iter(dataset_ids)),
            "fingerprint": next(iter(fingerprints)),
            "content_sha256": next(iter(contents)),
            "dataset_request_sha256s": sorted(protocol_dataset_requests[protocol_id]),
            "split_request_sha256s": sorted(protocol_split_requests[protocol_id]),
        }
    payload = {
        "schema_version": 1,
        "kind": "modssc.paper-dataset-observations",
        "protocols": protocols,
    }
    atomic_write_text(output_path, yaml.safe_dump(payload, sort_keys=False))
    return PaperDatasetObservationResult(
        output_path=str(output_path),
        protocol_count=len(protocols),
        prepared_request_count=len(prepared_requests),
        protocols=protocols,
    )


def create_dataset_lock(
    spec_path: Path,
    *,
    repo_root: Path,
    output_path: Path,
    dataset_cache_dir: Path | None = None,
    overwrite: bool = False,
) -> DatasetLockResult | PaperDatasetObservationResult:
    """Hash selected offline datasets for a standardized or paper campaign."""

    repo_root = repo_root.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignError(
            "E_CAMPAIGN_DATASET_LOCK_INVALID",
            "dataset lock output must be inside the repository",
        ) from exc
    if output_path.exists() and not overwrite:
        raise CampaignError(
            "E_CAMPAIGN_DATASET_LOCK_EXISTS",
            f"refusing to replace existing dataset lock: {output_path}",
        )
    spec = load_spec(spec_path.resolve())
    if spec["track"] == "paper":
        return _paper_observations(
            spec,
            repo_root=repo_root,
            output_path=output_path,
            dataset_cache_dir=dataset_cache_dir,
        )
    selected = _selected_configs(spec, repo_root=repo_root)
    prepared_requests: dict[str, tuple[str, dict[str, Any]]] = {}
    for config_path in selected:
        raw = load_yaml(config_path)
        cfg = ExperimentConfig.from_dict(raw)
        seeds = cfg.run.seeds or [cfg.run.seed]
        for seed in seeds:
            effective = apply_global_seed(raw, seed=seed, seeded_sections=cfg.run.seeded_sections)
            effective_cfg = ExperimentConfig.from_dict(effective)
            request_key = _prepared_request_key(effective, effective_cfg)
            prepared_requests.setdefault(request_key, (effective_cfg.dataset.id, effective))

    identities: dict[str, dict[str, set[str]]] = {}
    observed = _observe_prepared_requests(
        prepared_requests,
        dataset_cache_dir=dataset_cache_dir,
    )
    for identity in observed.values():
        dataset_id = identity["dataset_id"]
        entry = identities.setdefault(dataset_id, {"fingerprint": set(), "content_sha256": set()})
        entry["fingerprint"].add(identity["fingerprint"])
        entry["content_sha256"].add(identity["content_sha256"])

    locked: dict[str, dict[str, str]] = {}
    for dataset_id, values in sorted(identities.items()):
        if len(values["fingerprint"]) != 1 or len(values["content_sha256"]) != 1:
            raise CampaignError(
                "E_CAMPAIGN_DATASET_LOCK_DIVERGENCE",
                f"{dataset_id} resolves to multiple logical or content identities",
            )
        locked[dataset_id] = {
            "fingerprint": next(iter(values["fingerprint"])),
            "content_sha256": next(iter(values["content_sha256"])),
        }
    payload = {"schema_version": 2, "datasets": locked}
    atomic_write_text(output_path, yaml.safe_dump(payload, sort_keys=False))
    return DatasetLockResult(
        output_path=str(output_path.resolve()),
        dataset_count=len(locked),
        prepared_request_count=len(prepared_requests),
        datasets=locked,
    )


__all__ = [
    "DatasetLockResult",
    "PaperDatasetObservationResult",
    "create_dataset_lock",
]
