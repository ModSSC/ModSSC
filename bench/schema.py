from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from modssc.evaluation import (
    AcceptanceSpec,
    AcceptanceSpecError,
    list_metrics,
    parse_acceptance_spec,
)
from modssc.hpo import (
    RUNTIME_CONTRACT_FIELDS,
    HpoError,
    Space,
    validate_space_targets,
)
from modssc.runtime.artifacts import ArtifactContract, ArtifactContractError
from modssc.runtime.software import SoftwareProvenanceError, normalize_distribution_name


class BenchConfigError(ValueError):
    def __init__(self, message: str, *, code: str = "E_BENCH_CONFIG") -> None:
        self.code = str(code)
        self.message = str(message)
        super().__init__(f"{self.code}: {self.message}")


def _as_mapping(obj: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise BenchConfigError(f"{name} must be a mapping")
    return obj


def _check_unknown(data: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    extra = set(data.keys()) - allowed
    if extra:
        raise BenchConfigError(f"Unknown keys in {name}: {sorted(extra)}")


def _require_str(data: Mapping[str, Any], key: str, *, name: str) -> str:
    val = data.get(key)
    if not isinstance(val, str) or not val.strip():
        raise BenchConfigError(f"{name}.{key} must be a non-empty string")
    return val


def _require_bool(data: Mapping[str, Any], key: str, *, name: str) -> bool:
    val = data.get(key)
    if not isinstance(val, bool):
        raise BenchConfigError(f"{name}.{key} must be a bool")
    return bool(val)


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    val = data.get(key)
    if val is None:
        return None
    if not isinstance(val, str) or not val.strip():
        raise BenchConfigError(f"{key} must be a non-empty string when provided")
    return val


def _optional_sha256(data: Mapping[str, Any], key: str, *, name: str) -> str | None:
    value = _optional_str(data, key)
    if value is None:
        return None
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise BenchConfigError(f"{name}.{key} must be a lowercase SHA-256 digest")
    return value


def _optional_int(data: Mapping[str, Any], key: str) -> int | None:
    val = data.get(key)
    if val is None:
        return None
    if not isinstance(val, int):
        raise BenchConfigError(f"{key} must be an int when provided")
    return int(val)


def _optional_seed_list(data: Mapping[str, Any], key: str, *, name: str) -> list[int] | None:
    val = data.get(key)
    if val is None:
        return None
    if not isinstance(val, list):
        raise BenchConfigError(f"{name}.{key} must be a list[int] when provided")
    if not val:
        raise BenchConfigError(f"{name}.{key} must be non-empty when provided")
    out: list[int] = []
    seen: set[int] = set()
    for i, item in enumerate(val):
        if not isinstance(item, int):
            raise BenchConfigError(f"{name}.{key}[{i}] must be an int")
        seed = int(item)
        if seed in seen:
            raise BenchConfigError(f"{name}.{key} must not contain duplicates")
        seen.add(seed)
        out.append(seed)
    return out


def _optional_positive_int(data: Mapping[str, Any], key: str, *, name: str) -> int | None:
    val = _optional_int(data, key)
    if val is None:
        return None
    if int(val) <= 0:
        raise BenchConfigError(f"{name}.{key} must be > 0 when provided")
    return int(val)


def _optional_bool(data: Mapping[str, Any], key: str, *, default: bool) -> bool:
    val = data.get(key, default)
    if not isinstance(val, bool):
        raise BenchConfigError(f"{key} must be a bool")
    return bool(val)


def _optional_mapping(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    val = data.get(key, {})
    if val is None:
        return {}
    if not isinstance(val, Mapping):
        raise BenchConfigError(f"{key} must be a mapping")
    return dict(val)


def _optional_list(data: Mapping[str, Any], key: str) -> list[Any]:
    val = data.get(key, [])
    if val is None:
        return []
    if not isinstance(val, list):
        raise BenchConfigError(f"{key} must be a list")
    return list(val)


def _input_artifacts(data: Mapping[str, Any]) -> list[ArtifactContract]:
    raw = data.get("input_artifacts", [])
    if not isinstance(raw, list):
        raise BenchConfigError("run.input_artifacts must be a list")
    contracts: list[ArtifactContract] = []
    paths: set[str] = set()
    for index, value in enumerate(raw):
        item = _as_mapping(value, name=f"run.input_artifacts[{index}]")
        _check_unknown(
            item,
            {"path", "kind", "sha256"},
            name=f"run.input_artifacts[{index}]",
        )
        missing = {"path", "kind", "sha256"} - set(item)
        if missing:
            raise BenchConfigError(f"run.input_artifacts[{index}] missing keys: {sorted(missing)}")
        try:
            contract = ArtifactContract(
                path=item["path"],
                kind=item["kind"],
                sha256=item["sha256"],
            )
        except (ArtifactContractError, TypeError) as exc:
            raise BenchConfigError(f"run.input_artifacts[{index}] is invalid: {exc}") from exc
        if contract.path in paths:
            raise BenchConfigError(f"run.input_artifacts contains duplicate path: {contract.path}")
        paths.add(contract.path)
        contracts.append(contract)
    return contracts


def _software_dependencies(data: Mapping[str, Any]) -> list[str]:
    raw = _optional_list(data, "software_dependencies")
    normalized: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str):
            raise BenchConfigError(f"run.software_dependencies[{index}] must be a string")
        try:
            normalized.append(normalize_distribution_name(value))
        except SoftwareProvenanceError as exc:
            raise BenchConfigError(f"run.software_dependencies[{index}] is invalid: {exc}") from exc
    if len(set(normalized)) != len(normalized):
        raise BenchConfigError("run.software_dependencies must not contain duplicates")
    return sorted(normalized)


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    output_dir: str
    seeds: list[int] | None = None
    seeded_sections: list[str] | None = None
    model_seed: int | None = None
    fail_fast: bool = True
    log_level: str | None = None
    benchmark_mode: bool = False
    allow_custom_factories: bool = False
    resume_policy: str = "never"
    checkpoint_dir: str | None = None
    artifact_root: str | None = None
    input_artifacts: list[ArtifactContract] = field(default_factory=list)
    software_dependencies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LimitsConfig:
    profile: str | None = None
    max_preprocess_batch_size: int | None = None
    max_method_batch_size: int | None = None
    max_method_sup_batch_size: int | None = None
    max_graph_chunk_size: int | None = None
    max_train_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(frozen=True)
class DatasetIntegrityConfig:
    fingerprint: str | None = None
    content_sha256: str | None = None
    content_manifest_sha256: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    options: dict[str, Any] = field(default_factory=dict)
    download: bool = True
    cache_dir: str | None = None
    integrity: DatasetIntegrityConfig | None = None


@dataclass(frozen=True)
class SamplingConfig:
    seed: int | None
    plan: dict[str, Any]
    inductive_graph_policy: str = "reject"


@dataclass(frozen=True)
class PreprocessConfig:
    seed: int | None
    fit_on: str | None
    cache: bool
    plan: dict[str, Any]
    cache_dir: str | None = None


@dataclass(frozen=True)
class ViewsConfig:
    seed: int | None
    plan: dict[str, Any]


@dataclass(frozen=True)
class GraphConfig:
    enabled: bool
    seed: int | None
    cache: bool
    require_cache_hit: bool
    spec: dict[str, Any]
    cache_dir: str | None = None
    expected_fingerprint: str | None = None
    expected_preprocess_fingerprint: str | None = None


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    seed: int | None
    mode: str
    weak: dict[str, Any]
    strong: dict[str, Any]
    strong_views: int = 1
    modality: str | None = None
    online_augmenter_id: str | None = None
    online_augmenter_params: dict[str, Any] = field(default_factory=dict)
    online_augmenter_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    dtype: str


@dataclass(frozen=True)
class ModelConfig:
    factory: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    classifier_id: str | None = None
    classifier_backend: str | None = None
    classifier_params: dict[str, Any] = field(default_factory=dict)
    ema: bool | None = None


@dataclass(frozen=True)
class MethodConfig:
    kind: str
    method_id: str
    device: DeviceConfig
    params: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig | None = None
    profile: str = "standardized"


@dataclass(frozen=True)
class EvaluationConfig:
    report_splits: list[str]
    metrics: list[str]
    during_fit_splits: list[str] = field(default_factory=list)
    split_for_model_selection: str | None = None
    test_selection_policy: str = "forbid"


@dataclass(frozen=True)
class SearchObjectiveConfig:
    split: str
    metric: str
    direction: str
    aggregate: str


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool
    kind: str
    seed: int | None
    n_trials: int | None
    repeats: int
    objective: SearchObjectiveConfig
    space: dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    dataset: DatasetConfig
    sampling: SamplingConfig
    preprocess: PreprocessConfig
    method: MethodConfig
    evaluation: EvaluationConfig
    graph: GraphConfig | None = None
    views: ViewsConfig | None = None
    augmentation: AugmentationConfig | None = None
    search: SearchConfig | None = None
    limits: LimitsConfig | None = None
    acceptance: AcceptanceSpec | None = None

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        allow_resolved_acceptance_seed: bool = False,
    ) -> ExperimentConfig:
        data = _as_mapping(raw, name="config")
        _check_unknown(
            data,
            {
                "run",
                "limits",
                "dataset",
                "sampling",
                "preprocess",
                "views",
                "graph",
                "augmentation",
                "method",
                "evaluation",
                "search",
                "acceptance",
            },
            name="config",
        )

        run = _as_mapping(data.get("run", {}), name="run")
        _check_unknown(
            run,
            {
                "name",
                "seed",
                "seeds",
                "seeded_sections",
                "model_seed",
                "output_dir",
                "fail_fast",
                "log_level",
                "benchmark_mode",
                "allow_custom_factories",
                "resume_policy",
                "checkpoint_dir",
                "artifact_root",
                "input_artifacts",
                "software_dependencies",
            },
            name="run",
        )
        benchmark_mode = _optional_bool(run, "benchmark_mode", default=False)
        fail_fast = _optional_bool(run, "fail_fast", default=True)
        if benchmark_mode and not fail_fast:
            raise BenchConfigError(
                "run.fail_fast must be true when run.benchmark_mode=true",
                code="E_BENCH_FAIL_FAST_REQUIRED",
            )
        seeded_sections = None
        if "seeded_sections" in run:
            seeded_sections = [str(item) for item in _optional_list(run, "seeded_sections")]
        resume_policy = str(run.get("resume_policy", "never"))
        if resume_policy not in {"never", "auto", "required"}:
            raise BenchConfigError("run.resume_policy must be one of: auto, never, required")
        input_artifacts = _input_artifacts(run)
        artifact_root = _optional_str(run, "artifact_root")
        if input_artifacts and artifact_root is None:
            raise BenchConfigError(
                "run.artifact_root is required when run.input_artifacts is not empty"
            )

        run_cfg = RunConfig(
            name=_require_str(run, "name", name="run"),
            seed=int(run.get("seed", 0)),
            output_dir=str(run.get("output_dir", "modssc_cache/output")),
            seeds=_optional_seed_list(run, "seeds", name="run"),
            seeded_sections=seeded_sections,
            model_seed=_optional_int(run, "model_seed"),
            fail_fast=fail_fast,
            log_level=_optional_str(run, "log_level"),
            benchmark_mode=benchmark_mode,
            allow_custom_factories=_optional_bool(run, "allow_custom_factories", default=False),
            resume_policy=resume_policy,
            checkpoint_dir=_optional_str(run, "checkpoint_dir"),
            artifact_root=artifact_root,
            input_artifacts=input_artifacts,
            software_dependencies=_software_dependencies(run),
        )

        limits_cfg = None
        if "limits" in data:
            limits_raw = data.get("limits", {})
            if limits_raw is None:
                limits_raw = {}
            limits = _as_mapping(limits_raw, name="limits")
            _check_unknown(
                limits,
                {
                    "profile",
                    "max_preprocess_batch_size",
                    "max_method_batch_size",
                    "max_method_sup_batch_size",
                    "max_graph_chunk_size",
                    "max_train_samples",
                    "max_test_samples",
                },
                name="limits",
            )
            profile = _optional_str(limits, "profile")
            if profile is not None:
                profile = profile.lower()
                if profile not in {"auto", "v100", "h100"}:
                    raise BenchConfigError("limits.profile must be auto, v100, or h100")
            limits_cfg = LimitsConfig(
                profile=profile,
                max_preprocess_batch_size=_optional_positive_int(
                    limits, "max_preprocess_batch_size", name="limits"
                ),
                max_method_batch_size=_optional_positive_int(
                    limits, "max_method_batch_size", name="limits"
                ),
                max_method_sup_batch_size=_optional_positive_int(
                    limits, "max_method_sup_batch_size", name="limits"
                ),
                max_graph_chunk_size=_optional_positive_int(
                    limits, "max_graph_chunk_size", name="limits"
                ),
                max_train_samples=_optional_positive_int(
                    limits, "max_train_samples", name="limits"
                ),
                max_test_samples=_optional_positive_int(limits, "max_test_samples", name="limits"),
            )

        dataset = _as_mapping(data.get("dataset", {}), name="dataset")
        _check_unknown(
            dataset,
            {"id", "options", "download", "cache_dir", "integrity"},
            name="dataset",
        )
        integrity_cfg = None
        if "integrity" in dataset:
            integrity = _as_mapping(dataset.get("integrity"), name="dataset.integrity")
            _check_unknown(
                integrity,
                {"fingerprint", "content_sha256", "content_manifest_sha256"},
                name="dataset.integrity",
            )
            integrity_cfg = DatasetIntegrityConfig(
                fingerprint=_optional_sha256(integrity, "fingerprint", name="dataset.integrity"),
                content_sha256=_optional_sha256(
                    integrity, "content_sha256", name="dataset.integrity"
                ),
                content_manifest_sha256=_optional_sha256(
                    integrity, "content_manifest_sha256", name="dataset.integrity"
                ),
            )
            if not any(
                (
                    integrity_cfg.fingerprint,
                    integrity_cfg.content_sha256,
                    integrity_cfg.content_manifest_sha256,
                )
            ):
                raise BenchConfigError("dataset.integrity must pin at least one identity")
        ds_cfg = DatasetConfig(
            id=_require_str(dataset, "id", name="dataset"),
            options=_optional_mapping(dataset, "options"),
            download=_optional_bool(dataset, "download", default=True),
            cache_dir=_optional_str(dataset, "cache_dir"),
            integrity=integrity_cfg,
        )

        sampling = _as_mapping(data.get("sampling", {}), name="sampling")
        _check_unknown(
            sampling,
            {"seed", "plan", "inductive_graph_policy"},
            name="sampling",
        )
        plan = _optional_mapping(sampling, "plan")
        if not plan:
            raise BenchConfigError("sampling.plan must be provided")
        sampling_cfg = SamplingConfig(
            seed=_optional_int(sampling, "seed"),
            plan=plan,
            inductive_graph_policy=(_optional_str(sampling, "inductive_graph_policy") or "reject"),
        )

        preprocess = _as_mapping(data.get("preprocess", {}), name="preprocess")
        _check_unknown(
            preprocess, {"seed", "fit_on", "cache", "plan", "cache_dir"}, name="preprocess"
        )
        pre_plan = _optional_mapping(preprocess, "plan")
        if not pre_plan:
            raise BenchConfigError("preprocess.plan must be provided")
        preprocess_cfg = PreprocessConfig(
            seed=_optional_int(preprocess, "seed"),
            fit_on=_optional_str(preprocess, "fit_on"),
            cache=_optional_bool(preprocess, "cache", default=True),
            plan=pre_plan,
            cache_dir=_optional_str(preprocess, "cache_dir"),
        )

        views_cfg = None
        if "views" in data:
            views = _as_mapping(data.get("views", {}), name="views")
            _check_unknown(views, {"seed", "plan"}, name="views")
            views_plan = _optional_mapping(views, "plan")
            if not views_plan:
                raise BenchConfigError("views.plan must be provided when views is set")
            views_cfg = ViewsConfig(seed=_optional_int(views, "seed"), plan=views_plan)

        graph_cfg = None
        if "graph" in data:
            graph = _as_mapping(data.get("graph", {}), name="graph")
            _check_unknown(
                graph,
                {
                    "enabled",
                    "seed",
                    "cache",
                    "require_cache_hit",
                    "cache_dir",
                    "expected_fingerprint",
                    "expected_preprocess_fingerprint",
                    "spec",
                },
                name="graph",
            )
            graph_cfg = GraphConfig(
                enabled=_optional_bool(graph, "enabled", default=False),
                seed=_optional_int(graph, "seed"),
                cache=_optional_bool(graph, "cache", default=True),
                require_cache_hit=_optional_bool(graph, "require_cache_hit", default=False),
                cache_dir=_optional_str(graph, "cache_dir"),
                expected_fingerprint=_optional_str(graph, "expected_fingerprint"),
                expected_preprocess_fingerprint=_optional_str(
                    graph, "expected_preprocess_fingerprint"
                ),
                spec=_optional_mapping(graph, "spec"),
            )

        augmentation_cfg = None
        if "augmentation" in data:
            aug = _as_mapping(data.get("augmentation", {}), name="augmentation")
            _check_unknown(
                aug,
                {
                    "enabled",
                    "seed",
                    "mode",
                    "weak",
                    "strong",
                    "strong_views",
                    "modality",
                    "online_augmenter_id",
                    "online_augmenter_params",
                    "online_augmenter_metadata",
                },
                name="augmentation",
            )
            strong_views = _optional_int(aug, "strong_views")
            if strong_views is None:
                strong_views = 1
            if strong_views not in {1, 2}:
                raise BenchConfigError("augmentation.strong_views must be 1 or 2")
            augmentation_cfg = AugmentationConfig(
                enabled=_optional_bool(aug, "enabled", default=True),
                seed=_optional_int(aug, "seed"),
                mode=str(aug.get("mode", "fixed")),
                weak=_optional_mapping(aug, "weak"),
                strong=_optional_mapping(aug, "strong"),
                strong_views=strong_views,
                modality=_optional_str(aug, "modality"),
                online_augmenter_id=_optional_str(aug, "online_augmenter_id"),
                online_augmenter_params=_optional_mapping(aug, "online_augmenter_params"),
                online_augmenter_metadata=_optional_mapping(aug, "online_augmenter_metadata"),
            )
            if augmentation_cfg.online_augmenter_id is None and (
                augmentation_cfg.online_augmenter_params
                or augmentation_cfg.online_augmenter_metadata
            ):
                raise BenchConfigError(
                    "augmentation online augmenter params/metadata require online_augmenter_id"
                )
            if "seed" in augmentation_cfg.online_augmenter_params:
                raise BenchConfigError(
                    "augmentation.online_augmenter_params must not redefine seed; "
                    "use augmentation.seed or the run seed"
                )
            if (
                augmentation_cfg.online_augmenter_id is not None
                and augmentation_cfg.mode != "online"
            ):
                raise BenchConfigError(
                    "augmentation.online_augmenter_id requires augmentation.mode='online'"
                )
            if augmentation_cfg.mode == "online" and augmentation_cfg.strong_views != 1:
                raise BenchConfigError(
                    "augmentation.mode='online' supports exactly one strong view"
                )

        method = _as_mapping(data.get("method", {}), name="method")
        _check_unknown(
            method,
            {"kind", "id", "device", "profile", "params", "model"},
            name="method",
        )
        kind = _require_str(method, "kind", name="method")
        if kind not in {"inductive", "transductive"}:
            raise BenchConfigError("method.kind must be 'inductive' or 'transductive'")
        device_raw = _as_mapping(method.get("device", {}), name="method.device")
        _check_unknown(device_raw, {"device", "dtype"}, name="method.device")
        device = DeviceConfig(
            device=str(device_raw.get("device", "cpu")),
            dtype=str(device_raw.get("dtype", "float32")),
        )
        model_cfg = None
        model_raw = method.get("model")
        if model_raw is not None:
            model_map = _as_mapping(model_raw, name="method.model")
            has_factory = "factory" in model_map
            has_classifier = "classifier_id" in model_map
            if has_factory and has_classifier:
                raise BenchConfigError(
                    "method.model must use either factory or classifier_id, not both"
                )
            if not has_factory and not has_classifier:
                raise BenchConfigError("method.model must define factory or classifier_id")
            if has_factory:
                if not run_cfg.allow_custom_factories:
                    raise BenchConfigError(
                        "method.model.factory requires run.allow_custom_factories=true; "
                        "only enable this for trusted configs",
                        code="E_BENCH_CUSTOM_FACTORY_DISABLED",
                    )
                _check_unknown(model_map, {"factory", "params"}, name="method.model")
                model_cfg = ModelConfig(
                    factory=_require_str(model_map, "factory", name="method.model"),
                    params=_optional_mapping(model_map, "params"),
                )
            else:
                _check_unknown(
                    model_map,
                    {"classifier_id", "classifier_backend", "classifier_params", "ema"},
                    name="method.model",
                )
                backend = model_map.get("classifier_backend", "torch")
                if not isinstance(backend, str) or not backend.strip():
                    raise BenchConfigError(
                        "method.model.classifier_backend must be a non-empty string"
                    )
                ema_val = model_map.get("ema")
                if ema_val is None:
                    ema = None
                elif isinstance(ema_val, bool):
                    ema = bool(ema_val)
                else:
                    raise BenchConfigError("method.model.ema must be a bool when provided")
                model_cfg = ModelConfig(
                    classifier_id=_require_str(model_map, "classifier_id", name="method.model"),
                    classifier_backend=str(backend),
                    classifier_params=_optional_mapping(model_map, "classifier_params"),
                    ema=ema,
                )
        method_cfg = MethodConfig(
            kind=kind,
            method_id=_require_str(method, "id", name="method"),
            device=device,
            profile=_optional_str(method, "profile") or "standardized",
            params=_optional_mapping(method, "params"),
            model=model_cfg,
        )
        if "profile" in method_cfg.params:
            raise BenchConfigError(
                "declare the opaque execution profile only as method.profile; "
                "method.params.profile is ambiguous"
            )

        evaluation = _as_mapping(data.get("evaluation", {}), name="evaluation")
        _check_unknown(
            evaluation,
            {
                "report_splits",
                "metrics",
                "during_fit_splits",
                "split_for_model_selection",
                "test_selection_policy",
            },
            name="evaluation",
        )
        report_splits = [str(s) for s in _optional_list(evaluation, "report_splits")]
        metrics = [str(m) for m in _optional_list(evaluation, "metrics")]
        during_fit_splits = [
            str(split) for split in _optional_list(evaluation, "during_fit_splits")
        ]
        if not report_splits:
            raise BenchConfigError("evaluation.report_splits must be provided")
        if not metrics:
            raise BenchConfigError("evaluation.metrics must be provided")
        test_selection_policy = _optional_str(evaluation, "test_selection_policy") or "forbid"
        if test_selection_policy not in {"forbid", "paper_protocol"}:
            raise BenchConfigError(
                "evaluation.test_selection_policy must be 'forbid' or 'paper_protocol'"
            )
        evaluation_cfg = EvaluationConfig(
            report_splits=report_splits,
            metrics=metrics,
            during_fit_splits=during_fit_splits,
            split_for_model_selection=_optional_str(evaluation, "split_for_model_selection"),
            test_selection_policy=test_selection_policy,
        )

        search_cfg = None
        if "search" in data:
            search = _as_mapping(data.get("search", {}), name="search")
            _check_unknown(
                search,
                {"enabled", "kind", "seed", "n_trials", "repeats", "objective", "space"},
                name="search",
            )
            enabled = _require_bool(search, "enabled", name="search")

            kind = _require_str(search, "kind", name="search")
            if kind not in {"grid", "random"}:
                raise BenchConfigError("search.kind must be 'grid' or 'random'")

            seed = _optional_int(search, "seed")
            n_trials = _optional_int(search, "n_trials")
            if n_trials is not None and n_trials <= 0:
                raise BenchConfigError("search.n_trials must be > 0 when provided")
            if kind == "random" and seed is None:
                raise BenchConfigError("search.seed must be provided for random search")
            if kind == "random" and n_trials is None:
                raise BenchConfigError("search.n_trials must be provided for random search")

            repeats = _optional_int(search, "repeats")
            if repeats is None:
                repeats = 1
            if repeats <= 0:
                raise BenchConfigError("search.repeats must be > 0")

            objective_raw = _as_mapping(search.get("objective", {}), name="search.objective")
            _check_unknown(
                objective_raw,
                {"split", "metric", "direction", "aggregate"},
                name="search.objective",
            )
            split = _require_str(objective_raw, "split", name="search.objective")
            metric = _require_str(objective_raw, "metric", name="search.objective")
            direction = _require_str(objective_raw, "direction", name="search.objective")
            aggregate = _require_str(objective_raw, "aggregate", name="search.objective")

            if split not in {"train", "val", "test"}:
                raise BenchConfigError("search.objective.split must be train/val/test")
            if metric not in list_metrics():
                raise BenchConfigError(f"Unknown metric for search: {metric}")
            if direction not in {"maximize", "minimize"}:
                raise BenchConfigError("search.objective.direction must be maximize/minimize")
            if aggregate != "mean":
                raise BenchConfigError("search.objective.aggregate must be 'mean'")

            space = _optional_mapping(search, "space")
            if not space:
                raise BenchConfigError("search.space must be provided")
            _validate_search_space(space)
            try:
                parsed_space = Space.from_dict(space)
                if kind == "grid":
                    # Grid iteration is lazy. Advancing once validates that every
                    # leaf is a finite list or choice distribution without
                    # materializing the Cartesian product.
                    next(parsed_space.iter_grid())
            except HpoError as exc:
                raise BenchConfigError(f"search.space invalid: {exc}") from exc

            search_cfg = SearchConfig(
                enabled=enabled,
                kind=kind,
                seed=seed,
                n_trials=n_trials,
                repeats=repeats,
                objective=SearchObjectiveConfig(
                    split=split,
                    metric=metric,
                    direction=direction,
                    aggregate=aggregate,
                ),
                space=space,
            )

        acceptance_cfg = None
        if "acceptance" in data:
            acceptance_raw = _as_mapping(data.get("acceptance"), name="acceptance")
            try:
                acceptance_cfg = parse_acceptance_spec(acceptance_raw)
            except AcceptanceSpecError as exc:
                raise BenchConfigError(
                    f"acceptance is invalid: {exc}",
                    code="E_BENCH_ACCEPTANCE_SCHEMA",
                ) from exc
            if not run_cfg.benchmark_mode:
                raise BenchConfigError(
                    "acceptance requires run.benchmark_mode=true",
                    code="E_BENCH_ACCEPTANCE_STRICT_REQUIRED",
                )
            if acceptance_cfg.method_id != method_cfg.method_id:
                raise BenchConfigError(
                    "acceptance.method_id must equal method.id",
                    code="E_BENCH_ACCEPTANCE_METHOD_MISMATCH",
                )
            if run_cfg.seeds is None and not allow_resolved_acceptance_seed:
                raise BenchConfigError(
                    "acceptance requires run.seeds on an unresolved YAML card",
                    code="E_BENCH_ACCEPTANCE_REPETITIONS_MISMATCH",
                )
            if run_cfg.seeds is not None and len(run_cfg.seeds) != acceptance_cfg.repetitions:
                raise BenchConfigError(
                    "run.seeds must contain exactly acceptance.repetitions values",
                    code="E_BENCH_ACCEPTANCE_REPETITIONS_MISMATCH",
                )

        return cls(
            run=run_cfg,
            limits=limits_cfg,
            dataset=ds_cfg,
            sampling=sampling_cfg,
            preprocess=preprocess_cfg,
            method=method_cfg,
            evaluation=evaluation_cfg,
            graph=graph_cfg,
            views=views_cfg,
            augmentation=augmentation_cfg,
            search=search_cfg,
            acceptance=acceptance_cfg,
        )


def _validate_search_space(space: Mapping[str, Any]) -> None:
    try:
        validate_space_targets(
            space,
            allowed_prefix=("method", "params"),
            forbidden_leaf_names=RUNTIME_CONTRACT_FIELDS,
        )
    except HpoError as exc:
        raise BenchConfigError(f"search.space invalid: {exc}") from exc
