from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from modssc.evaluation import list_metrics
from modssc.hpo import HpoError, Space

from .partition_selection_schema import (
    DCL_PARTITION_SELECTION_KIND,
    PARTITION_SELECTION_TASK_FIELDS,
)


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
class DatasetConfig:
    id: str
    options: dict[str, Any] = field(default_factory=dict)
    download: bool = True
    cache_dir: str | None = None


@dataclass(frozen=True)
class SamplingConfig:
    seed: int | None
    plan: dict[str, Any]
    replay: dict[str, Any] | None = None


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
    modality: str | None = None
    reference_implementation: str | None = None
    reference_policy: dict[str, Any] = field(default_factory=dict)


_MATCH_PAPER_AUGMENTATION_CONTRACTS: dict[
    str,
    tuple[str, str, dict[str, Any], list[str]],
] = {
    "paper:sohn2020-cifar10-table2-250": (
        "fixmatch",
        "google_fixmatch_ra",
        {
            "source": ("google-research/fixmatch@d4985a158065947dba803e626ee9a6721709c570"),
            "strong_order": [
                "random_horizontal_flip",
                "reflect_pad_4_random_crop",
                "randaugment",
                "cutout",
            ],
            "randaugment": {
                "num_ops": 2,
                "configured_magnitude": 10,
                "magnitude_sampling": "integer_uniform_[1,10)",
            },
            "cutout": {"size_pixels": 16, "fill": 0},
        },
        [
            "vision.random_horizontal_flip",
            "vision.random_crop_pad",
            "vision.randaugment",
            "vision.cutout",
        ],
    ),
    "paper:zhang2021-cifar10-table1-250": (
        "flexmatch",
        "torchssl_ra",
        {
            "source": ("TorchSSL/TorchSSL@03193a1b7883727db1ce9c092e083091e18aedbb"),
            "strong_order": [
                "randaugment_with_cutout",
                "random_horizontal_flip",
                "reflect_pad_4_random_crop",
            ],
            "randaugment": {
                "num_ops": 3,
                "configured_magnitude": 5,
                "magnitude_sampling": "per_operation_uniform_full_range",
            },
            "cutout": {
                "size_fraction_sampling": "uniform_[0,0.5)",
                "fill_rgb": [125, 123, 114],
            },
        },
        [
            "vision.randaugment",
            "vision.random_horizontal_flip",
            "vision.random_crop_pad",
        ],
    ),
    "paper:wang2023-cifar10-table1-40": (
        "free_match",
        "torchssl_ra",
        {
            "source": ("TorchSSL/TorchSSL@03193a1b7883727db1ce9c092e083091e18aedbb"),
            "strong_order": [
                "randaugment_with_cutout",
                "random_horizontal_flip",
                "reflect_pad_4_random_crop",
            ],
            "randaugment": {
                "num_ops": 3,
                "configured_magnitude": 5,
                "magnitude_sampling": "per_operation_uniform_full_range",
            },
            "cutout": {
                "size_fraction_sampling": "uniform_[0,0.5)",
                "fill_rgb": [125, 123, 114],
            },
        },
        [
            "vision.randaugment",
            "vision.random_horizontal_flip",
            "vision.random_crop_pad",
        ],
    ),
    "paper:chen2023-cifar10-table2-250": (
        "softmatch",
        "torchssl_ra",
        {
            "source": ("TorchSSL/TorchSSL@03193a1b7883727db1ce9c092e083091e18aedbb"),
            "strong_order": [
                "randaugment_with_cutout",
                "random_horizontal_flip",
                "reflect_pad_4_random_crop",
            ],
            "randaugment": {
                "num_ops": 3,
                "configured_magnitude": 5,
                "magnitude_sampling": "per_operation_uniform_full_range",
            },
            "cutout": {
                "size_fraction_sampling": "uniform_[0,0.5)",
                "fill_rgb": [125, 123, 114],
            },
        },
        [
            "vision.randaugment",
            "vision.random_horizontal_flip",
            "vision.random_crop_pad",
        ],
    ),
}


def _augmentation_step_ids(plan: Mapping[str, Any]) -> list[str]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return []
    return [
        str(step.get("id") or step.get("op_id") or "")
        for step in steps
        if isinstance(step, Mapping)
    ]


def _validate_match_paper_augmentation(
    *,
    method_id: str,
    profile: str,
    augmentation: AugmentationConfig | None,
) -> None:
    contract = _MATCH_PAPER_AUGMENTATION_CONTRACTS.get(profile)
    if contract is None:
        return
    expected_method, implementation, policy, strong_ids = contract
    if method_id != expected_method:
        raise BenchConfigError(f"method.profile={profile!r} requires method.id={expected_method!r}")
    if (
        augmentation is None
        or not augmentation.enabled
        or augmentation.mode != "online"
        or augmentation.modality != "vision"
    ):
        raise BenchConfigError(f"method.profile={profile!r} requires online vision augmentation")
    if augmentation.reference_implementation != implementation:
        raise BenchConfigError(
            f"method.profile={profile!r} requires "
            f"augmentation.reference_implementation={implementation!r}"
        )
    if augmentation.reference_policy != policy:
        raise BenchConfigError(
            f"augmentation.reference_policy contradicts method.profile={profile!r}"
        )
    if _augmentation_step_ids(augmentation.weak) != [
        "vision.random_horizontal_flip",
        "vision.random_crop_pad",
    ]:
        raise BenchConfigError(f"augmentation.weak contradicts method.profile={profile!r}")
    if _augmentation_step_ids(augmentation.strong) != strong_ids:
        raise BenchConfigError(f"augmentation.strong contradicts method.profile={profile!r}")
    strong_steps = augmentation.strong["steps"]
    randaugment = next(step for step in strong_steps if step.get("id") == "vision.randaugment")
    expected_ra = {"num_ops": 2, "magnitude": 10, "num_magnitude_bins": 31}
    if implementation == "torchssl_ra":
        expected_ra = {"num_ops": 3, "magnitude": 5, "num_magnitude_bins": 31}
    if randaugment.get("params") != expected_ra:
        raise BenchConfigError(
            f"augmentation RandAugment parameters contradict method.profile={profile!r}"
        )
    if implementation == "google_fixmatch_ra":
        cutout = next(step for step in strong_steps if step.get("id") == "vision.cutout")
        if cutout.get("params") != {"length": 16, "n_holes": 1, "fill": 0.0}:
            raise BenchConfigError(
                f"augmentation Cutout parameters contradict method.profile={profile!r}"
            )


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    dtype: str
    resolved_device: str | None = None


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
    split_for_model_selection: str | None = None
    evaluation_interval_steps: int | None = None
    checkpoint_policy: str | None = None
    reporting_policy: str | None = None
    reporting_window_checkpoints: int | None = None


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

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
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
        _check_unknown(dataset, {"id", "options", "download", "cache_dir"}, name="dataset")
        ds_cfg = DatasetConfig(
            id=_require_str(dataset, "id", name="dataset"),
            options=_optional_mapping(dataset, "options"),
            download=_optional_bool(dataset, "download", default=True),
            cache_dir=_optional_str(dataset, "cache_dir"),
        )

        sampling = _as_mapping(data.get("sampling", {}), name="sampling")
        _check_unknown(sampling, {"seed", "plan", "replay"}, name="sampling")
        plan = _optional_mapping(sampling, "plan")
        if not plan:
            raise BenchConfigError("sampling.plan must be provided")
        replay_cfg: dict[str, Any] | None = None
        if "replay" in sampling:
            replay = _as_mapping(sampling.get("replay"), name="sampling.replay")
            replay_fields = PARTITION_SELECTION_TASK_FIELDS
            _check_unknown(replay, replay_fields, name="sampling.replay")
            missing = replay_fields - set(replay)
            if missing:
                raise BenchConfigError(
                    f"sampling.replay is missing required keys: {sorted(missing)}"
                )
            if replay.get("kind") != DCL_PARTITION_SELECTION_KIND:
                raise BenchConfigError(
                    "sampling.replay.kind must identify the DCL Vote partition selection"
                )
            for key in replay_fields - {"selection_rank"}:
                _require_str(replay, key, name="sampling.replay")
            selection_rank = replay.get("selection_rank")
            if (
                isinstance(selection_rank, bool)
                or not isinstance(selection_rank, int)
                or selection_rank <= 0
            ):
                raise BenchConfigError("sampling.replay.selection_rank must be a positive integer")
            replay_cfg = dict(replay)
        sampling_cfg = SamplingConfig(
            seed=_optional_int(sampling, "seed"),
            plan=plan,
            replay=replay_cfg,
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
                    "modality",
                    "reference_implementation",
                    "reference_policy",
                },
                name="augmentation",
            )
            augmentation_cfg = AugmentationConfig(
                enabled=_optional_bool(aug, "enabled", default=True),
                seed=_optional_int(aug, "seed"),
                mode=str(aug.get("mode", "fixed")),
                weak=_optional_mapping(aug, "weak"),
                strong=_optional_mapping(aug, "strong"),
                modality=_optional_str(aug, "modality"),
                reference_implementation=_optional_str(aug, "reference_implementation"),
                reference_policy=_optional_mapping(aug, "reference_policy"),
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
        _check_unknown(device_raw, {"device", "dtype", "resolved_device"}, name="method.device")
        device = DeviceConfig(
            device=str(device_raw.get("device", "cpu")),
            dtype=str(device_raw.get("dtype", "float32")),
            resolved_device=_optional_str(device_raw, "resolved_device"),
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
        _validate_match_paper_augmentation(
            method_id=method_cfg.method_id,
            profile=method_cfg.profile,
            augmentation=augmentation_cfg,
        )
        partition_raw = plan.get("partition")
        if isinstance(partition_raw, Mapping):
            artifact_raw = partition_raw.get("ordered_indices_artifact")
            if isinstance(artifact_raw, Mapping):
                inclusive_pool = artifact_raw.get("unlabeled_pool") == "includes_labeled"
                if inclusive_pool and not method_cfg.profile.startswith("paper:"):
                    raise BenchConfigError(
                        "partition.ordered_indices_artifact with "
                        "unlabeled_pool=includes_labeled is restricted to paper profiles"
                    )

        evaluation = _as_mapping(data.get("evaluation", {}), name="evaluation")
        _check_unknown(
            evaluation,
            {
                "report_splits",
                "metrics",
                "split_for_model_selection",
                "evaluation_interval_steps",
                "checkpoint_policy",
                "reporting_policy",
                "reporting_window_checkpoints",
            },
            name="evaluation",
        )
        report_splits = [str(s) for s in _optional_list(evaluation, "report_splits")]
        metrics = [str(m) for m in _optional_list(evaluation, "metrics")]
        if not report_splits:
            raise BenchConfigError("evaluation.report_splits must be provided")
        if not metrics:
            raise BenchConfigError("evaluation.metrics must be provided")
        evaluation_interval_steps = _optional_int(evaluation, "evaluation_interval_steps")
        reporting_window_checkpoints = _optional_int(
            evaluation,
            "reporting_window_checkpoints",
        )
        if evaluation_interval_steps is not None and evaluation_interval_steps <= 0:
            raise BenchConfigError("evaluation.evaluation_interval_steps must be positive")
        if reporting_window_checkpoints is not None and reporting_window_checkpoints <= 0:
            raise BenchConfigError("evaluation.reporting_window_checkpoints must be positive")
        evaluation_cfg = EvaluationConfig(
            report_splits=report_splits,
            metrics=metrics,
            split_for_model_selection=_optional_str(evaluation, "split_for_model_selection"),
            evaluation_interval_steps=evaluation_interval_steps,
            checkpoint_policy=_optional_str(evaluation, "checkpoint_policy"),
            reporting_policy=_optional_str(evaluation, "reporting_policy"),
            reporting_window_checkpoints=reporting_window_checkpoints,
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
                Space.from_dict(space)
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
        )


def _validate_search_space(space: Mapping[str, Any]) -> None:
    def _check_leaf(path: tuple[str, ...]) -> None:
        if len(path) < 3 or path[0] != "method" or path[1] != "params":
            joined = ".".join(path) if path else "<root>"
            raise BenchConfigError(
                f"search.space is limited to method.params.* in v1 (got leaf at {joined!r})"
            )

    def _walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, list):
            if not node:
                raise BenchConfigError("search.space leaves must be non-empty lists")
            _check_leaf(path)
            return
        if isinstance(node, Mapping):
            if not node:
                raise BenchConfigError("search.space cannot contain empty mappings")
            if "dist" in node:
                _check_leaf(path)
                return
            for key in sorted(node.keys()):
                if not isinstance(key, str) or not key:
                    raise BenchConfigError("search.space keys must be non-empty strings")
                _walk(node[key], path + (key,))
            return
        raise BenchConfigError("search.space leaves must be lists or dist specs")

    _walk(space, ())
