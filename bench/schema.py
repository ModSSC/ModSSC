from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from modssc.evaluation import list_metrics
from modssc.hpo import HpoError, Space


class BenchConfigError(ValueError):
    pass


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
    fail_fast: bool = True
    log_level: str | None = None


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


@dataclass(frozen=True)
class PreprocessConfig:
    seed: int | None
    fit_on: str | None
    cache: bool
    plan: dict[str, Any]


@dataclass(frozen=True)
class ViewsConfig:
    seed: int | None
    plan: dict[str, Any]


@dataclass(frozen=True)
class GraphConfig:
    enabled: bool
    seed: int | None
    cache: bool
    spec: dict[str, Any]
    cache_dir: str | None = None


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    seed: int | None
    mode: str
    weak: dict[str, Any]
    strong: dict[str, Any]
    modality: str | None = None


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


@dataclass(frozen=True)
class EvaluationConfig:
    report_splits: list[str]
    metrics: list[str]
    split_for_model_selection: str | None = None


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

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        data = _as_mapping(raw, name="config")
        _check_unknown(
            data,
            {
                "run",
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
        _check_unknown(run, {"name", "seed", "output_dir", "fail_fast", "log_level"}, name="run")
        run_cfg = RunConfig(
            name=_require_str(run, "name", name="run"),
            seed=int(run.get("seed", 0)),
            output_dir=str(run.get("output_dir", "runs")),
            fail_fast=_optional_bool(run, "fail_fast", default=True),
            log_level=_optional_str(run, "log_level"),
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
        _check_unknown(sampling, {"seed", "plan"}, name="sampling")
        plan = _optional_mapping(sampling, "plan")
        if not plan:
            raise BenchConfigError("sampling.plan must be provided")
        sampling_cfg = SamplingConfig(seed=_optional_int(sampling, "seed"), plan=plan)

        preprocess = _as_mapping(data.get("preprocess", {}), name="preprocess")
        _check_unknown(preprocess, {"seed", "fit_on", "cache", "plan"}, name="preprocess")
        pre_plan = _optional_mapping(preprocess, "plan")
        if not pre_plan:
            raise BenchConfigError("preprocess.plan must be provided")
        preprocess_cfg = PreprocessConfig(
            seed=_optional_int(preprocess, "seed"),
            fit_on=_optional_str(preprocess, "fit_on"),
            cache=_optional_bool(preprocess, "cache", default=True),
            plan=pre_plan,
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
            _check_unknown(graph, {"enabled", "seed", "cache", "cache_dir", "spec"}, name="graph")
            graph_cfg = GraphConfig(
                enabled=_optional_bool(graph, "enabled", default=False),
                seed=_optional_int(graph, "seed"),
                cache=_optional_bool(graph, "cache", default=True),
                cache_dir=_optional_str(graph, "cache_dir"),
                spec=_optional_mapping(graph, "spec"),
            )

        augmentation_cfg = None
        if "augmentation" in data:
            aug = _as_mapping(data.get("augmentation", {}), name="augmentation")
            _check_unknown(
                aug, {"enabled", "seed", "mode", "weak", "strong", "modality"}, name="augmentation"
            )
            augmentation_cfg = AugmentationConfig(
                enabled=_optional_bool(aug, "enabled", default=True),
                seed=_optional_int(aug, "seed"),
                mode=str(aug.get("mode", "fixed")),
                weak=_optional_mapping(aug, "weak"),
                strong=_optional_mapping(aug, "strong"),
                modality=_optional_str(aug, "modality"),
            )

        method = _as_mapping(data.get("method", {}), name="method")
        _check_unknown(method, {"kind", "id", "device", "params", "model"}, name="method")
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
            params=_optional_mapping(method, "params"),
            model=model_cfg,
        )

        evaluation = _as_mapping(data.get("evaluation", {}), name="evaluation")
        _check_unknown(
            evaluation, {"report_splits", "metrics", "split_for_model_selection"}, name="evaluation"
        )
        report_splits = [str(s) for s in _optional_list(evaluation, "report_splits")]
        metrics = [str(m) for m in _optional_list(evaluation, "metrics")]
        if not report_splits:
            raise BenchConfigError("evaluation.report_splits must be provided")
        if not metrics:
            raise BenchConfigError("evaluation.metrics must be provided")
        evaluation_cfg = EvaluationConfig(
            report_splits=report_splits,
            metrics=metrics,
            split_for_model_selection=_optional_str(evaluation, "split_for_model_selection"),
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
