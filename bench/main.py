from __future__ import annotations

import argparse
import logging
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_augmentation.utils import is_torch_tensor
from modssc.device import resolve_device_name
from modssc.evaluation import list_metrics
from modssc.hpo import deep_merge
from modssc.inductive.registry import get_method_class as get_inductive_method_class
from modssc.inductive.registry import get_method_info as get_inductive_method_info
from modssc.logging import configure_logging, resolve_log_level
from modssc.preprocess import step_info
from modssc.sampling.result import SamplingResult
from modssc.transductive.registry import get_method_info as get_transductive_method_info

from .context import RunContext
from .limits import apply_limits
from .orchestrators import augmentation as aug_orch
from .orchestrators import dataset as ds_orch
from .orchestrators import evaluation as eval_orch
from .orchestrators import graph as graph_orch
from .orchestrators import hpo as hpo_orch
from .orchestrators import method_inductive as inductive_orch
from .orchestrators import method_transductive as transductive_orch
from .orchestrators import preprocess as prep_orch
from .orchestrators import reporting as report_orch
from .orchestrators import sampling as sampling_orch
from .orchestrators import views as views_orch
from .schema import BenchConfigError, ExperimentConfig
from .utils.import_tools import check_extra_installed
from .utils.io import load_yaml

_ALLOWED_METRICS = set(list_metrics())
_ALLOWED_SPLITS = {"train", "val", "test"}
_LOGGER = logging.getLogger(__name__)


def _dataset_has_graph(dataset: Any) -> bool:
    return (
        getattr(dataset.train, "edges", None) is not None
        or getattr(dataset.train, "masks", None) is not None
    )


def _graph_sampling_to_inductive(sampling: SamplingResult) -> SamplingResult:
    masks = sampling.masks
    indices = {
        "train": np.where(np.asarray(masks["train"], dtype=bool))[0].astype(np.int64),
        "val": np.where(np.asarray(masks["val"], dtype=bool))[0].astype(np.int64),
        "test": np.where(np.asarray(masks["test"], dtype=bool))[0].astype(np.int64),
        "train_labeled": np.where(np.asarray(masks["labeled"], dtype=bool))[0].astype(np.int64),
        "train_unlabeled": np.where(np.asarray(masks["unlabeled"], dtype=bool))[0].astype(np.int64),
    }
    refs = {k: "train" for k in indices}
    return SamplingResult(
        schema_version=sampling.schema_version,
        created_at=sampling.created_at,
        dataset_fingerprint=sampling.dataset_fingerprint,
        split_fingerprint=sampling.split_fingerprint,
        plan=sampling.plan,
        indices=indices,
        refs=refs,
        masks={},
        stats=dict(sampling.stats),
    )


def _check_extra(extra: str) -> None:
    missing = check_extra_installed(extra)
    if missing:
        raise BenchConfigError(
            f"Missing optional dependency for extra '{extra}': {sorted(set(missing))}"
        )


def _method_requires_torch(method_id: str) -> bool:
    cls = get_inductive_method_class(method_id)
    try:
        inst = cls()
    except Exception:
        return False
    spec = getattr(inst, "spec", None)
    return bool(hasattr(spec, "model_bundle"))


def _resolve_method_device(requested: str, *, supports_gpu: bool) -> str:
    if requested != "auto":
        return requested
    if not supports_gpu:
        return "cpu"
    return resolve_device_name(requested)


def _record_resolved_device(
    raw: dict[str, Any],
    *,
    cfg: ExperimentConfig,
    resolved_device: str,
) -> None:
    method = raw.get("method")
    if not isinstance(method, dict):
        method = {}
        raw["method"] = method
    device = method.get("device")
    if not isinstance(device, dict):
        device = {}
        method["device"] = device
    device.setdefault("device", cfg.method.device.device)
    device.setdefault("dtype", cfg.method.device.dtype)
    device["resolved_device"] = resolved_device


def _preflight(
    *,
    cfg: ExperimentConfig,
    dataset_info: dict[str, Any],
    preprocess_steps: list[str],
    dataset_has_graph: bool,
) -> Any:
    extra = dataset_info.get("required_extra")
    if extra:
        _check_extra(str(extra))

    for step_id in preprocess_steps:
        info = step_info(step_id)
        extra = info.get("required_extra")
        if extra:
            _check_extra(str(extra))

    if cfg.method.kind == "inductive":
        method_info = get_inductive_method_info(cfg.method.method_id)
    else:
        method_info = get_transductive_method_info(cfg.method.method_id)

    if method_info.required_extra:
        _check_extra(str(method_info.required_extra))

    if (
        cfg.method.kind == "inductive"
        and cfg.method.method_id == "co_training"
        and cfg.views is None
    ):
        raise BenchConfigError("co_training requires views to be configured")

    if cfg.method.kind == "transductive":
        graph_enabled = bool(cfg.graph and cfg.graph.enabled)
        if not dataset_has_graph and not graph_enabled:
            raise BenchConfigError(
                "Transductive methods require a graph; set graph.enabled=true or use a graph dataset"
            )

    if cfg.augmentation is not None and cfg.augmentation.enabled:
        if cfg.augmentation.mode != "fixed":
            raise BenchConfigError("Only augmentation.mode='fixed' is supported")
        if not cfg.augmentation.weak or not cfg.augmentation.strong:
            raise BenchConfigError("augmentation.weak and augmentation.strong must be provided")

    for metric in cfg.evaluation.metrics:
        if metric not in _ALLOWED_METRICS:
            raise BenchConfigError(f"Unknown metric: {metric}")

    for split in cfg.evaluation.report_splits:
        if split not in _ALLOWED_SPLITS:
            raise BenchConfigError(f"Unknown split: {split}")
    return method_info


def _preprocess_step_ids(plan_dict: dict[str, Any]) -> list[str]:
    steps_raw = plan_dict.get("steps", [])
    if not isinstance(steps_raw, list):
        raise BenchConfigError("preprocess.plan.steps must be a list")
    ids: list[str] = []
    for item in steps_raw:
        if not isinstance(item, dict):
            raise BenchConfigError("Each preprocess step must be a mapping")
        step_id = str(item.get("id") or item.get("step_id") or "")
        if not step_id:
            raise BenchConfigError("Each preprocess step must define 'id'")
        ids.append(step_id)
    return ids


def _views_preprocess_step_ids(views_plan: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    views = views_plan.get("views", [])
    if not isinstance(views, list):
        return steps
    for view in views:
        if not isinstance(view, dict):
            continue
        pre = view.get("preprocess")
        if isinstance(pre, dict):
            steps.extend(_preprocess_step_ids(pre))
    return steps


def _requires_fit_indices(step_ids: list[str]) -> bool:
    for step_id in step_ids:
        info = step_info(step_id)
        if info.get("kind") == "fittable":
            return True
    return False


def _use_test_split(sampling: Any) -> bool:
    return sampling.refs.get("test", "train") == "test"


def _mask_map_from_sampling(
    sampling: Any, n_train: int, n_test: int | None
) -> dict[str, np.ndarray]:
    if sampling.is_graph():
        m = sampling.masks
        return {
            "train_mask": np.asarray(m["train"], dtype=bool),
            "val_mask": np.asarray(m["val"], dtype=bool),
            "test_mask": np.asarray(m["test"], dtype=bool),
            "unlabeled_mask": np.asarray(m["unlabeled"], dtype=bool),
            "labeled_mask": np.asarray(m["labeled"], dtype=bool),
        }
    return transductive_orch._build_masks_from_indices(
        n_train=n_train,
        n_test=n_test,
        indices=sampling.indices,
        refs=sampling.refs,
    )


def _expected_labeled_count(stats: Mapping[str, Any] | None) -> int | None:
    if not stats:
        return None
    train_labeled = stats.get("train_labeled")
    if isinstance(train_labeled, Mapping):
        n = train_labeled.get("n")
        if isinstance(n, (int, np.integer)):
            return int(n)
    labeled = stats.get("labeled")
    if isinstance(labeled, (int, np.integer)):
        return int(labeled)
    labeled_dist = stats.get("labeled_class_dist")
    if isinstance(labeled_dist, Mapping):
        n = labeled_dist.get("n")
        if isinstance(n, (int, np.integer)):
            return int(n)
    return None


def _select_rows(X: Any, idx: np.ndarray) -> Any:
    if isinstance(X, dict):
        out = {}
        # Special handling for Graph in main.py too!
        if "edge_index" in X:
            try:
                import importlib

                torch = importlib.import_module("torch")
                from torch_geometric.utils import subgraph

                ei = X["edge_index"]
                idx_t = torch.as_tensor(idx, device=ei.device, dtype=torch.long)
                sub_ei, _ = subgraph(idx_t, ei, relabel_nodes=True)
                out["edge_index"] = sub_ei
            except Exception:
                pass

        for k, v in X.items():
            if k == "edge_index":
                if "edge_index" not in out:
                    out[k] = v
                continue

            if is_torch_tensor(v):
                import importlib

                # ...
                torch = importlib.import_module("torch")
                # Heuristic: slice if dim 0 covers the indices and is not tiny (like edge_index's 2)
                if v.ndim > 0 and v.shape[0] > idx.max():
                    out[k] = v[torch.as_tensor(idx, device=v.device, dtype=torch.long)]
                else:
                    out[k] = v
            elif isinstance(v, (np.ndarray, list)):
                v_arr = np.array(v)
                if v_arr.ndim > 0 and v_arr.shape[0] > idx.max():
                    out[k] = v_arr[idx]
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    if is_torch_tensor(X):
        import importlib

        torch = importlib.import_module("torch")
        return X[torch.as_tensor(idx, device=X.device, dtype=torch.long)]
    return X[idx]


def _write_error_traceback(ctx: RunContext, tb: str) -> None:
    try:
        (ctx.run_dir / "error.txt").write_text(tb, encoding="utf-8")
    except Exception:
        _LOGGER.exception("Failed to write error.txt")


def _resolve_log_level_for_run(config_path: Path, cli_log_level: str | None) -> str:
    if cli_log_level is not None and str(cli_log_level).strip():
        return resolve_log_level(cli_log_level)
    raw = load_yaml(config_path)
    cfg = ExperimentConfig.from_dict(raw)
    return resolve_log_level(cfg.run.log_level)


def run_experiment(config_path: Path) -> int:
    raw = load_yaml(config_path)
    cfg = ExperimentConfig.from_dict(raw)
    raw, limit_changes, resolved_limits = apply_limits(raw, limits=cfg.limits)
    if limit_changes:
        profile = resolved_limits.profile if resolved_limits is not None else None
        profile_label = profile or "custom"
        _LOGGER.info(
            "Applied memory limits: profile=%s changes=%s", profile_label, len(limit_changes)
        )
        _LOGGER.debug("Limit adjustments: %s", limit_changes)
        cfg = ExperimentConfig.from_dict(raw)

    ctx = RunContext.from_run_config(
        name=cfg.run.name,
        seed=cfg.run.seed,
        output_dir=cfg.run.output_dir,
        config_path=config_path,
        fail_fast=cfg.run.fail_fast,
    )
    ctx.ensure_dirs()
    ctx.write_config_copy(raw)

    _LOGGER.info("Run started: %s", cfg.run.name)
    _LOGGER.info("Config: %s", config_path)
    _LOGGER.info("Run dir: %s", ctx.run_dir)

    artifacts: dict[str, Any] = {}
    metrics: dict[str, Any] | None = None
    hpo_summary: dict[str, Any] | None = None
    status = "success"
    error: str | None = None

    try:
        _LOGGER.info("Loading dataset: %s", cfg.dataset.id)
        dataset, dataset_info = ds_orch.load(cfg.dataset)
        dataset_has_graph = _dataset_has_graph(dataset)

        preprocess_steps = _preprocess_step_ids(cfg.preprocess.plan)
        view_preprocess_steps = (
            _views_preprocess_step_ids(cfg.views.plan) if cfg.views is not None else []
        )
        all_preprocess_steps = preprocess_steps + view_preprocess_steps

        _LOGGER.info("Preflight checks")
        if _requires_fit_indices(preprocess_steps) and cfg.preprocess.fit_on is None:
            raise BenchConfigError(
                "preprocess.fit_on must be set when the plan includes fittable steps"
            )
        if _requires_fit_indices(view_preprocess_steps) and cfg.preprocess.fit_on is None:
            raise BenchConfigError(
                "preprocess.fit_on must be set when views include fittable preprocess steps"
            )

        method_info = _preflight(
            cfg=cfg,
            dataset_info=dataset_info,
            preprocess_steps=all_preprocess_steps,
            dataset_has_graph=dataset_has_graph,
        )
        resolved_device = _resolve_method_device(
            cfg.method.device.device, supports_gpu=method_info.supports_gpu
        )
        _record_resolved_device(raw, cfg=cfg, resolved_device=resolved_device)
        ctx.write_config_copy(raw)
        artifacts["method"] = {
            "id": cfg.method.method_id,
            "kind": cfg.method.kind,
            "device": {
                "requested": cfg.method.device.device,
                "resolved": resolved_device,
                "dtype": cfg.method.device.dtype,
            },
        }
        _LOGGER.info(
            "Resolved device: requested=%s resolved=%s",
            cfg.method.device.device,
            resolved_device,
        )
        if (
            cfg.method.kind == "inductive"
            and _method_requires_torch(cfg.method.method_id)
            and "core.to_torch" not in preprocess_steps
        ):
            # Optimisation H100: on autorise l'absence de to_torch si on gère la conversion plus tard
            # (validation désactivée pour permettre le pipeline uint8)
            pass
            # raise BenchConfigError(
            #    "Torch inductive methods require preprocess step 'core.to_torch'"
            # )

        artifacts["dataset"] = {
            "id": cfg.dataset.id,
            "info": dataset_info,
            "fingerprint": dataset.meta.get("dataset_fingerprint"),
        }

        _LOGGER.info("Sampling splits")
        sampling_seed = ctx.seed_for("sampling", cfg.sampling.seed)
        sampling = sampling_orch.run(
            dataset,
            plan_dict=cfg.sampling.plan,
            seed=sampling_seed,
            dataset_id=cfg.dataset.id,
        )
        use_test = _use_test_split(sampling)
        artifacts["sampling"] = {
            "seed": sampling_seed,
            "plan": sampling.plan,
            "split_fingerprint": sampling.split_fingerprint,
            "stats": sampling.stats,
        }

        if cfg.method.kind == "inductive" and sampling.is_graph():
            _LOGGER.warning("Inductive method on graph dataset; converting graph masks to indices.")
            sampling = _graph_sampling_to_inductive(sampling)

        _LOGGER.info("Preprocess")
        fit_indices = prep_orch.resolve_fit_indices(
            dataset=dataset, sampling=sampling, fit_on=cfg.preprocess.fit_on
        )
        preprocess_seed = ctx.seed_for("preprocess", cfg.preprocess.seed)
        pre = prep_orch.run(
            dataset,
            plan_dict=cfg.preprocess.plan,
            seed=preprocess_seed,
            fit_indices=fit_indices,
            cache=cfg.preprocess.cache,
        )
        artifacts["preprocess"] = {
            "seed": preprocess_seed,
            "preprocess_fingerprint": pre.preprocess_fingerprint,
            "plan_fingerprint": pre.plan.fingerprint,
            "fit_fingerprint": pre.dataset.meta.get("preprocess_fit_fingerprint"),
            "cache_dir": pre.cache_dir,
        }

        views = None
        if cfg.views is not None:
            _LOGGER.info("Views")
            views_seed = ctx.seed_for("views", cfg.views.seed)
            views = views_orch.run(
                pre.dataset,
                plan_dict=cfg.views.plan,
                seed=views_seed,
                fit_indices=fit_indices,
                cache=cfg.preprocess.cache,
            )
            artifacts["views"] = {
                "seed": views_seed,
                "n_views": len(views.views),
                "meta": views.meta,
            }

        graph = None
        if cfg.graph is not None and cfg.graph.enabled:
            _LOGGER.info("Graph")
            if not cfg.graph.spec:
                raise BenchConfigError("graph.spec must be provided when graph.enabled=true")
            graph_seed = ctx.seed_for("graph", cfg.graph.seed)
            ds_fp = pre.dataset.meta.get("dataset_fingerprint")
            graph = graph_orch.build(
                pre,
                spec_dict=cfg.graph.spec,
                seed=graph_seed,
                dataset_fingerprint=ds_fp,
                cache=cfg.graph.cache,
                cache_dir=cfg.graph.cache_dir,
                include_test=use_test,
            )
            artifacts["graph"] = {
                "seed": graph_seed,
                "fingerprint": graph.meta.get("fingerprint"),
                "spec": cfg.graph.spec,
            }
        elif dataset_has_graph:
            _LOGGER.info("Graph (dataset-provided)")
            if pre.dataset.train.edges is None:
                raise BenchConfigError("Graph dataset is missing train.edges")
            # y can be a numpy array or a torch tensor on GPU; .shape works for both
            n_nodes = int(pre.dataset.train.y.shape[0])
            graph = transductive_orch.graph_from_dataset(pre.dataset, n_nodes)
            artifacts["graph"] = {
                "seed": None,
                "fingerprint": graph.meta.get("fingerprint"),
                "spec": None,
                "source": "dataset",
            }

        X_u_w = None
        X_u_s = None
        X_u_s_1 = None
        if (
            cfg.augmentation is not None
            and cfg.augmentation.enabled
            and cfg.method.kind == "inductive"
        ):
            _LOGGER.info("Augmentation")
            idx_u = np.asarray(sampling.indices["train_unlabeled"], dtype=np.int64)
            X_u = _select_rows(pre.dataset.train.X, idx_u)

            # For graph dictionaries, extract 'x' for tabular augmentation
            X_u_aug_input = X_u
            if isinstance(X_u, dict) and "x" in X_u:
                X_u_aug_input = X_u["x"]

            aug_seed = ctx.seed_for("augmentation", cfg.augmentation.seed)
            strong_views = 2 if cfg.method.method_id == "comatch" else 1
            X_u_w, X_u_s, X_u_s_1 = aug_orch.run(
                X_u_aug_input,
                weak_plan=cfg.augmentation.weak,
                strong_plan=cfg.augmentation.strong,
                seed=aug_seed,
                mode=cfg.augmentation.mode,
                modality=cfg.augmentation.modality,
                sample_ids=idx_u,
                strong_views=strong_views,
            )

            # If input was a graph dict, reconstruct dicts for augmented data
            if isinstance(X_u, dict) and "x" in X_u:

                def _wrapg(aug_x, ref):
                    if aug_x is None:
                        return None
                    d = ref.copy()
                    d["x"] = aug_x
                    return d

                X_u_w = _wrapg(X_u_w, X_u)
                X_u_s = _wrapg(X_u_s, X_u)
                X_u_s_1 = _wrapg(X_u_s_1, X_u)

            artifacts["augmentation"] = {"seed": aug_seed, "mode": cfg.augmentation.mode}

        masks = None
        expected_labeled_count = None
        if cfg.method.kind == "transductive":
            if graph is None:
                raise BenchConfigError("Transductive methods require a graph")
            n_train = int(pre.dataset.train.y.shape[0])
            n_test = (
                int(pre.dataset.test.y.shape[0])
                if use_test and pre.dataset.test is not None
                else None
            )
            masks = _mask_map_from_sampling(sampling, n_train=n_train, n_test=n_test)
            expected_labeled_count = _expected_labeled_count(sampling.stats)

        prepared_artifacts = {
            "pre": pre,
            "sampling": sampling,
            "views": views,
            "graph": graph,
            "X_u_w": X_u_w,
            "X_u_s": X_u_s,
            "X_u_s_1": X_u_s_1,
            "use_test": use_test,
            "masks": masks,
            "expected_labeled_count": expected_labeled_count,
        }

        if cfg.search is not None and cfg.search.enabled:
            _LOGGER.info("HPO: %s search", cfg.search.kind)
            best_patch, hpo_summary = hpo_orch.run_hpo(
                ctx=ctx,
                base_cfg=cfg,
                base_cfg_dict=raw,
                prepared_artifacts=prepared_artifacts,
            )
            patched_raw = deep_merge(raw, best_patch)
            patched_raw, hpo_limit_changes, _ = apply_limits(patched_raw, limits=cfg.limits)
            if hpo_limit_changes:
                _LOGGER.info("Applied memory limits after HPO: changes=%s", len(hpo_limit_changes))
                _LOGGER.debug("Limit adjustments: %s", hpo_limit_changes)
            cfg = ExperimentConfig.from_dict(patched_raw)

        if cfg.method.kind == "inductive":
            _LOGGER.info("Method: %s", cfg.method.method_id)
            method_seed = ctx.seed_for("method", None)
            method = inductive_orch.run(
                pre,
                sampling,
                views=views,
                X_u_w=X_u_w,
                X_u_s=X_u_s,
                X_u_s_1=X_u_s_1,
                cfg=cfg.method,
                seed=method_seed,
            )
            metrics = eval_orch.evaluate_inductive(
                method=method,
                pre=pre,
                sampling=sampling,
                report_splits=cfg.evaluation.report_splits,
                metrics=cfg.evaluation.metrics,
                method_id=cfg.method.method_id,
                views=views,
            )
        else:
            _LOGGER.info("Method: %s", cfg.method.method_id)
            method_seed = ctx.seed_for("method", None)
            method, data = transductive_orch.run(
                dataset=pre.dataset,
                graph=graph,
                masks=masks,
                cfg=cfg.method,
                seed=method_seed,
                use_test_split=use_test,
                expected_labeled_count=expected_labeled_count,
            )
            metrics = eval_orch.evaluate_transductive(
                method=method,
                data=data,
                report_splits=cfg.evaluation.report_splits,
                metrics=cfg.evaluation.metrics,
                masks=masks,
            )

    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        _write_error_traceback(ctx, traceback.format_exc())
        _LOGGER.exception("Run failed")
        if ctx.fail_fast:
            report_orch.write_run_summary(
                ctx=ctx,
                cfg=cfg,
                artifacts=artifacts,
                metrics=metrics,
                hpo=hpo_summary,
                status=status,
                error=error,
            )
            raise

    if metrics is not None:
        _LOGGER.info("Metrics: %s", metrics)
    _LOGGER.info("Run finished: %s", status)
    report_orch.write_run_summary(
        ctx=ctx,
        cfg=cfg,
        artifacts=artifacts,
        metrics=metrics,
        hpo=hpo_summary,
        status=status,
        error=error,
    )
    return 0 if status == "success" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ModSSC benchmark orchestration")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level: none, basic, detailed (aliases: quiet, full).",
    )
    args = parser.parse_args()
    try:
        resolved = _resolve_log_level_for_run(Path(args.config), args.log_level)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    configure_logging(resolved)
    return run_experiment(Path(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
