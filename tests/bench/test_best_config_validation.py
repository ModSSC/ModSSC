from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

from bench.schema import ExperimentConfig
from modssc.runtime.pipeline import MethodResolutionRequest, resolve_method

BEST_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "bench" / "configs" / "best"
YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
REPLICATION_INDUCTIVE_METHOD_IDS = frozenset(
    {
        "democratic_co_learning",
        "fixmatch",
        "flexmatch",
        "free_match",
        "pseudo_label",
        "softmatch",
        "tri_training",
    }
)
FROZEN_AET_METHOD_IDS = frozenset({"grand", "laplace_learning", "poisson_learning"})
FROZEN_AET_FEATURES_SHA256 = "02b21620c0c1448f00f87679efc0bd02b9cb4072b53a01277b1f6b03d2bbaeba"
FROZEN_AET_LABELS_SHA256 = "0ff873d4ee7578949152fa4305c30d62ace90df32e095fcc60855dc27ba591b5"
REGIME_LABELS_PER_CLASS = {
    "R1": 1,
    "R2": 3,
    "R3": 5,
    "R4": 10,
    "R5": 20,
    "R6": 50,
}


def _check(
    errors: list[str],
    path: Path,
    condition: bool,
    message: str,
) -> None:
    if not condition:
        errors.append(f"{path.relative_to(BEST_CONFIG_ROOT)}: {message}")


def _load_best_configs() -> list[tuple[Path, dict[str, Any]]]:
    configs: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(BEST_CONFIG_ROOT.rglob("*.yaml")):
        if path.name == "regime_manifest.yaml":
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        configs.append((path, raw))
    return configs


@cache
def _replication_method_requires_torch(method_id: str) -> bool:
    return resolve_method(
        MethodResolutionRequest(regime="inductive", method_id=method_id)
    ).requires_torch


def _config_uses_torch_classifier(raw: dict[str, Any]) -> bool:
    method = raw.get("method", {})
    model = method.get("model") or {}
    params = method.get("params") or {}
    backends = (
        model.get("classifier_backend"),
        params.get("classifier_backend"),
    )
    return any(isinstance(backend, str) and backend.lower() == "torch" for backend in backends)


def test_replication_torch_methods_declare_to_torch_in_best_configs() -> None:
    errors: list[str] = []

    for method_id in sorted(REPLICATION_INDUCTIVE_METHOD_IDS):
        runtime_requires_torch = _replication_method_requires_torch(method_id)
        paths = BEST_CONFIG_ROOT.glob(f"*/inductive/{method_id}/**/*.yaml")
        for path in sorted(paths):
            raw = yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_SAFE_LOADER)
            if not (runtime_requires_torch or _config_uses_torch_classifier(raw)):
                continue

            steps = raw.get("preprocess", {}).get("plan", {}).get("steps", [])
            step_ids = {step.get("id") for step in steps if isinstance(step, dict)}
            _check(
                errors,
                path,
                "core.to_torch" in step_ids,
                f"inductive method {method_id!r} requires preprocess step 'core.to_torch'",
            )

    assert not errors, "Missing Torch preprocessing in replication best configs:\n" + "\n".join(
        errors
    )


def test_replication_aet_configs_pin_official_artifacts() -> None:
    errors: list[str] = []

    for method_id in sorted(FROZEN_AET_METHOD_IDS):
        paths = BEST_CONFIG_ROOT.glob(f"*/transductive/{method_id}/vision/cifar10.yaml")
        for path in sorted(paths):
            raw = yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_SAFE_LOADER)
            steps = raw.get("preprocess", {}).get("plan", {}).get("steps", [])
            aet_steps = [
                step for step in steps if isinstance(step, dict) and step.get("id") == "vision.aet"
            ]
            _check(errors, path, len(aet_steps) == 1, "expected exactly one vision.aet step")
            if len(aet_steps) != 1:
                continue
            params = aet_steps[0].get("params") or {}
            _check(
                errors,
                path,
                params.get("source") == "precomputed",
                "replication vision.aet source must be precomputed",
            )
            _check(
                errors,
                path,
                params.get("expected_features_sha256") == FROZEN_AET_FEATURES_SHA256,
                "replication vision.aet must pin the official features SHA-256",
            )
            _check(
                errors,
                path,
                params.get("expected_labels_sha256") == FROZEN_AET_LABELS_SHA256,
                "replication vision.aet must pin the official labels SHA-256",
            )

    assert not errors, "Unpinned replication AET artifacts:\n" + "\n".join(errors)


def test_tsvm_best_configs_do_not_build_an_unused_graph() -> None:
    errors: list[str] = []
    paths = BEST_CONFIG_ROOT.glob("*/transductive/tsvm/**/*.yaml")

    for path in sorted(paths):
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_SAFE_LOADER)
        _check(
            errors,
            path,
            "graph" not in raw,
            "TSVM does not consume graph edges; an explicit graph block is an unused artifact",
        )

    assert not errors, "TSVM best configs still build graphs:\n" + "\n".join(errors)


def test_all_best_configs_follow_benchmark_contract() -> None:
    configs = _load_best_configs()
    errors: list[str] = []

    _check(
        errors, BEST_CONFIG_ROOT, len(configs) == 5285, f"expected 5285 configs, got {len(configs)}"
    )

    for path, raw in configs:
        try:
            ExperimentConfig.from_dict(raw)
        except Exception as exc:  # pragma: no cover - failure detail for malformed YAML
            errors.append(f"{path.relative_to(BEST_CONFIG_ROOT)}: schema error: {exc}")
            continue

        relative = path.relative_to(BEST_CONFIG_ROOT)
        regime = relative.parts[0]
        run = raw["run"]
        dataset = raw["dataset"]
        sampling = raw["sampling"]["plan"]
        split = sampling["split"]
        labeling = sampling["labeling"]
        evaluation = raw["evaluation"]
        seeds = run.get("seeds", [])

        _check(errors, path, run.get("benchmark_mode") is True, "benchmark_mode must be true")
        _check(errors, path, dataset.get("download") is False, "dataset.download must be false")
        _check(errors, path, len(seeds) == 5, f"expected five seeds, got {seeds!r}")
        _check(
            errors,
            path,
            len(seeds) == 5
            and all(right - left == 1 for left, right in zip(seeds, seeds[1:], strict=False)),
            f"seeds must be consecutive, got {seeds!r}",
        )

        _check(errors, path, split.get("kind") == "holdout", "split.kind must be holdout")
        _check(errors, path, split.get("test_fraction") == 0.2, "test_fraction must be 0.2")
        _check(errors, path, split.get("val_fraction") == 0.1, "val_fraction must be 0.1")
        _check(errors, path, split.get("stratify") is True, "split must be stratified")
        _check(errors, path, split.get("shuffle") is True, "split must be shuffled")

        expected_budget = REGIME_LABELS_PER_CLASS[regime]
        _check(errors, path, labeling.get("mode") == "per_class", "labeling.mode must be per_class")
        _check(errors, path, labeling.get("per_class") is True, "labeling.per_class must be true")
        _check(
            errors,
            path,
            labeling.get("value") == expected_budget,
            f"expected {expected_budget} labels/class for {regime}, got {labeling.get('value')!r}",
        )

        _check(
            errors,
            path,
            evaluation.get("split_for_model_selection") == "val",
            "model selection must use val",
        )
        _check(
            errors,
            path,
            evaluation.get("report_splits") == ["val", "test"],
            "report_splits must be [val, test]",
        )
        _check(
            errors,
            path,
            evaluation.get("metrics") == ["accuracy", "macro_f1"],
            "metrics must be [accuracy, macro_f1]",
        )

    assert not errors, "Best-config contract violations:\n" + "\n".join(errors)
