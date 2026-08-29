from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from bench.orchestrators import hpo
from bench.schema import BenchConfigError, ExperimentConfig, LimitsConfig
from modssc.hpo import HpoError


def _minimal_config() -> dict[str, object]:
    return {
        "run": {"name": "hpo", "seed": 1, "output_dir": "runs"},
        "dataset": {"id": "toy"},
        "sampling": {"plan": {"split": {"kind": "holdout"}}},
        "preprocess": {"plan": {"steps": []}},
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "params": {"alpha": 0.5},
        },
        "evaluation": {"report_splits": ["val"], "metrics": ["accuracy"]},
        "search": {
            "enabled": True,
            "kind": "grid",
            "objective": {
                "split": "val",
                "metric": "accuracy",
                "direction": "maximize",
                "aggregate": "mean",
            },
            "space": {"method": {"params": {"alpha": [0.0, 1.0]}}},
        },
    }


def test_grid_search_rejects_continuous_distribution_during_schema_validation() -> None:
    raw = _minimal_config()
    raw["search"]["space"]["method"]["params"]["alpha"] = {  # type: ignore[index]
        "dist": "uniform",
        "low": 0.0,
        "high": 1.0,
    }

    with pytest.raises(BenchConfigError, match="grid search requires list or choice"):
        ExperimentConfig.from_dict(raw)


def test_native_hpo_space_error_is_not_misreported_as_not_evaluable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = ExperimentConfig.from_dict(_minimal_config())
    monkeypatch.setattr(
        hpo,
        "run_search",
        lambda **_kwargs: (_ for _ in ()).throw(HpoError("invalid search space")),
    )
    ctx = SimpleNamespace(run_dir=tmp_path, seed_for=lambda _label: 1)

    with pytest.raises(BenchConfigError) as caught:
        hpo.run_hpo(
            ctx=ctx,
            base_cfg=cfg,
            base_cfg_dict=_minimal_config(),
            prepared_artifacts={},
        )

    assert caught.value.code == "E_BENCH_HPO_SPACE"


def test_inductive_hpo_preserves_during_fit_evaluation_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return object(), {}

    monkeypatch.setattr(hpo.inductive_orch, "run", fake_run)
    monkeypatch.setattr(
        hpo.eval_orch,
        "evaluate_inductive",
        lambda **_kwargs: {"validation": {"accuracy": 0.75}},
    )
    cfg = SimpleNamespace(
        method=SimpleNamespace(kind="inductive"),
        evaluation=SimpleNamespace(during_fit_splits=["test"]),
    )
    execution_input = SimpleNamespace(
        preprocess=object(),
        sampling=object(),
        views=None,
    )
    prepared = {
        "routed_input": SimpleNamespace(execution_input=execution_input),
        "use_test": False,
        "strict": False,
        "requires_torch": False,
    }

    result = hpo._objective_value(
        cfg=cfg,
        prepared_artifacts=prepared,
        seed=7,
        split="validation",
        metric="accuracy",
    )

    assert result == pytest.approx(0.75)
    assert captured["during_fit_splits"] == ["test"]


def test_hpo_scores_the_same_limit_clamped_params_used_by_final_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    search = SimpleNamespace(
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=1,
        objective=SimpleNamespace(
            split="val",
            metric="accuracy",
            direction="maximize",
            aggregate="mean",
        ),
        space={"method": {"params": {"batch_size": [512, 256]}}},
    )
    base_cfg = SimpleNamespace(
        search=search,
        limits=LimitsConfig(max_method_batch_size=128),
        run=SimpleNamespace(benchmark_mode=False),
    )
    base_raw = {"method": {"params": {"batch_size": 64}}}

    def parse(raw):
        return SimpleNamespace(
            method=SimpleNamespace(params=dict(raw["method"]["params"])),
        )

    monkeypatch.setattr(hpo.ExperimentConfig, "from_dict", parse)
    monkeypatch.setattr(
        hpo,
        "_objective_value",
        lambda *, cfg, **_kwargs: float(cfg.method.params["batch_size"]),
    )
    ctx = SimpleNamespace(
        run_dir=tmp_path,
        seed_for=lambda label: len(label),
    )

    best_patch, summary = hpo.run_hpo(
        ctx=ctx,
        base_cfg=base_cfg,
        base_cfg_dict=base_raw,
        prepared_artifacts={},
    )

    assert best_patch == {"method": {"params": {"batch_size": 512}}}
    assert summary["best_score"] == 128.0
    assert summary["best_effective_patch"]["method"]["params"]["batch_size"] == 128
    assert summary["best_limit_changes"] == ["method.params.batch_size: 512 -> 128"]


def test_hpo_re_resolves_torch_requirement_for_each_effective_trial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    raw = _minimal_config()
    cfg = ExperimentConfig.from_dict(raw)
    resolved: list[float] = []
    evaluated: list[tuple[float, bool]] = []

    def resolve_runtime(trial_cfg: ExperimentConfig):
        alpha = float(trial_cfg.method.params["alpha"])
        resolved.append(alpha)
        return SimpleNamespace(
            requires_torch=alpha == 1.0,
            to_dict=lambda: {
                "required_extras": ["inductive-torch"] if alpha == 1.0 else [],
                "resolved_backend": "torch" if alpha == 1.0 else "numpy",
                "requires_torch": alpha == 1.0,
            },
        )

    def objective(*, cfg: ExperimentConfig, requires_torch: bool, **_kwargs) -> float:
        alpha = float(cfg.method.params["alpha"])
        evaluated.append((alpha, requires_torch))
        return alpha

    monkeypatch.setattr(hpo, "_objective_value", objective)
    ctx = SimpleNamespace(run_dir=tmp_path, seed_for=lambda _label: 1)

    best_patch, _summary = hpo.run_hpo(
        ctx=ctx,
        base_cfg=cfg,
        base_cfg_dict=raw,
        prepared_artifacts={"requires_torch": False},
        method_runtime_resolver=resolve_runtime,
    )

    assert resolved == [0.0, 1.0]
    assert evaluated == [(0.0, False), (1.0, True)]
    assert best_patch == {"method": {"params": {"alpha": 1.0}}}
    trial_payloads = [
        json.loads(line)
        for line in (tmp_path / "hpo" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [payload["method_runtime"] for payload in trial_payloads] == [
        {
            "required_extras": [],
            "resolved_backend": "numpy",
            "requires_torch": False,
        },
        {
            "required_extras": ["inductive-torch"],
            "resolved_backend": "torch",
            "requires_torch": True,
        },
    ]


def test_hpo_serializes_non_finite_trial_as_not_evaluable_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    search = SimpleNamespace(
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=1,
        objective=SimpleNamespace(
            split="validation",
            metric="accuracy",
            direction="maximize",
            aggregate="mean",
        ),
        space={"method": {"params": {"alpha": [0.0, 1.0]}}},
    )
    base_cfg = SimpleNamespace(
        search=search,
        limits=LimitsConfig(),
        run=SimpleNamespace(benchmark_mode=False),
    )

    def parse(raw):
        return SimpleNamespace(method=SimpleNamespace(params=dict(raw["method"]["params"])))

    monkeypatch.setattr(hpo.ExperimentConfig, "from_dict", parse)
    monkeypatch.setattr(
        hpo,
        "_objective_value",
        lambda *, cfg, **_kwargs: math.nan if cfg.method.params["alpha"] == 0.0 else 0.75,
    )
    ctx = SimpleNamespace(run_dir=tmp_path, seed_for=lambda _label: 1)

    _best_patch, summary = hpo.run_hpo(
        ctx=ctx,
        base_cfg=base_cfg,
        base_cfg_dict={"method": {"params": {"alpha": 0.5}}},
        prepared_artifacts={},
    )

    lines = (tmp_path / summary["trials_path"]).read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0], parse_constant=lambda value: pytest.fail(value))
    assert first["status"] == "not_evaluable"
    assert first["reason"] == "non_finite_objective"
    assert first["objective"]["value"] is None
    assert first["objective"]["values"] == [None]


def test_hpo_returns_traceable_not_evaluable_summary_when_all_trials_are_non_finite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    search = SimpleNamespace(
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=1,
        objective=SimpleNamespace(
            split="validation",
            metric="accuracy",
            direction="maximize",
            aggregate="mean",
        ),
        space={"method": {"params": {"alpha": [0.0, 1.0]}}},
    )
    base_cfg = SimpleNamespace(
        search=search,
        limits=LimitsConfig(),
        run=SimpleNamespace(benchmark_mode=False),
    )
    monkeypatch.setattr(
        hpo.ExperimentConfig,
        "from_dict",
        lambda raw: SimpleNamespace(method=SimpleNamespace(params=dict(raw["method"]["params"]))),
    )
    monkeypatch.setattr(hpo, "_objective_value", lambda **_kwargs: math.nan)
    ctx = SimpleNamespace(run_dir=tmp_path, seed_for=lambda _label: 1)

    best_patch, summary = hpo.run_hpo(
        ctx=ctx,
        base_cfg=base_cfg,
        base_cfg_dict={"method": {"params": {"alpha": 0.5}}},
        prepared_artifacts={},
    )

    assert best_patch is None
    assert summary["status"] == "not_evaluable"
    assert summary["reason"] == "all_trial_objectives_non_finite"
    assert summary["best_index"] is None
    assert summary["best_score"] is None
    assert (tmp_path / summary["trials_path"]).is_file()
