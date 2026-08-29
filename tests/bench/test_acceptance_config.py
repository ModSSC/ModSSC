from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from bench.schema import BenchConfigError, ExperimentConfig
from bench.seed_sweep import apply_global_seed
from modssc.runtime.protocol import protocol_sha256


def _config() -> dict[str, Any]:
    return {
        "run": {
            "name": "paper-replication",
            "seed": 1,
            "seeds": [1, 2],
            "output_dir": "runs",
            "fail_fast": True,
            "benchmark_mode": True,
        },
        "dataset": {"id": "toy"},
        "sampling": {"seed": 1, "plan": {"split": {"kind": "holdout"}}},
        "preprocess": {
            "seed": 1,
            "fit_on": "train_labeled",
            "cache": False,
            "plan": {"steps": [{"id": "core.to_numpy"}]},
        },
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "cpu", "dtype": "float32"},
            "params": {},
        },
        "evaluation": {
            "split_for_model_selection": "val",
            "report_splits": ["test"],
            "metrics": ["accuracy"],
        },
        "acceptance": {
            "protocol_id": "paper-table-1",
            "method_id": "pseudo_label",
            "repetitions": 2,
            "fidelity_ceiling": "paper_matched",
            "conformity": {
                "status": "passed",
                "basis": "native implementation review",
                "evidence": ["tests/evaluation/test_acceptance.py"],
                "review": {
                    "reviewed_by": "test-suite",
                    "reviewed_at": "2026-08-29T00:00:00+00:00",
                },
            },
            "target": {
                "split": "test",
                "metric": "accuracy",
                "published_mean": 0.9,
                "margin_absolute": 0.1,
            },
        },
    }


def test_acceptance_is_parsed_with_matching_method_and_repetitions() -> None:
    cfg = ExperimentConfig.from_dict(_config())

    assert cfg.acceptance is not None
    assert cfg.acceptance.protocol_id == "paper-table-1"
    assert cfg.acceptance.method_id == cfg.method.method_id
    assert cfg.acceptance.repetitions == len(cfg.run.seeds or [])


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda raw: raw["run"].pop("seeds"), "E_BENCH_ACCEPTANCE_REPETITIONS_MISMATCH"),
        (
            lambda raw: raw["run"].update({"seeds": [1]}),
            "E_BENCH_ACCEPTANCE_REPETITIONS_MISMATCH",
        ),
        (
            lambda raw: raw["run"].update({"benchmark_mode": False}),
            "E_BENCH_ACCEPTANCE_STRICT_REQUIRED",
        ),
        (
            lambda raw: raw["acceptance"].update({"method_id": "tri_training"}),
            "E_BENCH_ACCEPTANCE_METHOD_MISMATCH",
        ),
    ],
)
def test_acceptance_rejects_ambiguous_card_binding(mutate: Any, code: str) -> None:
    raw = _config()
    mutate(raw)

    with pytest.raises(BenchConfigError) as raised:
        ExperimentConfig.from_dict(raw)

    assert raised.value.code == code


def test_seed_resolution_preserves_acceptance_but_requires_explicit_parser_mode() -> None:
    raw = _config()
    resolved = apply_global_seed(raw, seed=2, run_name="paper-replication-seed2")

    assert "seeds" not in resolved["run"]
    assert resolved["acceptance"] == raw["acceptance"]
    with pytest.raises(BenchConfigError) as raised:
        ExperimentConfig.from_dict(resolved)
    assert raised.value.code == "E_BENCH_ACCEPTANCE_REPETITIONS_MISMATCH"

    cfg = ExperimentConfig.from_dict(
        deepcopy(resolved),
        allow_resolved_acceptance_seed=True,
    )
    assert cfg.run.seed == 2
    assert cfg.acceptance is not None
    assert cfg.acceptance.repetitions == 2


def test_acceptance_spec_is_bound_into_the_seed_protocol_identity() -> None:
    first = apply_global_seed(_config(), seed=1)
    second = deepcopy(first)
    second["acceptance"]["target"]["margin_absolute"] = 0.2

    assert protocol_sha256(first) != protocol_sha256(second)
