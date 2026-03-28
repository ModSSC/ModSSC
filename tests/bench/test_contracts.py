from __future__ import annotations

from pathlib import Path

import pytest

from bench.schema import BenchConfigError, ExperimentConfig
from bench.utils.import_tools import check_extra_installed


def _base_config() -> dict:
    return {
        "run": {
            "name": "test",
            "seed": 0,
            "output_dir": "runs",
            "fail_fast": True,
        },
        "dataset": {"id": "toy"},
        "sampling": {
            "plan": {
                "split": {"kind": "holdout", "test_fraction": 0.0, "val_fraction": 0.2},
                "labeling": {"mode": "fraction", "value": 0.2},
            }
        },
        "preprocess": {
            "cache": True,
            "plan": {
                "output_key": "features.X",
                "steps": [{"id": "core.ensure_2d"}, {"id": "core.to_numpy"}],
            },
        },
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "cpu", "dtype": "float32"},
            "params": {"classifier_id": "knn", "classifier_backend": "numpy"},
        },
        "evaluation": {
            "report_splits": ["val"],
            "metrics": ["accuracy"],
        },
    }


def test_check_extra_installed_rejects_unknown_extra() -> None:
    with pytest.raises(ValueError, match="Unknown optional dependency extra"):
        check_extra_installed("does-not-exist", pyproject_path=Path("pyproject.toml"))


def test_schema_rejects_custom_factory_without_explicit_opt_in() -> None:
    cfg = _base_config()
    cfg["method"]["model"] = {"factory": "math:sqrt", "params": {}}

    with pytest.raises(BenchConfigError, match="allow_custom_factories"):
        ExperimentConfig.from_dict(cfg)


def test_schema_accepts_custom_factory_with_explicit_opt_in() -> None:
    cfg = _base_config()
    cfg["run"]["allow_custom_factories"] = True
    cfg["method"]["model"] = {"factory": "math:sqrt", "params": {}}

    parsed = ExperimentConfig.from_dict(cfg)
    assert parsed.run.allow_custom_factories is True
    assert parsed.method.model is not None
    assert parsed.method.model.factory == "math:sqrt"
