from __future__ import annotations

import json
from pathlib import Path

from bench.context import RunContext
from bench.orchestrators.reporting import write_run_summary
from bench.schema import ExperimentConfig


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "run": {
                "name": "reporting_info",
                "seed": 1,
                "output_dir": str(tmp_path),
                "fail_fast": True,
            },
            "limits": {"profile": "auto"},
            "dataset": {"id": "ag_news", "options": {"class_filter": None}},
            "sampling": {"seed": 1, "plan": {"split": {"kind": "holdout"}}},
            "preprocess": {
                "seed": 1,
                "fit_on": "train_labeled",
                "cache": True,
                "plan": {"output_key": "features.X", "steps": [{"id": "labels.encode"}]},
            },
            "graph": {
                "enabled": True,
                "seed": 1,
                "cache": True,
                "spec": {"scheme": "knn", "metric": "cosine", "k": 10},
            },
            "method": {
                "kind": "transductive",
                "id": "poisson_learning",
                "device": {"device": "auto", "dtype": "float32"},
                "params": {"backend": "numpy"},
            },
            "evaluation": {
                "split_for_model_selection": "val",
                "report_splits": ["val", "test"],
                "metrics": ["accuracy"],
            },
        }
    )


def test_write_run_summary_includes_task_graph_and_runtime_info(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ctx = RunContext.from_run_config(
        name="reporting_info",
        seed=1,
        run_id="abc",
        output_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        fail_fast=True,
    )
    ctx.ensure_dirs()
    artifacts = {
        "method": {
            "device": {"requested": "auto", "resolved": "cpu", "dtype": "float32"},
        },
        "sampling": {
            "stats": {
                "train_labeled": {"classes": {"0": 1, "1": 1, "2": 1, "3": 1}},
                "train": {"classes": {"0": 10, "1": 10, "2": 10, "3": 10}},
                "test": {"classes": {"0": 5, "1": 5, "2": 5, "3": 5}},
            }
        },
        "graph": {
            "info": {
                "n_nodes": 60,
                "n_edges": 600,
                "k": 10,
                "metric": "cosine",
                "connected_components": 1,
                "largest_component_fraction": 1.0,
            }
        },
    }

    write_run_summary(
        ctx=ctx,
        cfg=cfg,
        artifacts=artifacts,
        metrics={"test": {"accuracy": 0.25}},
        hpo=None,
        status="success",
        hashes={"config_hash": "c", "effective_config_hash": "e"},
        resolution={
            "device": {"requested": "auto", "resolved": "cpu"},
            "backend": {"requested": {}, "resolved": {}},
            "dtype": {"requested": {}, "resolved": {}},
            "normalization": {"requested": {}, "resolved": {}},
            "splits": {"requested": ["test"], "resolved": {}},
            "limits": {"requested": None, "resolved": None, "changes": []},
        },
        protocol={
            "kind": "transductive",
            "use_test_split": True,
            "report_splits": ["test"],
            "split_for_model_selection": "val",
        },
        versions={"python": "x", "modssc": "x", "numpy": "x", "git_sha": "x"},
        fallback_events=[],
    )

    payload = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["run_info"]["device_requested"] == "auto"
    assert payload["run_info"]["device_resolved"] == "cpu"
    assert payload["task_info"]["n_classes"] == 4
    assert payload["task_info"]["class_filter"] is None
    assert payload["task_info"]["train_labeled_per_class"] == 1
    assert payload["graph_info"]["connected_components"] == 1
