from __future__ import annotations

import pytest

from modssc.runtime.limits import (
    ResourceLimitError,
    ResourceLimits,
    apply_resource_limits,
    resolve_resource_limits,
)


def test_native_limits_materialize_every_pipeline_scope_without_mutating_input() -> None:
    config = {
        "dataset": {"options": {}},
        "preprocess": {"plan": {"steps": [{"id": "encoder", "params": {"batch_size": 128}}]}},
        "views": {
            "plan": {
                "views": [
                    {
                        "name": "v1",
                        "preprocess": {
                            "steps": [{"id": "view_encoder", "params": {"batch_size": 64}}]
                        },
                    }
                ]
            }
        },
        "method": {
            "params": {"batch_size": 512, "sup_batch_size": 256},
            "model": {"classifier_params": {"batch_size": 1024}},
        },
        "graph": {"spec": {}},
    }
    limits = ResourceLimits(
        max_preprocess_batch_size=32,
        max_method_batch_size=128,
        max_method_sup_batch_size=64,
        max_graph_chunk_size=512,
        max_train_samples=1000,
        max_test_samples=200,
    )

    effective, changes, resolved = apply_resource_limits(config, limits=limits)

    assert config["method"]["params"]["batch_size"] == 512
    assert effective["dataset"]["options"] == {
        "max_train_samples": 1000,
        "max_test_samples": 200,
    }
    assert effective["preprocess"]["plan"]["steps"][0]["params"]["batch_size"] == 32
    assert (
        effective["views"]["plan"]["views"][0]["preprocess"]["steps"][0]["params"]["batch_size"]
        == 32
    )
    assert effective["method"]["params"]["batch_size"] == 128
    assert effective["method"]["params"]["sup_batch_size"] == 64
    assert effective["method"]["model"]["classifier_params"]["batch_size"] == 128
    assert effective["graph"]["spec"]["chunk_size"] == 512
    assert changes
    assert resolved is not None


def test_native_limits_forbid_auto_in_strict_execution() -> None:
    with pytest.raises(ResourceLimitError, match="auto"):
        resolve_resource_limits(ResourceLimits(profile="auto"), strict=True)
