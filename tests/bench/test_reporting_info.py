from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bench.context import RunContext
from bench.errors import BenchRuntimeError
from bench.orchestrators import reporting
from bench.orchestrators.reporting import write_run_summary
from bench.report_schema import validate_run_payload
from bench.schema import ExperimentConfig
from modssc.runtime.execution import RunIdentity
from modssc.runtime.protocol import effective_config_sha256, protocol_sha256
from modssc.runtime.software import software_sha256


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


def _report_identity(
    effective_config: dict[str, Any],
    versions: dict[str, Any],
) -> dict[str, str]:
    identity = _execution_identity(effective_config, versions)
    return {
        "config_hash": "a" * 64,
        "effective_config_hash": effective_config_sha256(effective_config),
        "protocol_sha256": protocol_sha256(effective_config),
        "software_sha256": software_sha256(versions),
        "execution_identity_sha256": identity.sha256,
    }


def _execution_identity(
    effective_config: dict[str, Any],
    versions: dict[str, Any],
) -> RunIdentity:
    return RunIdentity(
        config_sha256=protocol_sha256(effective_config),
        seed=1,
        code_sha256=software_sha256(versions),
    )


def _write_minimal_resource_summary(
    tmp_path: Path,
    *,
    name: str,
    resource_measurement: reporting.RunResourceMeasurement,
) -> Path:
    cfg = _cfg(tmp_path)
    effective_config = {"run": {"seed": 1}, "method": {"params": {}}}
    versions = {"python": "x", "modssc": "x", "numpy": "x", "git_sha": "x"}
    identity = _execution_identity(effective_config, versions)
    ctx = RunContext.from_run_config(
        name=name,
        seed=1,
        run_id=identity.short_id,
        output_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        fail_fast=True,
    )
    ctx.ensure_dirs()
    write_run_summary(
        ctx=ctx,
        cfg=cfg,
        artifacts={"method": {"device": {"requested": "auto", "resolved": "cpu"}}},
        metrics=None,
        hpo=None,
        status="success",
        hashes=_report_identity(effective_config, versions),
        execution_identity=identity.to_dict(),
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
            "test_selection_policy": "forbid",
        },
        versions=versions,
        effective_config=effective_config,
        fallback_events=[],
        resource_measurement=resource_measurement,
    )
    return ctx.run_dir / "run.json"


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
    effective_config = {
        "run": {"seed": 1},
        "method": {"kind": "transductive", "params": {"backend": "numpy"}},
    }
    versions = {
        "python": "x",
        "modssc": "x",
        "numpy": "x",
        "git_sha": "x",
        "git_dirty": True,
        "git_diff_sha256": "a" * 64,
    }
    identity = _execution_identity(effective_config, versions)
    ctx.run_id = identity.short_id

    write_run_summary(
        ctx=ctx,
        cfg=cfg,
        artifacts=artifacts,
        metrics={"test": {"accuracy": 0.25}},
        hpo=None,
        status="success",
        hashes=_report_identity(effective_config, versions),
        execution_identity=identity.to_dict(),
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
            "test_selection_policy": "forbid",
        },
        versions=versions,
        effective_config=effective_config,
        fallback_events=[],
        resource_measurement=reporting.begin_run_resource_measurement(),
    )

    payload = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["run_info"]["device_requested"] == "auto"
    assert payload["run_info"]["device_resolved"] == "cpu"
    assert payload["task_info"]["n_classes"] == 4
    assert payload["task_info"]["class_filter"] is None
    assert payload["task_info"]["train_labeled_per_class"] == 1
    assert payload["graph_info"]["connected_components"] == 1
    assert payload["versions"]["git_dirty"] is True
    assert payload["versions"]["git_diff_sha256"] == "a" * 64
    assert payload["config"] == effective_config
    assert payload["hashes"]["effective_config_hash"] == effective_config_sha256(payload["config"])
    assert payload["hashes"]["protocol_sha256"] == protocol_sha256(payload["config"])
    assert payload["hashes"]["software_sha256"] == software_sha256(payload["versions"])
    assert payload["execution_identity"] == identity.to_dict()
    assert payload["hashes"]["execution_identity_sha256"] == identity.sha256
    assert payload["run"]["run_id"] == identity.short_id
    assert payload["protocol"]["test_selection_policy"] == "forbid"

    legacy_protocol_payload = json.loads(json.dumps(payload))
    legacy_protocol_payload["protocol"].pop("test_selection_policy")
    validate_run_payload(legacy_protocol_payload)

    invalid_protocol_payload = json.loads(json.dumps(payload))
    invalid_protocol_payload["protocol"]["test_selection_policy"] = "allow"
    with pytest.raises(BenchRuntimeError, match="test_selection_policy"):
        validate_run_payload(invalid_protocol_payload)

    legacy_payload = json.loads(json.dumps(payload))
    legacy_payload["versions"].pop("git_dirty")
    legacy_payload["versions"].pop("git_diff_sha256")
    validate_run_payload(legacy_payload)

    legacy_identity_payload = json.loads(json.dumps(payload))
    legacy_identity_payload.pop("execution_identity")
    legacy_identity_payload["hashes"].pop("execution_identity_sha256")
    validate_run_payload(
        legacy_identity_payload,
        require_execution_identity=False,
    )

    with pytest.raises(BenchRuntimeError, match="required for a modern run report"):
        validate_run_payload(legacy_identity_payload)

    partial_identity_payload = json.loads(json.dumps(payload))
    partial_identity_payload.pop("execution_identity")
    with pytest.raises(BenchRuntimeError, match="must be present together"):
        validate_run_payload(partial_identity_payload)

    partial_payload = json.loads(json.dumps(payload))
    partial_payload["versions"].pop("git_diff_sha256")
    with pytest.raises(BenchRuntimeError):
        validate_run_payload(partial_payload)

    invalid_payload = json.loads(json.dumps(payload))
    invalid_payload["versions"]["git_diff_sha256"] = "not-a-sha256"
    with pytest.raises(BenchRuntimeError):
        validate_run_payload(invalid_payload)

    tampered_identity = json.loads(json.dumps(payload))
    tampered_identity["execution_identity"]["seed"] += 1
    with pytest.raises(BenchRuntimeError, match="does not match"):
        validate_run_payload(tampered_identity)

    tampered_run_id = json.loads(json.dumps(payload))
    tampered_run_id["run"]["run_id"] = "host-dependent-id"
    with pytest.raises(BenchRuntimeError, match="portable execution identity"):
        validate_run_payload(tampered_run_id)


def test_run_summary_normalizes_linux_rss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rss_values = iter([2048, 3072])
    monkeypatch.setattr(reporting.sys, "platform", "linux")
    monkeypatch.setattr(
        reporting.resource,
        "getrusage",
        lambda _scope: SimpleNamespace(ru_maxrss=next(rss_values)),
    )
    monkeypatch.setattr(reporting, "_load_torch_safely", lambda: None)

    measurement = reporting.begin_run_resource_measurement()
    run_json = _write_minimal_resource_summary(
        tmp_path,
        name="reporting_linux_memory",
        resource_measurement=measurement,
    )

    payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert payload["run_info"]["peak_ram_bytes"] == 3072 * 1024
    usage = payload["run_info"]["resource_usage"]
    assert usage["peak_ram_at_start_bytes"] == 2048 * 1024
    assert usage["peak_ram_native_unit"] == "kibibytes"
    assert usage["cuda_measurement_status"] == "torch_unavailable"
    assert "peak_vram_bytes" not in payload["run_info"]


def test_peak_ram_normalization_uses_bytes_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporting.sys, "platform", "darwin")
    monkeypatch.setattr(
        reporting.resource,
        "getrusage",
        lambda _scope: SimpleNamespace(ru_maxrss=8192),
    )

    assert reporting._peak_ram_bytes() == (8192, "bytes")


class _FakeCuda:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 2

    @staticmethod
    def get_device_name(device_index: int) -> str:
        return f"Fake CUDA {device_index}"

    def reset_peak_memory_stats(self, device_index: int) -> None:
        self.events.append(("reset", device_index))

    def synchronize(self, device_index: int) -> None:
        self.events.append(("synchronize", device_index))

    def max_memory_allocated(self, device_index: int) -> int:
        self.events.append(("allocated", device_index))
        return (device_index + 1) * 1024

    def max_memory_reserved(self, device_index: int) -> int:
        self.events.append(("reserved", device_index))
        return (device_index + 1) * 2048


def test_cuda_peaks_are_reset_synchronized_and_report_reserved_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda = _FakeCuda()
    fake_torch: Any = SimpleNamespace(cuda=cuda)
    monkeypatch.setattr(reporting, "_load_torch_safely", lambda: fake_torch)
    monkeypatch.setattr(reporting, "_peak_ram_bytes", lambda: (4096, "kibibytes"))

    measurement = reporting.begin_run_resource_measurement()
    assert cuda.events == [("reset", 0), ("reset", 1)]

    usage = reporting.collect_run_resource_usage(measurement)

    assert usage["cuda_measurement_status"] == "ok"
    assert usage["max_gpu_memory_allocated_bytes"] == 2048
    assert usage["max_gpu_memory_reserved_bytes"] == 4096
    assert usage["peak_vram_bytes"] == 4096
    assert cuda.events == [
        ("reset", 0),
        ("reset", 1),
        ("synchronize", 0),
        ("allocated", 0),
        ("reserved", 0),
        ("synchronize", 1),
        ("allocated", 1),
        ("reserved", 1),
    ]

    run_json = _write_minimal_resource_summary(
        tmp_path,
        name="reporting_cuda_memory",
        resource_measurement=measurement,
    )
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert payload["run_info"]["peak_vram_bytes"] == 4096


def test_cuda_unavailable_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(
        reporting,
        "_load_torch_safely",
        lambda: SimpleNamespace(cuda=fake_cuda),
    )
    monkeypatch.setattr(reporting, "_peak_ram_bytes", lambda: (1024, "kibibytes"))

    measurement = reporting.begin_run_resource_measurement()
    usage = reporting.collect_run_resource_usage(measurement)

    assert usage["cuda_measurement_status"] == "cuda_unavailable"
    assert usage["cuda_counter_reset"] is False
    assert "peak_vram_bytes" not in usage
