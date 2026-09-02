from __future__ import annotations

from types import SimpleNamespace

import pytest

from bench import main as bench_main
from bench.errors import BenchRuntimeError, extract_error_code
from bench.schema import BenchConfigError
from modssc.data_loader import DatasetSelectionError


def test_extract_error_code_preserves_native_brick_code() -> None:
    error = DatasetSelectionError("outside population", code="E_DATA_SELECTION_INDEX_BOUNDS")

    assert extract_error_code(error) == "E_DATA_SELECTION_INDEX_BOUNDS"


def test_extract_error_code_preserves_runner_code() -> None:
    error = BenchRuntimeError("E_BENCH_CONFIG", "invalid")

    assert extract_error_code(error) == "E_BENCH_CONFIG"


def test_extract_error_code_rejects_unstructured_external_code() -> None:
    error = type("ExternalFailure", (RuntimeError,), {"code": "not-a-code"})("failed")

    assert extract_error_code(error) == "E_BENCH_RUNTIME"


def test_dependency_preflight_preserves_specific_bench_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(id="toy"),
        method=SimpleNamespace(model=None),
        graph=None,
        run=SimpleNamespace(software_dependencies=[]),
    )
    preprocess_plan = SimpleNamespace(enabled_step_ids=lambda: ())
    method = SimpleNamespace(
        required_extra="vision",
        required_extras=("inductive-torch", "supervised-torch"),
    )
    captured: dict[str, object] = {}

    def resolve(request):
        captured["request"] = request
        return SimpleNamespace(extras=("vision",))

    monkeypatch.setattr(
        bench_main,
        "resolve_pipeline_dependencies",
        resolve,
    )

    def missing(_extra: str) -> None:
        raise BenchConfigError("missing dependency", code="E_BENCH_DEPENDENCY_MISSING")

    monkeypatch.setattr(bench_main, "_check_extra", missing)

    with pytest.raises(BenchConfigError) as caught:
        bench_main._required_software_distributions(
            cfg=cfg,
            preprocess_plan=preprocess_plan,
            views_plan=None,
            method=method,
        )

    assert caught.value.code == "E_BENCH_DEPENDENCY_MISSING"
    request = captured["request"]
    assert request.method_required_extra == "vision"
    assert request.method_required_extras == ("inductive-torch", "supervised-torch")


def test_method_runtime_bridge_forwards_model_classifier_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(
        method=SimpleNamespace(
            kind="inductive",
            method_id="fixmatch",
            params={"threshold": 0.95},
            device=SimpleNamespace(device="cuda", dtype="float16"),
            model=SimpleNamespace(
                classifier_id="wide_resnet_cifar",
                classifier_backend="torch",
            ),
        ),
        run=SimpleNamespace(benchmark_mode=True),
    )
    captured: dict[str, object] = {}
    expected = object()

    def resolve(request):
        captured["request"] = request
        return expected

    monkeypatch.setattr(bench_main, "resolve_method", resolve)

    result = bench_main._resolve_method_runtime(
        cfg,
        preprocess_steps=["core.ensure_2d", "core.to_torch"],
    )

    assert result is expected
    request = captured["request"]
    assert request.model_classifier_id == "wide_resnet_cifar"
    assert request.model_classifier_backend == "torch"
    assert request.model_configured is True
    assert request.preprocess_step_ids == ("core.ensure_2d", "core.to_torch")
    assert request.strict is True
