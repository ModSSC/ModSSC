from __future__ import annotations

from types import SimpleNamespace

import pytest

from bench.errors import BenchRuntimeError
from bench.orchestrators import method_inductive
from bench.schema import DeviceConfig, MethodConfig, ModelConfig
from modssc.inductive import (
    InductiveExecutionError,
    InductiveExecutionInput,
    InductiveExecutionResult,
)


def _config() -> MethodConfig:
    return MethodConfig(
        kind="inductive",
        method_id="capture",
        device=DeviceConfig(device="cpu", dtype="float32"),
        params={"backend": "numpy"},
        model=ModelConfig(
            classifier_id="logreg",
            classifier_backend="sklearn",
            classifier_params={"max_iter": 3},
        ),
    )


def test_bench_inductive_runner_only_adapts_to_native_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fitted = object()

    def fake_execute(inputs, config):
        captured["inputs"] = inputs
        captured["config"] = config
        return InductiveExecutionResult(
            method=fitted,
            data=SimpleNamespace(),
            resolution={"backend": "numpy"},
        )

    monkeypatch.setattr(method_inductive, "execute_inductive_method", fake_execute)
    preprocess = SimpleNamespace(name="preprocess")
    sampling = SimpleNamespace(name="sampling")
    views = SimpleNamespace(name="views")
    online = SimpleNamespace(seed=19)
    context = SimpleNamespace(name="execution-context")

    inputs = InductiveExecutionInput(
        preprocess=preprocess,
        sampling=sampling,
        views=views,
        X_u_w="weak",
        X_u_s="strong",
        X_u_s_1="strong-1",
        online_augmentation=online,
        execution_context=context,
    )
    method, resolution = method_inductive.run(
        inputs,
        during_fit_splits=["test"],
        cfg=_config(),
        seed=7,
        strict=True,
        requires_torch=True,
    )

    assert method is fitted
    assert resolution == {"backend": "numpy"}
    inputs = captured["inputs"]
    assert inputs.preprocess is preprocess
    assert inputs.sampling is sampling
    assert inputs.views is views
    assert inputs.X_u_w == "weak"
    assert inputs.X_u_s == "strong"
    assert inputs.X_u_s_1 == "strong-1"
    assert inputs.online_augmentation is online
    assert inputs.execution_context is context
    config = captured["config"]
    assert config.method_id == "capture"
    assert config.params == {"backend": "numpy"}
    assert config.model.classifier_id == "logreg"
    assert config.model.classifier_params == {"max_iter": 3}
    assert config.device.device == "cpu"
    assert config.seed == 7
    assert config.strict is True
    assert config.requires_torch is True
    assert config.during_fit_splits == ("test",)


@pytest.mark.parametrize(
    ("kind", "expected_code"),
    [
        ("graph_sampling", "E_BENCH_GRAPH_SAMPLING_INVALID"),
        ("auto_backend", "E_BENCH_AUTO_FORBIDDEN"),
        ("dependency_missing", "E_BENCH_DEPENDENCY_MISSING"),
        ("labels_contract", "E_BENCH_LABELS_CONTRACT"),
        ("evaluation_split", "E_BENCH_EVAL_SPLIT_INVALID"),
        ("torch_required", "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED"),
        ("shape", "E_BENCH_SHAPE_CONTRACT"),
        ("dtype", "E_BENCH_DTYPE_CONTRACT"),
        ("method_contract", "E_BENCH_METHOD_CONTRACT"),
        ("method_introspection", "E_BENCH_METHOD_INTROSPECTION"),
        ("method_spec", "E_BENCH_METHOD_SPEC"),
        ("model_config", "E_BENCH_MODEL_CONFIG"),
        ("capability", "E_BENCH_CAPABILITY"),
        ("graph_contract", "E_BENCH_GRAPH_CONTRACT"),
    ],
)
def test_bench_inductive_runner_translates_native_errors_generically(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    expected_code: str,
) -> None:
    def fail(_inputs, _config):
        raise InductiveExecutionError(kind, "native failure")

    monkeypatch.setattr(method_inductive, "execute_inductive_method", fail)

    with pytest.raises(BenchRuntimeError) as caught:
        method_inductive.run(
            InductiveExecutionInput(
                preprocess=SimpleNamespace(),
                sampling=SimpleNamespace(),
            ),
            cfg=_config(),
            seed=0,
        )
    assert caught.value.code == expected_code
    assert caught.value.__cause__.kind == kind
