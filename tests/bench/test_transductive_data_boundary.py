from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bench.errors import BenchRuntimeError
from bench.orchestrators import method_transductive
from bench.schema import DeviceConfig, MethodConfig
from modssc.transductive import (
    TransductiveExecutionError,
    TransductiveExecutionInput,
    TransductiveExecutionResult,
)


def _config() -> MethodConfig:
    return MethodConfig(
        kind="transductive",
        method_id="capture",
        device=DeviceConfig(device="cpu", dtype="float32"),
        params={"backend": "numpy"},
    )


def test_bench_transductive_runner_only_adapts_to_native_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fitted = object()
    fit_data = SimpleNamespace(
        X=np.zeros((4, 2), dtype=np.float32),
        masks={
            "labeled_mask": np.array([True, True, False, False]),
            "train_all_mask": np.array([True, True, True, False]),
        },
    )
    prepared = SimpleNamespace(fit=fit_data)

    def fake_execute(inputs, config):
        captured["inputs"] = inputs
        captured["config"] = config
        return TransductiveExecutionResult(
            method=fitted,
            data=prepared,
            resolution={"backend": "numpy", "resolved_device": "cpu"},
        )

    monkeypatch.setattr(method_transductive, "execute_transductive_method", fake_execute)
    dataset = SimpleNamespace(train=SimpleNamespace(X=np.zeros((4, 2))))
    graph = SimpleNamespace(name="graph")
    masks = {"labeled_mask": np.array([True, True, False, False])}

    method, data, resolution = method_transductive.run(
        TransductiveExecutionInput(dataset=dataset, graph=graph, masks=masks),
        cfg=_config(),
        seed=7,
        use_test_split=True,
        expected_labeled_count=2,
        strict=True,
    )

    assert method is fitted
    assert data is prepared
    assert resolution == {"backend": "numpy", "resolved_device": "cpu"}
    inputs = captured["inputs"]
    assert inputs.dataset is dataset
    assert inputs.graph is graph
    assert inputs.masks == masks
    config = captured["config"]
    assert config.method_id == "capture"
    assert config.params == {"backend": "numpy"}
    assert config.device.device == "cpu"
    assert config.device.dtype == "float32"
    assert config.seed == 7
    assert config.strict is True
    assert config.use_test_split is True
    assert config.expected_labeled_count == 2


@pytest.mark.parametrize(
    ("kind", "expected_code"),
    [
        ("auto_backend", "E_BENCH_AUTO_FORBIDDEN"),
        ("dependency_missing", "E_BENCH_DEPENDENCY_MISSING"),
        ("method_contract", "E_BENCH_METHOD_CONTRACT"),
        ("method_introspection", "E_BENCH_METHOD_INTROSPECTION"),
        ("method_spec", "E_BENCH_METHOD_SPEC"),
        ("data_contract", "E_BENCH_MASK_CONTRACT"),
        ("augmentation_contract", "E_BENCH_AUGMENTATION_UNSUPPORTED"),
        ("capability", "E_BENCH_CAPABILITY"),
    ],
)
def test_bench_transductive_runner_translates_native_errors_generically(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    expected_code: str,
) -> None:
    def fail(_inputs, _config):
        raise TransductiveExecutionError(kind, "native failure")

    monkeypatch.setattr(method_transductive, "execute_transductive_method", fail)

    with pytest.raises(BenchRuntimeError) as caught:
        method_transductive.run(
            TransductiveExecutionInput(
                dataset=SimpleNamespace(train=SimpleNamespace(X=np.zeros((1, 1)))),
                graph=SimpleNamespace(),
                masks={},
            ),
            cfg=_config(),
            seed=0,
            use_test_split=False,
        )

    assert caught.value.code == expected_code
