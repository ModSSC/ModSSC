from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import modssc.inductive.model_binding as binding


@dataclass(frozen=True)
class _Spec:
    bundle: object | None = None
    shared: object | None = None
    heads: tuple[object, ...] | None = None


def test_model_binding_tensor_conversion_and_dtype_contracts() -> None:
    torch = pytest.importorskip("torch")
    assert binding._infer_num_classes(np.array([2, 2, 5])) == 2

    with pytest.raises(binding.ModelBindingError, match="requires a torch sample"):
        binding._as_torch_sample(np.ones((2, 2)), strict=True)
    uint8 = binding._as_torch_sample(np.array([[255]], dtype=np.uint8), strict=False)
    assert uint8.dtype == torch.float32
    assert uint8.item() == 1.0
    assert binding._as_torch_sample(np.ones((1, 1), dtype=np.float64), strict=False).dtype == (
        torch.float32
    )
    assert binding._as_torch_sample(np.ones((1, 1), dtype=np.int64), strict=False).dtype == (
        torch.int64
    )

    binding._validate_float_sample(torch.ones(1, dtype=torch.int64), strict=False)
    with pytest.raises(binding.ModelBindingError, match="floating tensor"):
        binding._validate_float_sample(torch.ones(1, dtype=torch.int64), strict=True)
    binding._validate_float_sample(torch.ones(1), strict=True)


def test_graph_probe_selects_first_node_and_preserves_metadata() -> None:
    small = {"x": np.ones((1, 2)), "num_nodes": 1}
    assert binding._first_graph_sample(small) == small

    graph = {
        "x": np.arange(6).reshape(3, 2),
        "edge_index": np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64),
        "node_values": np.arange(3),
        "node_names": ["a", "b", "c"],
        "num_nodes": 3,
        "meta": "kept",
    }
    selected = binding._first_graph_sample(graph)
    np.testing.assert_array_equal(selected["x"], graph["x"][:1])
    np.testing.assert_array_equal(selected["edge_index"], [[0], [0]])
    np.testing.assert_array_equal(selected["node_values"], [0])
    assert selected["node_names"] == ["a"]
    assert selected["num_nodes"] == 1
    assert selected["meta"] == "kept"

    opaque_edges = {"x": np.ones((2, 1)), "edge_index": "opaque", "num_nodes": 2}
    assert binding._first_graph_sample(opaque_edges)["edge_index"] == "opaque"


def test_shared_probe_and_head_extraction_cover_supported_protocols() -> None:
    torch = pytest.importorskip("torch")
    sample = torch.ones((2, 3))
    assert binding._shared_probe_sample(sample).shape == (1, 3)
    assert binding._shared_probe_sample(sample[:1]) is not None
    graph = {"x": torch.ones((2, 3)), "edge_index": torch.empty((2, 0), dtype=torch.long)}
    assert binding._shared_probe_sample(graph)["x"].shape == (1, 3)
    with pytest.raises(binding.ModelBindingError, match="requires torch inputs"):
        binding._shared_probe_sample(np.ones((2, 3)))

    output = torch.ones((1, 2), requires_grad=True)
    assert not binding._extract_head_sample(output).requires_grad
    assert not binding._extract_head_sample({"feat": None, "features": output}).requires_grad
    assert not binding._extract_head_sample((output, "metadata")).requires_grad
    with pytest.raises(binding.ModelBindingError, match="shared model"):
        binding._extract_head_sample({"feat": np.ones((1, 2))})


def test_binding_rejects_invalid_spec_and_model_contracts(monkeypatch) -> None:
    config = binding.ModelBuildConfig(factory=lambda: object())
    spec = _Spec()
    prebound = _Spec(bundle=object())
    assert (
        binding.bind_model_to_spec(
            prebound,
            None,
            binding=binding.ModelBindingSpec.single("bundle"),
            X_l=np.ones((2, 2)),
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )
        is prebound
    )
    with pytest.raises(binding.ModelBindingError, match="requires bound model fields"):
        binding.bind_model_to_spec(
            spec,
            None,
            binding=binding.ModelBindingSpec.single("bundle"),
            X_l=np.ones((2, 2)),
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )

    binding.bind_model_to_spec(
        _Spec(bundle=object()),
        None,
        binding=binding.ModelBindingSpec.pretrain_finetune(
            pretrain_field="bundle",
            finetune_field="shared",
        ),
        X_l=np.ones((2, 2)),
        y_l=np.array([0, 1]),
        default_ema=False,
        seed=0,
    )
    with pytest.raises(binding.ModelBindingError, match="at least one bound model bundle"):
        binding.bind_model_to_spec(
            spec,
            None,
            binding=binding.ModelBindingSpec.pretrain_finetune(
                pretrain_field="bundle",
                finetune_field="shared",
            ),
            X_l=np.ones((2, 2)),
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )

    with pytest.raises(binding.ModelBindingError, match="no dataclass"):
        binding.bind_model_to_spec(
            None,
            config,
            binding=binding.ModelBindingSpec.single("bundle"),
            X_l=np.ones((2, 2)),
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )
    with pytest.raises(binding.ModelBindingError, match="requires a dataclass"):
        binding.bind_model_to_spec(
            SimpleNamespace(bundle=None),
            config,
            binding=binding.ModelBindingSpec.single("bundle"),
            X_l=np.ones((2, 2)),
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )
    with pytest.raises(binding.ModelBindingError, match="does not declare"):
        binding.bind_model_to_spec(
            spec,
            config,
            binding=binding.NO_MODEL_BINDING,
            X_l=np.ones((2, 2)),
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )

    monkeypatch.setattr(binding, "build_torch_bundle_from_classifier", lambda **kwargs: kwargs)
    with pytest.raises(binding.ModelBindingError, match="classifier_id"):
        binding.bind_model_to_spec(
            spec,
            binding.ModelBuildConfig(),
            binding=binding.ModelBindingSpec.single("bundle"),
            X_l={"x": np.ones((2, 2), dtype=np.float32)},
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )


@pytest.mark.parametrize(
    ("model_binding", "message"),
    [
        (
            binding.ModelBindingSpec.shared_heads(
                shared_field="shared",
                heads_field="heads",
                head_count=1,
                head_classifier_ids=("mlp",),
                head_classifier_fallback="mlp",
            ),
            "do not support model.factory",
        ),
        (
            binding.ModelBindingSpec(kind="shared_heads", head_count=1),
            "fields must be declared",
        ),
        (
            binding.ModelBindingSpec(
                kind="shared_heads",
                shared_bundle_field="shared",
                head_bundles_field="heads",
                head_count=0,
                head_classifier_ids=("mlp",),
                head_classifier_fallback="mlp",
            ),
            "head_count",
        ),
        (
            binding.ModelBindingSpec(
                kind="shared_heads",
                shared_bundle_field="shared",
                head_bundles_field="heads",
                head_count=1,
                head_classifier_ids=("mlp",),
                head_classifier_fallback="logreg",
            ),
            "fallback",
        ),
    ],
)
def test_shared_head_binding_rejects_invalid_declarations(model_binding, message: str) -> None:
    factory = (lambda: object()) if "factory" in message else None
    with pytest.raises(binding.ModelBindingError, match=message):
        binding.bind_model_to_spec(
            _Spec(),
            binding.ModelBuildConfig(factory=factory, classifier_id="mlp"),
            binding=model_binding,
            X_l=np.ones((2, 2), dtype=np.float32),
            y_l=np.array([0, 1]),
            default_ema=False,
            seed=0,
        )


def test_shared_head_binding_keeps_declared_head_classifier(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    calls = []

    class Shared:
        def __call__(self, sample):
            return torch.ones((int(sample.shape[0]), 4))

    def build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=Shared() if len(calls) == 1 else object())

    monkeypatch.setattr(binding, "build_torch_bundle_from_classifier", build)
    result = binding.bind_model_to_spec(
        _Spec(),
        binding.ModelBuildConfig(classifier_id="mlp", classifier_backend="torch"),
        binding=binding.ModelBindingSpec.shared_heads(
            shared_field="shared",
            heads_field="heads",
            head_count=1,
            head_classifier_ids=("mlp", "logreg"),
            head_classifier_fallback="logreg",
        ),
        X_l=torch.ones((2, 3)),
        y_l=torch.tensor([0, 1]),
        default_ema=True,
        seed=0,
        strict=True,
    )

    assert result.heads is not None
    assert calls[1]["classifier_id"] == "mlp"
